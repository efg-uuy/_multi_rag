import os
import asyncio
import time
import json
import logging
import re
from typing import Dict, Any, List, Union, Optional, Tuple, Type
from abc import ABC, abstractmethod
from PIL import Image
from openai import AsyncOpenAI, APIError, Timeout, RateLimitError
from dotenv import load_dotenv
import torch
from torchvision import models, transforms
from torch.nn import functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
import mmdet
from mmdet.apis import init_detector, inference_detector
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 确保mmdet模型配置
try:
    MMDET_CONFIG = os.getenv("MMDET_CONFIG", "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")
    MMDET_CHECKPOINT = os.getenv("MMDET_CHECKPOINT", "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
    mmdet_model = init_detector(MMDET_CONFIG, MMDET_CHECKPOINT, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    MMDET_AVAILABLE = True
except Exception as e:
    logging.warning(f"mmdet初始化失败，图表区域检测功能将禁用: {str(e)}")
    MMDET_AVAILABLE = False

# 文本提取工具
from .prompts import get_image_analysis_prompt
from .image_analysis_utils import extract_json_content, image_to_base64_async, extract_text_from_pdf


# ===================== 监控指标管理 =====================
class MonitoringMetrics:
    """监控指标管理器"""
    def __init__(self):
        # API调用指标
        self.api_total_calls = 0
        self.api_failed_calls = 0
        # 检索指标
        self.retrieval_total_queries = 0
        self.retrieval_hits = 0
        # 缓存指标
        self.cache_total_requests = 0
        self.cache_hits = 0

    @property
    def api_failure_rate(self) -> float:
        """计算API失败率"""
        return self.api_failed_calls / self.api_total_calls if self.api_total_calls > 0 else 0.0

    @property
    def retrieval_hit_rate(self) -> float:
        """计算检索命中率"""
        return self.retrieval_hits / self.retrieval_total_queries if self.retrieval_total_queries > 0 else 0.0

    @property
    def cache_usage_rate(self) -> float:
        """计算缓存使用率"""
        return self.cache_hits / self.cache_total_requests if self.cache_total_requests > 0 else 0.0

    def reset_metrics(self) -> None:
        """重置所有指标"""
        self.api_total_calls = 0
        self.api_failed_calls = 0
        self.retrieval_total_queries = 0
        self.retrieval_hits = 0
        self.cache_total_requests = 0
        self.cache_hits = 0

    def __str__(self) -> str:
        """返回指标字符串表示"""
        return (f"API失败率: {self.api_failure_rate:.2%}, "
                f"检索命中率: {self.retrieval_hit_rate:.2%}, "
                f"缓存使用率: {self.cache_usage_rate:.2%}")


# ===================== 抽象基类定义 =====================
class BaseMultiModalAnalyzer(ABC):
    """多模态分析器抽象基类"""
    def __init__(self):
        self.metrics = MonitoringMetrics()

    @abstractmethod
    async def analyze_image(
            self,
            image_url: str = None,
            local_image_path: str = None,
            model: str = None,
            detail: str = "low",
            prompt: str = None,
            temperature: float = 0.1,
            text_context: Optional[str] = None,
            pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """分析图像并返回描述信息"""
        pass

    async def close(self):
        """关闭资源"""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ===================== OpenAI兼容多模态分析器 =====================
class OpenAIMultiModalAnalyzer(BaseMultiModalAnalyzer):
    """OpenAI兼容的多模态分析器实现"""
    def __init__(
            self,
            api_key: str,
            base_url: str,
            default_model: str,
            max_retries: int = 3
    ):
        super().__init__()
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.default_model = default_model
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, Timeout, ConnectionError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def analyze_image(
            self,
            image_url: str = None,
            local_image_path: str = None,
            model: str = None,
            detail: str = "low",
            prompt: str = None,
            temperature: float = 0.1,
            text_context: Optional[str] = None,
            pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """带重试机制的图像分析方法"""
        self.metrics.api_total_calls += 1  # 增加总调用次数
        model = model or self.default_model
        prompt = prompt or get_image_analysis_prompt()

        # 处理图像输入（本地路径转base64或使用URL）
        image_content = None
        if local_image_path:
            try:
                base64_img = await image_to_base64_async(local_image_path)
                image_content = f"data:image/jpeg;base64,{base64_img}"
            except Exception as e:
                logger.error(f"图像转换失败: {str(e)}")
                self.metrics.api_failed_calls += 1
                raise ValueError(f"无法处理图像路径: {local_image_path}") from e
        elif image_url:
            image_content = image_url
        else:
            self.metrics.api_failed_calls += 1
            raise ValueError("必须提供image_url或local_image_path")

        # 构建消息内容
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_content, "detail": detail}}
                ]
            }
        ]

        # 如果有文本上下文，添加到消息中
        if text_context:
            messages[0]["content"].insert(0, {
                "type": "text",
                "text": f"相关文本上下文: {text_context}"
            })

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            result = extract_json_content(response.choices[0].message.content)
            return {**result, "model_used": model}
        except Exception as e:
            self.metrics.api_failed_calls += 1  # 增加失败次数
            logger.error(f"API调用失败: {str(e)}")
            raise  # 让重试机制处理


# ===================== 图表类型校验工具 =====================
class ChartTypeClassifier:
    """图表类型分类器（基于微调ResNet-18）"""

    def __init__(self, model_weight_path: Optional[str] = None):
        self.classifier = None
        self.transform = None
        self.type_mapping = {
            0: "柱状图",
            1: "折线图",
            2: "饼图",
            3: "表格",
            4: "其他"
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model(model_weight_path)

    def _initialize_model(self, model_weight_path: Optional[str]) -> None:
        """初始化分类模型（带错误处理）"""
        try:
            # 初始化ResNet-18
            self.classifier = models.resnet18(pretrained=False)
            num_ftrs = self.classifier.fc.in_features
            self.classifier.fc = torch.nn.Linear(num_ftrs, 5)  # 5种图表类型

            # 加载微调权重
            if model_weight_path and os.path.exists(model_weight_path):
                self.classifier.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            else:
                logging.warning("未提供图表分类模型权重，使用随机初始化模型（效果有限）")

            self.classifier.eval()
            self.classifier.to(self.device)

            # 图像预处理管道
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            logger.info("图表分类器初始化成功")

        except Exception as e:
            logging.error(f"图表分类器初始化失败: {str(e)}")
            self.classifier = None
            self.transform = None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((IOError, RuntimeError)),
        before_sleep=before_sleep_log(logger, logging.DEBUG)
    )
    def predict(self, image_path: str) -> Optional[str]:
        """带重试机制的图表类型预测"""
        if not self.classifier or not self.transform:
            logger.warning("分类器未初始化，无法进行预测")
            return None

        try:
            with Image.open(image_path).convert("RGB") as img:
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.classifier(tensor)
                    pred_idx = F.softmax(output, dim=1).argmax().item()
                return self.type_mapping.get(pred_idx, "未知图表类型")
        except Exception as e:
            logger.error(f"图表类型预测失败: {str(e)}")
            raise  # 触发重试


# ===================== 审计日志与监控整合 =====================
class AuditLogManager:
    """审计日志管理器（整合监控指标）"""
    def __init__(self, metrics: Optional[MonitoringMetrics] = None):
        self.metrics = metrics or MonitoringMetrics()
        # 其他初始化逻辑（保持原有Redis等配置）
        try:
            import redis
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True
            )
            self.redis_client.ping()
            self.using_redis = True
            logger.info("✅ Redis审计日志连接成功")
        except Exception as e:
            logger.warning(f"⚠️ Redis连接失败，使用内存模拟: {str(e)}")
            self.redis_client = {}
            self.using_redis = False

    def log_api_call(self, success: bool) -> None:
        """记录API调用结果并更新指标"""
        self.metrics.api_total_calls += 1
        if not success:
            self.metrics.api_failed_calls += 1

    def log_retrieval(self, hit: bool) -> None:
        """记录检索结果并更新指标"""
        self.metrics.retrieval_total_queries += 1
        if hit:
            self.metrics.retrieval_hits += 1

    def log_cache(self, hit: bool) -> None:
        """记录缓存结果并更新指标"""
        self.metrics.cache_total_requests += 1
        if hit:
            self.metrics.cache_hits += 1

    def export_metrics(self) -> Dict[str, float]:
        """导出指标供监控系统使用"""
        return {
            "api_failure_rate": self.metrics.api_failure_rate,
            "retrieval_hit_rate": self.metrics.retrieval_hit_rate,
            "cache_usage_rate": self.metrics.cache_usage_rate,
            "api_total_calls": self.metrics.api_total_calls,
            "retrieval_total_queries": self.metrics.retrieval_total_queries
        }

    # 保持原有日志相关方法...
    def create_pre_log(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        log_id = os.urandom(16).hex()
        log_data = {
            "log_id": log_id,
            "question": question,
            "status": "processing",
            "timestamp": time.time(),
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {}
        }
        if self.using_redis:
            self.redis_client.setex(f"audit_log:{log_id}", 3600, json.dumps(log_data, ensure_ascii=False))
        else:
            self.redis_client[f"audit_log:{log_id}"] = (log_data, time.time() + 3600)
        return log_id