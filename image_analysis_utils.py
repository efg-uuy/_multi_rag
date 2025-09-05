#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分析工具类 - 包含多模态分析器适配器、自动降级及校验增强逻辑
"""
import json
import os
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import aiofiles
import base64
import logging
from enum import Enum

# 多模态模型依赖
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from PIL import Image

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("未安装transformers相关库，本地模型功能将禁用")

# 计算机视觉校验依赖
try:
    import mmdet
    from mmdet.apis import init_detector, inference_detector
    from mmcv.transforms import Compose

    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    logging.warning("未安装mmdet，图表区域检测功能将禁用")

# API调用依赖
try:
    from openai import AsyncOpenAI
    from zhipuai import AsyncZhipuAI

    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False
    logging.warning("未安装API客户端库，远程API功能将禁用")


# ===================== 配置与常量 =====================
class AnalyzerType(Enum):
    """分析器类型枚举"""
    LOCAL_QWEN_VL = "local_qwen_vl"
    ZHIPU = "zhipu"
    OPENAI = "openai"


DEFAULT_MODEL_PRIORITY = [
    AnalyzerType.LOCAL_QWEN_VL.value,
    AnalyzerType.ZHIPU.value,
    AnalyzerType.OPENAI.value
]

# 图表类型映射（与视觉模型输出对应）
CHART_TYPE_MAPPING = {
    0: "柱状图",
    1: "折线图",
    2: "饼图",
    3: "表格",
    4: "散点图",
    5: "其他"
}


# ===================== 抽象基类 =====================
class BaseMultiModalAnalyzer(ABC):
    """多模态分析器抽象基类"""

    @abstractmethod
    async def analyze_image(
            self,
            local_image_path: str,
            prompt: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        分析图片并返回结果

        参数:
            local_image_path: 本地图片路径
            prompt: 分析提示词
            kwargs: 其他参数

        返回:
            包含分析结果的字典，至少包含"title"和"description"键
        """
        pass


# ===================== 本地模型分析器 =====================
class LocalQwenVLAnalyzer(BaseMultiModalAnalyzer):
    """本地Qwen-VL模型分析器"""

    def __init__(self, model_path: str = "Qwen/Qwen-VL-Chat"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.initialized = False
        if TRANSFORMERS_AVAILABLE:
            self._init_model()
        else:
            logging.error("无法初始化本地模型：缺少transformers依赖")

    def _init_model(self):
        """初始化本地模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).eval()
            self.initialized = True
            logging.info(f"本地模型 {self.model_path} 初始化成功")
        except Exception as e:
            logging.error(f"本地模型初始化失败：{str(e)}")
            self.initialized = False

    async def analyze_image(
            self,
            local_image_path: str,
            prompt: str,
            **kwargs
    ) -> Dict[str, Any]:
        if not self.initialized:
            return {"error": "本地模型未初始化", "title": "", "description": ""}

        try:
            # 异步加载图片（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            img = await loop.run_in_executor(None, Image.open, local_image_path)

            # 构建输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": img}
                    ]
                }
            ]

            # 模型推理（异步执行CPU密集型操作）
            result = await loop.run_in_executor(
                None,
                lambda: self.model.chat(self.tokenizer, messages)
            )

            img.close()
            return extract_json_content(result)

        except Exception as e:
            logging.error(f"本地模型分析失败：{str(e)}")
            return {"error": f"本地模型错误：{str(e)}", "title": "", "description": ""}


# ===================== API分析器 =====================
class ZhipuAnalyzer(BaseMultiModalAnalyzer):
    """智谱AI分析器"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")
        self.client = None
        if API_CLIENT_AVAILABLE and self.api_key:
            self.client = AsyncZhipuAI(api_key=self.api_key)
        else:
            logging.warning("智谱API客户端未初始化：缺少API密钥或依赖")

    async def analyze_image(
            self,
            local_image_path: str,
            prompt: str,
            model: str = "glm-4v",
            **kwargs
    ) -> Dict[str, Any]:
        if not self.client:
            return {"error": "智谱API客户端未初始化", "title": "", "description": ""}

        try:
            # 转换图片为base64
            img_b64 = await image_to_base64_async(local_image_path)

            # 调用API
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": img_b64}
                        ]
                    }
                ]
            )

            result = response.choices[0].message.content
            return extract_json_content(result)

        except Exception as e:
            logging.error(f"智谱API调用失败：{str(e)}")
            return {"error": f"智谱API错误：{str(e)}", "title": "", "description": ""}


class OpenAIAnalyzer(BaseMultiModalAnalyzer):
    """OpenAI分析器"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE")
        self.client = None
        if API_CLIENT_AVAILABLE and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            logging.warning("OpenAI API客户端未初始化：缺少API密钥或依赖")

    async def analyze_image(
            self,
            local_image_path: str,
            prompt: str,
            model: str = "gpt-4o",
            **kwargs
    ) -> Dict[str, Any]:
        if not self.client:
            return {"error": "OpenAI API客户端未初始化", "title": "", "description": ""}

        try:
            # 转换图片为base64
            img_b64 = await image_to_base64_async(local_image_path)

            # 调用API
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ]
            )

            result = response.choices[0].message.content
            return extract_json_content(result)

        except Exception as e:
            logging.error(f"OpenAI API调用失败：{str(e)}")
            return {"error": f"OpenAI API错误：{str(e)}", "title": "", "description": ""}


# ===================== 分析器管理器（自动降级） =====================
class MultiModalAnalyzerManager:
    """多模态分析器管理器，实现自动降级逻辑"""

    def __init__(
            self,
            model_priority: List[str] = None,
            **kwargs
    ):
        self.model_priority = model_priority or DEFAULT_MODEL_PRIORITY
        self.analyzers = self._init_analyzers(**kwargs)
        self.available_analyzers = self._filter_available_analyzers()

    def _init_analyzers(self, **kwargs) -> Dict[str, BaseMultiModalAnalyzer]:
        """初始化所有分析器"""
        analyzers = {
            AnalyzerType.LOCAL_QWEN_VL.value: LocalQwenVLAnalyzer(
                model_path=kwargs.get("local_qwen_path", "Qwen/Qwen-VL-Chat")
            ),
            AnalyzerType.ZHIPU.value: ZhipuAnalyzer(
                api_key=kwargs.get("zhipu_api_key")
            ),
            AnalyzerType.OPENAI.value: OpenAIAnalyzer(
                api_key=kwargs.get("openai_api_key"),
                base_url=kwargs.get("openai_base_url")
            )
        }
        return analyzers

    def _filter_available_analyzers(self) -> List[BaseMultiModalAnalyzer]:
        """过滤出可用的分析器（按优先级排序）"""
        available = []
        for analyzer_type in self.model_priority:
            if analyzer_type not in self.analyzers:
                continue

            analyzer = self.analyzers[analyzer_type]
            # 简单可用性检查
            if (analyzer_type == AnalyzerType.LOCAL_QWEN_VL.value and
                    hasattr(analyzer, "initialized") and analyzer.initialized):
                available.append(analyzer)
            elif (analyzer_type in [AnalyzerType.ZHIPU.value, AnalyzerType.OPENAI.value] and
                  hasattr(analyzer, "client") and analyzer.client is not None):
                available.append(analyzer)

        if not available:
            logging.warning("没有可用的分析器，请检查配置")
        return available

    async def analyze_image(
            self,
            local_image_path: str,
            prompt: str,
            max_retries: int = 2,
            **kwargs
    ) -> Dict[str, Any]:
        """
        尝试所有可用分析器，实现自动降级

        参数:
            local_image_path: 本地图片路径
            prompt: 分析提示词
            max_retries: 每个分析器的重试次数
            kwargs: 传递给具体分析器的参数

        返回:
            分析结果字典
        """
        for analyzer in self.available_analyzers:
            for attempt in range(max_retries + 1):
                try:
                    result = await analyzer.analyze_image(
                        local_image_path=local_image_path,
                        prompt=prompt, **kwargs
                    )
                    # 检查是否有错误
                    if "error" not in result or not result["error"]:
                        # 添加分析器类型信息
                        result["analyzer_type"] = self._get_analyzer_type(analyzer)
                        return result
                except Exception as e:
                    logging.error(f"分析器 {self._get_analyzer_type(analyzer)} 第{attempt + 1}次尝试失败：{str(e)}")

                if attempt < max_retries:
                    await asyncio.sleep(1 * (attempt + 1))  # 指数退避

        return {
            "error": "所有分析器均失败",
            "title": "",
            "description": "",
            "analyzer_type": "none"
        }

    def _get_analyzer_type(self, analyzer: BaseMultiModalAnalyzer) -> str:
        """获取分析器类型字符串"""
        if isinstance(analyzer, LocalQwenVLAnalyzer):
            return AnalyzerType.LOCAL_QWEN_VL.value
        elif isinstance(analyzer, ZhipuAnalyzer):
            return AnalyzerType.ZHIPU.value
        elif isinstance(analyzer, OpenAIAnalyzer):
            return AnalyzerType.OPENAI.value
        return "unknown"


# ===================== 图表校验增强 =====================
class ChartValidator:
    """图表类型校验器（视觉特征+文本描述双向验证）"""

    def __init__(self, det_config: Optional[str] = None, det_checkpoint: Optional[str] = None):
        self.detector = None
        self.chart_classifier = None  # 可扩展为预训练图表分类模型
        if MMDET_AVAILABLE and det_config and det_checkpoint:
            self.detector = init_detector(det_config, det_checkpoint,
                                          device='cuda:0' if torch.cuda.is_available() else 'cpu')
            logging.info("图表区域检测器初始化成功")
        else:
            logging.warning("图表区域检测器未初始化：缺少mmdet依赖或配置")

    async def detect_chart_region(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        检测图片中是否包含图表区域

        返回:
            (是否为图表, 检测详情)
        """
        if not self.detector or not MMDET_AVAILABLE:
            return False, {"error": "图表检测器不可用"}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: inference_detector(self.detector, image_path)
            )

            # 简化处理：假设得分>0.5的为图表区域
            chart_boxes = []
            for cls_idx, bboxes in enumerate(result.pred_instances.bboxes):
                scores = result.pred_instances.scores
                for bbox, score in zip(bboxes, scores):
                    if score > 0.5:
                        chart_boxes.append({
                            "class": self.detector.CLASSES[cls_idx],
                            "bbox": bbox.tolist(),
                            "score": float(score)
                        })

            is_chart = len(chart_boxes) > 0
            return is_chart, {
                "is_chart": is_chart,
                "chart_boxes": chart_boxes,
                "chart_type_visual": self._infer_chart_type(chart_boxes)  # 从视觉特征推断类型
            }
        except Exception as e:
            logging.error(f"图表区域检测失败：{str(e)}")
            return False, {"error": str(e)}

    def _infer_chart_type(self, chart_boxes: List[Dict[str, Any]]) -> str:
        """从视觉特征推断图表类型（简化实现）"""
        # 实际应用中可结合预训练分类模型
        if not chart_boxes:
            return "未知"

        # 简单规则：根据检测到的类别推断
        class_names = [box["class"].lower() for box in chart_boxes]
        if "bar" in class_names:
            return "柱状图"
        elif "line" in class_names:
            return "折线图"
        elif "pie" in class_names:
            return "饼图"
        elif "table" in class_names:
            return "表格"
        return "其他"

    async def validate_chart_consistency(
            self,
            image_path: str,
            text_description: str
    ) -> Dict[str, Any]:
        """
        验证视觉特征与文本描述的一致性

        参数:
            image_path: 图片路径
            text_description: 文本描述

        返回:
            验证结果
        """
        is_chart, visual_info = await self.detect_chart_region(image_path)
        if not is_chart:
            return {
                "consistent": True,
                "message": "非图表内容，无需校验",
                "visual_info": visual_info,
                "text_info": {"extracted_type": "非图表"}
            }

        # 从文本描述提取图表类型
        text_type = "未知"
        for chart_type in CHART_TYPE_MAPPING.values():
            if chart_type in text_description:
                text_type = chart_type
                break

        visual_type = visual_info.get("chart_type_visual", "未知")
        consistent = visual_type == text_type or (
                visual_type != "未知" and text_type != "未知" and
                visual_type in text_type  # 允许部分匹配
        )

        return {
            "consistent": consistent,
            "message": f"视觉检测：{visual_type}，文本描述：{text_type}" +
                       ("，一致" if consistent else "，不一致"),
            "visual_info": visual_info,
            "text_info": {"extracted_type": text_type}
        }


# ===================== 原有工具函数 =====================
def extract_json_content(text: str) -> Dict[str, Any]:
    """
    从文本中提取JSON内容。

    参数:
        text (str): 可能包含JSON的文本

    返回:
        Dict[str, Any]: 解析后的JSON字典，如果解析失败则返回包含错误信息的字典
    """
    if not text:
        return {"error": "Empty response", "title": "", "description": ""}

    # 尝试寻找JSON的开始和结束位置
    json_start = text.find("{")
    json_end = text.rfind("}")

    if (json_start != -1 and json_end != -1 and json_end > json_start):
        try:
            json_text = text[json_start: json_end + 1]
            result = json.loads(json_text)
            # 确保返回的字典包含必要的键
            if "title" not in result:
                result["title"] = ""
            if "description" not in result:
                result["description"] = ""

            return result
        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败: {str(e)}", "title": "", "description": ""}

    try:
        result = json.loads(text)
        # 确保返回的字典包含必要的键
        if "title" not in result:
            result["title"] = ""
        if "description" not in result:
            result["description"] = ""
        return result
    except json.JSONDecodeError:
        # 尝试从文本中提取一些信息作为描述
        fallback_description = (
            text.strip().replace("```json", "").replace("```", "").strip()[:50]
        )
        return {
            "error": "无法提取JSON内容",
            "title": "",
            "description": fallback_description,
        }


async def image_to_base64_async(image_path: str) -> str:
    """
    异步将图像文件转换为base64编码字符串

    参数:
        image_path: 图像文件路径

    返回:
        base64编码的图像字符串
    """
    try:
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = await image_file.read()
            encoded_string = base64.b64encode(image_data).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {image_path}")