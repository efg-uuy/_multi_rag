import os
import asyncio
import time
import json
import logging
import re
from typing import Dict, Any, List, Union, Optional
from PIL import Image
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 新增：图表校验相关依赖（按需安装：pip install torch torchvision）
try:
    import torch
    from torchvision import models, transforms
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("未安装PyTorch相关库，图表类型校验功能将禁用。请执行 'pip install torch torchvision' 启用")
    TORCH_AVAILABLE = False

from .prompts import get_image_analysis_prompt
from .image_analysis_utils import extract_json_content, image_to_base64_async, extract_text_from_pdf  # 新增文本提取工具
load_dotenv()


# ===================== 新增：图表类型校验工具 =====================
class ChartTypeClassifier:
    """图表类型分类器（基于微调ResNet-18）"""
    def __init__(self, model_weight_path: Optional[str] = None):
        if not TORCH_AVAILABLE:
            self.classifier = None
            return
        
        # 初始化ResNet-18（微调后用于图表分类）
        self.classifier = models.resnet18(pretrained=False)
        # 调整输出层（假设分类4种图表：柱状图、折线图、饼图、表格）
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, 4)
        
        # 加载微调权重（用户需提供自己的微调模型）
        if model_weight_path and os.path.exists(model_weight_path):
            self.classifier.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
        else:
            logging.warning("未提供图表分类模型权重，使用随机初始化模型（效果有限）")
        
        self.classifier.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
        
        # 图像预处理（与训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 图表类型映射（与微调模型输出对应）
        self.type_mapping = {
            0: "柱状图",
            1: "折线图",
            2: "饼图",
            3: "表格"
        }

    def predict(self, image_path: str) -> Optional[str]:
        """预测图表类型"""
        if not TORCH_AVAILABLE or not self.classifier:
            return None
        
        try:
            with Image.open(image_path).convert("RGB") as img:
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.classifier(tensor)
                    pred_idx = F.softmax(output, dim=1).argmax().item()
                return self.type_mapping.get(pred_idx, "未知图表类型")
        except Exception as e:
            logging.error(f"图表类型预测失败: {str(e)}")
            return None


def check_chart_type_consistency(classifier: ChartTypeClassifier, image_path: str, vl_prediction: str) -> tuple[bool, str]:
    """
    校验VL模型图表类型预测与轻量分类器结果的一致性
    :param classifier: 图表分类器实例
    :param image_path: 本地图像路径
    :param vl_prediction: VL模型生成的描述文本
    :return: (是否一致, 校验信息)
    """
    if not classifier or not image_path:
        return True, "图表校验未执行（缺少分类器或图像路径）"
    
    # 从VL预测结果中提取图表类型（基于关键词匹配）
    vl_chart_type = "未知图表类型"
    for chart_type in classifier.type_mapping.values():
        if chart_type in vl_prediction:
            vl_chart_type = chart_type
            break
    
    # 轻量分类器预测结果
    true_chart_type = classifier.predict(image_path)
    if not true_chart_type:
        return True, "轻量分类器预测失败，跳过校验"
    
    # 对比一致性
    if vl_chart_type == true_chart_type:
        return True, f"图表类型校验通过（VL预测：{vl_chart_type}，实际：{true_chart_type}）"
    else:
        return False, f"图表类型校验失败（VL预测：{vl_chart_type}，实际：{true_chart_type}）"


# ===================== 新增：关键数字交叉验证 =====================
def extract_numbers(text: str) -> Dict[str, float]:
    """
    从文本中提取关键数字（支持百分比、小数、整数）
    :param text: 输入文本（图表描述或文本段落）
    :return: 指标-数字映射（如 {"毛利率": 30.5}）
    """
    numbers = {}
    # 匹配 "指标 数值" 格式（如 "毛利率 30.5%"、"营收 1200万"）
    patterns = [
        # 百分比模式（如 "毛利率 30.5%"）
        r"([^0-9]+?)\s*([0-9]+\.?[0-9]*%)",
        # 小数/整数模式（如 "营收 1200.5"、"成本 800"）
        r"([^0-9]+?)\s*([0-9]+\.?[0-9]*)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.strip())
        for indicator, num_str in matches:
            # 清理指标名称（去除特殊字符和空格）
            indicator = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", indicator).strip()
            if not indicator:
                continue
            # 转换数字格式
            try:
                if "%" in num_str:
                    num = float(num_str.replace("%", "")) / 100  # 百分比转为小数
                else:
                    num = float(num_str)
                numbers[indicator] = num
            except ValueError:
                continue
    return numbers


def cross_validate_numbers(chart_desc: str, text_context: str) -> List[str]:
    """
    交叉验证图表描述与文本段落中的关键数字
    :param chart_desc: VL模型生成的图表描述
    :param text_context: 相关文本段落（如PDF中的对应章节）
    :return: 验证警告列表
    """
    warnings = []
    if not chart_desc or not text_context:
        return warnings
    
    # 提取两边的数字
    chart_numbers = extract_numbers(chart_desc)
    text_numbers = extract_numbers(text_context)
    if not chart_numbers or not text_numbers:
        return ["关键数字提取失败，跳过交叉验证"]
    
    # 对比相同指标的数字差异
    common_indicators = set(chart_numbers.keys()) & set(text_numbers.keys())
    if not common_indicators:
        return ["未找到可对比的共同指标，跳过交叉验证"]
    
    for indicator in common_indicators:
        chart_num = chart_numbers[indicator]
        text_num = text_numbers[indicator]
        # 计算相对差异率（避免除以零）
        if text_num == 0:
            diff_rate = 1.0 if chart_num != 0 else 0.0
        else:
            diff_rate = abs(chart_num - text_num) / abs(text_num)
        
        # 差异超过5%则标记警告
        if diff_rate > 0.05:
            warnings.append(
                f"指标 '{indicator}' 数字差异超标：图表描述({chart_num:.4f}) vs 文本段落({text_num:.4f})，差异率 {diff_rate:.2%}"
            )
    return warnings if warnings else ["所有关键数字交叉验证通过"]


# ===================== 原有异步图像分析类（新增功能整合） =====================
class AsyncImageAnalysis:
    """
    异步图像文本提取器类，用于将图像内容转换为文本描述和标题。
    新增：图表类型预校验、关键数字交叉验证功能
    """

    # 预定义的配置
    PROVIDER_CONFIGS = {
        "guiji": {
            "api_key_env": "GUIJI_API_KEY",
            "base_url_env": "GUIJI_BASE_URL", 
            "model_env": "GUIJI_VISION_MODEL",
            "default_models": [ "Pro/Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-32B-Instruct",]
        },
        "zhipu": {
            "api_key_env": "ZHIPU_API_KEY",
            "base_url_env": "ZHIPU_BASE_URL",
            "model_env": "ZHIPU_VISION_MODEL", 
            "default_models": ["glm-4v-flash", "glm-4v"]
        },
        "volces": {
            "api_key_env": "VOLCES_API_KEY",
            "base_url_env": "VOLCES_BASE_URL",
            "model_env": "VOLCES_VISION_MODEL",
            "default_models": ["doubao-1.5-vision-lite-250315", "doubao-1.5-vision-pro-250328"]
        },
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "base_url_env": "OPENAI_API_BASE",
            "model_env": "OPENAI_VISION_MODEL",
            "default_models": ["gpt-4-vision-preview", "gpt-4o"]
        }
    }

    def __init__(
        self,
        provider: str = "zhipu",  # 默认使用智谱
        api_key: str = None,
        base_url: str = None,
        vision_model: str = None,
        prompt: Optional[str] = None,
        max_concurrent: int = 5,
        chart_classifier_weight: Optional[str] = None  # 新增：图表分类器权重路径
    ):
        """
        初始化图像分析器（新增图表分类器初始化）
        """
        self.provider = provider.lower()
        
        if self.provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"不支持的提供商: {provider}. 支持的提供商: {list(self.PROVIDER_CONFIGS.keys())}")
        
        config = self.PROVIDER_CONFIGS[self.provider]
        
        # 获取API密钥、基础URL、模型（原有逻辑）
        self.api_key = api_key or os.getenv(config["api_key_env"])
        if not self.api_key:
            raise ValueError(f"API密钥未提供，请设置 {config['api_key_env']} 环境变量，或传入api_key参数。")

        self.base_url = base_url or os.getenv(config["base_url_env"])
        if not self.base_url:
            raise ValueError(f"基础URL未提供，请设置 {config['base_url_env']} 环境变量，或传入base_url参数。")
        
        self.vision_model = (vision_model or 
                           os.getenv(config["model_env"]) or 
                           config["default_models"][0])
        
        # 初始化图表类型分类器（新增逻辑）
        self.chart_classifier = ChartTypeClassifier(chart_classifier_weight) if TORCH_AVAILABLE else None
        if self.chart_classifier and not self.chart_classifier.classifier:
            logging.warning("图表分类器初始化失败，将禁用图表类型校验功能")
            self.chart_classifier = None
        
        print(f"使用提供商: {self.provider}")
        print(f"API基础URL: {self.base_url}")
        print(f"视觉模型: {self.vision_model}")
        print(f"图表类型校验功能: {'启用' if self.chart_classifier else '禁用'}")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 设置提示词（原有逻辑）
        if prompt:
            self._prompt = prompt
        else:
            self._prompt = get_image_analysis_prompt(
                title_max_length=10,
                description_max_length=200,
            )
        
        # 设置并发限制（原有逻辑）
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_image(
        self,
        image_url: str = None,
        local_image_path: str = None,
        model: str = None,
        detail: str = "low",
        prompt: str = None,
        temperature: float = 0.1,
        text_context: Optional[str] = None,  # 新增：用于数字交叉验证的文本上下文
        pdf_path: Optional[str] = None     # 新增：若提供PDF路径，自动提取对应章节文本
    ) -> Dict[str, Any]:
        """
        异步分析图像并返回描述信息（新增校验功能）
        :param text_context: 相关文本段落（如PDF章节内容），用于数字交叉验证
        :param pdf_path: PDF文件路径，自动提取与图表相关的文本上下文
        """
        async with self.semaphore:
            # 基本参数检查（原有逻辑）
            if not image_url and not local_image_path:
                raise ValueError("必须提供一个图像来源：image_url或local_image_path")
            if image_url and local_image_path:
                raise ValueError("只能提供一个图像来源：image_url或local_image_path")

            # 处理图像来源（原有逻辑）
            final_image_url = image_url
            image_format = "jpeg"
            if local_image_path:
                try:
                    loop = asyncio.get_event_loop()
                    def get_image_format():
                        with Image.open(local_image_path) as img:
                            return img.format.lower() if img.format else "jpeg"
                    image_format = await loop.run_in_executor(None, get_image_format)
                except Exception as e:
                    logging.warning(f"无法识别图片格式 {local_image_path}: {e}, 使用默认jpeg")
                base64_image = await image_to_base64_async(local_image_path)
                final_image_url = f"data:image/{image_format};base64,{base64_image}"

            model_to_use = model or self.vision_model
            prompt_text = prompt or self._prompt

            # ===================== 新增：1. 调用VL模型获取原始结果 =====================
            try:
                response = await self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": final_image_url, "detail": detail}},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                    temperature=temperature,
                    max_tokens=300,
                )
                result_content = response.choices[0].message.content
                analysis_result = extract_json_content(result_content)
                # 确保基础字段存在
                analysis_result.setdefault("title", "")
                analysis_result.setdefault("description", "")
                analysis_result["validation_warnings"] = []  # 新增：校验警告列表
            except Exception as e:
                logging.error(f"API调用失败: {e}")
                return {
                    "error": f"API调用失败: {str(e)}",
                    "title": "",
                    "description": "",
                    "validation_warnings": [f"API调用失败，无法进行后续校验: {str(e)}"]
                }

            # ===================== 新增：2. 图表类型预校验 =====================
            if self.chart_classifier and local_image_path:  # 仅本地图像支持校验（需文件路径）
                vl_pred_desc = f"{analysis_result['title']} {analysis_result['description']}"
                is_consistent, check_msg = check_chart_type_consistency(
                    self.chart_classifier, local_image_path, vl_pred_desc
                )
                analysis_result["validation_warnings"].append(check_msg)
                # 校验失败触发熔断
                if not is_consistent:
                    analysis_result["title"] = "图表识别存疑"
                    analysis_result["description"] = "图表识别存疑，请参考原文"
                    # 后续数字验证不再执行（因描述已失效）
                    return analysis_result

            # ===================== 新增：3. 关键数字交叉验证 =====================
            # 自动提取PDF文本上下文（若提供PDF路径）
            if pdf_path and not text_context:
                try:
                    # 假设图像文件名包含PDF页码（如 "chart_page10.png"），提取页码并获取对应文本
                    page_num = None
                    if local_image_path:
                        # 从图像文件名提取页码（如 "chart_2024_page5.png" 提取 5）
                        page_match = re.search(r"page(\d+)", local_image_path, re.IGNORECASE)
                        if page_match:
                            page_num = int(page_match.group(1))
                    # 提取PDF对应页码的文本
                    if page_num:
                        text_context = await loop.run_in_executor(
                            None, extract_text_from_pdf, pdf_path, page_num
                        )
                        analysis_result["validation_warnings"].append(f"从PDF第{page_num}页提取文本上下文")
                except Exception as e:
                    logging.error(f"从PDF提取文本失败: {str(e)}")
                    analysis_result["validation_warnings"].append(f"PDF文本提取失败: {str(e)}")
            
            # 执行数字交叉验证
            if text_context:
                num_warnings = cross_validate_numbers(
                    chart_desc=analysis_result["description"],
                    text_context=text_context
                )
                analysis_result["validation_warnings"].extend(num_warnings)
            else:
                analysis_result["validation_warnings"].append("未提供文本上下文，跳过关键数字交叉验证")

            return analysis_result

    # 以下为原有方法（未修改）
    async def analyze_multiple_images(
        self,
        image_sources: List[Dict[str, Any]],
        model: str = None,
        detail: str = "low",
        prompt: str = None,
        temperature: float = 0.1,
        text_contexts: Optional[List[Optional[str]]] = None,  # 新增：批量文本上下文
        pdf_paths: Optional[List[Optional[str]]] = None     # 新增：批量PDF路径
    ) -> List[Dict[str, Any]]:
        tasks = []
        text_contexts = text_contexts or [None] * len(image_sources)
        pdf_paths = pdf_paths or [None] * len(image_sources)
        for idx, source in enumerate(image_sources):
            task = self.analyze_image(
                image_url=source.get("image_url"),
                local_image_path=source.get("local_image_path"),
                model=model,
                detail=detail,
                prompt=prompt,
                temperature=temperature,
                text_context=text_contexts[idx],
                pdf_path=pdf_paths[idx]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": f"处理第{i+1}张图像时出错: {str(result)}",
                    "title": "图片处理出错",
                    "description": "图片处理出错",
                    "validation_warnings": [f"处理失败: {str(result)}"]
                })
                print(f"处理第{i+1}张图像时出错: {str(result)}")
            else:
                processed_results.append(result)
        return processed_results

    async def close(self):
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()