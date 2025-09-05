from http import client
from openai import OpenAI
from dotenv import load_dotenv
import os
import hashlib
import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import json

# LOCAL_API_KEY,LOCAL_BASE_URL,LOCAL_TEXT_MODEL,LOCAL_EMBEDDING_MODEL

load_dotenv()  # 加载环境变量（可选，用户可自行读取）

# 初始化CLIP模型用于图像-文本相似度计算
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
except Exception as e:
    print(f"CLIP模型加载失败: {e}")
    clip_model = None
    clip_preprocess = None


def get_openai_client(api_key: str, base_url: str) -> OpenAI:
    """
    获取 OpenAI 客户端，必须传递 api_key 和 base_url
    """
    if not api_key or not base_url:
        raise ValueError("api_key 和 base_url 必须显式传递！")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


from tqdm import tqdm


def batch_get_embeddings(
        texts: List[str],
        batch_size: int = 64,
        api_key: str = None,
        base_url: str = None,
        embedding_model: str = None
) -> List[List[float]]:
    """
    批量获取文本的嵌入向量
    :param texts: 文本列表
    :param batch_size: 批处理大小
    :param api_key: 可选，自定义 API KEY
    :param base_url: 可选，自定义 BASE URL
    :param embedding_model: 可选，自定义嵌入模型
    :return: 嵌入向量列表
    """
    if not api_key or not base_url or not embedding_model:
        raise ValueError("api_key、base_url、embedding_model 必须显式传递！")
    all_embeddings = []
    client = get_openai_client(api_key, base_url)
    total = len(texts)
    if total == 0:
        return []
    iterator = range(0, total, batch_size)
    if total > 1:
        iterator = tqdm(iterator, desc="Embedding", unit="batch")
    import time
    from openai import RateLimitError
    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        retry_count = 0
        while True:
            try:
                response = client.embeddings.create(
                    model=embedding_model,
                    input=batch_texts
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except RateLimitError as e:
                retry_count += 1
                print(f"RateLimitError: {e}. 等待10秒后重试（第{retry_count}次）...")
                time.sleep(10)
    return all_embeddings


def get_image_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    使用CLIP模型获取图像的嵌入向量
    :param image_path: 图像文件路径
    :return: 图像嵌入向量，若失败则返回None
    """
    if not clip_model or not clip_preprocess:
        print("CLIP模型未初始化，无法获取图像嵌入")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = clip_model.encode_image(image_input)

        # 归一化并转换为numpy数组
        return image_embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"获取图像嵌入失败: {e}")
        return None


def calculate_text_similarity(image_embedding: np.ndarray, text_embeddings: List[np.ndarray]) -> List[float]:
    """
    计算图像嵌入与文本嵌入之间的余弦相似度
    :param image_embedding: 图像嵌入向量
    :param text_embeddings: 文本嵌入向量列表
    :return: 相似度列表
    """
    # 归一化向量
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    similarities = []

    for text_emb in text_embeddings:
        text_emb = text_emb / np.linalg.norm(text_emb)
        similarity = np.dot(image_embedding, text_emb)
        similarities.append(float(similarity))

    return similarities


def get_top_text_ids(image_embedding: np.ndarray, text_embeddings: List[np.ndarray], text_ids: List[str],
                     top_k: int = 3) -> List[str]:
    """
    获取与图像最相似的前k个文本ID
    :param image_embedding: 图像嵌入向量
    :param text_embeddings: 文本嵌入向量列表
    :param text_ids: 文本ID列表
    :param top_k: 取前k个结果
    :return: 最相似的文本ID列表
    """
    if len(text_embeddings) != len(text_ids):
        raise ValueError("文本嵌入与文本ID数量不匹配")

    similarities = calculate_text_similarity(image_embedding, text_embeddings)

    # 按相似度排序并获取前k个文本ID
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_k]

    return [text_ids[i] for i in top_indices]


def process_image_chunk(
        image_path: str,
        text_chunks: List[Dict[str, str]],  # 格式: [{"id": "text_id1", "content": "text_content1"}, ...]
        api_key: str,
        base_url: str,
        embedding_model: str
) -> Dict[str, Union[Optional[np.ndarray], List[str]]]:
    """
    处理图像Chunk，获取图像嵌入并找到最相似的前3个文本Chunk ID
    :param image_path: 图像路径
    :param text_chunks: 文本Chunk列表
    :param api_key: API密钥
    :param base_url: API基础URL
    :param embedding_model: 嵌入模型名称
    :return: 包含图像嵌入和关联文本ID的字典
    """
    # 获取图像嵌入
    image_embedding = get_image_embedding(image_path)

    if not image_embedding.any():
        return {"image_embedding": None, "related_text_ids": []}

    # 获取文本嵌入
    text_contents = [chunk["content"] for chunk in text_chunks]
    text_ids = [chunk["id"] for chunk in text_chunks]
    text_embeddings = get_text_embedding(
        text_contents,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model
    )

    # 转换为numpy数组以便计算相似度
    text_embeddings_np = [np.array(emb) for emb in text_embeddings]

    # 获取最相似的前3个文本ID
    top_text_ids = get_top_text_ids(image_embedding, text_embeddings_np, text_ids, top_k=3)

    return {
        "image_embedding": image_embedding.tolist(),
        "related_text_ids": top_text_ids
    }


def get_text_embedding(
        texts: List[str],
        api_key: str = None,
        base_url: str = None,
        embedding_model: str = None,
        batch_size: int = 64
) -> List[List[float]]:
    """
    获取文本的嵌入向量，支持批次处理，保持输出顺序与输入顺序一致
    :param texts: 文本列表
    :param api_key: 可选，自定义 API KEY
    :param base_url: 可选，自定义 BASE URL
    :param embedding_model: 可选，自定义嵌入模型
    :param batch_size: 批处理大小
    :return: 嵌入向量列表
    """
    if not api_key or not base_url or not embedding_model:
        raise ValueError("api_key、base_url、embedding_model 必须显式传递！")
    # 直接批量获取所有文本的embedding，不做缓存
    return batch_get_embeddings(
        texts,
        batch_size=batch_size,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model
    )