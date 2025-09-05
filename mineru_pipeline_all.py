import os
import json
import time
import hashlib
import uuid
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)
import redis
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# 加载环境变量
load_dotenv()


# ===================== 向量存储管理（含版本化缓存） =====================
class MilvusVectorStore:
    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", 19530)
        self.collection_name = os.getenv("MILVUS_COLLECTION", "rag_finance_chunks")
        self.dim = int(os.getenv("MILVUS_DIM", 1536))
        self.model_version = os.getenv("EMBEDDING_MODEL_VERSION", "bge-m3-v202405")
        self.redis_client = self._init_redis()
        self._init_milvus()

    def _init_redis(self):
        """初始化Redis用于缓存预热和频率统计"""
        try:
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True
            )
            r.ping()
            return r
        except Exception as e:
            print(f"Redis初始化失败，使用内存缓存: {e}")
            return defaultdict(int)  # 内存模拟

    def _init_milvus(self):
        """初始化Milvus集合，添加版本化字段"""
        connections.connect(host=self.host, port=self.port)

        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="model_version", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="pdf_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.TEXT),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="last_access", dtype=DataType.INT64)  # 用于缓存预热
            ]
            schema = CollectionSchema(fields, description="Versioned RAG chunks with PDF hash")
            Collection(self.collection_name, schema)

            # 创建索引
            collection = Collection(self.collection_name)
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
            collection.create_index("embedding", index_params)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def add_chunks_with_version(self, chunks: List[Dict], embeddings: List[List[float]]):
        """添加带版本和哈希的文档块"""
        entities = []
        for chunk, emb in zip(chunks, embeddings):
            # 计算PDF内容哈希
            pdf_path = chunk["metadata"].get("file_path")
            pdf_hash = self._compute_pdf_hash(pdf_path) if pdf_path else "unknown"

            # 更新访问时间（用于缓存预热）
            current_ts = int(time.time())

            entities.append({
                "id": chunk.get("id", str(uuid.uuid4())),
                "embedding": emb,
                "model_version": self.model_version,
                "pdf_hash": pdf_hash,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "last_access": current_ts
            })

        if entities:
            self.collection.insert(entities)
            self.collection.flush()
            # 更新Redis访问频率
            for entity in entities:
                self.redis_client.incr(f"pdf_freq:{entity['pdf_hash']}")

    def _compute_pdf_hash(self, pdf_path: str) -> str:
        """计算PDF内容的SHA-256哈希"""
        if not pdf_path or not os.path.exists(pdf_path):
            return "invalid_path"
        with open(pdf_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def invalidate_cache(self, pdf_hash: str):
        """主动失效指定PDF的缓存"""
        expr = f"pdf_hash == '{pdf_hash}'"
        self.collection.delete(expr)
        print(f"已删除PDF哈希为 {pdf_hash} 的缓存")
        # 清除Redis频率记录
        self.redis_client.delete(f"pdf_freq:{pdf_hash}")

    def cache_warmup(self, top_n: int = 10):
        """预热高频访问的PDF向量"""
        if isinstance(self.redis_client, redis.Redis):
            # 获取访问频率Top N的PDF哈希
            freq_dict = {k: int(v) for k, v in self.redis_client.hgetall("pdf_freq").items()}
            top_hashes = [k for k, _ in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]]

            for pdf_hash in top_hashes:
                expr = f"pdf_hash == '{pdf_hash}' && model_version == '{self.model_version}'"
                self.collection.query(expr, output_fields=["id", "embedding"])
                print(f"预热缓存: {pdf_hash}")
        else:
            print("使用内存缓存，跳过预热")

    def search(self, query_emb: List[float], limit: int = 5) -> List[Dict]:
        """检索符合当前模型版本的向量"""
        expr = f"model_version == '{self.model_version}'"
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["content", "metadata", "pdf_hash"]
        )

        # 更新访问时间
        hit_ids = [hit.id for hit in results[0]]
        if hit_ids:
            self.collection.update(
                expr=f"id in {hit_ids}",
                entities={"last_access": int(time.time())}
            )

        return [{"content": hit.entity.get("content"),
                 "metadata": hit.entity.get("metadata"),
                 "distance": hit.distance}
                for hit in results[0]]


# ===================== 嵌入向量获取 =====================
class EmbeddingClient:
    def __init__(self):
        self.api_key = os.getenv("LOCAL_API_KEY")
        self.base_url = os.getenv("LOCAL_BASE_URL")
        self.model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3-finance")
        self.client = self._get_client()

    def _get_client(self) -> OpenAI:
        if not self.api_key or not self.base_url:
            raise ValueError("请配置LOCAL_API_KEY和LOCAL_BASE_URL")
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embeddings(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """批量获取文本嵌入，带重试机制"""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="生成嵌入向量"):
            batch = texts[i:i + batch_size]
            retry_count = 0
            while retry_count < 3:
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch)
                    all_embeddings.extend([e.embedding for e in response.data])
                    break
                except RateLimitError:
                    time.sleep(2 ** retry_count)  # 指数退避
                    retry_count += 1
                except Exception as e:
                    raise RuntimeError(f"嵌入生成失败: {e}")
        return all_embeddings


# ===================== RAG核心逻辑 =====================
class SimpleRAG:
    def __init__(self):
        self.vector_store = MilvusVectorStore()
        self.embedding_client = EmbeddingClient()
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        self.llm = self._init_llm()
        self.vector_store.cache_warmup()  # 启动时预热缓存

    def _init_llm(self) -> OpenAI:
        """初始化LLM客户端"""
        return OpenAI(
            api_key=os.getenv("LOCAL_LLM_API_KEY"),
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        )

    def process_chunks(self, chunk_path: str = "all_pdf_page_chunks.json"):
        """处理文档块并添加到向量库"""
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        # 生成嵌入向量
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_client.get_embeddings(texts)

        # 添加到向量库（带版本和哈希）
        self.vector_store.add_chunks_with_version(chunks, embeddings)
        print(f"已处理 {len(chunks)} 个文档块")

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict]:
        """检索相关文档块并重新排序"""
        # 生成查询嵌入
        query_emb = self.embedding_client.get_embeddings([query])[0]

        # 向量检索
        candidates = self.vector_store.search(query_emb, limit=top_k * 2)

        # 重排序
        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.compute_score(pairs)
        scored_candidates = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

        return scored_candidates[:top_k]

    def generate_answer_with_source(self, context_chunks: List[Dict], question: str) -> str:
        """生成带标准化来源的答案"""
        if not context_chunks:
            return f"未找到相关信息回答问题：{question}\n来源：无"

        # 格式化来源信息
        sources = []
        for chunk in context_chunks:
            meta = chunk["metadata"]
            file_name = meta.get("file_name", "未知文件")
            start_page = meta.get("start_page", 0) + 1  # 转换为1基页码
            end_page = meta.get("end_page", 0) + 1

            page_str = f"P{start_page}" if start_page == end_page else f"P{start_page}-{end_page}"
            sources.append(f"{file_name}（{page_str}）")

        unique_sources = sorted(set(sources))
        source_str = "; ".join(unique_sources)

        # 构建提示词
        prompt = f"""
        基于以下参考内容回答问题，答案末尾必须以「来源：xxx」格式标注所有参考文档。
        要求：
        1. 来源必须完全匹配提供的可用来源列表，不得虚构
        2. 来源格式严格为「文件名（P页码）」或「文件名（P页码-页码）」
        3. 答案简洁准确，与问题强相关

        参考内容：
        {[c['content'][:500] for c in context_chunks]}  # 限制长度避免超限

        用户问题：{question}
        可用来源列表：{source_str}

        回答示例：
        2023年公司营收为120.5亿元，同比增长15.2%。
        来源：2023年报.pdf（P5）; 季度财报Q4.pdf（P2-3）
        """

        # 调用LLM
        try:
            response = self.llm.chat.completions.create(
                model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5-7b-instruct"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"生成答案失败：{str(e)}"

        # 校验并修复来源格式
        if "来源：" not in answer:
            answer += f"\n来源：{source_str}"
        else:
            if not re.search(r"来源：.*（P\d+(-\d+)?）", answer):
                answer = answer.split("来源：")[0].strip() + f"\n来源：{source_str}"

        return answer

    def pipeline(self, query: str) -> str:
        """完整RAG流程：检索→生成答案"""
        context = self.retrieve(query)
        return self.generate_answer_with_source(context, query)


# ===================== 缓存失效工具（用于PDF更新时） =====================
def handle_updated_pdf(pdf_path: str, vector_store: MilvusVectorStore):
    """处理更新后的PDF，失效旧缓存并重新解析"""
    pdf_hash = vector_store._compute_pdf_hash(pdf_path)
    vector_store.invalidate_cache(pdf_hash)

    # 触发重新解析（实际项目中应调用mineru_pipeline_all.py中的解析逻辑）
    print(f"PDF {pdf_path} 已更新，旧缓存已清除，建议重新运行解析流程")


# ===================== 测试代码 =====================
if __name__ == "__main__":
    # 初始化RAG
    rag = SimpleRAG()

    # 可选：处理文档块（首次运行时需要）
    # rag.process_chunks()

    # 测试查询
    test_query = "2023年公司营收是多少？"
    print("查询:", test_query)
    result = rag.pipeline(test_query)
    print("回答:\n", result)

    # 示例：处理更新的PDF（实际使用时调用）
    # handle_updated_pdf("datas/2023年报.pdf", rag.vector_store)