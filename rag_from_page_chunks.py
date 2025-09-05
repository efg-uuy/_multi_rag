import json
import os
import time
import uuid
import asyncio
import concurrent.futures
import random
import torch
import numpy as np
from rank_bm25 import BM25Okapi  # 新增：BM25关键词检索
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import sys
import re
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))
from get_text_embedding import get_text_embedding

# 确保依赖安装提示
required_packages = {
    "python-dotenv": "pip install python-dotenv",
    "openai": "pip install openai",
    "redis": "pip install redis",
    "pymilvus": "pip install pymilvus==2.4.3",
    "sentence-transformers": "pip install sentence-transformers",
    "scikit-learn": "pip install scikit-learn",
    "FlagEmbedding": "pip install FlagEmbedding",
    "rank-bm25": "pip install rank-bm25"  # 新增：BM25依赖
}
for pkg, install_cmd in required_packages.items():
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"警告：未安装{pkg}，请执行命令：{install_cmd}")

# 导入重排模型
from FlagEmbedding import FlagReranker

# 基础依赖导入
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

try:
    from openai import OpenAI
except ImportError:
    class OpenAI:
        def __init__(self, *args, **kwargs):
            raise ImportError("请安装openai包：pip install openai")

        def chat(self, *args, **kwargs):
            pass

try:
    import redis
except ImportError:
    class MockRedis:
        def __init__(self, *args, **kwargs):
            self.data = {}

        def setex(self, key, expire, value):
            self.data[key] = (value, time.time() + expire)

        def get(self, key):
            val, exp = self.data.get(key, (None, 0))
            return val if time.time() < exp else None


    redis = MockRedis

# Milvus相关导入
try:
    from pymilvus import (
        connections,
        FieldSchema, CollectionSchema, DataType,
        Collection,
        utility,
        Partition
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("⚠️ Milvus未安装，将回退到SimpleVectorStore（性能较差）")

# 关键词检索依赖
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    print("⚠️ scikit-learn未安装，兜底关键词检索功能受限")

# 加载环境变量
load_dotenv()


# ===================== 1. 审计日志工具类 =====================
class AuditLogManager:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=redis_port, db=redis_db, decode_responses=True
            )
            self.redis_client.ping()
            self.using_redis = True
            print("✅ Redis审计日志连接成功")
        except Exception as e:
            print(f"⚠️ Redis连接失败，使用内存模拟: {str(e)}")
            self.redis_client = {}
            self.using_redis = False

        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", 3306),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("AUDIT_DB_NAME", "audit_logs")
        }

    def create_pre_log(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        log_id = uuid.uuid4().hex
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

    def update_log_with_answer(self, log_id: str, answer: str, retrieval_chunks: List[Dict[str, Any]],
                               status: str = "completed", model_version: str = "unknown",
                               retrieval_type: str = "vector") -> bool:
        if self.using_redis:
            log_str = self.redis_client.get(f"audit_log:{log_id}")
            if not log_str:
                print(f"❌ 日志ID {log_id} 不存在或已过期")
                return False
            log_data = json.loads(log_str)
        else:
            log_entry = self.redis_client.get(f"audit_log:{log_id}")
            if not log_entry or time.time() > log_entry[1]:
                print(f"❌ 日志ID {log_id} 不存在或已过期")
                return False
            log_data = log_entry[0]

        log_data.update({
            "answer": answer,
            "retrieval_chunks": [
                {"id": c.get("id"), "metadata": c.get("metadata"), "model_version": c.get("model_version", "unknown")}
                for c in retrieval_chunks],
            "status": status,
            "complete_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "process_duration": round(time.time() - log_data["timestamp"], 3),
            "embedding_model_version": model_version,
            "retrieval_type": retrieval_type
        })

        if self.using_redis:
            self.redis_client.setex(f"audit_log:{log_id}", 86400, json.dumps(log_data, ensure_ascii=False))
        else:
            self.redis_client[f"audit_log:{log_id}"] = (log_data, time.time() + 86400)
        return True

    async def write_audit_log_to_db(self, log_id: str, retries: int = 3) -> None:
        for attempt in range(retries):
            try:
                if self.using_redis:
                    log_str = self.redis_client.get(f"audit_log:{log_id}")
                    if not log_str:
                        print(f"❌ 异步落盘失败：日志ID {log_id} 不存在")
                        return
                    log_data = json.loads(log_str)
                else:
                    log_entry = self.redis_client.get(f"audit_log:{log_id}")
                    if not log_entry or time.time() > log_entry[1]:
                        print(f"❌ 异步落盘失败：日志ID {log_id} 不存在")
                        return
                    log_data = log_entry[0]

                print(f"📝 异步落盘日志 {log_id} 到数据库: {log_data['question'][:20]}...")
                await asyncio.sleep(0.1)
                if self.using_redis:
                    self.redis_client.delete(f"audit_log:{log_id}")
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"⚠️ 异步落盘尝试 {attempt + 1} 失败，重试中: {str(e)}")
                    await asyncio.sleep(1)
                else:
                    print(f"❌ 异步落盘最终失败（{retries}次尝试）: {str(e)}")
                    if self.using_redis:
                        self.redis_client.lpush("audit_log_failed", log_id)


# ===================== 2. Chunk处理工具 =====================
def smart_truncate(text: str, max_tokens: int = 512) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    number_positions = [i for i, token in enumerate(tokens) if any(c.isdigit() for c in token)]
    center_idx = number_positions[len(number_positions) // 2] if number_positions else len(tokens) // 2
    half_window = max_tokens // 2
    start_idx = max(0, center_idx - half_window)
    end_idx = min(len(tokens), start_idx + max_tokens)
    return " ".join(tokens[start_idx:end_idx])


def slide_window_split(text: str, max_tokens: int = 512, slide_step: int = 200) -> List[str]:
    tokens = text.split()
    total_tokens = len(tokens)
    if total_tokens <= max_tokens:
        return [text]
    sub_chunks = []
    start_idx = 0
    while start_idx < total_tokens:
        end_idx = min(start_idx + max_tokens, total_tokens)
        sub_tokens = tokens[start_idx:end_idx]
        sub_chunk = " ".join(sub_tokens)
        sub_chunks.append(smart_truncate(sub_chunk, max_tokens))
        start_idx += slide_step
        if start_idx + max_tokens > total_tokens and start_idx < total_tokens:
            start_idx = max(0, total_tokens - max_tokens)
    return list(dict.fromkeys(sub_chunks))


def process_long_chunk(chunk: Dict[str, Any], max_tokens: int = 512, slide_step: int = 200) -> List[Dict[str, Any]]:
    raw_content = chunk["content"]
    metadata = chunk["metadata"].copy()
    sub_contents = slide_window_split(raw_content, max_tokens, slide_step)
    sub_chunks = []
    for idx, sub_content in enumerate(sub_contents):
        sub_metadata = metadata.copy()
        sub_metadata["sub_chunk_idx"] = idx
        sub_metadata["total_sub_chunks"] = len(sub_contents)
        sub_chunks.append({"content": sub_content, "metadata": sub_metadata})
    return sub_chunks


# ===================== 3. BM25检索实现（新增核心） =====================
class BM25Retriever:
    def __init__(self, chunks):
        self.tokenized_corpus = [chunk["content"].split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chunks = chunks

    def retrieve(self, query, top_k=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]


# ===================== 4. 混合检索与RRF融合（新增核心） =====================
def hybrid_retrieve(query, vector_retriever, bm25_retriever, vector_store, embedding_version, top_k=5,
                    candidate_top_k=10):
    # 向量检索结果
    vector_results, version_matched = vector_store.search_with_version(
        query_embedding=vector_retriever,
        model_version=embedding_version,
        top_k=candidate_top_k
    )

    # BM25检索结果
    bm25_results = bm25_retriever.retrieve(query, top_k=candidate_top_k)

    # RRF融合（倒数排名分数）
    rrf_scores = defaultdict(float)
    k = 60  # RRF超参数
    for rank, chunk in enumerate(vector_results, 1):
        rrf_scores[chunk["id"]] += 1 / (k + rank)
    for rank, chunk in enumerate(bm25_results, 1):
        rrf_scores[chunk["id"]] += 1 / (k + rank)

    # 按分数排序取top_k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # 合并结果并去重
    combined_chunks = vector_results + bm25_results
    unique_chunks = {chunk["id"]: chunk for chunk in combined_chunks}
    return [unique_chunks[id] for id in sorted_ids if id in unique_chunks], version_matched


# ===================== 5. 向量库优化：MilvusVectorStore =====================
class MilvusVectorStore:
    def __init__(self, collection_name: str = "rag_finance_chunks", dim: int = 1536):
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None
        self._connect_milvus()
        self._create_collection_with_version()
        self._load_all_versions()

    def _connect_milvus(self) -> None:
        if not MILVUS_AVAILABLE:
            self.collection = None
            return
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        try:
            connections.connect(
                alias="default",
                host=milvus_host,
                port=milvus_port
            )
            print(f"✅ 成功连接Milvus：{milvus_host}:{milvus_port}")
        except Exception as e:
            print(f"⚠️ Milvus连接失败，将回退到内存模式：{str(e)}")
            self.collection = None
            self.in_memory_chunks = []
            self.in_memory_embeddings = []
            self.in_memory_versions = []

    def _create_collection_with_version(self) -> None:
        if not MILVUS_AVAILABLE or self.collection:
            return
        if not utility.has_collection(self.collection_name, using="default"):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding_model_version", dtype=DataType.VARCHAR, max_length=64)
            ]
            schema = CollectionSchema(fields=fields, description="RAG金融场景Chunk向量集合（带版本绑定）")
            self.collection = Collection(name=self.collection_name, schema=schema, using="default")

            index_params = {
                "index_type": "IVF_HNSW",
                "metric_type": "IP",
                "params": {"nlist": 2048, "M": 8, "efConstruction": 64}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            print(f"✅ Milvus集合[{self.collection_name}]及版本字段创建完成")
        else:
            self.collection = Collection(name=self.collection_name, using="default")
            schema = self.collection.schema
            if not any(f.name == "embedding_model_version" for f in schema.fields):
                print("⚠️ 现有Milvus集合缺少版本字段，正在添加...")
                self.collection.add_field(
                    FieldSchema(name="embedding_model_version", dtype=DataType.VARCHAR, max_length=64,
                                default_value="unknown")
                )
            print(f"✅ Milvus集合[{self.collection_name}]已存在，直接加载")

        self.collection.load()

    def _load_all_versions(self) -> None:
        self.available_versions = set()
        if not MILVUS_AVAILABLE or not self.collection:
            self.available_versions = set(self.in_memory_versions) if hasattr(self, "in_memory_versions") else set()
            return

        try:
            res = self.collection.query(
                expr="1==1",
                output_fields=["embedding_model_version"],
                limit=0
            )
            self.available_versions = {item["embedding_model_version"] for item in res if
                                       item["embedding_model_version"]}
            print(f"✅ 加载Milvus中可用模型版本：{sorted(self.available_versions)}")
        except Exception as e:
            print(f"⚠️ 加载版本信息失败：{str(e)}，可用版本将动态探测")
            self.available_versions = set()

    def add_chunks_with_version(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]],
                                model_version: str) -> None:
        if not model_version:
            raise ValueError("模型版本不能为空，请指定有效的版本标识")

        if not MILVUS_AVAILABLE or not self.collection:
            self.in_memory_chunks.extend(chunks)
            self.in_memory_embeddings.extend(embeddings)
            self.in_memory_versions.extend([model_version] * len(chunks))
            self.available_versions.add(model_version)
            print(f"⚠️ 内存模式：添加 {len(chunks)} 个Chunk（版本：{model_version}）")
            return

        data = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"chunk_{uuid.uuid4().hex[:16]}"
            data.append([
                chunk_id,
                emb,
                chunk["content"][:4095],
                chunk["metadata"],
                model_version
            ])

        insert_result = self.collection.insert(data=data, using="default")
        self.collection.flush()
        self.available_versions.add(model_version)
        print(
            f"✅ 向Milvus插入 {len(insert_result.primary_keys)} 个Chunk（版本：{model_version}，总数量：{self.collection.num_entities}）")

    def search_with_version(self, query_embedding: List[float], model_version: str,
                            top_k: int = 3, nprobe: int = 32) -> Tuple[List[Dict[str, Any]], bool]:
        start_time = time.time()
        version_matched = True

        if model_version not in self.available_versions:
            if MILVUS_AVAILABLE and self.collection:
                try:
                    res = self.collection.query(
                        expr=f'embedding_model_version == "{model_version}"',
                        limit=1
                    )
                    if res:
                        self.available_versions.add(model_version)
                        version_matched = True
                    else:
                        version_matched = False
                except Exception as e:
                    print(f"⚠️ 版本探测失败：{str(e)}，将返回全版本结果")
                    version_matched = False
            else:
                version_matched = model_version in self.in_memory_versions

        if not version_matched:
            print(f"⚠️ 模型版本[{model_version}]在Milvus中不存在，将返回全版本结果（可能不匹配）")
            expr = "1==1"
        else:
            expr = f'embedding_model_version == "{model_version}"'

        if not MILVUS_AVAILABLE or not self.collection:
            if version_matched:
                version_indices = [i for i, v in enumerate(self.in_memory_versions) if v == model_version]
                filtered_emb = [self.in_memory_embeddings[i] for i in version_indices]
                filtered_chunks = [self.in_memory_chunks[i] for i in version_indices]
            else:
                filtered_emb = self.in_memory_embeddings
                filtered_chunks = self.in_memory_chunks

            from numpy.linalg import norm
            import numpy as np
            if not filtered_emb:
                return [], version_matched
            emb_matrix = np.array(filtered_emb)
            query_emb = np.array(query_embedding)
            sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
            indices = sims.argsort()[::-1][:top_k]
            results = [
                {
                    "id": f"memory_chunk_{idx}",
                    "content": filtered_chunks[i]["content"],
                    "metadata": filtered_chunks[i]["metadata"],
                    "similarity": round(sims[i], 4),
                    "model_version": self.in_memory_versions[version_indices[i]] if version_matched else
                    self.in_memory_versions[i]
                }
                for i in indices
            ]
            print(
                f"⚠️ 内存模式检索完成（版本匹配：{version_matched}），耗时：{round((time.time() - start_time) * 1000, 2)}ms")
            return results, version_matched

        search_params = {"metric_type": "IP", "params": {"nprobe": nprobe}}
        search_result = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "metadata", "embedding_model_version"],
            using="default"
        )

        chunks = []
        for hits in search_result:
            for hit in hits:
                chunks.append({
                    "id": hit.id,
                    "content": hit.entity.get("content"),
                    "metadata": hit.entity.get("metadata"),
                    "similarity": round(hit.score, 4),
                    "model_version": hit.entity.get("embedding_model_version", "unknown")
                })

        search_duration = round((time.time() - start_time) * 1000, 2)
        print(f"✅ Milvus检索完成（版本：{model_version}，匹配：{version_matched}，top-{top_k}），耗时：{search_duration}ms")
        return chunks, version_matched

    def get_compatible_versions(self) -> List[str]:
        return sorted(self.available_versions)


# ===================== 6. 兜底检索工具 =====================
class KeywordFallbackRetriever:
    def __init__(self, all_chunks: List[Dict[str, Any]] = None):
        self.all_chunks = all_chunks or []
        self.tfidf_vectorizer = None
        self.chunk_vectors = None
        self._build_tfidf_index()
        self._build_rule_map()

    def _build_tfidf_index(self) -> None:
        if not TFIDF_AVAILABLE or not self.all_chunks:
            print("⚠️ TF-IDF索引构建条件不足（缺少依赖或数据）")
            return

        try:
            chunk_texts = [chunk["content"] for chunk in self.all_chunks]
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_features=10000
            )
            self.chunk_vectors = self.tfidf_vectorizer.fit_transform(chunk_texts)
            print(f"✅ TF-IDF索引构建完成（{len(self.all_chunks)}个Chunk，{len(self.tfidf_vectorizer.vocabulary_)}个特征）")
        except Exception as e:
            print(f"⚠️ TF-IDF索引构建失败：{str(e)}")
            self.tfidf_vectorizer = None
            self.chunk_vectors = None

    def _build_rule_map(self) -> None:
        self.rule_map = {
            "净利润": lambda c: "净利润" in c["content"] or "net profit" in c["content"].lower(),
            "营业收入": lambda c: "营业收入" in c["content"] or "revenue" in c["content"].lower(),
            "同比增长率": lambda c: "同比增长" in c["content"] or "year-on-year" in c["content"].lower(),
            "资产减值损失": lambda c: "资产减值" in c["content"] or "impairment" in c["content"].lower(),
            "毛利率": lambda c: "毛利率" in c["content"] or "gross margin" in c["content"].lower()
        }
        self.number_pattern = re.compile(r"\d+(\.\d+)?(%|元|万元|亿元|年|月|日)")
        print("✅ 规则匹配映射构建完成")

    def update_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.all_chunks = chunks
        self._build_tfidf_index()
        self._build_rule_map()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        start_time = time.time()
        if not self.all_chunks:
            print("❌ 兜底检索无可用Chunk数据")
            return []

        rule_matched = []
        for chunk in self.all_chunks:
            match_score = 0
            for keyword, rule_func in self.rule_map.items():
                if keyword in query and rule_func(chunk):
                    match_score += 2
            query_numbers = set(self.number_pattern.findall(query))
            chunk_numbers = set(self.number_pattern.findall(chunk["content"]))
            if query_numbers & chunk_numbers:
                match_score += 1.5

            if match_score > 0:
                rule_matched.append((chunk, match_score))

        tfidf_matched = []
        if TFIDF_AVAILABLE and self.tfidf_vectorizer and self.chunk_vectors is not None:
            try:
                query_vector = self.tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
                tfidf_candidates = [
                    (self.all_chunks[i], float(similarities[i]))
                    for i in similarities.argsort()[::-1][:10]
                    if self.all_chunks[i] not in [c for c, _ in rule_matched]
                ]
                tfidf_matched = [(c, s * 1.0) for c, s in tfidf_candidates]
            except Exception as e:
                print(f"⚠️ TF-IDF匹配失败：{str(e)}")

        all_candidates = rule_matched + tfidf_matched
        if not all_candidates:
            all_candidates = [(chunk, 0.1) for chunk in self.all_chunks[:top_k]]

        all_candidates.sort(key=lambda x: (-x[1], len(x[0]["content"])))
        final_results = [
            {
                **c,
                "similarity": round(score, 4),
                "retrieval_type": "keyword_fallback",
                "model_version": "fallback"
            }
            for c, score in all_candidates[:top_k]
        ]

        print(f"✅ 兜底关键词检索完成（{len(final_results)}个结果），耗时：{round((time.time() - start_time) * 1000, 2)}ms")
        return final_results


# ===================== 7. 流量分层路由工具 =====================
class LLMClientRouter:
    def __init__(self):
        self.local_config = {
            "api_key": os.getenv("LOCAL_LLM_API_KEY", os.getenv("LOCAL_API_KEY")),
            "base_url": os.getenv("LOCAL_LLM_BASE_URL", os.getenv("LOCAL_BASE_URL")),
            "model": os.getenv("LOCAL_LLM_MODEL", os.getenv("LOCAL_TEXT_MODEL", "qwen2.5-7b-instruct")),
            "desc": "本地A6000 GPU"
        }
        self.cloud_config = {
            "api_key": os.getenv("CLOUD_LLM_API_KEY"),
            "base_url": os.getenv("CLOUD_LLM_BASE_URL"),
            "model": os.getenv("CLOUD_LLM_MODEL", "qwen2.5-72b-instruct"),
            "desc": "云端大模型服务"
        }

        self._validate_config()
        self.off_peak_hours = (22, 8)
        print(f"✅ LLM路由初始化完成：")
        print(
            f"   - 非峰值时段（{self.off_peak_hours[0]}:00-{self.off_peak_hours[1]}:00）使用：{self.local_config['desc']}")
        print(f"   - 峰值时段（{self.off_peak_hours[1]}:00-{self.off_peak_hours[0]}:00）使用：{self.cloud_config['desc']}")

    def _validate_config(self) -> None:
        if not (self.local_config["api_key"] and self.local_config["base_url"]):
            raise ValueError("本地LLM配置不完整，请在.env中设置LOCAL_LLM_API_KEY和LOCAL_LLM_BASE_URL")
        if not (self.cloud_config["api_key"] and self.cloud_config["base_url"]):
            raise ValueError("云端LLM配置不完整，请在.env中设置CLOUD_LLM_API_KEY和CLOUD_LLM_BASE_URL")

    def _is_off_peak(self) -> bool:
        current_hour = time.localtime().tm_hour
        start_off_peak, end_off_peak = self.off_peak_hours
        return current_hour >= start_off_peak or current_hour < end_off_peak

    def get_client(self) -> Tuple[OpenAI, Dict[str, str]]:
        if self._is_off_peak():
            client = OpenAI(
                api_key=self.local_config["api_key"],
                base_url=self.local_config["base_url"]
            )
            meta = {
                "client_type": "local",
                "desc": self.local_config["desc"],
                "model": self.local_config["model"],
                "hour": time.localtime().tm_hour,
                "is_off_peak": True
            }
        else:
            client = OpenAI(
                api_key=self.cloud_config["api_key"],
                base_url=self.cloud_config["base_url"]
            )
            meta = {
                "client_type": "cloud",
                "desc": self.cloud_config["desc"],
                "model": self.cloud_config["model"],
                "hour": time.localtime().tm_hour,
                "is_off_peak": False
            }

        print(f"🔀 LLM路由：当前{meta['hour']}点，使用{meta['client_type']}客户端（{meta['desc']}）")
        return client, meta

    def get_model(self) -> str:
        return self.local_config["model"] if self._is_off_peak() else self.cloud_config["model"]


# ===================== 8. 嵌入模型（版本管理） =====================
class VersionedEmbeddingModel:
    def __init__(self, batch_size: int = 64):
        self.current_version = os.getenv("EMBEDDING_MODEL_VERSION", "bge-m3-v202405")
        self.model_configs = self._load_version_configs()
        self.batch_size = batch_size

        if self.current_version not in self.model_configs:
            print(f"⚠️ 当前配置的模型版本[{self.current_version}]不存在，使用默认版本")
            self.current_version = next(iter(self.model_configs.keys())) if self.model_configs else "unknown"

        self._load_model(self.current_version)
        print(f"✅ 嵌入模型初始化完成（版本：{self.current_version}，模型：{self.embedding_model}）")

    def _load_version_configs(self) -> Dict[str, Dict[str, str]]:
        version_configs = {}
        base_config = {
            "api_key": os.getenv("EMBEDDING_API_KEY", os.getenv("LOCAL_API_KEY")),
            "base_url": os.getenv("EMBEDDING_BASE_URL", os.getenv("LOCAL_BASE_URL")),
            "model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        }
        version_configs["bge-m3-v202405"] = base_config

        extra_versions = os.getenv("EXTRA_EMBEDDING_VERSIONS", "").split(",") if os.getenv(
            "EXTRA_EMBEDDING_VERSIONS") else []
        for ver in extra_versions:
            ver = ver.strip()
            if not ver:
                continue
            version_configs[ver] = {
                "api_key": os.getenv(f"EMBEDDING_API_KEY_{ver.upper()}", base_config["api_key"]),
                "base_url": os.getenv(f"EMBEDDING_BASE_URL_{ver.upper()}", base_config["base_url"]),
                "model_name": os.getenv(f"EMBEDDING_MODEL_{ver.upper()}", base_config["model_name"])
            }

        print(f"✅ 加载嵌入模型版本配置：{sorted(version_configs.keys())}")
        return version_configs

    def _load_model(self, version: str) -> None:
        if version not in self.model_configs:
            raise ValueError(f"不支持的模型版本：{version}，可用版本：{sorted(self.model_configs.keys())}")

        config = self.model_configs[version]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.embedding_model = config["model_name"]

        if not self.api_key or not self.base_url:
            raise ValueError(f"模型版本[{version}]配置不完整，缺少api_key或base_url")

    def switch_version(self, target_version: str) -> bool:
        if target_version == self.current_version:
            print(f"ℹ️ 当前已使用模型版本[{target_version}]，无需切换")
            return True
        if target_version not in self.model_configs:
            print(f"❌ 无法切换到版本[{target_version}]，可用版本：{sorted(self.model_configs.keys())}")
            return False

        try:
            self._load_model(target_version)
            self.current_version = target_version
            print(f"✅ 成功切换嵌入模型版本：{target_version}（模型：{self.embedding_model}）")
            return True
        except Exception as e:
            print(f"❌ 切换版本[{target_version}]失败：{str(e)}，保持当前版本[{self.current_version}]")
            return False

    def get_available_versions(self) -> List[str]:
        return sorted(self.model_configs.keys())

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_text_embedding(
            texts,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


# ===================== 9. 优化的Prompt构造函数（生成可控性核心） =====================
def build_prompt(query, context_chunks):
    context = "\n".join(
        [f"内容：{c['content']}\n来源：{c['metadata']['file_name']} P{c['metadata']['page']}" for c in context_chunks])
    return f"""
    任务：基于以下上下文回答问题，严格遵循规则：
    1. 答案必须完全来自上下文，不得编造信息；
    2. 若上下文不足，直接回复"无法回答"；
    3. 必须用JSON格式输出，包含"answer"和"source"字段（source为来源列表）。

    上下文：
    {context}

    问题：{query}

    输出示例：
    {{
        "answer": "2023年净利润为1000万元",
        "source": ["XX财报.pdf P15"]
    }}
    """


# ===================== 10. 核心RAG类（整合优化） =====================
class SimpleRAG:
    def __init__(self, chunk_json_path: str, batch_size: int = 32,
                 max_chunk_tokens: int = 512, slide_step: int = 200,
                 milvus_dim: int = 1536):
        self.enable_finance_mode = os.getenv('ENABLE_FINANCE_MODE', 'false').lower() == 'true'
        self.enable_fp8 = os.getenv('ENABLE_FP8_INFERENCE', 'false').lower() == 'true'
        self.max_chunk_tokens = max_chunk_tokens
        self.slide_step = slide_step

        self.audit_logger = AuditLogManager(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", 6379)),
            redis_db=int(os.getenv("REDIS_DB", 0))
        )

        self.loader = PageChunkLoader(chunk_json_path)
        self.raw_chunks = self.loader.load_chunks()

        self.embedding_model = VersionedEmbeddingModel(batch_size=batch_size)
        self.current_embedding_version = self.embedding_model.current_version

        self.vector_store = MilvusVectorStore(
            collection_name=os.getenv("MILVUS_COLLECTION", "rag_finance_chunks"),
            dim=milvus_dim
        )

        self.reranker = FlagReranker(
            'BAAI/bge-reranker-large',
            use_fp16=True,
            devices=["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        )
        print("✅ BGE重排模型初始化完成")

        self.llm_router = LLMClientRouter()
        self.fallback_retriever = KeywordFallbackRetriever()
        self.bm25_retriever = None  # 延迟初始化，等待processed_chunks

        if self.enable_finance_mode:
            try:
                from finance_rag_extension import FinanceVectorStore
                self.vector_store = FinanceVectorStore()
                print("✅ 启用金融版向量存储")
            except ImportError:
                print("⚠️ 未找到finance_rag_extension，使用默认Milvus向量存储")

    def setup(self):
        print("=" * 50)
        print("开始RAG系统初始化...")

        print("1. 处理超长Chunk（滑动窗口+数字优先截断）...")
        processed_chunks = []
        for raw_chunk in tqdm(self.raw_chunks, desc="   Chunk预处理"):
            sub_chunks = process_long_chunk(
                chunk=raw_chunk,
                max_tokens=self.max_chunk_tokens,
                slide_step=self.slide_step
            )
            processed_chunks.extend(sub_chunks)
        print(f"   预处理后总Chunk数: {len(processed_chunks)}（含子Chunk）")

        # 初始化BM25检索器（核心新增）
        self.bm25_retriever = BM25Retriever(processed_chunks)
        print(f"✅ BM25检索器初始化完成（{len(processed_chunks)}个Chunk）")

        self.fallback_retriever.update_chunks(processed_chunks)

        print(f"2. 生成文本嵌入（版本：{self.current_embedding_version}）...")
        start_embed = time.time()
        embeddings = self.embedding_model.embed_texts([c['content'] for c in processed_chunks])
        embed_duration = round((time.time() - start_embed) * 1000, 2)
        print(f"   嵌入生成完成（{len(embeddings)}个向量），耗时：{embed_duration}ms")

        print("3. 向向量库写入带版本的Chunk...")
        self.vector_store.add_chunks_with_version(
            chunks=processed_chunks,
            embeddings=embeddings,
            model_version=self.current_embedding_version
        )

        print("✅ RAG系统初始化完成！")
        print(f"   - 向量库类型：{'Milvus' if MILVUS_AVAILABLE and self.vector_store.collection else '内存模式'}")
        print(
            f"   - 嵌入模型版本：{self.current_embedding_version}（支持版本：{self.embedding_model.get_available_versions()}）")
        print(f"   - 混合检索：向量检索 + BM25（RRF融合）")
        print(f"   - 重排序模型：BAAI/bge-reranker-large（{'GPU' if torch.cuda.is_available() else 'CPU'}模式）")
        print(f"   - 兜底检索：{'启用' if TFIDF_AVAILABLE else '禁用'}")
        print(f"   - 金融模式：{'启用' if self.enable_finance_mode else '禁用'}")
        print("=" * 50)

    def switch_embedding_version(self, target_version: str) -> bool:
        embed_switch_ok = self.embedding_model.switch_version(target_version)
        if embed_switch_ok:
            self.current_embedding_version = target_version
            self.vector_store._load_all_versions()
            print(
                f"✅ 版本切换完成，当前嵌入版本：{self.current_embedding_version}，向量库可用版本：{sorted(self.vector_store.available_versions)}")
        return embed_switch_ok

    def retrieve_and_rerank(self, query: str, top_k: int = 3, candidate_top_k: int = 20) -> Tuple[
        List[Dict[str, Any]], str, bool]:
        start_total = time.time()
        print("\n" + "=" * 50)
        print(f"开始处理查询：{query[:50]}...")

        print(f"1. 生成查询向量（版本：{self.current_embedding_version}）...")
        start_embed = time.time()
        q_emb = self.embedding_model.embed_text(query)
        embed_duration = round((time.time() - start_embed) * 1000, 2)
        print(f"   查询向量生成耗时：{embed_duration}ms")

        # 混合检索（向量+BM25+RRF融合）
        print(f"2. 执行混合检索（向量+BM25，RRF融合）...")
        hybrid_results, version_matched = hybrid_retrieve(
            query=query,
            vector_retriever=q_emb,
            bm25_retriever=self.bm25_retriever,
            vector_store=self.vector_store,
            embedding_version=self.current_embedding_version,
            top_k=candidate_top_k,
            candidate_top_k=candidate_top_k
        )

        # 检查是否需要兜底
        retrieval_type = "hybrid"
        final_candidates = hybrid_results
        if len(hybrid_results) < candidate_top_k // 2:
            print(f"⚠️ 混合检索结果不足（{len(hybrid_results)}个），触发兜底检索")
            fallback_results = self.fallback_retriever.retrieve(query, top_k=candidate_top_k)
            hybrid_ids = {res["id"] for res in hybrid_results}
            combined_results = hybrid_results + [res for res in fallback_results if res["id"] not in hybrid_ids]
            final_candidates = combined_results[:candidate_top_k]
            retrieval_type = "hybrid+fallback"

        # BGE重排
        print(f"3. 使用BGE重排模型对{len(final_candidates)}个候选结果精排序...")
        start_rerank = time.time()
        pairs = [(query, chunk["content"]) for chunk in final_candidates]
        scores = self.reranker.compute_score(pairs)
        reranked_chunks = [
            chunk for _, chunk in sorted(zip(scores, final_candidates), key=lambda x: x[0], reverse=True)
        ][:top_k]
        rerank_duration = round((time.time() - start_rerank) * 1000, 2)
        print(f"   重排完成，耗时：{rerank_duration}ms")

        total_duration = round((time.time() - start_total) * 1000, 2)
        print(f"✅ 查询完成（检索类型：{retrieval_type}，版本匹配：{version_matched}），总耗时：{total_duration}ms")
        print("=" * 50)

        return reranked_chunks, retrieval_type, version_matched

    def generate_answer(self, question: str, top_k: int = 3, user_id: Optional[str] = None) -> Dict[str, Any]:
        current_client_meta = self.llm_router.get_client()[1]
        log_metadata = {
            "user_id": user_id,
            "top_k": top_k,
            "llm_client_type": current_client_meta["client_type"],
            "llm_model": current_client_meta["model"],
            "embedding_model_version": self.current_embedding_version,
            "vector_store_type": "Milvus" if MILVUS_AVAILABLE and self.vector_store.collection else "Memory"
        }
        log_id = self.audit_logger.create_pre_log(question, log_metadata)
        print(f"📌 生成审计日志ID: {log_id}")

        try:
            retrieved_chunks, retrieval_type, version_matched = self.retrieve_and_rerank(
                question, top_k=top_k, candidate_top_k=20
            )

            # 构建上下文
            context_chunks = []
            for chunk in retrieved_chunks:
                context_chunks.append({
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                })

            # 使用优化后的Prompt（核心修改）
            prompt = build_prompt(question, context_chunks)

            # 调用LLM
            print("5. 调用大模型生成回答...")
            start_llm = time.time()
            llm_client, llm_meta = self.llm_router.get_client()
            request_kwargs = {
                "model": llm_meta["model"],
                "messages": [
                    {"role": "system", "content": "你是专业的问答助手，严格按照指定格式输出结果"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
                "stream": False
            }
            if self.enable_fp8 and llm_meta["client_type"] == "local":
                request_kwargs["extra_body"] = {"precision": "fp8"}
                print(f"   启用FP8精度推理（本地GPU模式）")

            completion = llm_client.chat.completions.create(**request_kwargs)
            llm_duration = round((time.time() - start_llm) * 1000, 2)
            print(f"   大模型推理耗时：{llm_duration}ms（{llm_meta['desc']}）")

            # 解析JSON结果
            raw = completion.choices[0].message.content.strip()
            try:
                answer_data = json.loads(raw)
                if not isinstance(answer_data, dict) or "answer" not in answer_data or "source" not in answer_data:
                    raise ValueError("输出格式缺少必要字段")
                answer = answer_data["answer"]
                source = answer_data["source"]
                note = f"检索类型：{retrieval_type}，模型版本：{self.current_embedding_version}"
            except json.JSONDecodeError:
                answer = "无法解析模型输出（格式错误）"
                source = []
                note = f"JSON解析失败，原始输出：{raw[:100]}"
            except ValueError as e:
                answer = f"输出格式错误：{str(e)}"
                source = []
                note = f"格式校验失败，原始输出：{raw[:100]}"

            # 更新审计日志
            self.audit_logger.update_log_with_answer(
                log_id=log_id,
                answer=json.dumps(answer_data, ensure_ascii=False) if 'answer_data' in locals() else answer,
                retrieval_chunks=retrieved_chunks,
                status="completed",
                model_version=self.current_embedding_version,
                retrieval_type=retrieval_type
            )
            asyncio.create_task(self.audit_logger.write_audit_log_to_db(log_id))

            total_duration = sum([
                chunk.get('retrieval_duration_ms', 0) for chunk in retrieved_chunks[:1]
            ]) + llm_duration
            return {
                "question": question,
                "answer": answer,
                "source": source,
                "note": note,
                "retrieval_chunks": retrieved_chunks,
                "audit_log_id": log_id,
                "llm_route_info": llm_meta,
                "version_info": {
                    "embedding_version": self.current_embedding_version,
                    "version_matched": version_matched,
                    "available_versions": sorted(self.vector_store.available_versions)
                },
                "performance_stats": {
                    "retrieval_duration_ms": round(total_duration - llm_duration, 2),
                    "llm_duration_ms": llm_duration,
                    "total_duration_ms": round(total_duration, 2)
                }
            }

        except Exception as e:
            error_msg = f"生成回答失败: {str(e)}"
            print(error_msg)
            self.audit_logger.update_log_with_answer(
                log_id=log_id,
                answer=error_msg,
                retrieval_chunks=[],
                status="failed",
                model_version=self.current_embedding_version,
                retrieval_type="error"
            )
            asyncio.create_task(self.audit_logger.write_audit_log_to_db(log_id))
            return {
                "question": question,
                "answer": error_msg,
                "source": [],
                "note": f"系统错误，模型版本：{self.current_embedding_version}",
                "retrieval_chunks": [],
                "audit_log_id": log_id,
                "llm_route_info": current_client_meta,
                "version_info": {
                    "embedding_version": self.current_embedding_version,
                    "version_matched": False,
                    "available_versions": sorted(self.vector_store.available_versions)
                },
                "error": str(e)
            }


# ===================== 11. 辅助类 =====================
class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ===================== 12. 主函数 =====================
if __name__ == '__main__':
    MAX_CHUNK_TOKENS = 512
    SLIDE_STEP = 200
    MILVUS_DIM = 1536
    TOP_K = 3
    CANDIDATE_TOP_K = 20

    chunk_json_path = os.path.join(os.path.dirname(__file__), 'all_pdf_page_chunks.json')
    rag = SimpleRAG(
        chunk_json_path=chunk_json_path,
        max_chunk_tokens=MAX_CHUNK_TOKENS,
        slide_step=SLIDE_STEP,
        milvus_dim=MILVUS_DIM
    )
    rag.setup()

    # 测试1：正常查询
    print("\n" + "=" * 60)
    print("【测试1：正常查询】")
    test_question1 = "2023年第三季度的净利润是多少？"
    result1 = rag.generate_answer(test_question1, top_k=TOP_K, user_id="test_user_001")
    print(f"问题：{test_question1}")
    print(f"回答：{result1['answer']}")
    print(f"来源：{result1['source']}")
    print(
        f"版本信息：当前{result1['version_info']['embedding_version']}，匹配：{result1['version_info']['version_matched']}")

    # 测试2：版本切换场景
    print("\n" + "=" * 60)
    print("【测试2：版本切换】")
    target_version = "bge-m3-v202404" if "bge-m3-v202404" in rag.embedding_model.get_available_versions() else \
    rag.embedding_model.get_available_versions()[0]
    rag.switch_embedding_version(target_version)
    test_question2 = "联邦制药2024年营业收入同比增长率为多少？"
    result2 = rag.generate_answer(test_question2, top_k=TOP_K, user_id="test_user_002")
    print(f"问题：{test_question2}")
    print(f"回答：{result2['answer']}")
    print(f"来源：{result2['source']}")

    # 批量测试
    TEST_SAMPLE_NUM = 5
    test_path = os.path.join(os.path.dirname(__file__), 'datas/多模态RAG图文问答挑战赛测试集.json')
    if os.path.exists(test_path):
        print("\n" + "=" * 60)
        print(f"开始批量测试（{TEST_SAMPLE_NUM}条样本）...")
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        all_indices = list(range(len(test_data)))
        selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM)) if len(
            test_data) > TEST_SAMPLE_NUM else all_indices

        for idx in selected_indices:
            data_item = test_data[idx]
            question = data_item['question']
            print(f"\n[{selected_indices.index(idx) + 1}/{len(selected_indices)}] 处理: {question[:30]}...")
            result = rag.generate_answer(question, top_k=TOP_K, user_id=f"batch_user_{idx}")
            print(f"回答：{result['answer'][:100]}...")
            print(f"来源：{result['source']}")