# ===================== 基于章节边界的分块逻辑与检索工具 =====================
import re
import time
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi  # 需确保安装rank-bm25

# 检查TF-IDF依赖是否可用
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False


# ===================== 新增：滑动窗口分割函数 =====================
def slide_window_split(text: str, max_tokens: int, slide_step: int) -> List[str]:
    """使用滑动窗口分割超长文本"""
    tokens = text.split()
    sub_paras = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        sub_para = ' '.join(tokens[start:end])
        sub_paras.append(sub_para)
        # 防止步长为0导致死循环
        start += slide_step if slide_step > 0 else max_tokens
    return sub_paras


# ===================== 新增：基于章节逻辑的分块函数 =====================
def process_chunks_by_logical_structure(
        logical_pages: List[Dict[str, Any]],
        max_tokens: int = 512,
        slide_step: int = 200
) -> List[Dict[str, Any]]:
    """
    基于文档逻辑结构（章节、段落）进行分块，保留表格和图表的完整性
    优先以logical_pages（章节）为单位，超长章节内部再细分
    """
    processed_chunks = []

    for page in logical_pages:
        # 提取章节级内容与元数据
        content = page["content"].strip()
        metadata = {
            "logical_id": page["logical_id"],  # 章节唯一标识
            "physical_page": page["physical_page"],  # 原始物理页码
            "para_simhashes": page["para_simhashes"],  # 段落特征值
            "block_types": page.get("block_types", [])  # 块类型（表格/图片等）
        }

        # 为特殊元素添加标记（提升检索精度）
        if "table" in metadata["block_types"]:
            content = f"[TABLE]\n{content}\n[/TABLE]"
        if "image" in metadata["block_types"] or "figure" in metadata["block_types"]:
            content = f"[CHART]\n{content}\n[/CHART]"

        # 按空行分割段落（保留段落逻辑）
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]  # 过滤空段落

        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_tokens = para.split()
            para_length = len(para_tokens)

            # 处理超长段落（单个段落超过最大长度）
            if para_length > max_tokens:
                if current_chunk:
                    # 先保存当前累积的合法块
                    processed_chunks.append({
                        "content": '\n\n'.join(current_chunk),
                        "metadata": metadata.copy()
                    })
                    current_chunk = []
                    current_length = 0

                # 滑动窗口分割超长段落
                sub_paras = slide_window_split(para, max_tokens, slide_step)
                for sub_para in sub_paras:
                    processed_chunks.append({
                        "content": sub_para,
                        "metadata": metadata.copy()
                    })
                continue

            # 累积段落至当前块，超出长度则保存
            if current_length + para_length > max_tokens:
                processed_chunks.append({
                    "content": '\n\n'.join(current_chunk),
                    "metadata": metadata.copy()
                })
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length

        # 添加当前章节剩余内容
        if current_chunk:
            processed_chunks.append({
                "content": '\n\n'.join(current_chunk),
                "metadata": metadata.copy()
            })

    return processed_chunks


# ===================== 兜底检索工具（增强BM25融合） =====================
class KeywordFallbackRetriever:
    """兜底检索器：结合TF-IDF和BM25，增强关键词密集型问题召回"""

    def __init__(self, all_chunks: List[Dict[str, Any]] = None):
        self.all_chunks = all_chunks or []
        self.tfidf_vectorizer = None
        self.chunk_vectors = None
        self.bm25 = None
        self.bm25_tokenized_corpus = []
        self._build_tfidf_index()
        self._build_bm25_index()
        self._build_rule_map()

    def _build_bm25_index(self) -> None:
        """构建BM25索引，优化关键词检索"""
        if not self.all_chunks:
            print("⚠️ BM25索引构建条件不足（缺少数据）")
            return

        try:
            self.bm25_tokenized_corpus = [chunk["content"].split() for chunk in self.all_chunks]
            self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
            print(f"✅ BM25索引构建完成（{len(self.all_chunks)}个Chunk）")
        except Exception as e:
            print(f"⚠️ BM25索引构建失败：{str(e)}")
            self.bm25 = None

    def _build_tfidf_index(self) -> None:
        """构建TF-IDF索引，用于关键词相似度匹配"""
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
        """构建规则匹配映射（针对金融场景优化）"""
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
        """更新Chunk数据并重建所有索引"""
        self.all_chunks = chunks
        self._build_tfidf_index()
        self._build_bm25_index()
        self._build_rule_map()

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """BM25关键词检索"""
        if not self.bm25 or not self.all_chunks:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.all_chunks[i] for i in top_indices]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """兜底检索执行：融合BM25+TF-IDF+规则"""
        start_time = time.time()
        if not self.all_chunks:
            print("❌ 兜底检索无可用Chunk数据")
            return []

        # 1. BM25检索（关键词密集型问题优先）
        bm25_results = self.bm25_search(query, top_k=top_k * 2)

        # 2. 规则匹配（优先级最高）
        rule_matched = []
        for chunk in bm25_results:
            match_score = 0
            # 金融关键词规则匹配
            for keyword, rule_func in self.rule_map.items():
                if keyword in query and rule_func(chunk):
                    match_score += 2
            # 数字/日期匹配
            query_numbers = set(self.number_pattern.findall(query))
            chunk_numbers = set(self.number_pattern.findall(chunk["content"]))
            if query_numbers & chunk_numbers:
                match_score += 1.5

            if match_score > 0:
                rule_matched.append((chunk, match_score))

        # 3. TF-IDF关键词相似度匹配
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

        # 4. 综合排序
        all_candidates = rule_matched + \
                         [(c, 1.0) for c in bm25_results if c not in [rc[0] for rc in rule_matched]] + \
                         tfidf_matched
        if not all_candidates:
            all_candidates = [(chunk, 0.1) for chunk in self.all_chunks[:top_k]]

        # 按得分排序，相同得分优先短文本
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


# ===================== 主流程：加载逻辑页并处理 =====================
def main():
    import json
    from pathlib import Path

    # 配置路径（根据实际项目结构调整）
    input_base_dir = Path("_multi_rag/data_base_json_content")  # mineru解析结果目录
    output_chunks_path = Path("_multi_rag/all_pdf_logical_chunks.json")  # 输出分块结果

    # 1. 加载mineru生成的logical_pages
    logical_pages = []
    for pdf_dir in input_base_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        # 查找middle_json文件（包含logical_pages）
        middle_json_path = pdf_dir / "auto" / f"{pdf_dir.name}_middle.json"
        if not middle_json_path.exists():
            print(f"⚠️ 未找到{pdf_dir.name}的middle_json文件，跳过")
            continue

        with open(middle_json_path, "r", encoding="utf-8") as f:
            middle_json = json.load(f)
            # 提取logical_pages并补充block_types信息
            for page in middle_json.get("logical_pages", []):
                # 从原始页面信息中获取block_types
                page_idx = page["physical_page"]
                for raw_page in middle_json["pdf_info"]["pages"]:
                    if raw_page["page_idx"] == page_idx:
                        page["block_types"] = [b["type"] for b in raw_page["blocks"]]
                        break
            logical_pages.extend(middle_json.get("logical_pages", []))

    if not logical_pages:
        print("❌ 未加载到任何logical_pages数据")
        return

    # 2. 使用新分块函数处理
    chunks = process_chunks_by_logical_structure(
        logical_pages=logical_pages,
        max_tokens=512,
        slide_step=200
    )
    print(f"✅ 分块完成，共生成{len(chunks)}个Chunk")

    # 3. 保存分块结果
    output_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ 分块结果已保存至：{output_chunks_path}")

    # 4. 初始化检索器
    retriever = KeywordFallbackRetriever(chunks)
    # 示例检索
    test_query = "净利润同比增长率"
    results = retriever.retrieve(test_query, top_k=3)
    print(f"\n测试检索 '{test_query}' 结果：")
    for i, res in enumerate(results, 1):
        print(f"\n结果{i}（相似度：{res['similarity']}）：")
        print(f"章节ID：{res['metadata']['logical_id']}")
        print(f"内容预览：{res['content'][:200]}...")


if __name__ == "__main__":
    main()
