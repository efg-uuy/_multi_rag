from typing import List, Dict
from transformers import pipeline  # 或根据实际使用的LLM库调整


# 假设已初始化LLM（根据实际场景替换，如OpenAI、本地化模型等）
def init_llm():
    """初始化LLM模型（示例使用Hugging Face pipeline）"""
    return pipeline(
        "text-generation",
        model=os.getenv("LLM_MODEL", "Qwen/Qwen2-7B-Instruct"),
        torch_dtype=torch.float16,
        device_map="auto"
    )


llm = init_llm()


def generate_answer_with_source(context_chunks: List[Dict], question: str) -> str:
    """
    生成包含标准化来源标注的答案
    :param context_chunks: 检索到的分块列表（含完整metadata）
    :param question: 用户问题
    :return: 带标准化来源的答案
    """
    if not context_chunks:
        return f"未找到相关信息回答问题：{question}\n来源：无"

    # 1. 提取并格式化来源（严格匹配metadata中的字段）
    sources = []
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "未知文件")
        start_page = meta.get("start_page", 0) + 1  # 转为显示用页码（原page_idx为索引，+1后为第N页）
        end_page = meta.get("end_page", 0) + 1

        # 统一格式：单页（P1）、跨页（P1-2）
        if start_page == end_page:
            page_str = f"P{start_page}"
        else:
            page_str = f"P{start_page}-{end_page}"

        sources.append(f"{file_name}（{page_str}）")

    # 2. 来源去重（避免同一内容重复标注）
    unique_sources = list(sorted(set(sources)))
    source_str = "; ".join(unique_sources)

    # 3. 构建带来源约束的Prompt（强制LLM遵循格式）
    prompt = f"""
    任务：基于提供的参考内容回答用户问题，答案末尾必须以「来源：xxx」格式标注所有参考的文档来源。
    要求：
    1. 来源必须完全匹配提供的「可用来源列表」，不得虚构或遗漏；
    2. 来源格式严格为「文件名（P页码）」或「文件名（P页码-页码）」；
    3. 答案需简洁准确，与问题强相关。

    参考内容：
    {[chunk['content'] for chunk in context_chunks]}

    用户问题：{question}

    可用来源列表：{source_str}

    回答示例：
    2023年公司营收为120.5亿元，同比增长15.2%。
    来源：2023年报.pdf（P5）; 季度财报Q4.pdf（P2-3）
    """

    # 4. 调用LLM生成答案
    try:
        response = llm(
            prompt,
            max_new_tokens=1024,
            temperature=0.3,  # 降低随机性，确保来源格式正确
            stop=["<|end_of_text|>"]
        )
        answer = response[0]["generated_text"].strip()
    except Exception as e:
        print(f"LLM调用失败：{str(e)}")
        answer = f"无法生成答案：{str(e)}"

    # 5. 二次校验：确保来源标注存在且格式正确
    if "来源：" not in answer:
        # 兜底补充来源（避免遗漏）
        answer += f"\n来源：{source_str}"
    else:
        # 简单校验格式（可根据需要增强正则匹配）
        import re
        source_pattern = r"来源：.*（P\d+(-\d+)?）"
        if not re.search(source_pattern, answer):
            # 格式错误，替换为标准格式
            answer = answer.split("来源：")[0].strip() + f"\n来源：{source_str}"

    return answer


def retrieve_chunks(query: str, chunk_path: str = "all_pdf_page_chunks.json") -> List[Dict]:
    """
    模拟检索逻辑（实际场景需替换为向量数据库检索，如FAISS、Milvus等）
    :param query: 用户查询
    :param chunk_path: 分块文件路径
    :return: 匹配的分块列表
    """
    with open(chunk_path, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    # 简单匹配（实际需用向量相似度计算）
    matched_chunks = [chunk for chunk in all_chunks if any(keyword in chunk['content'] for keyword in query.split())]
    return matched_chunks[:3]  # 返回前3个匹配分块


# 示例：完整RAG流程调用
def rag_pipeline(query: str):
    """完整RAG流程：检索 → 生成答案（带来源）"""
    # 1. 检索相关分块
    context_chunks = retrieve_chunks(query)
    # 2. 生成带来源的答案
    answer = generate_answer_with_source(context_chunks, query)
    return answer


# 测试
if __name__ == "__main__":
    test_query = "2023年公司营收是多少？"
    result = rag_pipeline(test_query)
    print("RAG回答结果：")
    print(result)