import fitz  # PyMuPDF
import json
from pathlib import Path

def process_pdfs_to_chunks(datas_dir: Path, output_json_path: Path):
    """
    使用 PyMuPDF 直接从 PDF 提取每页文本，并生成最终的 JSON 文件。
    
    Args:
        datas_dir (Path): 包含 PDF 文件的输入目录。
        output_json_path (Path): 最终输出的 JSON 文件路径。
    """
    all_chunks = []
    
    # 递归查找 datas_dir 目录下的所有 .pdf 文件
    pdf_files = list(datas_dir.rglob('*.pdf'))
    if not pdf_files:
        print(f"警告：在目录 '{datas_dir}' 中未找到任何 PDF 文件。")
        return

    print(f"找到 {len(pdf_files)} 个 PDF 文件，开始处理...")

    for pdf_path in pdf_files:
        file_name_stem = pdf_path.stem  # 文件名（不含扩展名）
        full_file_name = pdf_path.name  # 完整文件名（含扩展名）
        print(f"  - 正在处理: {full_file_name}")

        try:
            # 使用 with 语句确保文件被正确关闭
            with fitz.open(pdf_path) as doc:
                # 遍历 PDF 的每一页
                for page_idx, page in enumerate(doc):
                    # 提取当前页面的所有文本
                    content = page.get_text("text")
                    
                    # 如果页面没有文本内容，则跳过
                    if not content.strip():
                        continue

                    # 构建符合最终格式的 chunk 字典
                    chunk = {
                        "id": f"{file_name_stem}_page_{page_idx}",
                        "content": content,
                        "metadata": {
                            "page": page_idx,  # 0-based page index
                            "file_name": full_file_name
                        }
                    }
                    all_chunks.append(chunk)
        except Exception as e:
            print(f"处理文件 '{pdf_path}' 时发生错误: {e}")

    # 确保输出目录存在
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 将所有 chunks 写入一个 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！所有内容已保存至: {output_json_path}")

def main():
    base_dir = Path(__file__).parent
    datas_dir = base_dir / 'datas'
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'
    
    process_pdfs_to_chunks(datas_dir, chunk_json_path)

if __name__ == '__main__':
    main()