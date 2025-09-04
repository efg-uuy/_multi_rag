import os
import time
from pathlib import Path
import json
from collections import defaultdict
import asyncio
from dotenv import load_dotenv  # 新增：导入dotenv
from image_utils.async_image_analysis import AsyncImageAnalysis
# 新增：导入金融插件
from finance_parser_plugin import fix_finance_cross_page_tables, analyze_finance_chart

# 新增：加载环境变量
load_dotenv()


def parse_all_pdfs(datas_dir, output_base_dir):
    """
    步骤1：解析所有PDF，输出内容到 data_base_json_content/
    """
    from mineru_parse_pdf import do_parse
    datas_dir = Path(datas_dir)
    output_base_dir = Path(output_base_dir)
    pdf_files = list(datas_dir.rglob('*.pdf'))
    if not pdf_files:
        print(f"未找到PDF文件于: {datas_dir}")
        return
    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="pipeline",
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True
        )
        print(f"已输出: {output_dir / 'auto' / (file_name + '_content_list.json')}")


def group_by_page(content_list):
    pages = defaultdict(list)
    for item in content_list:
        page_idx = item.get('page_idx', 0)
        pages[page_idx].append(item)
    return dict(pages)


def build_element_relation(page_elements: list) -> list:
    """
    为表格/图片添加相邻文本关联信息
    :param page_elements: 单页包含的所有元素列表（文本、表格、图片）
    :return: 处理后的元素列表
    """
    for i, elem in enumerate(page_elements):
        # 仅处理表格和图片
        if elem["type"] not in ["table", "image"]:
            continue
        
        # 查找前一个文本元素（标题或段落）
        prev_text = ""
        for j in range(i-1, -1, -1):
            if page_elements[j]["type"] == "text":
                prev_text = page_elements[j].get("text", "")[:50]  # 取前50字作为摘要
                break
        
        # 查找后一个文本元素
        next_text = ""
        for j in range(i+1, len(page_elements)):
            if page_elements[j]["type"] == "text":
                next_text = page_elements[j].get("text", "")[:50]
                break
        
        # 添加关联信息
        elem["related_context"] = {
            "prev_text": prev_text,
            "next_text": next_text,
            "position": f"第{i+1}个元素"  # 元素在页面中的位置索引
        }
    return page_elements


def item_to_markdown(item, enable_image_caption=True):
    """
    enable_image_caption: 是否启用多模态视觉分析（图片caption补全），默认True。
    """
    # 默认API参数：硅基流动Qwen/Qwen2.5-VL-32B-Instruct
    vision_provider = "guiji"
    vision_model = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
    vision_api_key = os.getenv("LOCAL_API_KEY")
    vision_base_url = os.getenv("LOCAL_BASE_URL")

    if item['type'] == 'text':
        level = item.get('text_level', 0)
        text = item.get('text', '')
        if level == 1:
            return f"# {text}\n\n"
        elif level == 2:
            return f"## {text}\n\n"
        else:
            return f"{text}\n\n"
    elif item['type'] == 'image':
        captions = item.get('image_caption', [])
        caption = captions[0] if captions else ''
        img_path = item.get('img_path', '')
        # 如果没有caption，且允许视觉分析，调用多模态API补全
        if enable_image_caption and not caption and img_path and os.path.exists(img_path):
            max_retries = 3
            retry_delay = 2  # 秒
            for attempt in range(max_retries):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    async def get_caption():
                        async with AsyncImageAnalysis(
                                provider=vision_provider,
                                api_key=vision_api_key,
                                base_url=vision_base_url,
                                vision_model=vision_model
                        ) as analyzer:
                            result = await analyzer.analyze_image(local_image_path=img_path)
                            return result.get('title') or result.get('description') or ''

                    caption = loop.run_until_complete(get_caption())
                    loop.close()
                    if caption:
                        item['image_caption'] = [caption]
                    break  # 成功则退出重试
                except Exception as e:
                    print(f"图片解释失败（尝试{attempt+1}/{max_retries}）: {img_path}, {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
        md = f"![{caption}]({img_path})\n"
        return md + "\n"
    elif item['type'] == 'table':
        captions = item.get('table_caption', [])
        caption = captions[0] if captions else ''
        table_html = item.get('table_body', '')
        img_path = item.get('img_path', '')
        md = ''
        if caption:
            md += f"**{caption}**\n"
        if img_path:
            md += f"![{caption}]({img_path})\n"
        md += f"{table_html}\n\n"
        return md
    else:
        return '\n'


def assemble_pages_to_markdown(pages, pdf_path):
    # 强制启用金融模式
    enable_finance_mode = os.getenv('ENABLE_FINANCE_MODE', 'true').lower() == 'true'

    page_md = {}
    for page_idx in sorted(pages.keys()):
        # 处理金融表格（如果启用）
        if enable_finance_mode:
            # 提取当前页的表格并修复跨页表格
            tables = [item for item in pages[page_idx] if item['type'] == 'table']
            if tables:
                fixed_tables = fix_finance_cross_page_tables(tables, page_idx, pdf_path)
                # 更新页面中的表格数据
                table_idx = 0
                for i, item in enumerate(pages[page_idx]):
                    if item['type'] == 'table' and table_idx < len(fixed_tables):
                        pages[page_idx][i] = fixed_tables[table_idx]
                        table_idx += 1

        # 处理金融图表分析（如果启用）
        if enable_finance_mode:
            images = [item for item in pages[page_idx] if item['type'] == 'image']
            if images:
                chart_analyzer = AsyncImageAnalysis(provider="zhipu")
                analyzed_images = [analyze_finance_chart(img, chart_analyzer) for img in images]
                # 更新页面中的图表数据
                img_idx = 0
                for i, item in enumerate(pages[page_idx]):
                    if item['type'] == 'image' and img_idx < len(analyzed_images):
                        pages[page_idx][i] = analyzed_images[img_idx]
                        img_idx += 1

        # 生成当前页的Markdown内容
        md = ''
        for item in pages[page_idx]:
            md += item_to_markdown(item, enable_image_caption=True)
        page_md[page_idx] = md

    return page_md


def process_all_pdfs_to_page_json(input_base_dir, output_base_dir):
    """
    步骤2：将content_list.json按页组织，生成page_content.json（含Markdown内容）
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    pdf_dirs = [d for d in input_base_dir.iterdir() if d.is_dir()]
    if not pdf_dirs:
        print(f"⚠️  未在 {input_base_dir} 找到PDF解析目录")
        return

    for pdf_dir in pdf_dirs:
        file_name = pdf_dir.name
        pdf_file_name = f"{file_name}.pdf"

        # 定位content_list.json
        content_list_path = pdf_dir / 'auto' / f'{file_name}_content_list.json'
        if not content_list_path.exists():
            content_list_path = pdf_dir / file_name / 'auto' / f'{file_name}_content_list.json'
        if not content_list_path.exists():
            print(f"❌ 未找到 {file_name} 的content_list.json，跳过")
            continue

        # 读取并处理content_list
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        
        # 按页分组
        pages = group_by_page(content_list)
        
        # 新增：为每一页的元素添加关联信息
        for page_idx in pages:
            pages[page_idx] = build_element_relation(pages[page_idx])
        
        # 生成每页Markdown
        page_md = assemble_pages_to_markdown(pages, pdf_file_name)

        # 输出page_content.json（包含关联信息）
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = output_dir / f'{file_name}_page_content.json'
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(page_md, f, ensure_ascii=False, indent=2)

        print(f"✅ 页面内容生成完成: {output_json_path}")


def process_page_content_to_chunks(input_base_dir, output_json_path):
    """
    步骤3：将 page_content.json 合并为 all_pdf_page_chunks.json
    """
    input_base_dir = Path(input_base_dir)
    all_chunks = []
    for pdf_dir in input_base_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        file_name = pdf_dir.name
        page_content_path = pdf_dir / f"{file_name}_page_content.json"
        if not page_content_path.exists():
            sub_dir = pdf_dir / file_name
            page_content_path2 = sub_dir / f"{file_name}_page_content.json"
            if page_content_path2.exists():
                page_content_path = page_content_path2
            else:
                print(f"未找到: {page_content_path} 也未找到: {page_content_path2}")
                continue
        with open(page_content_path, 'r', encoding='utf-8') as f:
            page_dict = json.load(f)
        for page_idx, content in page_dict.items():
            chunk = {
                "id": f"{file_name}_page_{page_idx}",
                "content": content,
                "metadata": {
                    "page": page_idx,
                    "file_name": file_name + ".pdf"
                }
            }
            all_chunks.append(chunk)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"已输出: {output_json_path}")


def main():
    base_dir = Path(__file__).parent
    datas_dir = base_dir / 'datas'
    content_dir = base_dir / 'data_base_json_content'
    page_dir = base_dir / 'data_base_json_page_content'
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'
    # 步骤1：PDF → content_list.json
    parse_all_pdfs(datas_dir, content_dir)
    # 步骤2：content_list.json → page_content.json
    process_all_pdfs_to_page_json(content_dir, page_dir)
    # 步骤3：page_content.json → all_pdf_page_chunks.json
    process_page_content_to_chunks(page_dir, chunk_json_path)
    print("全部处理完成！")


if __name__ == '__main__':
    main()
