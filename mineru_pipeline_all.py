import os
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
            except Exception as e:
                print(f"图片解释失败: {img_path}, {e}")
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
    # 新增：从环境变量获取金融模式开关状态，默认关闭
    enable_finance_mode = os.getenv('ENABLE_FINANCE_MODE', 'false').lower() == 'true'

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
    步骤2：将 content_list.json 转为 page_content.json
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    pdf_dirs = [d for d in input_base_dir.iterdir() if d.is_dir()]
    for pdf_dir in pdf_dirs:
        file_name = pdf_dir.name
        json_path = pdf_dir / 'auto' / f'{file_name}_content_list.json'
        # 构建PDF路径用于金融插件
        pdf_path = Path(input_base_dir.parent) / 'datas' / f'{file_name}.pdf'

        if not json_path.exists():
            sub_dir = pdf_dir / file_name
            json_path2 = sub_dir / 'auto' / f'{file_name}_content_list.json'
            if json_path2.exists():
                json_path = json_path2
            else:
                print(f"未找到: {json_path} 也未找到: {json_path2}")
                continue
        with open(json_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        pages = group_by_page(content_list)
        # 传递pdf_path给assemble_pages_to_markdown
        page_md = assemble_pages_to_markdown(pages, pdf_path)
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = output_dir / f'{file_name}_page_content.json'
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(page_md, f, ensure_ascii=False, indent=2)
        print(f"已输出: {output_json_path}")


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
