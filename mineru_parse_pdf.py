# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path
import hashlib
from simhash import Simhash

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

os.environ['MINERU_MODEL_SOURCE'] = "modelscope"


# ===================== 新增：段落级Simhash生成 =====================
def paragraph_simhash(content, window_size=3):
    """生成段落级Simhash特征，降低排版变动影响"""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return [Simhash(p).value for p in paragraphs[:window_size]]  # 取前N段特征


# ===================== 新增：基于章节的逻辑页ID生成 =====================
def generate_logical_page_id(middle_json):
    logical_pages = []
    for page in middle_json["pdf_info"]["pages"]:
        # 提取章节标题（优先取一级标题，无则取所有标题的拼接）
        headings = [b["content"] for b in page["blocks"] if b["type"] == "heading"]
        if headings:
            # 用标题内容的MD5哈希确保唯一性，避免章节名重复导致ID冲突
            heading_hash = hashlib.md5(headings[0].encode()).hexdigest()
            logical_id = f"chap_{heading_hash}_{page['page_idx']}"
        else:
            logical_id = f"no_heading_{page['page_idx']}"

        # 生成段落级Simhash特征
        para_simhashes = paragraph_simhash(page["content"])

        logical_pages.append({
            "logical_id": logical_id,
            "content": page["content"],
            "physical_page": page["page_idx"],  # 保留物理页信息用于溯源
            "para_simhashes": para_simhashes  # 新增段落Simhash特征
        })
    return logical_pages


def do_parse(
        output_dir,  # 解析结果输出目录
        pdf_file_names: list[str],  # 待解析的 PDF 文件名列表
        pdf_bytes_list: list[bytes],  # 待解析的 PDF 字节流列表
        p_lang_list: list[str],  # 每个 PDF 的语言列表，默认为 'ch'（中文）
        backend="pipeline",  # 解析 PDF 的后端，默认为 'pipeline'
        parse_method="auto",  # 解析 PDF 的方法，默认为 'auto'
        formula_enable=True,  # 是否启用公式解析
        table_enable=True,  # 是否启用表格解析
        server_url=None,  # 用于 vlm-sglang-client 后端的服务器地址
        f_draw_layout_bbox=True,  # 是否绘制版面布局框
        f_draw_span_bbox=True,  # 是否绘制文本块框
        f_dump_md=True,  # 是否导出 markdown 文件
        f_dump_middle_json=True,  # 是否导出中间 JSON 文件
        f_dump_model_output=True,  # 是否导出模型输出文件
        f_dump_orig_pdf=True,  # 是否导出原始 PDF 文件
        f_dump_content_list=True,  # 是否导出内容列表文件
        f_make_md_mode=MakeMode.MM_MD,  # 生成 markdown 内容的模式，默认为 MM_MD
        start_page_id=0,  # 解析起始页码，默认为 0
        end_page_id=None,  # 解析结束页码，默认为 None（解析到文档结尾）
):
    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list,
                                                                                                         p_lang_list,
                                                                                                         parse_method=parse_method,
                                                                                                         formula_enable=formula_enable,
                                                                                                         table_enable=table_enable)

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang,
                                                         _ocr_enable, formula_enable)

            # 生成逻辑页ID（含章节锚点和段落Simhash）
            logical_pages = generate_logical_page_id(middle_json)
            middle_json["logical_pages"] = logical_pages  # 将逻辑页信息注入中间JSON

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend,
                                                        server_url=server_url)

            # 生成逻辑页ID（含章节锚点和段落Simhash）
            logical_pages = generate_logical_page_id(middle_json)
            middle_json["logical_pages"] = logical_pages  # 将逻辑页信息注入中间JSON

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

            if f_draw_span_bbox:
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="pipeline",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    """
        参数说明：
        path_list: 待解析的文档路径列表，可以是 PDF 或图片文件。
        output_dir: 解析结果输出目录。
        lang: 语言选项，默认为 'ch'，可选值包括['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
            如果已知 PDF 内的语言可填写此项以提升 OCR 准确率，选填。
            仅当 backend 设置为 "pipeline" 时生效。
        backend: 解析 pdf 的后端：
            pipeline: 通用。
            vlm-transformers: 通用。
            vlm-sglang-engine: 更快（engine）。
            vlm-sglang-client: 更快（client）。
            未指定 method 时默认使用 pipeline。
        method: 解析 pdf 的方法：
            auto: 根据文件类型自动判断。
            txt: 使用文本提取方法。
            ocr: 针对图片型 PDF 使用 OCR 方法。
            未指定 method 时默认使用 'auto'。
            仅当 backend 设置为 "pipeline" 时生效。
        server_url: 当 backend 为 `sglang-client` 时需指定服务器地址，例如：`http://127.0.0.1:30000`
        start_page_id: 解析起始页码，默认为 0
        end_page_id: 解析结束页码，默认为 None（解析到文档结尾）
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    # 确保simhash库已安装：pip install simhash
    try:
        from simhash import Simhash
    except ImportError:
        print("警告：未安装simhash库，请执行 'pip install simhash' 以启用段落级Simhash功能")

    # 参数设置
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(__dir__, "output")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    """如果您由于网络问题无法下载模型，可以设置环境变量 MINERU_MODEL_SOURCE 为 modelscope，使用免代理仓库下载模型"""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    """如环境不支持 VLM，可使用 pipeline 模式"""
    parse_doc(doc_path_list, output_dir, backend="pipeline")

    """如需启用 VLM 模式，将 backend 改为 'vlm-xxx'"""
    # parse_doc(doc_path_list, output_dir, backend="vlm-transformers")  # 通用。
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-engine")  # 更快（engine）。
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-client", server_url="http://127.0.0.1:30000")  # 更快（client）。