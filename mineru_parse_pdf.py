# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path
import hashlib
from simhash import Simhash

from loguru import logger

# 新增：CLIP相关导入
try:
    import torch
    import clip
    from PIL import Image
    from torch.nn.functional import cosine_similarity

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

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


# ===================== 新增：语义块生成（基于文档结构） =====================
def generate_semantic_chunks(middle_json):
    """
    基于文档语义层级生成结构化语义块
    以一级标题为单元，打包其下的段落、表格和图像
    """
    semantic_chunks = []
    current_chunk = {
        "chunk_id": "",
        "title": "",
        "content": [],
        "images": [],
        "tables": [],
        "start_page": None,
        "end_page": None,
        "para_simhashes": []
    }

    for page in middle_json["pdf_info"]["pages"]:
        page_idx = page["page_idx"]

        for block in page["blocks"]:
            # 一级标题作为新块的起点
            if block["type"] == "heading" and block.get("level") == 1:
                # 保存当前块（如果有内容）
                if current_chunk["content"] or current_chunk["images"] or current_chunk["tables"]:
                    # 生成chunk_id
                    title_hash = hashlib.md5(current_chunk["title"].encode()).hexdigest()
                    current_chunk["chunk_id"] = f"chunk_{title_hash}_{current_chunk['start_page']}"
                    # 生成段落Simhash
                    full_content = "\n\n".join(current_chunk["content"])
                    current_chunk["para_simhashes"] = paragraph_simhash(full_content)
                    semantic_chunks.append(current_chunk)

                # 初始化新块
                current_chunk = {
                    "chunk_id": "",
                    "title": block["content"],
                    "content": [],
                    "images": [],
                    "tables": [],
                    "start_page": page_idx,
                    "end_page": page_idx,
                    "para_simhashes": []
                }

            # 其他类型内容添加到当前块
            else:
                if current_chunk["start_page"] is None:
                    current_chunk["start_page"] = page_idx
                current_chunk["end_page"] = page_idx

                if block["type"] == "text":
                    current_chunk["content"].append(block["content"])
                elif block["type"] == "image":
                    current_chunk["images"].append({
                        **block,
                        "page_idx": page_idx
                    })
                elif block["type"] == "table":
                    current_chunk["tables"].append({
                        **block,
                        "page_idx": page_idx
                    })

    # 添加最后一个块
    if current_chunk["content"] or current_chunk["images"] or current_chunk["tables"]:
        title_hash = hashlib.md5(current_chunk["title"].encode()).hexdigest() if current_chunk[
            "title"] else hashlib.md5(str(current_chunk["start_page"]).encode()).hexdigest()
        current_chunk["chunk_id"] = f"chunk_{title_hash}_{current_chunk['start_page']}"
        full_content = "\n\n".join(current_chunk["content"])
        current_chunk["para_simhashes"] = paragraph_simhash(full_content)
        semantic_chunks.append(current_chunk)

    return semantic_chunks


# ===================== 新增：CLIP图文关联增强 =====================
class CLIPImageTextAssociator:
    """使用CLIP模型实现图文向量关联"""

    def __init__(self):
        if not CLIP_AVAILABLE:
            self.model = None
            self.preprocess = None
            logger.warning("CLIP未安装，无法使用图文关联功能。请安装：pip install torch clip-by-openai pillow")
            return

        try:
            # 加载CLIP模型
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info(f"CLIP模型加载成功，使用设备：{self.device}")
        except Exception as e:
            self.model = None
            self.preprocess = None
            logger.error(f"CLIP模型加载失败：{str(e)}")

    def get_image_embedding(self, image_path):
        """生成图像的CLIP向量"""
        if not self.model or not os.path.exists(image_path):
            return None

        try:
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
            return image_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"生成图像向量失败：{str(e)}")
            return None

    def get_text_embedding(self, text):
        """生成文本的CLIP向量"""
        if not self.model:
            return None

        try:
            text = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
            return text_features.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"生成文本向量失败：{str(e)}")
            return None

    def associate_images_with_texts(self, semantic_chunks, image_dir):
        """为语义块中的图像关联最相关的文本内容"""
        if not self.model:
            return semantic_chunks

        # 处理每个语义块
        for chunk in semantic_chunks:
            # 生成块内文本的向量
            chunk_text = chunk["title"] + "\n\n" + "\n\n".join(chunk["content"])
            chunk_text_embedding = self.get_text_embedding(chunk_text)

            # 处理块内每个图像
            for image in chunk["images"]:
                img_path = os.path.join(image_dir, image.get("img_path", ""))
                image_embedding = self.get_image_embedding(img_path)

                if image_embedding and chunk_text_embedding:
                    # 计算当前图像与块文本的相似度
                    similarity = cosine_similarity(
                        torch.tensor([image_embedding]),
                        torch.tensor([chunk_text_embedding])
                    ).item()

                    image["clip_embedding"] = image_embedding
                    image["text_embedding_similarity"] = round(similarity, 4)

        return semantic_chunks


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
        f_enable_semantic_chunks=True,  # 新增：是否启用语义块生成
        f_enable_clip_association=False,  # 新增：是否启用CLIP图文关联
):
    # 初始化CLIP图文关联器
    clip_associator = CLIPImageTextAssociator() if f_enable_clip_association else None

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

            # 新增：生成语义块
            if f_enable_semantic_chunks:
                semantic_chunks = generate_semantic_chunks(middle_json)

                # 新增：应用CLIP图文关联
                if f_enable_clip_association and clip_associator and CLIP_AVAILABLE:
                    semantic_chunks = clip_associator.associate_images_with_texts(
                        semantic_chunks,
                        local_image_dir
                    )

                middle_json["semantic_chunks"] = semantic_chunks
                logger.info(f"生成语义块完成：{len(semantic_chunks)}个块")

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

            # 新增：生成语义块
            if f_enable_semantic_chunks:
                semantic_chunks = generate_semantic_chunks(middle_json)

                # 新增：应用CLIP图文关联
                if f_enable_clip_association and clip_associator and CLIP_AVAILABLE:
                    semantic_chunks = clip_associator.associate_images_with_texts(
                        semantic_chunks,
                        local_image_dir
                    )

                middle_json["semantic_chunks"] = semantic_chunks
                logger.info(f"生成语义块完成：{len(semantic_chunks)}个块")

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
        end_page_id=None,
        enable_semantic_chunks=True,  # 新增参数
        enable_clip_association=False  # 新增参数
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
        enable_semantic_chunks: 是否启用语义块生成，默认为True
        enable_clip_association: 是否启用CLIP图文关联，默认为False
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
            end_page_id=end_page_id,
            f_enable_semantic_chunks=enable_semantic_chunks,
            f_enable_clip_association=enable_clip_association
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    # 确保simhash库已安装：pip install simhash
    try:
        from simhash import Simhash
    except ImportError:
        print("警告：未安装simhash库，请执行 'pip install simhash' 以启用段落级Simhash功能")

    # 检查CLIP依赖
    if CLIP_AVAILABLE:
        print("CLIP库已安装，可使用图文关联功能")
    else:
        print("CLIP库未安装，如需使用图文关联功能，请执行 'pip install torch clip-by-openai pillow'")

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
    # 基础模式：仅生成语义块
    parse_doc(doc_path_list, output_dir, backend="pipeline")

    # 增强模式：生成语义块并启用CLIP图文关联（需要GPU支持）
    # parse_doc(doc_path_list, output_dir, backend="pipeline", enable_clip_association=True)

    """如需启用 VLM 模式，将 backend 改为 'vlm-xxx'"""
    # parse_doc(doc_path_list, output_dir, backend="vlm-transformers")  # 通用。
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-engine")  # 更快（engine）。
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-client", server_url="http://127.0.0.1:30000")  # 更快（client）。