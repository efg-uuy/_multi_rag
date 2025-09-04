
# Spark-Multi-modal RAG 图文问答项目

本项目实现了一个完整的多模态 RAG（Retrieval-Augmented Generation）图文问答系统，支持从 PDF 文档解析、内容结构化、向量检索到大模型问答的全流程。

## 快速开始

```sh
git clone https://github.com/li-xiu-qi/spark_multi_rag.git
cd spark_multi_rag
```

## 核心功能

- **PDF 解析**：支持 MinerU（复杂布局）和 PyMuPDF（纯文本）两种解析方案
- **向量检索**：基于 bge-m3 模型的语义检索
- **智能问答**：集成 Qwen 系列模型的生成式问答
- **模型微调**：提供完整的模型训练和微调 Notebook
- **批量评测**：支持测试集批量处理和结果输出

## 项目结构

```
spark_multi_rag/
├── README.md                          # 项目说明文档
├── requirements.txt                   # 基础依赖包
├── pyproject.toml                     # 完整项目配置（包含高级功能）
├── .env.example                       # 环境变量配置示例
│
├── 数据处理脚本/
│   ├── mineru_pipeline_all.py         # MinerU方案：PDF一键处理（推荐）
│   ├── fitz_pipeline_all.py           # PyMuPDF方案：轻量级PDF处理
│   ├── mineru_parse_pdf.py            # MinerU解析工具
│   └── get_text_embedding.py          # 文本向量化工具
│
├── RAG问答系统/
│   ├── rag_from_page_chunks.py        # RAG检索与问答主脚本
│   └── extract_json_array.py          # JSON结构提取工具
│
├── 模型训练Notebook/
│   ├── spark_data_process.ipynb       # 数据预处理
│   ├── spark_model_finetune.ipynb     # Qwen2.5-7B微调
│   ├── Qwen2_5_7B_Alpaca_fintune.ipynb # Alpaca数据集微调示例
│   └── Qwen3_14B_finetune.ipynb       # Qwen3-14B微调示例
│
├── 数据目录/
│   ├── datas/                         # 原始PDF及测试集
│   │   ├── 多模态RAG图文问答挑战赛训练集.json
│   │   ├── 多模态RAG图文问答挑战赛测试集.json
│   │   └── 财报数据库/                # PDF文档
│   └── all_pdf_page_chunks.json      # 处理后的分页内容
│
└── 输出目录/
    ├── data_base_json_content/        # PDF解析中间结果
    ├── data_base_json_page_content/   # 分页中间结果
    ├── output/                        # 评测结果输出
    ├── model_train_outputs/           # 模型训练输出
    └── caches/                        # 向量缓存
```

## 环境配置

### 1. 依赖安装

推荐使用 Python 3.11+：

```bash
pip install -r requirements.txt
```

如需完整功能（包含 MinerU 和模型训练），建议使用 uv 安装：

```bash
uv sync
```

### 2. 数据准备

下载比赛数据并放入 `datas/` 目录：

- `多模态RAG图文问答挑战赛训练集.json`
- `多模态RAG图文问答挑战赛测试集.json`
- `财报数据库/`（PDF文档集合）

数据下载地址：[讯飞挑战赛官网](https://challenge.xfyun.cn/topic/info?type=Multimodal-RAG-QA&option=stsj&ch=dwsf2517)

### 3. 环境变量与模型配置

请在 `.env` 文件中配置本地或云端API参数，示例：

```env
LOCAL_API_KEY=anything（基于xinference部署或者是硅基流动的api key）# your guiji api
LOCAL_BASE_URL=http://localhost:10002/v1（基于xinference或者是基于硅基流动的baseurl）# GUIJI_BASE_URL=https://api.siliconflow.cn/v1
LOCAL_TEXT_MODEL=qwen3（基于xinference或者是硅基流动的对话模型名称） # Qwen/Qwen3-8B
LOCAL_EMBEDDING_MODEL=bge-m3（基于xinference） # BAAI/bge-m3
```

> 推荐使用 xinference 本地部署或硅基流动 API。本地部署建议 A6000 等高性能显卡。

## 使用流程

### 步骤1：PDF 数据处理

#### 方案一：MinerU（推荐，支持复杂布局）

```bash
python mineru_pipeline_all.py
```

自动完成：

- 遍历 `datas/` 下所有PDF，解析为结构化内容
- 分页处理并生成 `all_pdf_page_chunks.json`
- 支持图片caption自动补全

#### 方案二：PyMuPDF（轻量级，纯文本PDF）

```bash
python fitz_pipeline_all.py
```

适用于纯文本PDF的快速批量处理，无需GPU。

### 步骤2：RAG 检索与问答

```bash
python rag_from_page_chunks.py
```

功能包括：

- 加载分页内容，构建向量索引
- 批量处理测试集，输出评测结果
- 支持交互式问答模式

### 步骤3：模型训练（可选）

项目提供完整的模型微调流程：

1. **数据预处理**：运行 `spark_data_process.ipynb`
2. **模型微调**：运行 `spark_model_finetune.ipynb`

支持的模型：Qwen2.5-7B、Qwen3-14B 等，采用 LoRA 技术节省显存。

## 模型部署方案

### 推荐方案

1. **硅基流动 API**（免费，快速上手）
   - 多模态模型：Qwen/Qwen2.5-VL-32B-Instruct
   - 向量模型：BAAI/bge-m3、BAAI/bge-reranker-v2-m3
   - 无需本地GPU，开箱即用
   - 官网：[硅基流动](https://cloud.siliconflow.cn/i/FcjKykMn)

> 硅基流动提供了部分免费的大模型资源，比如embedding模型，小参数chat模型等等，Embedding模型：如 BAAI/bge-m3、重排序模型 BAAI/bge-reranker-v2-m3，均可免费调用 ，可以用我的邀请码FcjKykMn，或者点击链接：<https://cloud.siliconflow.cn/i/FcjKykMn>
   这样我会有2000wtoken的奖励，hh。

2. **Xinference 本地部署**（性能最佳）
   - 支持统一管理多种模型
   - 推荐 A6000 等高性能显卡
   - 文档：[Xinference 官方文档](https://inference.readthedocs.io/en/latest/)

### 模型配置

在 `.env` 文件中配置对应的API参数即可自动适配不同部署方案。

## 主要特性

- ✅ **多PDF解析**：支持MinerU（复杂布局）和PyMuPDF（纯文本）
- ✅ **语义检索**：基于bge-m3的高质量向量检索
- ✅ **智能问答**：集成Qwen系列模型的生成式问答
- ✅ **模型微调**：完整的LoRA微调流程和Notebook
- ✅ **批量评测**：支持测试集自动化处理
- ✅ **缓存优化**：向量缓存和结果缓存提升效率

## 适用场景

- 📊 **金融分析**：财报、研究报告的智能问答
- 📚 **学术研究**：论文、文献的内容检索
- 📋 **企业文档**：内部文档的知识管理
- 🏢 **法律咨询**：法规、合同的条款查询

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。
