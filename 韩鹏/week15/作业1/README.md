# 图文知识库 MVP

这是一个基于 `Streamlit + SQLite + Kafka + Milvus + 本地/缓存模型` 的最小可用图文知识库项目，支持：

- 上传 `PDF` / `DOCX` 文档
- 异步切分文本与图片并写入向量库
- 在“图文对话”页面进行检索增强问答

## 初始化前先确认

第一次启动前，建议先确认下面这些配置项：

- Python 环境：默认使用 `conda` 的 `py310` 环境
- Kafka：默认 `KAFKA_BOOTSTRAP_SERVERS=localhost:9092`
- Milvus / Zilliz：确认 `MILVUS_URI`、`MILVUS_TOKEN`、`MILVUS_COLLECTION_NAME`
- LLM：确认 `LLM_MODEL`、`LLM_API_KEY`、`LLM_BASE_URL`
- Hugging Face 缓存目录：默认使用 `HF_HUB_CACHE=~/.cache/huggingface/hub`
- 模型自动下载：默认 `AUTO_DOWNLOAD_MODELS=true`
- 缓存优先策略：默认 `PREFER_HF_CACHE=true`
- 文本模型：默认 `TEXT_MODEL_NAME_OR_PATH=BAAI/bge-small-zh-v1.5`
- 多模态模型：默认 `MULTIMODAL_MODEL_NAME_OR_PATH=jinaai/jina-clip-v2`

如果是 Windows 且首次自动下载模型，建议额外确认：

- 当前网络可以访问 Hugging Face
- `HF_HUB_CACHE` 所在磁盘空间充足
- 如果不想看到 symlink 警告，可以开启 Windows Developer Mode，或设置 `HF_HUB_DISABLE_SYMLINKS_WARNING=1`

## 模型加载策略

项目内的模型获取逻辑已经改成以下顺序：

1. 先检查 Hugging Face 本地缓存目录是否已有模型快照
2. 如果缓存不完整或不存在，则在运行时自动下载缺失文件
3. 如果 Hugging Face 不可用，则回退到项目内保留的兜底模型文件

其中 `jina-clip-v2` 比较特殊：

- 模型权重、tokenizer、预处理配置优先来自 Hugging Face 缓存
- 实际推理代码仍然使用项目内已经打补丁的 Jina CLIP 实现
- 这样既能利用缓存/自动下载，又能避免原始实现与当前 `torch/transformers` 组合下的兼容性问题

## 运行方式

启动 Web 应用：

```powershell
conda run -n py310 streamlit run app.py
```

启动异步处理 worker：

```powershell
conda run -n py310 python worker.py
```

只处理一条任务：

```powershell
conda run -n py310 python worker.py --once
```

运行测试：

```powershell
conda run -n py310 python -m unittest discover -s tests -v
```

## 目录说明

- `app.py`：Streamlit 入口
- `worker.py`：异步任务消费者
- `src/rag_mvp/`：核心业务代码
- `tests/`：单元测试
- `uploads/`：上传文件与派生资源
- `data/`：本地队列与本地向量索引等运行数据
- `models/`：项目内保留的模型补丁代码和兜底模型资源

## 降级策略

- Kafka 不可用：自动写入本地队列目录
- Milvus 不可用：自动切换到本地 JSON 向量索引
- 大模型不可用：返回检索结果摘要，页面仍可继续使用
- 多模态模型不可用：图片检索回退到文本描述向量

## 当前建议

- 首次运行前先确认 Kafka、Milvus、LLM 三类外部配置是否可用
- 如果希望完全离线运行，先提前把模型下载到本地 Hugging Face 缓存
- 如果希望强制只使用项目内兜底模型，可以将 `AUTO_DOWNLOAD_MODELS=false`
