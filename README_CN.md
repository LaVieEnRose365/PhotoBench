# PhotoBench（中文）

PhotoBench 是一个使用了隐私保护措施的照片检索与推理评测集。提供展示用子集（samples）与机器可读特征（protected），支持研究者在本地完成检索与提交，无需公开原始图片。

## 内容结构

- 数据集
  - `dataset/samples/`：每个相册 20 条的展示子集（含可视化图片）。
  - `dataset/protected/`：本地评测用特征（caption/embedding 索引、location/time 元数据、匿名化 face 信息）。
- 脚本
  - `scripts/eval_embedding.py`：基于图像嵌入的检索并生成提交文件。
  - `scripts/eval_caption.py`：基于 caption 索引的检索并生成提交文件。
  - `scripts/eval_agent.py`：基于代理的检索流水线并生成提交文件。

## 提交格式

所有评测脚本写出到 `results/<language>/<model_name>_submission.json`，结构如下：

```
{
  "model_name": "<model>",
  "language": "cn|en",
  "results": [
    { "query_id": "<id>", "predictions": ["<filename>", ...] }
  ]
}
```

## 最小使用方式

Embedding 检索：
- 输入：`index_dir` 下的图像嵌入索引（`index.faiss` 与 `metadata.json`）以及包含 `query_cn` 或 `query_en` 的查询文件。
- 运行：执行 `scripts/eval_embedding.py`，指定 `--model_name`、`--index_dir`、`--query_file`、`--language`（可选 `--device`、`--output_folder`）。

Caption 检索：
- 输入：`index_dir` 下的 caption 索引与查询文件。
- 运行：执行 `scripts/eval_caption.py`，指定 `--index_dir`、`--query_file`、`--language`（可选 `--model_name`、`--device`、`--output_folder`）。

Agent 流水线：
- 输入：嵌入索引、face 信息、metadata、查询文件。
- 运行：执行 `scripts/eval_agent.py`，指定 `--face_info_path`、`--metadata_path`、`--model_name`、`--index_dir`、`--query_file`、`--language`（可选 `--output_folder`、`--device`、`--llm_model`、`--concurrent`）。

## 说明

- samples 用于可视化；protected 提供本地评测所需的特征，不包含原始图片。
- 将生成的 JSON 提交到评测服务，即可获得相对于隐藏 GroundTruth 的评分。
