# PhotoBench 数据集概览（中文）

本目录提供两类数据以支持评测与研究：
- samples：用于可视化展示的小规模子集（包含展示用图片）。
- protected：用于本地评测的机器可读特征（不包含原始图片）。

## 目录结构

```
dataset/
├── README.md            # 英文说明
├── README_CN.md         # 中文说明（本文件）
├── samples/             # 展示子集（每个相册 20 条，含展示图片）
│   ├── album1/
│   │   ├── query.json   # 20 条查询（含 GroundTruth 图片文件名，仅用于展示）
│   │   └── images/      # 与 GroundTruth 对应的展示图片
│   ├── album2/
│   └── album3/
│   ├── README.md
│   └── sample_queries.schema.json
└── protected/           # 本地评测用特征（不含原始图片）
    ├── album1/
    │   ├── query.json   # 全量查询（不含任何 GroundTruth 字段）
    │   └── images/
    │       ├── captions/    # 按模型的文本索引
    │       ├── embeddings/  # 按模型的向量索引
    │       ├── metadata/
    │       │   └── metadata.json  # 隐私保护的位置信息与时间
    │       └── faceid/
    │           ├── face_info_cn.json
    │           ├── face_info_en.json
    │           └── face_id_images/
    ├── album2/
    └── album3/
```

## 文件与字段说明

### samples/query.json（每个相册 20 条）
- 类型：`list`，元素为查询对象。
- 关键字段：
  - `Query`：检索意图文本。
  - `Query_id`：查询唯一标识。
  - `Album_id`：相册标识。
  - `Source_type`：来源类别标签。
  - `Reasoning_type`：推理类别（例如 location/time/social）。
  - `GroundTruth`：用于展示的图片文件名列表。
- 用途：面向读者与审稿人进行可视化理解；不用于训练与完整评分。

### protected/query.json（每个相册的全量查询）
- 类型：`list`，元素为查询对象。
- 说明：与 samples 中的查询字段一致，但不包含任何 GroundTruth 相关字段。

### protected/images/captions/
- 作用：按模型的文本索引，用于基于 caption 的检索或重排。
- 每个 `metadata.json` 的字段：
  - `model_name`：模型名称。
  - `filenames`：索引中包含的图片文件名列表。

### protected/images/embeddings/
- 作用：按模型的向量索引，用于基于图像嵌入的检索。
- 每个 `metadata.json` 的字段：
  - `model_name`
  - `filenames`

### protected/images/metadata/metadata.json
- 作用：隐私保护的地理位置与时间信息。
- 字段：
  - `latitude`：十进制度表示，四舍五入至约 10 米粒度。
  - `longitude`：十进制度表示，四舍五入至约 10 米粒度。
  - `time`：ISO 格式时间字符串。

### protected/images/faceid/
- 作用：匿名化的人物关系信息，支持社交/身份相关推理，不暴露真实身份。
- 文件：
  - `face_info_cn.json` / `face_info_en.json` 包含：
    - `face_id_to_nicknames`：face id 到匿名名字/昵称列表的映射。
    - `image_to_face_ids`：图片文件名到出现的 face id 列表的映射。
  - `face_id_images/`：出现过的 face id 的代表图片。

## 使用与评测
- samples：用于展示与定性理解，不用于模型训练。
- protected：用于本地算法在机器可读特征上运行与评测。
- 评测方式：本地生成提交文件（例如 `submission.json`），由评测端对比隐藏的 GroundTruth 进行评分。

隐私与合规说明可参考项目说明文档。
