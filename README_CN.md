# PhotoBench (中文)

<p align="center">
  🏠 <strong>GitHub</strong> ·
  📄 <a href="https://arxiv.org/abs/2603.01493v1"><strong>arXiv</strong></a> ·
  🏅 <a href="https://huggingface.co/spaces/SorrowTea/PhotoBench/"><strong>Leaderboard</strong></a> ·
  📦 <a href="https://huggingface.co/datasets/SorrowTea/PhotoBench"><strong>Dataset</strong></a> ·
  🏅 <a href="https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/"><strong>Protected Leaderboard</strong></a> ·
  📦 <a href="https://huggingface.co/datasets/SorrowTea/PhotoBench-Protected"><strong>Protected Dataset</strong></a> ·
  🖼️ <a href="https://sbox.myoas.com/l/Be5be4053f6b43840"><strong>Raw Images</strong></a>
</p>

PhotoBench 是从真实个人相册构建的首个评测基准，旨在将范式从视觉匹配转变为个性化多源意图驱动的照片检索。本仓库提供数据集、评测工具与榜单入口。

## 🎉 最新动态

- **📦 完整数据集正式上架 HuggingFace！**
  从最初仅有 Protected 版（预计算 caption / embedding / 元数据，专注 agent 规划能力），到现在**完整版 RAW 图像全面开放**——你可以自由选择模型、自由提取特征、无拘无束地做检索！
  [→ 完整数据集](https://huggingface.co/datasets/SorrowTea/PhotoBench) · [→ Protected 数据集](https://huggingface.co/datasets/SorrowTea/PhotoBench-Protected)

- **🏅 在线评测榜单上线！**
  PhotoBench 全量榜与 Protected 榜正式接入 HuggingFace，支持一键提交、实时排名！
  [→ 全量榜](https://huggingface.co/spaces/SorrowTea/PhotoBench/) · [→ Protected 榜](https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/)

## 数据集设计

为了解决现有数据集的局限性，我们引入了 PhotoBench，它专为真实世界的相册检索和推理而设计。与以前的文本到图像评测基准相比，其主要区别总结如下：

![表1: 与其他数据集的比较](./table1.png)

## 数据集下载

您可以直接从我们的 Gitee 仓库获取完整数据集：[https://gitee.com/sorrowtea/PhotoBench](https://gitee.com/sorrowtea/PhotoBench)

数据集包含以下内容：
- **完整的图片**：3个相册的全部原始图片
- **validation.json**：附带已公开 Ground Truth 的验证查询，用于本地自测。每个相册100条，共300条。
- **test.json**：测试查询，用于榜单提交。相册1: 382条，相册2: 236条，相册3: 269条，共887条。Ground Truth 不公开，由 [PhotoBench 全量榜](https://huggingface.co/spaces/SorrowTea/PhotoBench/) 统一评测。

## 评测结果

下表总结了各种模型在 PhotoBench 评测基准上的性能。这提供了当前最先进技术的概览以及仍然存在的挑战。

![表2: 主要结果](./table2.png)

## 评测与提交

当您使用自己的模型生成了提交文件后，您可以将其提交到我们的在线平台以获得最终分数。

**评测榜单：** [PhotoBench 全量榜](https://huggingface.co/spaces/SorrowTea/PhotoBench/) · [Protected 榜](https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/)

评测流程如下:

1.  **生成提交文件**: 运行您的模型以产出 `_submission.json` 文件。
2.  **提交您的文件**: 将生成的 JSON 文件上传到榜单。
3.  **接收分数**: 评测结果将即时展示，并发送到您在网站上提供的邮箱地址。

### 提交文件格式

数据集为每个相册提供一个 `albumN_test.json`。提交前，你必须**将所有相册合并为一个 JSON 数组**，并为每个查询对象加上 `album_id` 字段。

```python
import json

submission = []
for album_id in ["1", "2", "3"]:
    with open(f"album{album_id}_test.json") as f:
        queries = json.load(f)
    for q in queries:
        submission.append({
            "album_id": album_id,
            "query_en": q["query_en"],
            "pred": ["IMG_0001.JPG", "IMG_0002.JPG", ...]  # 你的预测结果
        })

with open("submission.json", "w") as f:
    json.dump(submission, f, indent=2)
```

**最终提交格式：**

```json
[
  {
    "album_id": "1",
    "query_en": "cluttered desk",
    "pred": ["IMG_1234.jpg", "IMG_5678.jpg", ...]
  }
]
```

**必填字段：**
- `album_id`：相册编号（`"1"`、`"2"` 或 `"3"` —— 字符串）
- `query_en`：英文查询文本（必须完全匹配，区分大小写）
- `pred`：预测图片文件名的有序列表（顺序对 NDCG 有影响）

只有**完整提交**（全部 3 个相册、全部测试查询）才有资格进入公开榜单排名。部分提交可以参与评测，但不会出现在榜单上。

## 仓库结构

-   `dataset/`: 完整的数据集，包括图片和查询文件。
    -   `album1/`, `album2/`, `album3/`: 完整的相册。
    -   `validation.json`: 附带已公开 Ground Truth 的验证查询，用于本地自测。每个相册100条，共300条。
    -   `test.json`: 测试查询，用于榜单提交。相册1: 382条，相册2: 236条，相册3: 269条，共887条。Ground Truth 不公开。
-   `scripts/`: 用于生成提交文件的评测脚本。
    -   `eval_embedding.py`: 用于基于 Embedding 的检索。
    -   `eval_caption.py`: 用于基于 Caption 索引的检索。
    -   `eval_agent.py`: 用于基于 Agent 的检索流水线。
