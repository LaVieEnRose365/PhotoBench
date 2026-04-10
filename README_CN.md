# PhotoBench (中文)

这里是 PhotoBench 的官方仓库，这是一个为照片检索与推理设计的全新评测基准。本仓库为研究人员提供了必要的数据集和工具，以便在真实的相册场景中评估其模型的性能。

## 数据集设计

为了解决现有数据集的局限性，我们引入了 PhotoBench，它专为真实世界的相册检索和推理而设计。与以前的文本到图像评测基准相比，其主要区别总结如下：

![表1: 与其他数据集的比较](./table1.png)

## 数据集下载

您可以直接从我们的 Gitee 仓库获取完整数据集：[https://gitee.com/sorrowtea/PhotoBench](https://gitee.com/sorrowtea/PhotoBench)

数据集包含以下内容：
- **完整的图片**：3个相册的全部原始图片
- **train.json**：带有Ground Truth标注的训练查询
- **test.json**：测试查询（包含带Ground Truth和不带Ground Truth的）

## 评测结果

下表总结了各种模型在 PhotoBench 评测基准上的性能。这提供了当前最先进技术的概览以及仍然存在的挑战。

![表2: 主要结果](./table2.png)

## 评测与提交

当您使用自己的模型生成了提交文件后，您可以将其提交到我们的在线平台以获得最终分数。

**评测网址:** [https://photo-bench-site.vercel.app](https://photo-bench-site.vercel.app) 
**国内访问网址：** [https://sorrowcloud.tech](https://sorrowcloud.tech)

评测流程如下:

1.  **生成提交文件**: 运行您的模型以产出 `_submission.json` 文件。
2.  **提交您的文件**: 将生成的 JSON 文件上传到我们的评测网站。
3.  **接收分数**: 评测结果将会发送到您在网站上提供的邮箱地址。

### 提交文件格式

所有提交文件应遵循以下结构：

```json
{
  "model_name": "<your_model_name>",
  "language": "cn" or "en",
  "results": [
    { "query_id": "<query_id>", "predictions": ["<filename_1>", "<filename_2>", ...] }
  ]
}
```

## 仓库结构

-   `dataset/`: 完整的数据集，包括图片和查询文件。
    -   `album1/`, `album2/`, `album3/`: 完整的相册。
    -   `train.json`: 带有Ground Truth的训练查询。
    -   `test.json`: 测试查询。
-   `scripts/`: 用于生成提交文件的评测脚本。
    -   `eval_embedding.py`: 用于基于 Embedding 的检索。
    -   `eval_caption.py`: 用于基于 Caption 索引的检索。
    -   `eval_agent.py`: 用于基于 Agent 的检索流水线。
