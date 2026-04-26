# PhotoBench

<p align="center">
  🏠 <strong>GitHub</strong> ·
  📄 <a href="https://arxiv.org/abs/2603.01493v1"><strong>arXiv</strong></a> ·
  🏅 <a href="https://huggingface.co/spaces/SorrowTea/PhotoBench/"><strong>Leaderboard</strong></a> ·
  📦 <a href="https://huggingface.co/datasets/SorrowTea/PhotoBench"><strong>Dataset</strong></a> ·
  🏅 <a href="https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/"><strong>Protected Leaderboard</strong></a> ·
  📦 <a href="https://huggingface.co/datasets/SorrowTea/PhotoBench-Protected"><strong>Protected Dataset</strong></a> ·
  🖼️ <a href="https://sbox.myoas.com/l/Be5be4053f6b43840"><strong>Raw Images</strong></a>
</p>

PhotoBench is the first benchmark constructed from authentic, personal albums, designed to shift the paradigm from visual matching to personalized multi-source intent-driven photo retrieval. This repository hosts the datasets, evaluation tools, and leaderboard interfaces.

## Dataset Design

To address the limitations of existing datasets, we introduce PhotoBench, which is specifically designed for real-world photo album retrieval and reasoning. The key differences from previous text-to-image benchmarks are summarized below:

![Table 1: Comparison with other datasets](./table1.png)

## Dataset Download

You can get the complete dataset directly from our Gitee repository: [https://gitee.com/sorrowtea/PhotoBench](https://gitee.com/sorrowtea/PhotoBench)

The dataset includes:
- **Complete images**: 3 photo albums with all original images
- **validation.json**: Queries with ground truth are provided to help users conveniently evaluate the performance of their own methods. 100\*3 queries in total.
- **test.json**: Test queries without ground truth. To evaluate your performance on this split, please submit your results to our test platform: [https://huggingface.co/spaces/SorrowTea/PhotoBench](https://huggingface.co/spaces/SorrowTea/PhotoBench)

## Benchmark Results

The following table summarizes the performance of various models on the PhotoBench benchmark. This provides an overview of the current state-of-the-art and the challenges that still remain.

![Table 2: Main Results](./table2.png)

## Evaluation and Submission

Once you have generated a submission file by using your own model, you can submit it to our online platform to get the final scores.

**Evaluation Website:** [https://photo-bench-site.vercel.app](https://photo-bench-site.vercel.app)
or [https://sorrowcloud.tech](https://sorrowcloud.tech)

The process is as follows:

1.  **Generate Submission File**: Run your model to produce a `_submission.json` file.
2.  **Submit Your File**: Upload the generated JSON file to our evaluation website.
3.  **Receive Scores**: The evaluation results will be sent to the email address you provide on the website.

### Submission File Format

All submissions should adhere to the following file structure:

```json
{
  "model_name": "<your_model_name>",
  "language": "cn" or "en",
  "results": [
    { "query_id": "<query_id>", "predictions": ["<filename_1>", "<filename_2>", ...] }
  ]
}
```

## Repository Structure

-   `dataset/`: Complete dataset including images and query files.
    -   `album1/`, `album2/`, `album3/`: Complete photo albums.
    -   `validation.json`: Queries with ground truth. 100\*3 queries in total.
    -   `test.json`: Test queries.
-   `scripts/`: Evaluation scripts to generate submission files.
    -   `eval_embedding.py`: For embedding-based retrieval.
    -   `eval_caption.py`: For caption-index-based retrieval.
    -   `eval_agent.py`: For an agent-based retrieval pipeline.
