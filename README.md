# PhotoBench

This is the official repository for PhotoBench, a new benchmark for photo retrieval and reasoning. This repository provides the necessary datasets and tools for researchers to evaluate their models' performance in a realistic photo album scenario.

## Dataset Design

To address the limitations of existing datasets, we introduce PhotoBench, which is specifically designed for real-world photo album retrieval and reasoning. The key differences from previous text-to-image benchmarks are summarized below:

![Table 1: Comparison with other datasets](./table1.png)

## Dataset Download

You can get the complete dataset directly from our Gitee repository: [https://gitee.com/sorrowtea/PhotoBench](https://gitee.com/sorrowtea/PhotoBench)

The dataset includes:
- **Complete images**: 3 photo albums with all original images
- **train.json**: Training queries with ground truth annotations
- **test.json**: Test queries (with and without ground truth)

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
    -   `train.json`: Training queries with ground truth.
    -   `test.json`: Test queries.
-   `scripts/`: Evaluation scripts to generate submission files.
    -   `eval_embedding.py`: For embedding-based retrieval.
    -   `eval_caption.py`: For caption-index-based retrieval.
    -   `eval_agent.py`: For an agent-based retrieval pipeline.
