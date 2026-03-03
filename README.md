# PhotoBench

This is the official repository for PhotoBench, a new benchmark for photo retrieval and reasoning. This repository provides the necessary datasets and tools for researchers to evaluate their models' performance in a realistic photo album scenario.

Due to the private nature of personal photo albums, and to protect the privacy of the individuals depicted, we provide the dataset in two parts: a small, visual sample set for qualitative review, and a larger, feature-extracted protected set for quantitative evaluation.

## Dataset Design

To address the limitations of existing datasets, we introduce PhotoBench, which is specifically designed for real-world photo album retrieval and reasoning. The key differences from previous text-to-image benchmarks are summarized below:

![Table 1: Comparison with other datasets](./table1.png)

PhotoBench is designed with privacy as a first priority. Instead of providing the full image dataset, we provide:

1.  **A small sample set (`dataset/samples/`)**: This subset contains a few dozen images and queries per album, allowing for qualitative analysis and a better understanding of the dataset's nature.
2.  **Protected data for evaluation (`dataset/protected/`)**: For the full dataset, we provide extracted, privacy-safe features. These features cover all the models mentioned in the main table of our paper, facilitating the reproduction of our results. This includes:
    *   Image and text embedding indices.
    *   Anonymized face information.
    *   Location and time metadata.

This design allows you to run evaluation scripts locally using these features to generate a submission file.

***The dataset is currently undergoing safety scrutiny by the company, and we will open-source it as soon as it passes the review process.***

## Benchmark Results

The following table summarizes the performance of various models on the PhotoBench benchmark. This provides an overview of the current state-of-the-art and the challenges that still remain.

![Table 2: Main Results](./table2.png)

## Reproducing Results

You can reproduce the results from our paper by running the provided evaluation scripts. These scripts use the pre-computed features from the `dataset/protected/` directory to generate a submission file in the correct format.

### Caption-based Retrieval

To run the evaluation for a caption-based model, you need to provide the path.

**Example Command:**
```bash
python scripts/caption/eval_caption.py \
    --model_path /path/to/your/model \
    --index_dir dataset/protected/album1/images/captions/bge-m3 \
    --query_file dataset/samples/album1/query.json \
    --language cn
```
This command will generate a `_submission.json` file in the `results/cn/` directory.

### Embedding-based Retrieval

To run the evaluation for an image-embedding model, provide the model's architecture name and the path to its checkpoint.

**Example Command:**
```bash
python scripts/embedding/eval_embedding.py \
    --model_name "ops_mm_embedding_v1" \
    --model_path /path/to/your/model_checkpoint.pth \
    --index_dir dataset/protected/album1/images/embeddings/chinese-clip-vit-base-patch16 \
    --query_file dataset/samples/album1/query.json \
    --language cn
```
This will also generate a `_submission.json` file in the `results/cn/` directory.

## Evaluation and Submission

Once you have generated a submission file—either by running our evaluation scripts as shown above or by using your own custom agent—you can submit it to our online platform to get the final scores.

**Evaluation Website:** [https://photo-bench-site.vercel.app](https://photo-bench-site.vercel.app)
or [https://sorrowcloud.tech](https://sorrowcloud.tech)

The process is as follows:

1.  **Generate Submission File**: Run the scripts to produce a `_submission.json` file.
2.  **Submit Your File**: Upload the generated JSON file to our evaluation website.
3.  **Receive Scores**: The evaluation results will be sent to the email address you provide on the website.

### Submission File Format

All evaluation scripts generate a file with the following structure, which you should adhere to for custom submissions:

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

-   `dataset/samples/`: A small display subset for each album (e.g., 20 queries with corresponding images for visualization).
-   `dataset/protected/`: Features for local evaluation (caption/embedding indices, location/time metadata, anonymized face info).
-   `scripts/`: Evaluation scripts to generate submission files.
    -   `eval_embedding.py`: For embedding-based retrieval.
    -   `eval_caption.py`: For caption-index-based retrieval.
    -   `eval_agent.py`: For an agent-based retrieval pipeline.
