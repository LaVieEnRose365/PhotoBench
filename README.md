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

## 🎉 What's New

- **📦 Full Dataset Release on HuggingFace!**  
  Building upon our initial Protected release (pre-computed captions, embeddings, and metadata for agent-based research), the **complete raw images** are now openly available—choose your own models, extract your own features, and retrieve without limits!  
  [→ Full Dataset](https://huggingface.co/datasets/SorrowTea/PhotoBench) · [→ Protected Dataset](https://huggingface.co/datasets/SorrowTea/PhotoBench-Protected)

- **🏅 Online Leaderboards are Live!**  
  Submit your results and see your model on the leaderboard in real time.  
  [→ Full Leaderboard](https://huggingface.co/spaces/SorrowTea/PhotoBench/) · [→ Protected Leaderboard](https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/)

## Dataset Design

To address the limitations of existing datasets, we introduce PhotoBench, which is specifically designed for real-world photo album retrieval and reasoning. The key differences from previous text-to-image benchmarks are summarized below:

![Table 1: Comparison with other datasets](./table1.png)

## Two Variants

PhotoBench is released in two variants to support different research directions:

| | PhotoBench (Full) | PhotoBench-Protected |
|---|---|---|
| **Images** | Raw original photos (~11 GB) | Not included |
| **Features** | Use your own models (CLIP, SigLIP, etc.) | Pre-computed captions & embeddings provided |
| **Metadata** | Extract your own (EXIF, timestamps, etc.) | Pre-computed metadata provided |
| **Focus** | Unrestricted retrieval: embedding, caption, or agent | Agent planning only |
| **Leaderboard** | [PhotoBench](https://huggingface.co/spaces/SorrowTea/PhotoBench/) | [PhotoBench-Protected](https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/) |
| **Dataset** | [SorrowTea/PhotoBench](https://huggingface.co/datasets/SorrowTea/PhotoBench) | [SorrowTea/PhotoBench-Protected](https://huggingface.co/datasets/SorrowTea/PhotoBench-Protected) |

- **PhotoBench (Full)** — For researchers who want to experiment with their own vision encoders, caption generators, or end-to-end agent pipelines. You get the raw images and complete freedom.
- **PhotoBench-Protected** — For researchers focusing exclusively on **agent planning and reasoning**. No raw images are provided; you must work with pre-computed captions, embeddings, and metadata. This isolates the planning component from visual representation learning.

## Dataset Download

You can get the complete dataset directly from our Gitee repository: [https://gitee.com/sorrowtea/PhotoBench](https://gitee.com/sorrowtea/PhotoBench)

The dataset includes:
- **Complete images**: 3 photo albums with all original images
- **validation.json**: Queries with released ground truth for local self-evaluation. 100 queries per album (300 total).
- **test.json**: Test queries for leaderboard submission. Album 1: 382, Album 2: 236, Album 3: 269 (887 total). Ground truth is held out and evaluated on the [PhotoBench Leaderboard](https://huggingface.co/spaces/SorrowTea/PhotoBench/).

## Benchmark Results

The following table summarizes the performance of various models on the PhotoBench benchmark. This provides an overview of the current state-of-the-art and the challenges that still remain.

![Table 2: Main Results](./table2.png)

## Evaluation and Submission

Once you have generated a submission file by using your own model, you can submit it to our online platform to get the final scores.

**Leaderboard:** [PhotoBench Leaderboard](https://huggingface.co/spaces/SorrowTea/PhotoBench/) · [Protected Leaderboard](https://huggingface.co/spaces/SorrowTea/PhotoBench-Protected/)

The process is as follows:

1.  **Generate Submission File**: Run your model to produce a `_submission.json` file.
2.  **Submit Your File**: Upload the generated JSON file to the leaderboard.
3.  **Receive Scores**: The evaluation results will be displayed instantly and sent to the email address you provide.

### Submission File Format

The dataset provides one `albumN_test.json` per album. You must **combine all albums into a single JSON array** and add the `album_id` field to each query before submitting.

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
            "pred": ["IMG_0001.JPG", "IMG_0002.JPG", ...]  # your predictions
        })

with open("submission.json", "w") as f:
    json.dump(submission, f, indent=2)
```

**Final submission format:**

```json
[
  {
    "album_id": "1",
    "query_en": "cluttered desk",
    "pred": ["IMG_1234.jpg", "IMG_5678.jpg", ...]
  }
]
```

**Required fields:**
- `album_id`: Album number (`"1"`, `"2"`, or `"3"` — string)
- `query_en`: The English query text (must match exactly, case-sensitive)
- `pred`: Ordered list of predicted image filenames (order matters for NDCG)

Only **full submissions** (all 3 albums, all test queries) are eligible for public leaderboard ranking. Partial submissions are accepted for evaluation but will not appear on the leaderboard.

## Repository Structure

-   `dataset/`: Complete dataset including images and query files.
    -   `album1/`, `album2/`, `album3/`: Complete photo albums.
    -   `validation.json`: Queries with released ground truth for local self-evaluation. 100 queries per album (300 total).
    -   `test.json`: Test queries for leaderboard submission. Album 1: 382, Album 2: 236, Album 3: 269 (887 total). Ground truth is held out.
-   `scripts/`: Evaluation scripts to generate submission files.
    -   `eval_embedding.py`: For embedding-based retrieval.
    -   `eval_caption.py`: For caption-index-based retrieval.
    -   `eval_agent.py`: For an agent-based retrieval pipeline.
