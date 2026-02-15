# PhotoBench

PhotoBench is a privacy-preserving benchmark for photo retrieval and reasoning. It provides a display subset (samples) and machine-readable features (protected) to enable local evaluation and submission without exposing raw images.

## What’s included

- Dataset
	- `dataset/samples/`: small display subset per album (20 queries with visualization images).
	- `dataset/protected/`: features for local evaluation (caption/embedding indices, location/time metadata, anonymized face info).
- Scripts
	- `scripts/eval_embedding.py`: run embedding-based retrieval and save submission.
	- `scripts/eval_caption.py`: run caption-index retrieval and save submission.
	- `scripts/eval_agent.py`: run an agent-based retrieval pipeline and save submission.

## Submission format

All evaluation scripts write to `results/<language>/<model_name>_submission.json` with:

```
{
	"model_name": "<model>",
	"language": "cn|en",
	"results": [
		{ "query_id": "<id>", "predictions": ["<filename>", ...] }
	]
}
```

## Minimal usage

Embedding retrieval:
- Inputs: an image embedding index at `index_dir` (with `index.faiss` and `metadata.json`), a query file containing `query_cn` or `query_en`.
- Run: `scripts/eval_embedding.py` with `--model_name`, `--index_dir`, `--query_file`, `--language` (and optional `--device`, `--output_folder`).

Caption retrieval:
- Inputs: a caption index at `index_dir` and a query file.
- Run: `scripts/eval_caption.py` with `--index_dir`, `--query_file`, `--language` (and optional `--model_name`, `--device`, `--output_folder`).

Agent pipeline:
- Inputs: embedding index, face info, metadata, query file.
- Run: `scripts/eval_agent.py` with `--face_info_path`, `--metadata_path`, `--model_name`, `--index_dir`, `--query_file`, `--language` (optional `--output_folder`, `--device`, `--llm_model`, `--concurrent`).

## Notes

- Samples are for display; protected features enable local evaluation without raw images.
- Submit the generated JSON to the evaluation service to obtain scores against hidden ground truth.
