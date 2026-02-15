import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import faiss

from config import model_paths
from text_embedding import TextEmbedder


class CaptionSearch:
    def __init__(self, index_dir: str, device: str = "cuda"):
        self.index_dir = Path(index_dir)
        self.device = device
        self._embedder: Optional[TextEmbedder] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._filenames: List[str] = []
        self._model_name: str = ""
        self._load()

    def _load(self):
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        self._index = faiss.read_index(str(index_path))
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            self._filenames = metadata["filenames"]
            self._model_name = metadata.get("model_name", "bge-m3")
        self._load_model()

    def _load_model(self):
        if self._model_name not in model_paths:
            raise ValueError(f"Unknown model: {self._model_name}. Available: {list(model_paths.keys())}")
        self._embedder = TextEmbedder(model_name=self._model_name, device=self.device)

    def search(self, query: str, top_k: int = 100) -> List[Dict]:
        if self._index is None or self._embedder is None:
            raise ValueError("Index or embedder not loaded")
        query_emb = self._embedder.encode([query], batch_size=1, is_query=True)
        faiss.normalize_L2(query_emb)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({"filename": self._filenames[idx], "score": float(score)})
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=False)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_folder", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
    parser.add_argument("--language", choices=["en", "cn"], required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_folder) / args.language
    output_dir.mkdir(parents=True, exist_ok=True)
    out_model = args.model_name or "caption_index"
    output_file = output_dir / f"{out_model}_submission.json"

    searcher = CaptionSearch(index_dir=args.index_dir, device=args.device)
    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    submissions = []
    qfield = f"query_{args.language}"
    for q in queries:
        q_text = q.get(qfield) or q.get("query")
        if q_text is None:
            continue
        results = searcher.search(q_text, top_k=100)
        preds = [r["filename"] for r in results]
        submissions.append({"query_id": q.get("Query_id") or q.get("query_id"), "predictions": preds})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model_name": out_model, "language": args.language, "results": submissions}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
