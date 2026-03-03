"""
Text Embedding Utilities (Caption)

Provides a small, shared embedder to avoid duplicated model/encode logic.
Uses sentence_transformers for Qwen3-Embedding, FlagEmbedding for BGE.
"""

from typing import List

import numpy as np
from tqdm import tqdm

from config import model_paths


class TextEmbedder:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
        self._model = None
        self._kind = None  # "bge-m3" | "bge" | "qwen3" | "e5"
        
        # Load model immediately during initialization
        self._load()

    def _load(self):
        """Load the embedding model."""
        if self._model is not None:
            return

        if self.model_name not in model_paths:
            raise ValueError(f"Unknown model: {self.model_name}. Available: {list(model_paths.keys())}")

        model_path = model_paths[self.model_name]

        # 1. BGE M3
        if self.model_name == "bge-m3":
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(model_path, use_fp16=True, device=self.device)
            self._kind = "bge-m3"
            print(f"Loaded BGE-M3 model from {model_path}")
            return

        # 2. Other BGE models
        if self.model_name.startswith("bge-"):
            from FlagEmbedding import FlagModel

            self._model = FlagModel(model_path, use_fp16=True, device=self.device)
            self._kind = "bge"
            print(f"Loaded BGE model from {model_path}")
            return

        # 3. Qwen Series
        if self.model_name.startswith("Qwen3-Embedding"):
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
            self._kind = "qwen3"
            print(f"Loaded Qwen3-Embedding model from {model_path}")
            return

        # 4. E5 Series
        if "e5" in self.model_name:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
            self._kind = "e5"
            print(f"Loaded E5 model from {model_path}")
            return

        raise ValueError(f"Unsupported model type: {self.model_name}")

    @staticmethod
    def _apply_e5_prefix(batch: List[str], is_query: bool) -> List[str]:
        """Ensure E5 inputs have the expected 'query:' or 'passage:' prefix."""
        prefix = "query:" if is_query else "passage:"
        prefixed = []
        for txt in batch:
            low = txt.lstrip().lower()
            if low.startswith("query:") or low.startswith("passage:"):
                prefixed.append(txt)
            else:
                prefixed.append(f"{prefix} {txt}")
        return prefixed

    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        show_progress: bool = False,
        is_query: bool = False,  # NEW: 区分 query 和 document
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            is_query: If True, encode as query (for retrieval). 
                      For Qwen3-Embedding, this will use prompt_name="query".
                      For BGE models, this has no effect.
        """
        if not texts:
            return np.array([], dtype=np.float32)
        
        if self._model is None:
            raise ValueError("Model not loaded")

        vecs = []
        # Unified batch loop
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch = texts[i : i + batch_size]

            if self._kind == "bge-m3":
                out = self._model.encode(batch)
                vecs.append(out["dense_vecs"])

            elif self._kind == "bge":
                out = self._model.encode(batch)
                vecs.append(out)

            elif self._kind == "e5":
                batch_prefixed = self._apply_e5_prefix(batch, is_query=is_query)
                out = self._model.encode(
                    batch_prefixed,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                vecs.append(out)

            else:  # qwen3
                # Disable internal progress bar since we have an outer one
                # Use prompt_name="query" for query encoding (retrieval)
                if is_query:
                    out = self._model.encode(batch, show_progress_bar=False, prompt_name="query")
                else:
                    out = self._model.encode(batch, show_progress_bar=False)
                vecs.append(out)

        return np.vstack(vecs).astype(np.float32)
