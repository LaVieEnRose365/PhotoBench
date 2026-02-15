import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import torch
from PIL import Image

from ops_embedding import OpsMMEmbeddingV1, fetch_image, create_model
from config import model_paths

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tiff", ".gif"}

class ImageSearchEngine:
    def __init__(self, model: torch.nn.Module, model_name: str, index_dir: str):
        self.model = model
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._filenames: List[str] = []

    @classmethod
    def from_pretrained(cls, model_name: str, index_dir: str, device: str = "cuda", **model_kwargs):
        if model_name not in model_paths:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(model_paths.keys())}")
        model_path = model_paths[model_name]
        model = create_model(model_name=model_name, model_path=model_path, device=device, **model_kwargs)
        return cls(model, model_name, index_dir)

    def build_index(self, images_dir: str, batch_size: int = 8, force_rebuild: bool = False, instruction: str = None, prompt_suffix: str = None) -> None:
        if not force_rebuild and (self.index_dir / "index.faiss").exists():
            self.load()
            return
        images_path = Path(images_dir)
        all_files = [f for f in images_path.iterdir() if f.is_file() and not f.name.startswith('.')]
        unsupported = [f.name for f in all_files if f.suffix.lower() not in SUPPORTED_EXTENSIONS]
        if unsupported:
            raise ValueError(f"Unsupported file formats: {unsupported}. Supported: {SUPPORTED_EXTENSIONS}")
        image_files = sorted(all_files)
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        all_embeddings = []
        self._filenames = []
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            batch_images, batch_names = [], []
            for img_path in batch_files:
                try:
                    batch_images.append(fetch_image(str(img_path)))
                    batch_names.append(img_path.name)
                except Exception:
                    pass
            if batch_images:
                with torch.no_grad():
                    embeddings = self.model.get_image_embeddings(images=batch_images, instruction=instruction, prompt_suffix=prompt_suffix, batch_size=len(batch_images), show_progress=False)
                all_embeddings.append(embeddings.float().cpu().numpy())
                self._filenames.extend(batch_names)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if not all_embeddings:
            raise ValueError("No valid images processed")
        embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
        self._index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        self._index.add(embeddings_matrix)
        self.save()

    def save(self) -> None:
        if self._index is None:
            raise ValueError("No index to save")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "filenames": self._filenames, "created_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        self._index = faiss.read_index(str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "metadata.json", "r", encoding="utf-8") as f:
            self._filenames = json.load(f)["filenames"]

    def retrieve(self, text: Optional[str] = None, image: Optional[Union[str, Image.Image]] = None, top_k: int = 10, instruction: Optional[str] = None, prompt_suffix: Optional[str] = None) -> List[dict]:
        if self._index is None:
            raise ValueError("No index loaded")
        if text is None and image is None:
            raise ValueError("Must provide text or image or both")
        texts = [text] if text else None
        images = [fetch_image(image) if isinstance(image, str) else image] if image else None
        with torch.no_grad():
            query_emb = self.model.get_fused_embeddings(texts=texts, images=images, instruction=instruction, prompt_suffix=prompt_suffix, batch_size=1, show_progress=False)
        scores, indices = self._index.search(query_emb.float().cpu().numpy(), min(top_k, self._index.ntotal))
        return [{"filename": self._filenames[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0]) if idx >= 0]
