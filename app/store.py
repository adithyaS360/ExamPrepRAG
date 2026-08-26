from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import faiss
import numpy as np
from .domain import Chunk


class FaissStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.index_path = directory / "chunks.faiss"
        self.meta_path = directory / "chunks.json"

    def _faiss_path(self) -> Path:
        """FAISS Windows wheels cannot open non-ASCII paths; stage only the binary index."""
        if os.name == "nt" and not str(self.index_path).isascii():
            location = Path(tempfile.gettempdir()) / "vtu-rag-faiss"
            location.mkdir(exist_ok=True)
            return location / "chunks.faiss"
        return self.index_path

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity because vectors are normalized
        index.add(vectors)
        faiss.write_index(index, str(self._faiss_path()))
        self.meta_path.write_text(json.dumps([c.as_dict() for c in chunks], indent=2), encoding="utf-8")

    def load(self) -> tuple[faiss.Index, list[Chunk]]:
        if not self._faiss_path().exists() or not self.meta_path.exists():
            raise FileNotFoundError("No index. Run: python -m app.ingest data/raw")
        index = faiss.read_index(str(self._faiss_path()))
        items = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return index, [Chunk(**item) for item in items]

    def ready(self) -> bool:
        return self._faiss_path().exists() and self.meta_path.exists()

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        index, chunks = self.load()
        scores, ids = index.search(query_vector.reshape(1, -1), min(k, len(chunks)))
        return [(chunks[i], float(score)) for score, i in zip(scores[0], ids[0]) if i >= 0]
