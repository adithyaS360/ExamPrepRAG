from __future__ import annotations
import time
from functools import lru_cache
import hashlib
import re
import numpy as np


@lru_cache(maxsize=1)
def model(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install sentence-transformers to use EMBEDDING_BACKEND=sentence_transformer") from error
    return SentenceTransformer(name)


def _hash_embed(texts: list[str], dimensions: int = 768) -> np.ndarray:
    """Dependency-free smoke-test embedding; lexical, not a substitute for MiniLM."""
    vectors = np.zeros((len(texts), dimensions), dtype="float32")
    for row, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % dimensions
            vectors[row, bucket] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed(texts: list[str], model_name: str, backend: str = "hash") -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    vectors = (_hash_embed(texts) if backend == "hash"
               else model(model_name).encode(texts, normalize_embeddings=True, show_progress_bar=False))
    return np.asarray(vectors, dtype="float32"), (time.perf_counter() - started) * 1000
