from __future__ import annotations

import hashlib
import re
import time
from functools import lru_cache

import numpy as np


WORD_DIMENSIONS = 1024
CHAR_DIMENSIONS = 1024
TOTAL_DIMENSIONS = WORD_DIMENSIONS + CHAR_DIMENSIONS


def _normalize_token(token: str) -> str:
    """
    Lightweight normalization so related word forms share features.

    Examples:
        dependencies -> depend
        dependency   -> depend
        normalized   -> normal
        normalization -> normal
    """
    token = token.lower()

    if len(token) <= 4:
        return token

    replacements = [
        ("ies", "y"),
        ("ation", ""),
        ("tions", ""),
        ("tion", ""),
        ("ments", ""),
        ("ment", ""),
        ("ing", ""),
        ("ed", ""),
        ("es", ""),
        ("s", ""),
    ]

    for suffix, replacement in replacements:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + replacement
            break

    return token


def _hash_to_bucket(value: str, dimensions: int) -> int:
    digest = hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(digest, "little") % dimensions


def _hybrid_hash_embed(
    texts: list[str],
) -> np.ndarray:
    """
    Dependency-free retrieval embedding.

    Combines:
      1. normalized word features
      2. character n-gram features

    This is still lexical retrieval, not a neural semantic model,
    but it is substantially stronger than the original token-only
    hash embedding and remains extremely lightweight.
    """

    vectors = np.zeros(
        (len(texts), TOTAL_DIMENSIONS),
        dtype="float32",
    )

    for row, text in enumerate(texts):
        text = text.lower()

        # -----------------------------
        # WORD FEATURES
        # -----------------------------
        tokens = re.findall(r"[a-z0-9]+", text)

        for token in tokens:
            normalized = _normalize_token(token)

            # Original token
            bucket = _hash_to_bucket(
                "word:" + token,
                WORD_DIMENSIONS,
            )
            vectors[row, bucket] += 1.0

            # Normalized token
            bucket = _hash_to_bucket(
                "stem:" + normalized,
                WORD_DIMENSIONS,
            )
            vectors[row, bucket] += 1.5

        # -----------------------------
        # CHARACTER N-GRAM FEATURES
        # -----------------------------
        compact = re.sub(r"[^a-z0-9]+", " ", text)

        for word in compact.split():
            if len(word) < 3:
                continue

            padded = f"^{word}$"

            for n in (3, 4, 5):
                for i in range(len(padded) - n + 1):
                    gram = padded[i : i + n]

                    bucket = _hash_to_bucket(
                        f"char:{n}:{gram}",
                        CHAR_DIMENSIONS,
                    )

                    vectors[
                        row,
                        WORD_DIMENSIONS + bucket,
                    ] += 0.35

    # L2 normalize so FAISS inner-product search
    # behaves like cosine similarity.
    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    vectors /= np.maximum(norms, 1e-12)

    return vectors


@lru_cache(maxsize=1)
def model(name: str):
    """
    Optional SentenceTransformer backend.

    Kept for local experimentation, but Render should use
    EMBEDDING_BACKEND=hash to avoid loading Torch.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install sentence-transformers to use "
            "EMBEDDING_BACKEND=sentence_transformer"
        ) from error

    return SentenceTransformer(name)


def embed(
    texts: list[str],
    model_name: str,
    backend: str = "hash",
) -> tuple[np.ndarray, float]:

    started = time.perf_counter()

    if backend == "hash":
        vectors = _hybrid_hash_embed(texts)

    else:
        vectors = model(model_name).encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return (
        np.asarray(vectors, dtype="float32"),
        (time.perf_counter() - started) * 1000,
    )