from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    index_dir: Path
    embedding_model: str
    embedding_backend: str
    groq_api_key: str | None
    groq_model: str
    top_k: int
    cache_similarity_threshold: float
    llm_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        return cls(
            index_dir=ROOT / "data" / "index",
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "sentence_transformer"),
            groq_api_key=os.getenv("GROQ_API_KEY") or None,
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            top_k=int(os.getenv("TOP_K", "5")),
            cache_similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.97")),
            llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "12")),
        )
