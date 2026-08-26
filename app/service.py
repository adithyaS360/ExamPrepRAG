from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from .config import Settings
from .embeddings import embed
from .llm import LlmUnavailable, answer
from .store import FaissStore


@dataclass
class CacheEntry:
    vector: np.ndarray
    response: dict


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = FaissStore(settings.index_dir)
        self.cache: list[CacheEntry] = []  # process-local: replace with Redis when horizontally scaling
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.store.ready():
            from .ingest import ingest
            raw = Path("data/raw")
            fixtures = Path("data/fixtures")
            if not (raw.exists() and list(raw.glob("*.pdf"))):
                try:
                    from scripts.fetch_vtu_source import main as fetch_vtu
                    fetch_vtu()
                except Exception as e:
                    print("Could not fetch VTU PDF automatically:", e)
            ingested = False
            if raw.exists() and (list(raw.glob("*.pdf")) or list(raw.glob("*.txt"))):
                try:
                    ingest(raw)
                    ingested = True
                except Exception as e:
                    print("Raw ingestion failed, falling back to fixtures:", e)
            if not ingested and fixtures.exists() and list(fixtures.glob("*.txt")):
                ingest(fixtures)




    def query(self, question: str, force_fallback: bool = False) -> dict:
        started = time.perf_counter()
        vector, embedding_ms = embed([question], self.settings.embedding_model, self.settings.embedding_backend)
        for entry in self.cache:
            similarity = float(np.dot(vector[0], entry.vector))
            if similarity >= self.settings.cache_similarity_threshold:
                result = dict(entry.response)
                result["cache"] = {"hit": True, "similarity": round(similarity, 4)}
                result["latency_ms"] = {"embedding": round(embedding_ms, 2), "retrieval": 0.0, "llm": 0.0,
                    "total": round((time.perf_counter() - started) * 1000, 2)}
                return result
        retrieval_started = time.perf_counter()
        matches = self.store.search(vector[0], self.settings.top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        llm_started = time.perf_counter()
        fallback_reason = None
        try:
            if force_fallback:
                raise LlmUnavailable("forced fallback for smoke test")
            generated = answer(question, matches, self.settings.groq_api_key, self.settings.groq_model,
                               self.settings.llm_timeout_seconds)
            mode = "llm"
        except LlmUnavailable as error:
            generated = None
            mode = "retrieval_fallback"
            fallback_reason = str(error)
        llm_ms = (time.perf_counter() - llm_started) * 1000
        result = {"question": question, "answer": generated, "mode": mode,
                  "fallback_reason": fallback_reason,
                  "citations": [chunk.citation() | {"score": round(score, 4)} for chunk, score in matches],
                  "retrieved_chunks": [{"text": chunk.text, "citation": chunk.citation(), "score": round(score, 4)}
                                       for chunk, score in matches],
                  "cache": {"hit": False},
                  "latency_ms": {"embedding": round(embedding_ms, 2), "retrieval": round(retrieval_ms, 2),
                                 "llm": round(llm_ms, 2), "total": round((time.perf_counter() - started) * 1000, 2)}}
        self.cache.append(CacheEntry(vector[0], result))
        return result
