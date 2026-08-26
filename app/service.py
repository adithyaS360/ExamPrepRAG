from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .embeddings import embed
from .llm import LlmUnavailable, answer
from .store import FaissStore


@dataclass
class CacheEntry:
    vector: np.ndarray
    subject: str
    module: int | None
    response: dict


class RagService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = FaissStore(
            settings.index_dir
        )

        self.cache: list[CacheEntry] = []

        self._ensure_index()

    def _ensure_index(self) -> None:

        if self.store.ready():
            return

        from .ingest import ingest

        raw = Path("data/raw")
        fixtures = Path("data/fixtures")

        documents = []

        if raw.exists():
            documents = [
                *raw.rglob("*.pdf"),
                *raw.rglob("*.PDF"),
                *raw.rglob("*.txt"),
                *raw.rglob("*.TXT"),
            ]

        if not documents:

            try:
                from scripts.fetch_vtu_source import main as fetch_vtu

                fetch_vtu()

            except Exception as error:
                print(
                    "Could not fetch VTU PDF automatically:",
                    error,
                )

            if raw.exists():
                documents = [
                    *raw.rglob("*.pdf"),
                    *raw.rglob("*.PDF"),
                    *raw.rglob("*.txt"),
                    *raw.rglob("*.TXT"),
                ]

        ingested = False

        if documents:

            try:
                ingest(raw)
                ingested = True

            except Exception as error:
                print(
                    "Raw ingestion failed, "
                    "falling back to fixtures:",
                    error,
                )

        if (
            not ingested
            and fixtures.exists()
            and (
                list(fixtures.glob("*.txt"))
                or list(fixtures.glob("*.pdf"))
            )
        ):
            ingest(fixtures)

    def _extract_subject(
        self,
        question: str,
    ) -> tuple[str | None, str]:

        patterns = [
            r"^\s*(?:for\s+)?(CNS|BDA|PC|IOT)\s*[:,-]\s*(.+)$",
            r"^\s*(?:for\s+)?(CNS|BDA|PC|IOT)\s+(.+)$",
        ]

        for pattern in patterns:

            match = re.match(
                pattern,
                question,
                re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1).upper(),
                    match.group(2).strip(),
                )

        return None, question.strip()

    def _extract_module(
        self,
        question: str,
    ) -> tuple[int | None, str]:

        patterns = [
            r"\bmodule\s*[-:]?\s*(\d+)\b",
            r"\bmod\s*[-:]?\s*(\d+)\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                question,
                re.IGNORECASE,
            )

            if match:

                module = int(
                    match.group(1)
                )

                clean_question = re.sub(
                    pattern,
                    "",
                    question,
                    flags=re.IGNORECASE,
                )

                clean_question = re.sub(
                    r"\s+",
                    " ",
                    clean_question,
                ).strip(" :-,")

                return (
                    module,
                    clean_question,
                )

        return None, question.strip()

    def query(
        self,
        question: str,
        force_fallback: bool = False,
    ) -> dict:

        started = time.perf_counter()

        subject, clean_question = (
            self._extract_subject(question)
        )

        if subject is None:

            return {
                "question": question,
                "answer": None,
                "mode": "subject_required",
                "fallback_reason": None,
                "error": (
                    "Please specify a subject: "
                    "CNS, BDA, PC, or IOT."
                ),
                "citations": [],
                "retrieved_chunks": [],
                "cache": {
                    "hit": False
                },
                "latency_ms": {
                    "embedding": 0.0,
                    "retrieval": 0.0,
                    "llm": 0.0,
                    "total": round(
                        (
                            time.perf_counter()
                            - started
                        ) * 1000,
                        2,
                    ),
                },
            }

        module, clean_question = (
            self._extract_module(clean_question)
        )

        vector, embedding_ms = embed(
            [clean_question],
            self.settings.embedding_model,
            self.settings.embedding_backend,
        )
        # -------------------------
        # RETRIEVAL
        # -------------------------
        retrieval_started = (
            time.perf_counter()
        )

        matches = self.store.search(
            vector[0],
            self.settings.top_k,
            subject=subject,
            module=module,
        )

        retrieval_ms = (
            time.perf_counter()
            - retrieval_started
        ) * 1000

        # -------------------------
        # LLM
        # -------------------------
        llm_started = (
            time.perf_counter()
        )

        fallback_reason = None

        try:

            if force_fallback:
                raise LlmUnavailable(
                    "forced fallback for smoke test"
                )

            generated = answer(
                clean_question,
                matches,
                self.settings.groq_api_key,
                self.settings.groq_model,
                self.settings.llm_timeout_seconds,
            )

            mode = "llm"

        except LlmUnavailable as error:

            generated = None
            mode = "retrieval_fallback"
            fallback_reason = str(error)

        llm_ms = (
            time.perf_counter()
            - llm_started
        ) * 1000

        # -------------------------
        # RESPONSE
        # -------------------------
        result = {
            "question": question,
            "subject": subject,
            "module": module,
            "answer": generated,
            "mode": mode,
            "fallback_reason": fallback_reason,

            "citations": [
                chunk.citation()
                | {
                    "score": round(
                        score,
                        4,
                    )
                }
                for chunk, score in matches
            ],

            "retrieved_chunks": [
                {
                    "text": chunk.text,
                    "citation": chunk.citation(),
                    "score": round(
                        score,
                        4,
                    ),
                }
                for chunk, score in matches
            ],

            "cache": {
                "hit": False
            },

            "latency_ms": {
                "embedding": round(
                    embedding_ms,
                    2,
                ),
                "retrieval": round(
                    retrieval_ms,
                    2,
                ),
                "llm": round(
                    llm_ms,
                    2,
                ),
                "total": round(
                    (
                        time.perf_counter()
                        - started
                    ) * 1000,
                    2,
                ),
            },
        }

        #self.cache.append(
            #CacheEntry(
              #  vector=vector[0],
                #subject=subject,
               # module=module,
               # response=result,
            #)
      #  )

        if len(self.cache) > 100:
            self.cache.pop(0)

        return result