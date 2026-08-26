from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import faiss
import numpy as np

from .domain import Chunk


class FaissStore:

    def __init__(self, directory: Path):
        self.directory = directory
        self.index_path = directory / "chunks.faiss"
        self.meta_path = directory / "chunks.json"

    def _faiss_path(self) -> Path:

        if os.name == "nt" and not str(
            self.index_path
        ).isascii():

            location = (
                Path(tempfile.gettempdir())
                / "vtu-rag-faiss"
            )

            location.mkdir(exist_ok=True)

            return location / "chunks.faiss"

        return self.index_path

    def build(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> None:

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        index.add(
            vectors.astype("float32")
        )

        faiss.write_index(
            index,
            str(self._faiss_path()),
        )

        self.meta_path.write_text(
            json.dumps(
                [
                    chunk.as_dict()
                    for chunk in chunks
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(
        self,
    ) -> tuple[faiss.Index, list[Chunk]]:

        if (
            not self._faiss_path().exists()
            or not self.meta_path.exists()
        ):
            raise FileNotFoundError(
                "No index. Run: "
                "python -m app.ingest data/raw"
            )

        index = faiss.read_index(
            str(self._faiss_path())
        )

        items = json.loads(
            self.meta_path.read_text(
                encoding="utf-8"
            )
        )

        return (
            index,
            [
                Chunk(**item)
                for item in items
            ],
        )

    def ready(self) -> bool:
        return (
            self._faiss_path().exists()
            and self.meta_path.exists()
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        subject: str | None = None,
        module: int | None = None,
    ) -> list[tuple[Chunk, float]]:

        index, chunks = self.load()

        if not chunks:
            return []

        # Search ALL chunks before applying subject/module filters.
        # This prevents relevant chunks from being excluded simply
        # because they were not in the global top-k candidates.
        candidate_count = len(chunks)

        scores, ids = index.search(
            query_vector.reshape(1, -1),
            candidate_count,
        )

        results = []
        seen_text = set()

        for score, index_id in zip(
            scores[0],
            ids[0],
        ):

            if index_id < 0:
                continue

            chunk = chunks[index_id]

            # -------------------------
            # SUBJECT FILTER
            # -------------------------
            if (
                subject is not None
                and chunk.subject.upper() != subject.upper()
            ):
                continue

            # -------------------------
            # MODULE FILTER
            # -------------------------
            if module is not None:

                if not chunk.heading:
                    continue

                import re

                match = re.search(
                    r"module\s*[-:]?\s*(\d+)",
                    chunk.heading,
                    re.IGNORECASE,
                )

                if not match:
                    continue

                chunk_module = int(
                    match.group(1)
                )

                if chunk_module != module:
                    continue

            # -------------------------
            # DUPLICATE FILTER
            # -------------------------
            normalized_text = " ".join(
                chunk.text.split()
            ).lower()

            if normalized_text in seen_text:
                continue

            seen_text.add(normalized_text)

            results.append(
                (
                    chunk,
                    float(score),
                )
            )

            if len(results) >= k:
                break

        return results