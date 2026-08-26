from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from pypdf import PdfReader
from .chunking import chunk_page
from .config import Settings
from .embeddings import embed
from .store import FaissStore


def extract(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        return [(page.extract_text() or "") for page in PdfReader(path).pages]
    if path.suffix.lower() == ".txt":
        # Fixture-friendly, and \f makes pages testable without a PDF generator.
        return path.read_text(encoding="utf-8").split("\f")
    return []


def ingest(raw_dir: Path) -> dict[str, object]:
    settings = Settings.from_env()
    chunks = []
    sources = sorted([*raw_dir.glob("*.pdf"), *raw_dir.glob("*.txt")])
    if not sources:
        raise ValueError(f"No .pdf or .txt sources in {raw_dir}")
    for path in sources:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for page, text in enumerate(extract(path), start=1):
            chunks.extend(chunk_page(path.name, page, text, digest))
    vectors, embedding_ms = embed([c.text for c in chunks], settings.embedding_model, settings.embedding_backend)
    FaissStore(settings.index_dir).build(chunks, vectors)
    return {"documents": len(sources), "chunks": len(chunks), "embedding_ms": round(embedding_ms, 2)}


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    print(ingest(source))
