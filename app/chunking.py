from __future__ import annotations
import re
from .domain import Chunk

MODULE = re.compile(r"(?im)^\s*(module\s*[-:]?\s*\d+[^\n]*)")


def detect_type(text: str) -> str:
    return "question_paper" if re.search(r"question\s+paper|answer any|\bquestion\s*\d", text, re.I) else "syllabus"


def chunk_page(document: str, page: int, text: str, source_hash: str, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Structure-aware chunks: never cross PDF pages; prefer module boundaries first."""
    clean = re.sub(r"[ \t]+", " ", text).strip()
    if not clean:
        return []
    boundaries = [m.start() for m in MODULE.finditer(clean)] + [len(clean)]
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    pieces: list[tuple[str, str | None]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        section = clean[start:end].strip()
        heading_match = MODULE.match(section)
        heading = heading_match.group(1) if heading_match else None
        while len(section) > max_chars:
            cut = section.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            pieces.append((section[:cut].strip(), heading))
            section = section[max(0, cut - overlap):].strip()
        if section:
            pieces.append((section, heading))
    source_type = detect_type(clean)
    return [Chunk(id=f"{source_hash}:{page}:{i}", document=document, page=page, text=body,
                  source_type=source_type, source_hash=source_hash, heading=heading)
            for i, (body, heading) in enumerate(pieces)]
