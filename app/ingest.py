from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from .chunking import chunk_page
from .config import Settings
from .embeddings import embed
from .store import FaissStore


VALID_SUBJECTS = {
    "CNS",
    "BDA",
    "PC",
    "IOT",
}


# Names/identifiers used to recognize the four subjects
SUBJECT_PATTERNS = {
    "CNS": [
        r"\bBCS703\b",
        r"cryptography\s+and\s+network\s+security",
        r"\bCNS\b",
    ],
    "IOT": [
        r"\bBCS701\b",
        r"internet\s+of\s+things",
        r"\bIOT\b",
    ],
    "PC": [
        r"\bBCS702\b",
        r"parallel\s+computing",
        r"\bPC\b",
    ],
    "BDA": [
        r"big\s+data\s+analytics",
        r"\bBDA\b",
    ],
}


def extract(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        return [
            page.extract_text() or ""
            for page in PdfReader(path).pages
        ]

    if path.suffix.lower() == ".txt":
        return path.read_text(
            encoding="utf-8"
        ).split("\f")

    return []


def detect_subject(
    path: Path,
    raw_dir: Path,
) -> str:
    """
    Determine subject from the directory structure.

    Files inside:

        data/raw/CNS/
        data/raw/BDA/
        data/raw/PC/
        data/raw/IOT/

    are automatically assigned that subject.
    """

    try:
        relative = path.relative_to(raw_dir)

        if relative.parts:
            folder_subject = relative.parts[0].upper()

            if folder_subject in VALID_SUBJECTS:
                return folder_subject

    except ValueError:
        pass

    # Files directly inside data/raw can also use:
    # CNS_xxx.pdf, BDA_xxx.pdf, PC_xxx.pdf, IOT_xxx.pdf
    name = path.stem.upper()

    for subject in VALID_SUBJECTS:
        if (
            name.startswith(subject + "_")
            or name.startswith(subject + "-")
            or name == subject
        ):
            return subject

    raise ValueError(
        f"Could not determine subject for '{path.name}'. "
        f"Put subject documents inside "
        f"data/raw/CNS, data/raw/BDA, "
        f"data/raw/PC, or data/raw/IOT."
    )


def detect_syllabus_subject(text: str) -> str | None:
    """
    Detect one of the four target subjects from a syllabus page.

    The combined VTU syllabus PDFs contain many subjects, so we only
    recognize CNS, BDA, PC and IOT.
    """

    if not text:
        return None

    # Normalize whitespace so phrases split across lines can match.
    normalized = " ".join(text.split())

    for subject, patterns in SUBJECT_PATTERNS.items():
        for pattern in patterns:
            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                return subject

    return None


def is_syllabus_file(
    path: Path,
    raw_dir: Path,
) -> bool:
    """
    Files inside data/raw/syllabus/ are treated as combined
    syllabus documents.
    """

    try:
        relative = path.relative_to(raw_dir)

        return (
            bool(relative.parts)
            and relative.parts[0].lower() == "syllabus"
        )

    except ValueError:
        return False


def ingest(raw_dir: Path) -> dict[str, object]:
    settings = Settings.from_env()

    chunks = []

    sources = sorted(
        [
            *raw_dir.rglob("*.pdf"),
            *raw_dir.rglob("*.PDF"),
            *raw_dir.rglob("*.txt"),
            *raw_dir.rglob("*.TXT"),
        ]
    )

    if not sources:
        raise ValueError(
            f"No .pdf or .txt sources in {raw_dir}"
        )

    indexed_documents = 0
    skipped_syllabus_pages = 0

    for path in sources:

        # ---------------------------------------------------------
        # NORMAL SUBJECT DOCUMENT
        # ---------------------------------------------------------
        if not is_syllabus_file(path, raw_dir):

            subject = detect_subject(
                path,
                raw_dir,
            )

            digest = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()[:16]

            pages = extract(path)

            for page_number, text in enumerate(
                pages,
                start=1,
            ):

                page_chunks = chunk_page(
                    path.name,
                    page_number,
                    text,
                    digest,
                    subject,
                )

                chunks.extend(page_chunks)

            indexed_documents += 1
            continue

        # ---------------------------------------------------------
        # COMBINED SYLLABUS DOCUMENT
        # ---------------------------------------------------------
        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()[:16]

        pages = extract(path)

        current_subject: str | None = None

        for page_number, text in enumerate(
            pages,
            start=1,
        ):

            detected = detect_syllabus_subject(text)

            if detected is not None:
                current_subject = detected

            # If this page does not contain enough information to
            # determine a target subject, don't guess.
            if current_subject is None:
                skipped_syllabus_pages += 1
                continue

            page_chunks = chunk_page(
                path.name,
                page_number,
                text,
                digest,
                current_subject,
            )

            chunks.extend(page_chunks)

        indexed_documents += 1

    if not chunks:
        raise ValueError(
            "Documents were found, but no text chunks "
            "could be extracted for CNS, BDA, PC or IOT."
        )

    # -------------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # -------------------------------------------------------------
    unique_chunks = []
    seen = set()

    for chunk in chunks:

        key = (
            chunk.subject,
            chunk.document,
            chunk.page,
            " ".join(
                chunk.text.split()
            ).lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    chunks = unique_chunks

    # -------------------------------------------------------------
    # CREATE EMBEDDINGS
    # -------------------------------------------------------------
    vectors, embedding_ms = embed(
        [chunk.text for chunk in chunks],
        settings.embedding_model,
        settings.embedding_backend,
    )

    # -------------------------------------------------------------
    # BUILD FAISS INDEX
    # -------------------------------------------------------------
    FaissStore(
        settings.index_dir
    ).build(
        chunks,
        vectors,
    )

    # -------------------------------------------------------------
    # COUNT CHUNKS PER SUBJECT
    # -------------------------------------------------------------
    subject_counts: dict[str, int] = {}

    for chunk in chunks:
        subject_counts[chunk.subject] = (
            subject_counts.get(chunk.subject, 0) + 1
        )

    return {
        "documents": indexed_documents,
        "chunks": len(chunks),
        "subjects": subject_counts,
        "skipped_syllabus_pages": skipped_syllabus_pages,
        "embedding_ms": round(
            embedding_ms,
            2,
        ),
    }


if __name__ == "__main__":

    source = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("data/raw")
    )

    print(ingest(source))