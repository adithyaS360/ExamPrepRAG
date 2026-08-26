from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    page: int
    text: str
    source_type: str
    source_hash: str
    subject: str
    heading: str | None = None

    def citation(self) -> dict[str, object]:
        return {
            "document": self.document,
            "page": self.page,
            "heading": self.heading,
            "subject": self.subject,
        }

    def as_dict(self) -> dict[str, object]:
        return asdict(self)