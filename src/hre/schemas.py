"""Typed data models shared across the engine and the HTTP service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single indexable document.

    `id` must be unique within a corpus. `text` is the body used for ranking.
    `title` is optional and, when present, is weighted more heavily by the
    analyzer-aware BM25F-style field boosting.
    """

    id: str
    text: str
    title: str = ""
    meta: dict = field(default_factory=dict)

    def content(self) -> str:
        """Concatenated searchable content (title repeated for a light boost)."""
        if self.title:
            return f"{self.title} {self.title} {self.text}"
        return self.text


@dataclass(frozen=True)
class Hit:
    """One ranked result returned to a caller."""

    doc_id: str
    score: float
    rank: int
    title: str = ""
    snippet: str = ""

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "title": self.title,
            "snippet": self.snippet,
        }
