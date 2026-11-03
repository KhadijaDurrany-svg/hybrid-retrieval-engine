"""Hybrid Retrieval Engine (hre).

A compact, production-minded hybrid search engine that combines a hand-written
BM25 lexical index with a dense semantic index, fuses the two ranked lists, and
optionally reranks the top candidates. Every ranking stage is pluggable and the
whole pipeline runs fully offline.
"""

from .engine import HybridEngine, SearchResult, StageTimings
from .schemas import Document, Hit

__all__ = [
    "HybridEngine",
    "SearchResult",
    "StageTimings",
    "Document",
    "Hit",
]

__version__ = "0.1.0"
