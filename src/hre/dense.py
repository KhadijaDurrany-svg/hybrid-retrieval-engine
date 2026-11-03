"""Dense semantic index.

Stores one L2-normalized vector per document and ranks by cosine similarity
(a dot product, since vectors are normalized). The default backend is an exact
NumPy scan, which is the right choice at this corpus size and keeps results
deterministic. An approximate-nearest-neighbour backend (e.g. hnswlib) can be
dropped in behind the same `search` signature without touching callers.
"""

from __future__ import annotations

import numpy as np

from .embedders import Embedder


class DenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.doc_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def fit(self, doc_ids: list[str], texts: list[str]) -> DenseIndex:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must be the same length")
        self.doc_ids = list(doc_ids)
        self._matrix = self.embedder.encode(texts)
        return self

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("index must be fit() before search()")
        if k <= 0 or self._matrix.shape[0] == 0:
            return []
        q = self.embedder.encode([query])[0]
        sims = self._matrix @ q  # cosine similarity (rows are normalized)
        k = min(k, sims.shape[0])
        # argpartition for the top-k, then sort just those by score desc.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(self.doc_ids[int(i)], float(sims[int(i)])) for i in top]
