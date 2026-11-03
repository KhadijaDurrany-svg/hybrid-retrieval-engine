"""A hand-written BM25 inverted index with WAND top-k pruning.

The index is built from scratch (no Lucene, no rank-bm25) so the scoring and the
pruning are fully under our control and can be checked against a brute-force
reference in the test-suite. `search` uses the WAND algorithm (Broder et al.,
2003) with per-term score upper bounds; it returns the exact same top-k scores
as an exhaustive scan while skipping documents that cannot enter the result.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .analysis import Analyzer

_INF_DOC = math.inf


@dataclass
class BM25Params:
    k1: float = 1.2
    b: float = 0.75


class BM25Index:
    def __init__(self, analyzer: Analyzer | None = None, params: BM25Params | None = None) -> None:
        self.analyzer = analyzer or Analyzer()
        self.params = params or BM25Params()
        self.doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._term_ub: dict[str, float] = {}
        self._frozen = False

    # ------------------------------------------------------------------ build
    def add(self, doc_id: str, text: str) -> None:
        if self._frozen:
            raise RuntimeError("index is frozen; create a new index to add documents")
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        terms = self.analyzer.analyze(text)
        self._doc_len.append(len(terms))
        tf: dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            self._postings.setdefault(term, []).append((idx, freq))
            self._df[term] = self._df.get(term, 0) + 1

    def build(self) -> BM25Index:
        """Finalize statistics and precompute per-term score upper bounds."""
        n = len(self.doc_ids)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        for term, df in self._df.items():
            # BM25 idf, always non-negative so WAND upper bounds stay valid.
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for term, postings in self._postings.items():
            postings.sort(key=lambda p: p[0])  # by doc index, required by WAND
            ub = 0.0
            for doc_idx, freq in postings:
                ub = max(ub, self._term_score(term, freq, self._doc_len[doc_idx]))
            self._term_ub[term] = ub
        self._frozen = True
        return self

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    # ---------------------------------------------------------------- scoring
    def _term_score(self, term: str, tf: int, dl: int) -> float:
        k1, b = self.params.k1, self.params.b
        idf = self._idf.get(term, 0.0)
        denom = tf + k1 * (1.0 - b + b * (dl / self._avgdl if self._avgdl else 0.0))
        return idf * (tf * (k1 + 1.0)) / denom if denom else 0.0

    def _query_terms(self, query: str) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.analyzer.analyze(query):
            if t in self._postings:
                seen.setdefault(t, None)
        return list(seen.keys())

    # -------------------------------------------------------------- reference
    def search_bruteforce(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Exhaustive scan. Used as the correctness oracle for WAND."""
        terms = self._query_terms(query)
        scores: dict[int, float] = {}
        for term in terms:
            for doc_idx, freq in self._postings[term]:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + self._term_score(
                    term, freq, self._doc_len[doc_idx]
                )
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.doc_ids[i], s) for i, s in ranked]

    # ------------------------------------------------------------------- WAND
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        terms = self._query_terms(query)
        if not terms or k <= 0:
            return []

        postings = [self._postings[t] for t in terms]
        ubs = [self._term_ub[t] for t in terms]
        cursors = [0] * len(terms)

        def current_doc(i: int) -> float:
            c = cursors[i]
            pl = postings[i]
            return pl[c][0] if c < len(pl) else _INF_DOC

        # Min-heap of the current top-k as (score, doc_idx).
        heap: list[tuple[float, int]] = []
        threshold = 0.0

        def full() -> bool:
            return len(heap) >= k

        order = list(range(len(terms)))

        while True:
            order.sort(key=current_doc)
            if current_doc(order[0]) == _INF_DOC:
                break

            # Find the pivot term: first point where accumulated UB beats theta.
            acc = 0.0
            pivot = -1
            for oi in order:
                if current_doc(oi) == _INF_DOC:
                    break
                acc += ubs[oi]
                if acc > threshold:
                    pivot = oi
                    break
            if pivot == -1:
                break  # no document can beat the threshold any more

            pivot_doc = current_doc(pivot)
            if current_doc(order[0]) == pivot_doc:
                # All terms before the pivot are aligned on pivot_doc: score it.
                score = 0.0
                for i in range(len(terms)):
                    if current_doc(i) == pivot_doc:
                        freq = postings[i][cursors[i]][1]
                        score += self._term_score(terms[i], freq, self._doc_len[int(pivot_doc)])
                        cursors[i] += 1
                if not full():
                    heapq.heappush(heap, (score, int(pivot_doc)))
                    if full():
                        threshold = heap[0][0]
                elif score > threshold:
                    heapq.heapreplace(heap, (score, int(pivot_doc)))
                    threshold = heap[0][0]
            else:
                # Advance one lagging cursor up to pivot_doc.
                for oi in order:
                    if current_doc(oi) < pivot_doc:
                        pl = postings[oi]
                        lo, hi = cursors[oi], len(pl)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if pl[mid][0] < pivot_doc:
                                lo = mid + 1
                            else:
                                hi = mid
                        cursors[oi] = lo
                        break

        ranked = sorted(heap, key=lambda s: (-s[0], s[1]))
        return [(self.doc_ids[i], s) for s, i in ranked]
