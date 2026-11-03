"""Candidate rerankers.

A reranker takes the fused top-N candidates (with their text) and reorders them
with a more expensive, query-aware signal. The interface is deliberately small
so a real cross-encoder can be swapped in for the lightweight default.

`LexicalOverlapReranker` is a dependency-free default that rewards exact term
coverage and proximity - cheap, deterministic, and genuinely useful as a
tie-breaker over the fused list. `IdentityReranker` leaves order untouched.
"""

from __future__ import annotations

from typing import Protocol

from .analysis import Analyzer

Candidate = tuple[str, str]  # (doc_id, text)


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Candidate]) -> list[tuple[str, float]]:
        ...


class IdentityReranker:
    def rerank(self, query: str, candidates: list[Candidate]) -> list[tuple[str, float]]:
        n = len(candidates)
        return [(doc_id, float(n - i)) for i, (doc_id, _text) in enumerate(candidates)]


class LexicalOverlapReranker:
    """Score = coverage of query terms + a small contiguous-span bonus."""

    def __init__(self, analyzer: Analyzer | None = None, span_bonus: float = 0.25) -> None:
        self.analyzer = analyzer or Analyzer()
        self.span_bonus = span_bonus

    def rerank(self, query: str, candidates: list[Candidate]) -> list[tuple[str, float]]:
        q_terms = self.analyzer.analyze(query)
        q_set = set(q_terms)
        if not q_set:
            return [(doc_id, 0.0) for doc_id, _ in candidates]
        scored: list[tuple[str, float]] = []
        for doc_id, text in candidates:
            d_terms = self.analyzer.analyze(text)
            d_set = set(d_terms)
            coverage = len(q_set & d_set) / len(q_set)
            span = self._best_span_bonus(q_terms, d_terms)
            scored.append((doc_id, coverage + self.span_bonus * span))
        # Stable sort keeps the fused order among ties.
        order = sorted(range(len(scored)), key=lambda i: -scored[i][1])
        return [scored[i] for i in order]

    def _best_span_bonus(self, q_terms: list[str], d_terms: list[str]) -> float:
        """Fraction of the query found as a contiguous run in the document."""
        if not q_terms or not d_terms:
            return 0.0
        q_set = set(q_terms)
        best = 0
        run = 0
        for t in d_terms:
            if t in q_set:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return best / len(q_terms)
