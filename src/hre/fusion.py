"""Rank fusion strategies for combining lexical and dense result lists.

Two strategies are provided:

* Reciprocal Rank Fusion (RRF) - rank-based, scale-free and robust. It needs no
  score normalization, which is why it is the default: BM25 scores and cosine
  similarities live on completely different scales.
* Weighted score fusion - min-max normalizes each list then takes a convex
  combination. Useful when absolute scores carry signal worth keeping.
"""

from __future__ import annotations

from collections.abc import Sequence

Ranked = Sequence[tuple[str, float]]


def reciprocal_rank_fusion(lists: Sequence[Ranked], k: int = 10, c: float = 60.0) -> list[tuple[str, float]]:
    """Combine ranked lists with RRF: score(d) = sum 1 / (c + rank_i(d)).

    `c` damps the influence of very high ranks; 60 is the value from the
    original Cormack et al. paper and a sane default.
    """
    if c <= 0:
        raise ValueError("c must be positive")
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (c + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:k] if k > 0 else fused


def _min_max(ranked: Ranked) -> dict[str, float]:
    if not ranked:
        return {}
    vals = [s for _, s in ranked]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return {d: 1.0 for d, _ in ranked}
    return {d: (s - lo) / span for d, s in ranked}


def weighted_fusion(
    lexical: Ranked,
    dense: Ranked,
    k: int = 10,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """Convex combination of min-max normalized scores.

    `alpha` is the lexical weight; `1 - alpha` the dense weight. A document
    missing from one list contributes 0 from that side.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    lex = _min_max(lexical)
    den = _min_max(dense)
    docs = set(lex) | set(den)
    scores = {d: alpha * lex.get(d, 0.0) + (1.0 - alpha) * den.get(d, 0.0) for d in docs}
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:k] if k > 0 else fused
