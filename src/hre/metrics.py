"""Information-retrieval metrics and a bootstrap confidence interval.

Metrics take a ranked list of doc ids and a relevance map {doc_id: grade}.
Grades are non-negative integers (0 = not relevant). All functions are pure.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    gains = [float(relevant.get(doc_id, 0)) for doc_id in ranked[:k]]
    ideal = sorted((float(g) for g in relevant.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    rel_docs = {d for d, g in relevant.items() if g > 0}
    if not rel_docs:
        return 0.0
    hit = sum(1 for d in ranked[:k] if d in rel_docs)
    return hit / len(rel_docs)


def reciprocal_rank(ranked: Sequence[str], relevant: dict[str, int]) -> float:
    rel_docs = {d for d, g in relevant.items() if g > 0}
    for i, d in enumerate(ranked):
        if d in rel_docs:
            return 1.0 / (i + 1)
    return 0.0


def bootstrap_ci(
    per_query: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) with a percentile bootstrap over per-query scores."""
    vals = list(per_query)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int((1.0 - confidence) / 2.0 * n_resamples)
    hi_idx = int((1.0 + confidence) / 2.0 * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return (mean, means[lo_idx], means[hi_idx])
