"""Benchmark runner shared by the CLI and the standalone bench script.

Builds the engine once, runs every configuration over the held-out query set,
and reports nDCG@10, Recall@10, Recall@50 and MRR with bootstrap confidence
intervals, broken down by query family so wins and losses are both visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .analysis import Analyzer
from .corpus import load_corpus, load_qrels, load_queries
from .engine import HybridEngine
from .metrics import bootstrap_ci, ndcg_at_k, recall_at_k, reciprocal_rank
from .rerank import LexicalOverlapReranker

# name -> (mode, fusion, rerank)
CONFIGS: dict[str, tuple[str, str, bool]] = {
    "lexical_bm25": ("lexical", "rrf", False),
    "dense_lsa": ("dense", "rrf", False),
    "hybrid_rrf": ("hybrid", "rrf", False),
    "hybrid_weighted": ("hybrid", "weighted", False),
    "hybrid_rrf_rerank": ("hybrid", "rrf", True),
}


@dataclass
class ConfigReport:
    name: str
    metrics: dict[str, dict[str, float]]
    by_family: dict[str, dict[str, float]]
    mean_latency_ms: float


def _aggregate(scores: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric, vals in scores.items():
        mean, lo, hi = bootstrap_ci(vals)
        out[metric] = {"mean": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return out


def run_benchmark(data_dir: str | Path, k: int = 10) -> dict:
    data_dir = Path(data_dir)
    docs = load_corpus(data_dir / "corpus.jsonl")
    queries = load_queries(data_dir / "queries.jsonl")
    qrels = load_qrels(data_dir / "qrels.jsonl")

    analyzer = Analyzer()
    engine = HybridEngine(analyzer=analyzer, reranker=LexicalOverlapReranker(analyzer))
    t_index = time.perf_counter()
    engine.index(docs)
    index_ms = (time.perf_counter() - t_index) * 1000.0

    reports: list[ConfigReport] = []
    for name, (mode, fusion, rerank) in CONFIGS.items():
        engine.fusion = fusion  # type: ignore[assignment]
        overall: dict[str, list[float]] = {"ndcg@10": [], "recall@10": [], "recall@50": [], "mrr": []}
        per_family: dict[str, dict[str, list[float]]] = {}
        latencies: list[float] = []
        for q in queries:
            res = engine.search(q.text, k=max(k, 50), mode=mode, rerank=rerank)  # type: ignore[arg-type]
            ranked = [h.doc_id for h in res.hits]
            rel = qrels[q.id]
            m = {
                "ndcg@10": ndcg_at_k(ranked, rel, 10),
                "recall@10": recall_at_k(ranked, rel, 10),
                "recall@50": recall_at_k(ranked, rel, 50),
                "mrr": reciprocal_rank(ranked, rel),
            }
            for key, val in m.items():
                overall[key].append(val)
                per_family.setdefault(q.family, {}).setdefault(key, []).append(val)
            latencies.append(res.timings.total_ms)
        by_family = {
            fam: {key: round(sum(v) / len(v), 4) for key, v in metrics.items()}
            for fam, metrics in per_family.items()
        }
        reports.append(
            ConfigReport(
                name=name,
                metrics=_aggregate(overall),
                by_family=by_family,
                mean_latency_ms=round(sum(latencies) / len(latencies), 3),
            )
        )

    return {
        "corpus": {"n_docs": len(docs), "n_queries": len(queries), "index_ms": round(index_ms, 1)},
        "k": k,
        "configs": [r.__dict__ for r in reports],
    }


def format_table(result: dict) -> str:
    lines: list[str] = []
    c = result["corpus"]
    lines.append(f"corpus: {c['n_docs']} docs, {c['n_queries']} queries, built in {c['index_ms']} ms")
    header = f"{'config':<20} {'nDCG@10':>18} {'Recall@10':>11} {'Recall@50':>11} {'MRR':>8} {'ms':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in result["configs"]:
        m = r["metrics"]
        nd = m["ndcg@10"]
        ndcg_str = f"{nd['mean']:.3f} [{nd['ci_lo']:.3f},{nd['ci_hi']:.3f}]"
        lines.append(
            f"{r['name']:<20} {ndcg_str:>18} "
            f"{m['recall@10']['mean']:>11.3f} {m['recall@50']['mean']:>11.3f} "
            f"{m['mrr']['mean']:>8.3f} {r['mean_latency_ms']:>7.2f}"
        )
    lines.append("")
    lines.append("by query family (nDCG@10):")
    for r in result["configs"]:
        fam = r["by_family"]
        parts = ", ".join(f"{k}={v.get('ndcg@10', 0):.3f}" for k, v in sorted(fam.items()))
        lines.append(f"  {r['name']:<20} {parts}")
    return "\n".join(lines)
