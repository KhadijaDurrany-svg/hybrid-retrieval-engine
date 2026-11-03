"""Command line interface: generate data, query the engine, run the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import Analyzer
from .corpus import load_corpus, write_corpus
from .engine import HybridEngine
from .evaluate import format_table, run_benchmark
from .rerank import LexicalOverlapReranker


def _build_engine(data_dir: Path) -> HybridEngine:
    docs = load_corpus(data_dir / "corpus.jsonl")
    analyzer = Analyzer()
    engine = HybridEngine(analyzer=analyzer, reranker=LexicalOverlapReranker(analyzer))
    engine.index(docs)
    return engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hre", description="Hybrid Retrieval Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data", help="write the bundled corpus/queries/qrels")
    g.add_argument("--data-dir", default="data")
    g.add_argument("--docs-per-topic", type=int, default=8)
    g.add_argument("--seed", type=int, default=7)

    q = sub.add_parser("query", help="run a single query against the bundled corpus")
    q.add_argument("text")
    q.add_argument("--data-dir", default="data")
    q.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="hybrid")
    q.add_argument("--k", type=int, default=10)
    q.add_argument("--rerank", action="store_true")

    b = sub.add_parser("bench", help="evaluate every configuration and report metrics")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--out", default="")

    args = parser.parse_args(argv)

    if args.cmd == "generate-data":
        write_corpus(args.data_dir, docs_per_topic=args.docs_per_topic, seed=args.seed)
        print(f"wrote corpus, queries and qrels to {args.data_dir}/")
        return 0

    if args.cmd == "query":
        engine = _build_engine(Path(args.data_dir))
        res = engine.search(args.text, k=args.k, mode=args.mode, rerank=args.rerank)
        print(json.dumps(res.as_dict(), indent=2))
        return 0

    if args.cmd == "bench":
        result = run_benchmark(args.data_dir, k=args.k)
        print(format_table(result))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
