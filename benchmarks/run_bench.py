"""Run the full benchmark and write benchmarks/results.json.

Usage:
    python benchmarks/run_bench.py            # uses ./data, writes results.json
    python benchmarks/run_bench.py --data-dir data --out benchmarks/results.json

Every number in the README is produced by this script. Re-run it after any
change to the ranking code; the committed results.json is the source of truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hre.evaluate import format_table, run_benchmark


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="benchmarks/results.json")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    result = run_benchmark(args.data_dir, k=args.k)
    print(format_table(result))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
