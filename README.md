# Hybrid Retrieval Engine

A compact, production-minded search engine that combines a hand-written **BM25**
lexical index with a **dense semantic** index, fuses the two ranked lists, and
optionally reranks the top candidates. It ships with a reproducible benchmark so
every number below can be regenerated from the committed data with one command.

The point of the project is not a new algorithm. It is an honest, testable
reference implementation: the pruning is checked against a brute-force oracle,
the metrics come with confidence intervals, and the evaluation reports where the
design **does not** help as plainly as where it does.

```
query ─┬─▶ BM25 (WAND top-k) ─┐
       └─▶ Dense (LSA/ANN) ───┴─▶ fusion (RRF | weighted) ─▶ rerank ─▶ hits + per-stage timings
```

## Why hybrid

Lexical search is precise on exact terms but blind to vocabulary mismatch: if a
relevant document does not contain the query's words, BM25 cannot find it. A
dense index ranks by meaning and recovers those documents, at the cost of being
fuzzier on rare exact tokens. Fusing the two keeps the precision of one and the
recall of the other. The benchmark below shows exactly that effect.

## What is in the box

- **`hre.bm25`** — an inverted index built from scratch with BM25 scoring and
  **WAND** top-k pruning (Broder et al., 2003). A property test asserts the
  pruned top-k is identical to an exhaustive scan.
- **`hre.embedders`** — a default **offline** embedder (TF-IDF → truncated SVD,
  i.e. Latent Semantic Analysis), deterministic and dependency-light. A neural
  `OnnxEmbedder` implements the same interface for users who want it.
- **`hre.dense`** — a cosine-similarity dense index (exact NumPy scan; an ANN
  backend drops in behind the same `search` signature).
- **`hre.fusion`** — Reciprocal Rank Fusion (default) and weighted score fusion.
- **`hre.rerank`** — a dependency-free lexical-overlap reranker and an identity
  baseline, behind a `Reranker` protocol a cross-encoder can replace.
- **`hre.engine`** — orchestrates the stages and reports per-stage latency.
- **`hre.service`** — a FastAPI app (`/search`, `/index`, `/healthz`).
- **`hre.evaluate` / `benchmarks/`** — the IR metrics and the benchmark runner.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10+.

## Quickstart

```bash
hre generate-data --data-dir data          # write the bundled corpus
hre query "semantic similarity retrieval" --mode hybrid --rerank
hre bench --data-dir data                   # reproduce the numbers below
uvicorn hre.service:app --reload            # serve it; GET /healthz, POST /search
```

Library use:

```python
from hre import HybridEngine, Document

engine = HybridEngine().index([
    Document(id="1", text="retrieval augmented generation grounds answers in sources", title="RAG"),
    Document(id="2", text="bm25 ranks documents by term frequency and inverse document frequency"),
])
result = engine.search("grounded question answering", k=5, mode="hybrid")
for hit in result.hits:
    print(hit.rank, hit.doc_id, round(hit.score, 4))
print(result.timings.as_dict())
```

## Benchmark

A synthetic, hand-grounded corpus of **128 documents across 16 topics** with
**32 queries** in two families: *keyword* queries (exact topic terms) and
*paraphrase* queries (the topic's alternate vocabulary). Documents are sparse on
purpose, so a relevant document often lacks the query's exact words — the
vocabulary-mismatch case hybrid search is meant to fix. Relevance is graded by
topic membership (seed document = 2, other topic documents = 1). Confidence
intervals are a percentile bootstrap over per-query scores.

Reproduce with `hre bench --data-dir data` (deterministic). Source of truth:
[`benchmarks/results.json`](benchmarks/results.json).

| Config | nDCG@10 (95% CI) | Recall@10 | Recall@50 | MRR | ms/query |
|---|---|---|---|---|---|
| lexical (BM25) | 0.673 [0.599, 0.744] | 0.652 | 0.664 | 0.969 | 0.04 |
| dense (LSA) | 0.681 [0.607, 0.750] | 0.664 | **0.809** | 0.970 | 0.76 |
| hybrid (RRF) | **0.682** [0.609, 0.752] | 0.664 | **0.809** | 0.970 | 0.78 |
| hybrid (weighted) | 0.677 [0.604, 0.747] | 0.664 | 0.809 | 0.970 | 0.98 |
| hybrid (RRF + rerank) | 0.682 [0.609, 0.752] | 0.664 | 0.809 | 0.970 | 1.49 |

### How to read this honestly

- **The real win is deep recall.** Adding the dense signal lifts **Recall@50
  from 0.664 to 0.809** — about 15 points. Those are relevant documents the pure
  keyword index never retrieves because they share no surface terms with the
  query. This is the headline result and it is unambiguous.
- **nDCG@10 gains are within the noise.** Hybrid's 0.682 vs lexical's 0.673 sits
  well inside the overlapping confidence intervals, so on this corpus hybrid does
  not reliably beat BM25 for top-10 *ordering* — it mainly finds more relevant
  documents deeper down. Reported as a tie, not a win.
- **Reranking did not move nDCG@10 here.** The lexical-overlap reranker is a
  cheap tie-breaker; on this data it left the top-10 metric unchanged. A real
  cross-encoder (via the same interface) is where to look for ordering gains.
- **Latency** is sub-millisecond for lexical and ~0.8 ms for hybrid at this
  corpus size; the dense stage and rerank are the cost, as the per-stage timings
  in every response show.

The default embedder is LSA, not a transformer. LSA can only relate terms it has
seen co-occur, so it will not bridge truly out-of-vocabulary synonyms — swap in
`OnnxEmbedder` for that. The benchmark reflects the honest default, not a
best-case model.

## Testing

```bash
pytest -q          # 33 tests
ruff check .
```

The suite includes property-based tests (Hypothesis): WAND pruning is checked
against a brute-force oracle on randomly generated indexes, and every returned
score is verified to be a valid member of the true top-k.

## Design notes

Architecture decisions are recorded under [`docs/adr/`](docs/adr/). Short
version: RRF is the default fusion because it needs no score normalization
across the very different BM25 and cosine scales; LSA is the default embedder
because it keeps the project fully offline and reproducible.

## License

MIT — see [LICENSE](LICENSE).
