# Benchmark methodology

## Corpus
Synthetic but hand-grounded, generated deterministically by `hre.corpus`:

- 16 topics, each with a set of **primary** terms and a set of **alternate**
  (paraphrase) terms.
- 8 documents per topic (128 total). Each document is **sparse**: it carries
  only one or two primary terms and one or two alternate terms, plus a filler
  sentence. Sparsity is deliberate — it manufactures vocabulary mismatch between
  queries and relevant documents, which is the case hybrid retrieval targets.

## Queries
Two per topic (32 total):

- **keyword**: built from the topic's primary terms (favours exact-match BM25);
- **paraphrase**: built from the topic's alternate terms (favours the dense
  signal, when the alternate terms co-occur with primary terms elsewhere).

## Relevance (qrels)
Graded by topic membership: the seed document the query was derived from is
highly relevant (grade 2), every other document of the same topic is relevant
(grade 1), all others are non-relevant.

## Metrics
- **nDCG@10** — graded ranking quality in the top 10.
- **Recall@10 / Recall@50** — fraction of relevant documents retrieved.
- **MRR** — reciprocal rank of the first relevant document.
- **Confidence intervals** — percentile bootstrap (2000 resamples) over the
  per-query scores.

## Configurations
`lexical_bm25`, `dense_lsa`, `hybrid_rrf`, `hybrid_weighted`,
`hybrid_rrf_rerank`. All share one indexed engine; only the retrieval mode,
fusion rule and rerank flag change, so the comparison is apples-to-apples.

## Reproducing
```bash
hre generate-data --data-dir data   # deterministic (seed=7)
hre bench --data-dir data           # prints the table, writes results.json
```

## Known limitations
- The corpus is synthetic. It is designed to isolate the lexical-vs-dense
  trade-off, not to stand in for a production distribution. Swap in your own
  `corpus.jsonl` / `queries.jsonl` / `qrels.jsonl` to evaluate on real data.
- The default embedder is LSA, which cannot relate terms it never saw co-occur.
  Out-of-vocabulary paraphrase is where a neural embedder would widen the gap.
