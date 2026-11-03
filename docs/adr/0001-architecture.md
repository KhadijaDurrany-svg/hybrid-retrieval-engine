# ADR 0001: Two-stage hybrid retrieval with pluggable fusion and rerank

## Status
Accepted.

## Context
We need a search engine that is precise on exact-term queries and robust to
vocabulary mismatch, while staying small enough to read in one sitting and
reproducible without network access.

A pure lexical index (BM25) is strong on exact terms and cheap, but cannot
retrieve a relevant document that shares no surface words with the query. A pure
dense index handles paraphrase but is fuzzier on rare tokens and depends on an
embedding model. Neither alone is sufficient.

## Decision
Use a two-stage pipeline:

1. **Candidate generation** from two independent indexes — a hand-written BM25
   index with WAND pruning, and a dense cosine index — each returning the top
   `candidate_k`.
2. **Fusion** of the two ranked lists into one.
3. **Optional rerank** of the fused shortlist with a more expensive, query-aware
   scorer.

Every stage sits behind a small interface (`Embedder`, `Reranker`, fusion
functions), so each can be replaced without touching callers. The engine reports
per-stage latency on every query.

## Consequences
- The same engine serves lexical-only, dense-only and hybrid modes, which is
  what makes the benchmark an apples-to-apples comparison.
- Correctness of the hot path (WAND pruning) is pinned by property tests against
  a brute-force oracle rather than trusted by inspection.
- The cost is a slightly larger surface than a single-index search; the
  interfaces keep that surface honest.
