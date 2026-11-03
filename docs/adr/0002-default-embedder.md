# ADR 0002: LSA as the default embedder, RRF as the default fusion

## Status
Accepted.

## Context
The dense index needs vectors, and the fused result needs a way to combine two
score distributions that live on different scales (BM25 is unbounded and
non-negative; cosine similarity is in [-1, 1]).

Two choices had to be made with reproducibility and honesty in mind: which
embedder ships as the default, and which fusion rule.

## Decision

**Default embedder: TF-IDF + truncated SVD (Latent Semantic Analysis).**
It is deterministic, needs no model download, and runs anywhere the package
installs. It gives the dense index a genuine term-correlation signal learned
from the corpus itself. A transformer backend (`OnnxEmbedder`) implements the
same interface for callers who want neural embeddings and can supply a model.

**Default fusion: Reciprocal Rank Fusion (RRF).**
RRF combines by rank, not score, so it needs no normalization between the BM25
and cosine scales and has one stable hyper-parameter (`c`, default 60 from the
original paper). Weighted score fusion is available for callers who want to keep
absolute scores, but it requires per-list min-max normalization and an `alpha`
to tune.

## Consequences
- The project is fully offline and the benchmark is reproducible on any machine.
- The honest limitation, stated in the README: LSA only relates terms it has
  seen co-occur, so it does not bridge truly out-of-vocabulary synonyms. That is
  the reason the paraphrase-family gains are modest; a neural embedder is the
  upgrade path, not a change to the architecture.
- Choosing RRF as the default means the headline numbers do not depend on a
  fragile score-normalization step.
