"""Bundled evaluation corpus: generation and loading.

The corpus is synthetic but hand-grounded. Each topic ships with a set of
primary terms and a set of alternate (synonym / paraphrase) terms, and every
document mixes both so the dense model can learn the association from
co-occurrence. Two query families are generated per topic:

* a *keyword* query built from the primary terms (favours exact-match BM25);
* a *paraphrase* query built from the alternate terms (favours the dense
  semantic signal).

Relevance judgements (qrels) are derived from topic membership: documents of a
query's topic are relevant (grade 1), and the document the query was seeded
from is highly relevant (grade 2). This gives a reproducible IR benchmark with
a genuine before/after, without downloading an external dataset.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .schemas import Document

# (topic_id, title, primary terms, alternate terms, filler sentence fragments)
TOPICS: list[tuple[str, str, list[str], list[str], list[str]]] = [
    ("vecsearch", "Vector search indexing",
     ["vector", "embedding", "nearest", "neighbour", "ann", "index"],
     ["semantic", "dense", "similarity", "retrieval", "proximity"],
     ["graphs connect close points", "recall trades against latency", "quantization shrinks memory"]),
    ("bm25", "Lexical ranking with BM25",
     ["bm25", "inverted", "term", "frequency", "tokeniser", "posting"],
     ["keyword", "lexical", "exact", "match", "sparse"],
     ["document length normalisation matters", "idf weights rare words", "pruning skips cheap documents"]),
    ("rag", "Retrieval augmented generation",
     ["retrieval", "augmented", "generation", "grounding", "context", "prompt"],
     ["cited", "grounded", "knowledge", "assistant", "answer"],
     ["the model reads retrieved passages", "citations let a reader verify", "it abstains when unsure"]),
    ("eval", "Ranking evaluation metrics",
     ["ndcg", "recall", "precision", "mrr", "relevance", "judgement"],
     ["measure", "benchmark", "quality", "offline", "metric"],
     ["gains are discounted by rank", "a held out query set is fixed", "bootstrap gives error bars"]),
    ("rerank", "Cross encoder reranking",
     ["rerank", "crossencoder", "candidate", "reorder", "pairwise"],
     ["refine", "rescoring", "second", "stage", "precision"],
     ["a slower model scores few pairs", "it runs over the shortlist only", "latency budgets bound the depth"]),
    ("chunking", "Document chunking strategies",
     ["chunk", "segment", "overlap", "window", "passage", "split"],
     ["section", "piece", "boundary", "granularity", "paragraph"],
     ["overlap preserves context across cuts", "headings make natural boundaries", "too small loses meaning"]),
    ("hybrid", "Hybrid fusion of rankers",
     ["hybrid", "fusion", "reciprocal", "combine", "blend", "weighted"],
     ["merge", "ensemble", "mix", "union", "rank"],
     ["rank based fusion needs no scaling", "normalisation aligns score ranges", "both signals cover each other"]),
    ("latency", "Serving latency and caching",
     ["latency", "throughput", "cache", "batch", "percentile", "tail"],
     ["speed", "fast", "responsive", "queue", "concurrency"],
     ["warm caches cut repeated work", "batching amortises overhead", "the tail dominates user experience"]),
    ("embedding_train", "Training text embeddings",
     ["contrastive", "triplet", "negatives", "encoder", "pooling", "finetune"],
     ["training", "learn", "representation", "objective", "pairs"],
     ["hard negatives sharpen the model", "mean pooling summarises tokens", "temperature scales the loss"]),
    ("safety", "Guardrails and safety checks",
     ["guardrail", "injection", "pii", "redaction", "filter", "policy"],
     ["safety", "protect", "block", "sanitise", "check"],
     ["prompt injection hijacks instructions", "pii is masked before logging", "outputs are validated by rules"]),
    ("sql_agent", "Text to SQL agents",
     ["sql", "schema", "query", "database", "join", "column"],
     ["table", "relational", "select", "structured", "data"],
     ["generated sql is validated first", "read only access limits blast radius", "the plan is explained to a user"]),
    ("forecasting", "Demand forecasting",
     ["forecast", "horizon", "seasonality", "quantile", "demand", "series"],
     ["predict", "trend", "future", "projection", "timeseries"],
     ["probabilistic bands beat point guesses", "leakage ruins a backtest", "holidays shift the baseline"]),
    ("fraud", "Fraud detection models",
     ["fraud", "anomaly", "imbalanced", "score", "ring", "transaction"],
     ["risk", "suspicious", "detection", "abuse", "alert"],
     ["rare positives need careful sampling", "graph features expose rings", "cost sensitive thresholds matter"]),
    ("ocr", "Document OCR and extraction",
     ["ocr", "extract", "field", "invoice", "form", "confidence"],
     ["scan", "parse", "capture", "digitise", "reading"],
     ["low confidence fields go to review", "layout helps locate values", "tables are harder than prose"]),
    ("routing", "LLM provider routing",
     ["router", "provider", "failover", "budget", "quota", "gateway"],
     ["switch", "fallback", "spend", "limit", "route"],
     ["circuit breakers ride out outages", "per tenant budgets cap spend", "one api fronts many vendors"]),
    ("memory", "Agent context memory",
     ["memory", "context", "eviction", "summary", "recall", "window"],
     ["remember", "history", "archive", "retention", "buffer"],
     ["old turns are summarised not dropped", "important facts are pinned", "search brings archived notes back"]),
]


@dataclass
class Query:
    id: str
    text: str
    family: str  # "keyword" or "paraphrase"
    topic: str
    seed_doc: str


def _make_doc_text(rng: random.Random, primary: list[str], alt: list[str], fillers: list[str]) -> str:
    # Sparse documents on purpose: each note carries only a few of its topic's
    # terms. That creates genuine vocabulary mismatch between a query and a
    # relevant document, which is exactly where a dense signal can help a pure
    # keyword index that would otherwise miss the document.
    n_p = rng.randint(1, 2)
    n_a = rng.randint(1, 2)
    p = rng.sample(primary, k=min(n_p, len(primary)))
    a = rng.sample(alt, k=min(n_a, len(alt)))
    f = rng.sample(fillers, k=1)
    words = p + a
    rng.shuffle(words)
    body = " ".join(words)
    return f"This note covers {body}. {' '.join(f)}."


def generate(docs_per_topic: int = 6, seed: int = 7) -> tuple[list[Document], list[Query]]:
    rng = random.Random(seed)
    docs: list[Document] = []
    queries: list[Query] = []
    for topic_id, title, primary, alt, fillers in TOPICS:
        topic_docs: list[str] = []
        for j in range(docs_per_topic):
            did = f"{topic_id}-{j}"
            text = _make_doc_text(rng, primary, alt, fillers)
            docs.append(Document(id=did, text=text, title=title, meta={"topic": topic_id}))
            topic_docs.append(did)
        seed_doc = topic_docs[0]
        kw = " ".join(rng.sample(primary, k=min(2, len(primary))))
        pp = " ".join(rng.sample(alt, k=min(2, len(alt))))
        queries.append(Query(f"{topic_id}-kw", kw, "keyword", topic_id, seed_doc))
        queries.append(Query(f"{topic_id}-pp", pp, "paraphrase", topic_id, seed_doc))
    return docs, queries


def qrels_for(queries: list[Query], docs: list[Document]) -> dict[str, dict[str, int]]:
    by_topic: dict[str, list[str]] = {}
    for d in docs:
        by_topic.setdefault(d.meta.get("topic", ""), []).append(d.id)
    qrels: dict[str, dict[str, int]] = {}
    for q in queries:
        rel = {doc_id: 1 for doc_id in by_topic.get(q.topic, [])}
        rel[q.seed_doc] = 2
        qrels[q.id] = rel
    return qrels


# ----------------------------------------------------------------- IO helpers
def write_corpus(data_dir: str | Path, docs_per_topic: int = 8, seed: int = 7) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    docs, queries = generate(docs_per_topic=docs_per_topic, seed=seed)
    qrels = qrels_for(queries, docs)
    with (data_dir / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({"id": d.id, "title": d.title, "text": d.text, "meta": d.meta}) + "\n")
    with (data_dir / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps({"id": q.id, "text": q.text, "family": q.family,
                                 "topic": q.topic, "seed_doc": q.seed_doc}) + "\n")
    with (data_dir / "qrels.jsonl").open("w", encoding="utf-8") as fh:
        for qid, rel in qrels.items():
            fh.write(json.dumps({"query_id": qid, "relevant": rel}) + "\n")


def load_corpus(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                docs.append(Document(id=r["id"], text=r["text"], title=r.get("title", ""), meta=r.get("meta", {})))
    return docs


def load_queries(path: str | Path) -> list[Query]:
    out: list[Query] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out.append(Query(r["id"], r["text"], r["family"], r["topic"], r["seed_doc"]))
    return out


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["query_id"]] = {k: int(v) for k, v in r["relevant"].items()}
    return out
