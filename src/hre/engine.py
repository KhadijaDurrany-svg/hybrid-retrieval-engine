"""The hybrid engine: orchestrates lexical + dense retrieval, fusion and rerank.

Each `search` reports per-stage latency so a caller can see where time goes.
The retrieval `mode` selects which signals run, which is what the benchmark
uses to compare lexical-only, dense-only and hybrid on the same corpus.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from .analysis import Analyzer
from .bm25 import BM25Index, BM25Params
from .dense import DenseIndex
from .embedders import Embedder, HashingLSAEmbedder
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .rerank import Reranker
from .schemas import Document, Hit

Mode = Literal["lexical", "dense", "hybrid"]
FusionName = Literal["rrf", "weighted"]


@dataclass
class StageTimings:
    lexical_ms: float = 0.0
    dense_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass
class SearchResult:
    query: str
    mode: str
    hits: list[Hit] = field(default_factory=list)
    timings: StageTimings = field(default_factory=StageTimings)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "hits": [h.as_dict() for h in self.hits],
            "timings": self.timings.as_dict(),
        }


class HybridEngine:
    def __init__(
        self,
        analyzer: Analyzer | None = None,
        embedder: Embedder | None = None,
        bm25_params: BM25Params | None = None,
        fusion: FusionName = "rrf",
        rrf_c: float = 60.0,
        alpha: float = 0.5,
        candidate_k: int = 50,
        reranker: Reranker | None = None,
    ) -> None:
        self.analyzer = analyzer or Analyzer()
        self.embedder = embedder or HashingLSAEmbedder(analyzer=self.analyzer)
        self.bm25 = BM25Index(analyzer=self.analyzer, params=bm25_params)
        self.dense = DenseIndex(self.embedder)
        self.fusion = fusion
        self.rrf_c = rrf_c
        self.alpha = alpha
        self.candidate_k = candidate_k
        self.reranker = reranker
        self._texts: dict[str, str] = {}
        self._titles: dict[str, str] = {}
        self._built = False

    # ------------------------------------------------------------------ index
    def index(self, documents: list[Document]) -> HybridEngine:
        if not documents:
            raise ValueError("cannot index an empty document set")
        ids = [d.id for d in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("document ids must be unique")
        contents = [d.content() for d in documents]
        for d, c in zip(documents, contents, strict=True):
            self.bm25.add(d.id, c)
            self._texts[d.id] = d.text
            self._titles[d.id] = d.title
        self.bm25.build()
        if isinstance(self.embedder, HashingLSAEmbedder) and not getattr(self.embedder, "_fitted", False):
            self.embedder.fit(contents)
        self.dense.fit(ids, contents)
        self._built = True
        return self

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 10, mode: Mode = "hybrid", rerank: bool = False) -> SearchResult:
        if not self._built:
            raise RuntimeError("engine must index() documents before search()")
        t0 = time.perf_counter()
        timings = StageTimings()
        cand_k = max(k, self.candidate_k)

        lexical: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []

        if mode in ("lexical", "hybrid"):
            s = time.perf_counter()
            lexical = self.bm25.search(query, cand_k)
            timings.lexical_ms = (time.perf_counter() - s) * 1000.0

        if mode in ("dense", "hybrid"):
            s = time.perf_counter()
            dense = self.dense.search(query, cand_k)
            timings.dense_ms = (time.perf_counter() - s) * 1000.0

        s = time.perf_counter()
        if mode == "lexical":
            fused = lexical[:cand_k]
        elif mode == "dense":
            fused = dense[:cand_k]
        elif self.fusion == "weighted":
            fused = weighted_fusion(lexical, dense, k=cand_k, alpha=self.alpha)
        else:
            fused = reciprocal_rank_fusion([lexical, dense], k=cand_k, c=self.rrf_c)
        timings.fusion_ms = (time.perf_counter() - s) * 1000.0

        if rerank and self.reranker is not None and fused:
            s = time.perf_counter()
            candidates = [(doc_id, self._texts.get(doc_id, "")) for doc_id, _ in fused]
            fused = self.reranker.rerank(query, candidates)
            timings.rerank_ms = (time.perf_counter() - s) * 1000.0

        hits = [self._to_hit(doc_id, score, rank) for rank, (doc_id, score) in enumerate(fused[:k])]
        timings.total_ms = (time.perf_counter() - t0) * 1000.0
        return SearchResult(query=query, mode=mode, hits=hits, timings=timings)

    def _to_hit(self, doc_id: str, score: float, rank: int) -> Hit:
        text = self._texts.get(doc_id, "")
        snippet = (text[:157] + "...") if len(text) > 160 else text
        return Hit(doc_id=doc_id, score=score, rank=rank, title=self._titles.get(doc_id, ""), snippet=snippet)

    @property
    def n_docs(self) -> int:
        return self.bm25.n_docs
