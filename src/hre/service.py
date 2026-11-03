"""FastAPI service exposing the hybrid engine over HTTP.

Endpoints:
  GET  /healthz           liveness and index size
  POST /index             (re)build the index from a list of documents
  POST /search            run a query, returns hits and per-stage timings

On startup the bundled corpus is loaded if present, so the service answers
queries immediately. Build the app with `create_app()`; run with uvicorn:

    uvicorn hre.service:app --reload
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .analysis import Analyzer
from .corpus import load_corpus
from .engine import HybridEngine
from .rerank import LexicalOverlapReranker
from .schemas import Document


class DocIn(BaseModel):
    id: str
    text: str
    title: str = ""


class IndexRequest(BaseModel):
    documents: list[DocIn] = Field(..., min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=100)
    mode: str = Field("hybrid", pattern="^(lexical|dense|hybrid)$")
    rerank: bool = False


def create_app(data_dir: str | None = None) -> FastAPI:
    analyzer = Analyzer()
    state: dict[str, HybridEngine | None] = {"engine": None}

    def _new_engine() -> HybridEngine:
        return HybridEngine(analyzer=analyzer, reranker=LexicalOverlapReranker(analyzer))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        path = Path(data_dir or os.environ.get("HRE_DATA_DIR", "data")) / "corpus.jsonl"
        if path.exists():
            engine = _new_engine()
            engine.index(load_corpus(path))
            state["engine"] = engine
        yield

    app = FastAPI(title="Hybrid Retrieval Engine", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        engine = state["engine"]
        return {"status": "ok", "indexed": engine is not None, "n_docs": engine.n_docs if engine else 0}

    @app.post("/index")
    def index(req: IndexRequest) -> dict:
        engine = _new_engine()
        try:
            engine.index([Document(id=d.id, text=d.text, title=d.title) for d in req.documents])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state["engine"] = engine
        return {"status": "indexed", "n_docs": engine.n_docs}

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        engine = state["engine"]
        if engine is None:
            raise HTTPException(status_code=409, detail="no index; POST /index or provide a bundled corpus first")
        res = engine.search(req.query, k=req.k, mode=req.mode, rerank=req.rerank)  # type: ignore[arg-type]
        return res.as_dict()

    return app


app = create_app()
