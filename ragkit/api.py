"""REST API for the RAG pipeline (FastAPI). Run: uvicorn ragkit.api:app --port 8000"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chunking import chunk_by_section
from .embedders import CachedEmbedder, LSAEmbedder
from .loader import load_manuals
from .pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="RAG pipeline", version="1.0.0")
_pipeline: RAGPipeline | None = None


def pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        chunks = chunk_by_section(load_manuals(os.environ.get("RAG_DOCS", ROOT / "data" / "manuals")))
        _pipeline = RAGPipeline(chunks, CachedEmbedder(LSAEmbedder()), index_kind=os.environ.get("RAG_INDEX", "flat")).build()
    return _pipeline


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health():
    p = pipeline()
    return {"status": "ok", "chunks": len(p.chunks), "index": p.index_kind, "embedder": p.embedder.name}


@app.post("/search")
def search(body: Question):
    hits = pipeline().retrieve(body.question, body.k)
    return {"results": [{"citation": h.chunk.citation, "score": round(h.score, 4), "text": h.chunk.body} for h in hits]}


@app.post("/ask")
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question is empty")
    a = pipeline().ask(body.question, body.k)
    return {"question": a.question, "answer": a.answer, "citations": a.citations,
            "grounded": a.grounded, "cached": a.cached, "latency_ms": round(a.latency_ms, 3)}
