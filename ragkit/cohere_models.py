"""Cohere Embed and Rerank for the RAG pipeline, called through Cohere's v2 REST API.

Needs COHERE_API_KEY (a free trial key works: https://dashboard.cohere.com/api-keys). Never commit the key.
  - CohereEmbedder: documents are embedded with input_type="search_document", questions with "search_query",
    which is how Cohere's embed models are meant to be used for search.
  - CohereReranker: re-scores the top retrieved chunks with a rerank model (a cross-encoder that reads the question
    and each chunk together), so the best chunk comes first.
"""
from __future__ import annotations

import os
import time

import httpx
import numpy as np

API_URL = "https://api.cohere.com/v2"
EMBED_MODEL = os.environ.get("COHERE_EMBED_MODEL", "embed-v4.0")
RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")
MAX_TEXTS_PER_CALL = 96


class CohereError(RuntimeError):
    pass


class CohereClient:
    def __init__(self, api_key: str | None = None, base_url: str = API_URL, transport: httpx.BaseTransport | None = None,
                 max_retries: int = 3, backoff_s: float = 1.0):
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise CohereError("Set COHERE_API_KEY to use Cohere models (free trial key: dashboard.cohere.com/api-keys).")
        self.http = httpx.Client(base_url=base_url, transport=transport, timeout=30.0,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.calls = 0

    def _post(self, path: str, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            self.calls += 1
            r = self.http.post(path, json=body)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(self.backoff_s * 2**attempt)  # rate limited or temporary error: wait and retry
                continue
            if r.status_code == 401:
                raise CohereError("Cohere rejected the API key (401). Check COHERE_API_KEY.")
            raise CohereError(f"Cohere {path} failed with HTTP {r.status_code}: {r.text[:200]}")
        raise CohereError(f"Cohere {path} still failing after {self.max_retries} retries")

    def embed(self, texts: list[str], input_type: str, model: str = EMBED_MODEL) -> np.ndarray:
        out = []
        for i in range(0, len(texts), MAX_TEXTS_PER_CALL):
            body = {"model": model, "texts": texts[i:i + MAX_TEXTS_PER_CALL], "input_type": input_type, "embedding_types": ["float"]}
            out.extend(self._post("/embed", body)["embeddings"]["float"])
        return np.asarray(out, dtype=np.float32)

    def rerank(self, query: str, documents: list[str], top_n: int, model: str = RERANK_MODEL) -> list[tuple[int, float]]:
        data = self._post("/rerank", {"model": model, "query": query, "documents": documents, "top_n": top_n})
        return [(r["index"], float(r["relevance_score"])) for r in data["results"]]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class CohereEmbedder:
    name = "cohere"

    def __init__(self, client: CohereClient | None = None, model: str = EMBED_MODEL):
        self.client = client or CohereClient()
        self.model = model
        self.fitted = True

    def fit(self, texts: list[str]) -> CohereEmbedder:
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _normalise(self.client.embed(texts, "search_document", self.model))

    def encode_query(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.client.embed(texts, "search_query", self.model))


class CohereReranker:
    name = "cohere-rerank"

    def __init__(self, client: CohereClient | None = None, model: str = RERANK_MODEL):
        self.client = client or CohereClient()
        self.model = model

    def rerank(self, question: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        return self.client.rerank(question, texts, top_n, self.model)
