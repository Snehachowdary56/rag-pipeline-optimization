"""Cohere Embed and Rerank, tested against a fake Cohere v2 API (no key or network needed)."""
import hashlib
import json
import re

import httpx
import numpy as np
import pytest

from ragkit.chunking import chunk_by_section
from ragkit.cohere_eval import ROOT, run
from ragkit.cohere_models import CohereClient, CohereEmbedder, CohereError, CohereReranker
from ragkit.embedders import CachedEmbedder, LSAEmbedder
from ragkit.loader import load_manuals
from ragkit.pipeline import NOT_FOUND, RAGPipeline

WORD = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "and", "the", "of", "to", "in", "is", "it", "on", "for", "does", "do", "which", "what", "who", "how", "use"}


def words(text):
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


class FakeCohere:
    """Mimics /v2/embed and /v2/rerank. Embeddings are bag-of-words hashes; rerank scores word overlap."""

    def __init__(self, fail_first=0, status=429):
        self.requests = []
        self.fail_first = fail_first
        self.status = status

    def vector(self, text, dim=256):
        v = np.zeros(dim)
        for w in WORD.findall(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
        return v.tolist()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, dict(request.headers), body))
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(self.status, json={"message": "slow down"})
        if request.url.path.endswith("/embed"):
            assert body["input_type"] in ("search_document", "search_query")
            assert body["embedding_types"] == ["float"] and len(body["texts"]) <= 96
            return httpx.Response(200, json={"embeddings": {"float": [self.vector(t) for t in body["texts"]]}})
        q = words(body["query"])
        scores = [len(q & words(d)) / (len(q) or 1) for d in body["documents"]]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: body["top_n"]]
        return httpx.Response(200, json={"results": [{"index": i, "relevance_score": scores[i]} for i in order]})


def client(fake, **kw):
    return CohereClient(api_key="test-key", transport=httpx.MockTransport(fake), backoff_s=0, **kw)


def test_embed_uses_document_and_query_modes_and_batches():
    fake = FakeCohere()
    emb = CohereEmbedder(client(fake))
    docs = [f"document number {i}" for i in range(200)]
    v = emb.encode(docs)
    assert v.shape == (200, 256) and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert [len(b["texts"]) for _, _, b in fake.requests] == [96, 96, 8]
    emb.encode_query(["where is document 3"])
    assert fake.requests[-1][2]["input_type"] == "search_query"
    assert fake.requests[0][2]["input_type"] == "search_document"
    assert fake.requests[0][1]["authorization"] == "Bearer test-key"
    assert fake.requests[0][2]["model"] == "embed-v4.0"


def test_retries_when_rate_limited_then_succeeds():
    fake = FakeCohere(fail_first=2, status=429)
    c = client(fake)
    assert c.embed(["hello"], "search_document").shape == (1, 256)
    assert c.calls == 3


def test_clear_errors_for_bad_key_and_missing_key(monkeypatch):
    with pytest.raises(CohereError, match="rejected the API key"):
        client(FakeCohere(fail_first=1, status=401)).embed(["x"], "search_document")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(CohereError, match="Set COHERE_API_KEY"):
        CohereClient()


def test_cached_embedder_passes_queries_to_query_mode():
    fake = FakeCohere()
    cached = CachedEmbedder(CohereEmbedder(client(fake)))
    cached.encode(["a b c"])
    cached.encode(["a b c"])
    assert cached.hits == 1
    cached.encode_query(["a b"])
    assert fake.requests[-1][2]["input_type"] == "search_query"


@pytest.fixture(scope="module")
def chunks():
    return chunk_by_section(load_manuals(ROOT / "data" / "manuals"))


def test_rerank_reorders_first_stage_and_keeps_refusal(chunks):
    fake = FakeCohere()
    p = RAGPipeline(chunks, CachedEmbedder(LSAEmbedder()), reranker=CohereReranker(client(fake)), rerank_candidates=10).build()
    got = p.retrieve("Which TCP port does third-party router control use?", k=3)
    assert len(got) == 3 and got[0].score >= got[1].score >= got[2].score
    path, _, body = fake.requests[-1]
    assert path.endswith("/rerank") and body["model"] == "rerank-v3.5" and len(body["documents"]) == 10 and body["top_n"] == 3
    assert "52116" in p.ask("Which TCP port does third-party router control use?").answer
    assert p.ask("Who won the World Cup?").answer == NOT_FOUND


def test_cohere_eval_runs_all_four_setups(chunks, tmp_path):
    rows = run(client(FakeCohere()), tmp_path)
    assert [r["setup"].split()[0] for r in rows] == ["LSA", "LSA", "Cohere", "Cohere"]
    assert all(0 <= r["recall@1"] <= 1 for r in rows)
    assert (tmp_path / "benchmark_cohere.md").read_text().startswith("# Cohere Embed and Rerank")
