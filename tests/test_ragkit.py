from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ragkit.api import app
from ragkit.chunking import chunk_by_section, chunk_by_window
from ragkit.embedders import CachedEmbedder, LSAEmbedder
from ragkit.evaluate import evaluate, load_questions
from ragkit.index import FlatIndex, HNSWIndex, QdrantIndex
from ragkit.loader import load_manuals
from ragkit.pipeline import NOT_FOUND, RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = load_manuals(ROOT / "data" / "manuals")
QUESTIONS = load_questions(ROOT / "data" / "eval_questions.jsonl")


@pytest.fixture(scope="module")
def pipe():
    return RAGPipeline(chunk_by_section(SECTIONS), CachedEmbedder(LSAEmbedder())).build()


# ---------- loading and chunking ----------

def test_loader_reads_all_sections():
    assert len(SECTIONS) == 30
    assert {s.doc for s in SECTIONS} == {"switcher", "camera", "graphics", "replay", "router", "gateway"}


def test_section_chunks_keep_title_for_search_and_body_for_answers():
    c = chunk_by_section(SECTIONS)[0]
    assert c.text.startswith(c.title) and c.body in c.text and not c.body.startswith(c.title)


def test_window_chunks_respect_size_and_overlap():
    chunks = chunk_by_window(SECTIONS, size=40, overlap=10)
    assert all(len(c.body.split()) <= 40 for c in chunks)
    same_doc = [c for c in chunks if c.doc == "switcher"]
    first, second = same_doc[0].body.split(), same_doc[1].body.split()
    assert first[-10:] == second[:10]          # 10 words of overlap


@pytest.mark.parametrize("size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_window_rejects_bad_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_by_window(SECTIONS, size, overlap)


def test_every_question_points_at_a_real_section():
    keys = {(s.doc, s.section) for s in SECTIONS}
    assert len(QUESTIONS) == 30
    assert all((q["doc"], q["section"]) in keys for q in QUESTIONS)


# ---------- embeddings and caching ----------

def test_embeddings_are_unit_length():
    emb = LSAEmbedder().fit([s.text for s in SECTIONS])
    v = emb.encode(["router port", "camera fiber"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_embedding_cache_skips_unchanged_text():
    emb = CachedEmbedder(LSAEmbedder()).fit([s.text for s in SECTIONS])
    texts = [s.text for s in SECTIONS]
    first = emb.encode(texts)
    second = emb.encode(texts)
    assert emb.misses == 30 and emb.hits == 30
    assert np.array_equal(first, second)


# ---------- indexes ----------

@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((500, 32)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.mark.parametrize("cls", [HNSWIndex, QdrantIndex])
def test_indexes_agree_with_exact_search(cls, vectors):
    flat, other = FlatIndex(32), cls(32)
    flat.add(vectors)
    other.add(vectors)
    for q in vectors[:20]:
        assert flat.search(q, 1)[0][0] == other.search(q, 1)[0][0]   # a vector's nearest neighbour is itself


def test_flat_search_returns_sorted_scores(vectors):
    idx = FlatIndex(32)
    idx.add(vectors)
    _, scores = idx.search(vectors[0], 5)
    assert list(scores) == sorted(scores, reverse=True)


# ---------- pipeline ----------

def test_retrieval_quality_meets_target(pipe):
    m = evaluate(pipe, QUESTIONS)
    assert m["recall@1"] >= 0.85 and m["recall@3"] >= 0.95 and m["answer_hit"] >= 0.9


def test_answer_is_grounded_and_cited(pipe):
    a = pipe.ask("Which TCP port does third-party router control use?")
    assert "52116" in a.answer and a.citations == ["router#Control Protocol"] and a.grounded


def test_off_topic_question_is_refused(pipe):
    a = pipe.ask("Who won the football world cup in 2018?")
    assert a.answer == NOT_FOUND and not a.grounded and a.citations == []


def test_repeated_question_is_served_from_cache(pipe):
    pipe.ask("How long can the camera fiber cable be?")
    again = pipe.ask("  how long can the camera fiber cable be?  ")
    assert again.cached


def test_hybrid_and_hnsw_pipelines_still_find_answers():
    for p in (RAGPipeline(chunk_by_section(SECTIONS), CachedEmbedder(LSAEmbedder()), hybrid_alpha=0.5).build(),
              RAGPipeline(chunk_by_section(SECTIONS), CachedEmbedder(LSAEmbedder()), index_kind="hnsw").build()):
        assert evaluate(p, QUESTIONS)["recall@3"] >= 0.9


# ---------- REST API ----------

def test_api_health_ask_and_search():
    client = TestClient(app)
    assert client.get("/health").json()["chunks"] == 30
    r = client.post("/ask", json={"question": "How are remote production streams encrypted?"})
    assert r.status_code == 200 and "AES-256" in r.json()["answer"]
    s = client.post("/search", json={"question": "replay export formats", "k": 2})
    assert s.status_code == 200 and len(s.json()["results"]) == 2


def test_api_rejects_bad_input():
    client = TestClient(app)
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 50}).status_code == 422
