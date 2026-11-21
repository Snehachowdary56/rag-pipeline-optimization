"""Run the performance experiments and write results/benchmark.md and results/benchmark.json."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .chunking import chunk_by_section, chunk_by_window
from .embedders import CachedEmbedder, LSAEmbedder
from .evaluate import evaluate, load_questions
from .index import FlatIndex, HNSWIndex, QdrantIndex
from .loader import load_manuals
from .pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent


def make_embedder(name: str):
    if name == "minilm":
        from .embedders import SBERTEmbedder
        return CachedEmbedder(SBERTEmbedder())
    return CachedEmbedder(LSAEmbedder())


def chunking_and_hybrid(sections, questions, embedder_name):
    rows = []
    setups = [
        ("section chunks, dense", chunk_by_section(sections), None),
        ("40-word windows (overlap 10), dense", chunk_by_window(sections, 40, 10), None),
        ("80-word windows (overlap 20), dense", chunk_by_window(sections, 80, 20), None),
        ("section chunks, hybrid (dense + keyword)", chunk_by_section(sections), 0.5),
    ]
    for label, chunks, alpha in setups:
        p = RAGPipeline(chunks, make_embedder(embedder_name), hybrid_alpha=alpha).build()
        rows.append({"setup": label, "chunks": len(chunks), **evaluate(p, questions)})
    return rows


def clustered_vectors(n: int, dim: int, clusters: int, rng, latent: int = 32, proj=None) -> np.ndarray:
    """Simulated embeddings: clustered points in a 32-dimension space, projected up to `dim` dimensions.
    Real sentence embeddings have a much lower intrinsic dimension than their size, so this is closer to
    real data than uniform random noise (the worst case for approximate search)."""
    centres = rng.standard_normal((clusters, latent)).astype(np.float32)
    labels = rng.integers(0, clusters, n)
    z = centres[labels] + 0.5 * rng.standard_normal((n, latent)).astype(np.float32)
    data = z @ proj + 0.02 * rng.standard_normal((n, dim)).astype(np.float32)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def index_speed(n: int = 50_000, dim: int = 384, queries: int = 200, k: int = 10, seed: int = 0,
                ef_values=(16, 64, 256), include_qdrant: bool = False):
    """Compare exact search with HNSW (at several ef_search settings) and, optionally, Qdrant."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((32, dim)).astype(np.float32)
    data = clustered_vectors(n, dim, 200, np.random.default_rng(seed + 1), proj=proj)
    qs = clustered_vectors(queries, dim, 200, np.random.default_rng(seed + 2), proj=proj)

    def run(idx, label):
        res, times = [], []
        for q in qs:
            t1 = time.perf_counter()
            ids, _ = idx.search(q, k)
            times.append((time.perf_counter() - t1) * 1000)
            res.append(set(ids.tolist()))
        return label, res, times

    rows = []
    flat = FlatIndex(dim)
    t0 = time.perf_counter()
    flat.add(data)
    fb = time.perf_counter() - t0
    _, truth, ft = run(flat, "flat")
    rows.append(("flat (exact)", fb, ft, truth))
    hnsw = HNSWIndex(dim, max_elements=n)
    t0 = time.perf_counter()
    hnsw.add(data)
    hb = time.perf_counter() - t0
    for ef in ef_values:
        hnsw.index.set_ef(max(ef, k))
        _, r, t = run(hnsw, "hnsw")
        rows.append((f"hnsw ef_search={ef}", hb, t, r))
    if include_qdrant:
        qd = QdrantIndex(dim)
        t0 = time.perf_counter()
        qd.add(data)
        qb = time.perf_counter() - t0
        _, r, t = run(qd, "qdrant")
        rows.append(("qdrant (local mode)", qb, t, r))
    out = []
    for label, build, times, res in rows:
        out.append({"index": label, "vectors": n, "build_s": round(build, 2),
                    "query_ms_p50": round(float(np.percentile(times, 50)), 3),
                    "query_ms_p95": round(float(np.percentile(times, 95)), 3),
                    f"recall@{k}_vs_exact": round(float(np.mean([len(a & b) / k for a, b in zip(res, truth, strict=True)])), 3)})
    return out


def caching(sections, questions, embedder_name):
    chunks = chunk_by_section(sections)
    emb = make_embedder(embedder_name)
    emb.fit([c.text for c in chunks])
    t0 = time.perf_counter()
    emb.encode([c.text for c in chunks])
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    emb.encode([c.text for c in chunks])
    warm = time.perf_counter() - t0
    p = RAGPipeline(chunks, make_embedder(embedder_name)).build()
    first = [p.ask(q["q"]).latency_ms for q in questions]
    repeat = [p.ask(q["q"]).latency_ms for q in questions]
    return {
        "reindex_cold_ms": round(cold * 1000, 3), "reindex_warm_ms": round(warm * 1000, 3),
        "embedding_cache_hits_on_reindex": emb.hits,
        "ask_first_ms_p50": round(float(np.percentile(first, 50)), 3),
        "ask_repeat_ms_p50": round(float(np.percentile(repeat, 50)), 4),
    }


def table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["lsa", "minilm"], default="lsa")
    ap.add_argument("--vectors", type=int, default=50_000)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    sections = load_manuals(ROOT / "data" / "manuals")
    questions = load_questions(ROOT / "data" / "eval_questions.jsonl")
    res = {"embedder": a.embedder,
           "chunking_and_hybrid": chunking_and_hybrid(sections, questions, a.embedder),
           "index_speed": index_speed(n=a.vectors),
           "index_speed_small": index_speed(n=5_000, queries=100, include_qdrant=True),
           "caching": caching(sections, questions, a.embedder)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"benchmark_{a.embedder}.json").write_text(json.dumps(res, indent=2))
    md = [f"# Benchmark results ({a.embedder} embeddings)", "",
          "## Retrieval quality: chunking and hybrid search (30 questions, k = 3)", "", table(res["chunking_and_hybrid"]), "",
          f"## Index speed ({a.vectors:,} simulated 384-dimension embeddings, top 10)", "", table(res["index_speed"]), "",
          "## Index speed (5,000 vectors, includes Qdrant)", "", table(res["index_speed_small"]), "",
          "## Caching", "", table([res["caching"]]), ""]
    (a.out / f"benchmark_{a.embedder}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
