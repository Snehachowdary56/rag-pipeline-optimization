"""Compare retrieval with and without Cohere models on the 30-question set. Needs COHERE_API_KEY.

  COHERE_API_KEY=... python -m ragkit.cohere_eval

Setups: offline LSA embeddings (baseline), LSA + Cohere Rerank, Cohere Embed, Cohere Embed + Cohere Rerank.
Writes results/benchmark_cohere.md and .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_by_section
from .cohere_models import EMBED_MODEL, RERANK_MODEL, CohereClient, CohereEmbedder, CohereReranker
from .embedders import CachedEmbedder, LSAEmbedder
from .evaluate import evaluate, load_questions
from .loader import load_manuals
from .pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent


def run(client: CohereClient, out: Path | None = ROOT / "results") -> list[dict]:
    sections = load_manuals(ROOT / "data" / "manuals")
    questions = load_questions(ROOT / "data" / "eval_questions.jsonl")
    chunks = chunk_by_section(sections)
    reranker = CohereReranker(client)
    setups = [
        ("LSA embeddings (offline baseline)", CachedEmbedder(LSAEmbedder()), None),
        (f"LSA + Cohere {RERANK_MODEL}", CachedEmbedder(LSAEmbedder()), reranker),
        (f"Cohere {EMBED_MODEL}", CachedEmbedder(CohereEmbedder(client)), None),
        (f"Cohere {EMBED_MODEL} + {RERANK_MODEL}", CachedEmbedder(CohereEmbedder(client)), reranker),
    ]
    rows = []
    for name, emb, rr in setups:
        p = RAGPipeline(chunks, emb, reranker=rr, min_score=0.0).build()
        m = evaluate(p, questions, k=3)
        rows.append({"setup": name, **{k: m[k] for k in ("recall@1", "recall@3", "mrr")}})
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_cohere.json").write_text(json.dumps(rows, indent=2))
        lines = ["# Cohere Embed and Rerank (30 questions, section chunks, k = 3)", "",
                 "| setup | recall@1 | recall@3 | MRR |", "|---|---|---|---|"]
        lines += [f"| {r['setup']} | {r['recall@1']} | {r['recall@3']} | {r['mrr']} |" for r in rows]
        (out / "benchmark_cohere.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    run(CohereClient(), ap.parse_args(argv).out)


if __name__ == "__main__":
    main()
