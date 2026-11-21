"""Measure retrieval quality and speed on the question set."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(pipeline, questions: list[dict], k: int = 3) -> dict:
    """recall@1 and recall@k: share of questions whose answer section is in the top 1 or top k chunks.
    MRR: average of 1/rank of the first correct chunk. answer_hit: share whose answer text contains the expected fact."""
    hits1 = hitsk = rr = ans_hits = 0
    times = []
    for item in questions:
        t0 = time.perf_counter()
        got = pipeline.retrieve(item["q"], k)
        times.append((time.perf_counter() - t0) * 1000)
        ranks = [i for i, r in enumerate(got) if r.chunk.doc == item["doc"] and item["section"] in r.chunk.sections]
        if ranks:
            hitsk += 1
            hits1 += ranks[0] == 0
            rr += 1 / (ranks[0] + 1)
        answer = pipeline.ask(item["q"], k).answer
        ans_hits += item["answer"].lower() in answer.lower()
    n = len(questions)
    return {
        "questions": n,
        "recall@1": round(hits1 / n, 3),
        f"recall@{k}": round(hitsk / n, 3),
        "mrr": round(rr / n, 3),
        "answer_hit": round(ans_hits / n, 3),
        "retrieve_ms_p50": round(float(np.percentile(times, 50)), 3),
        "retrieve_ms_p95": round(float(np.percentile(times, 95)), 3),
    }
