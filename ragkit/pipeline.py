"""The RAG pipeline: retrieve chunks, then answer only from what was retrieved, with citations."""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .index import make_index

NOT_FOUND = "I could not find this in the manuals."


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str]
    retrieved: list[Retrieved]
    latency_ms: float
    cached: bool = False
    grounded: bool = True


@dataclass
class RAGPipeline:
    chunks: list[Chunk]
    embedder: object
    index_kind: str = "flat"
    hybrid_alpha: float | None = None   # None = dense only; 0.0-1.0 = weight of dense score in hybrid search
    min_score: float = 0.15             # below this the pipeline says it does not know
    cache_size: int = 256
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)

    def build(self) -> "RAGPipeline":
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        self.index = make_index(self.index_kind, vectors.shape[1])
        self.index.add(vectors)
        if self.hybrid_alpha is not None:
            self._lex = TfidfVectorizer(sublinear_tf=True, stop_words="english")
            self._lex_matrix = self._lex.fit_transform(texts)
        return self

    def retrieve(self, question: str, k: int = 3) -> list[Retrieved]:
        q = self.embedder.encode([question])[0]
        if self.hybrid_alpha is None:
            ids, scores = self.index.search(q, k)
            return [Retrieved(self.chunks[i], float(s)) for i, s in zip(ids, scores, strict=True)]
        # Hybrid search: re-score a wider dense candidate set with keyword (TF-IDF) similarity.
        ids, dense = self.index.search(q, min(len(self.chunks), max(k * 5, 10)))
        lex = (self._lex_matrix[ids] @ self._lex.transform([question]).T).toarray().ravel()
        combined = self.hybrid_alpha * dense + (1 - self.hybrid_alpha) * lex
        order = np.argsort(-combined)[:k]
        return [Retrieved(self.chunks[ids[i]], float(combined[i])) for i in order]

    def ask(self, question: str, k: int = 3) -> Answer:
        start = time.perf_counter()
        key = (question.strip().lower(), k)
        if key in self._cache:
            self._cache.move_to_end(key)
            hit = self._cache[key]
            return Answer(hit.question, hit.answer, hit.citations, hit.retrieved,
                          (time.perf_counter() - start) * 1000, cached=True, grounded=hit.grounded)
        retrieved = self.retrieve(question, k)
        if not retrieved or retrieved[0].score < self.min_score:
            ans = Answer(question, NOT_FOUND, [], retrieved, 0.0, grounded=False)
        else:
            text = llm_answer(question, retrieved)
            if text:   # the model may use any retrieved chunk, so cite them all
                cites = list(dict.fromkeys(r.chunk.citation for r in retrieved))
            else:      # the offline answer uses only the best chunk
                text = extractive_answer(question, retrieved)
                cites = [retrieved[0].chunk.citation]
            ans = Answer(question, text, cites, retrieved, 0.0)
        ans.latency_ms = (time.perf_counter() - start) * 1000
        self._cache[key] = ans
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return ans


def extractive_answer(question: str, retrieved: list[Retrieved], max_words: int = 80) -> str:
    """Offline answer: return the text of the best-matching chunk, cited. Nothing is added that is not in the manuals."""
    if not retrieved:
        return NOT_FOUND
    best = retrieved[0].chunk
    words = (best.body or best.text).split()
    text = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    return f"{text} [{best.citation}]"


def llm_answer(question: str, retrieved: list[Retrieved]) -> str | None:
    """Optional: if ANTHROPIC_API_KEY is set, ask Claude to answer only from the retrieved text."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    context = "\n\n".join(f"[{r.chunk.citation}] {r.chunk.text}" for r in retrieved)
    prompt = ("Answer the question using only the context. Cite sources like [doc#Section]. "
              f"If the context does not contain the answer, reply exactly: {NOT_FOUND}\n\n"
              f"Context:\n{context}\n\nQuestion: {question}")
    client = anthropic.Anthropic()
    msg = client.messages.create(model=os.environ.get("RAG_MODEL", "claude-haiku-4-5"), max_tokens=300,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip() or None
