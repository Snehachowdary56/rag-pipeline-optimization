"""Embedding models plus a cache so unchanged text is never embedded twice."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class LSAEmbedder:
    """TF-IDF followed by truncated SVD (latent semantic analysis). Runs offline with no model download."""

    name = "lsa"

    def __init__(self, dims: int = 64, seed: int = 0):
        self.dims = dims
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        self.svd = TruncatedSVD(n_components=dims, random_state=seed)
        self.fitted = False

    def fit(self, texts: list[str]) -> "LSAEmbedder":
        tfidf = self.vectorizer.fit_transform(texts)
        n = min(self.dims, tfidf.shape[1] - 1, len(texts) - 1)
        if n < self.dims:
            self.svd = TruncatedSVD(n_components=max(n, 2), random_state=0)
        self.svd.fit(tfidf)
        self.fitted = True
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before encode()")
        return _normalise(self.svd.transform(self.vectorizer.transform(texts)))


class SBERTEmbedder:
    """Neural sentence embeddings (all-MiniLM-L6-v2). Needs the sentence-transformers package and a model download."""

    name = "minilm"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported only when used
        self.model = SentenceTransformer(model)
        self.fitted = True

    def fit(self, texts: list[str]) -> "SBERTEmbedder":
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return _normalise(np.asarray(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)))


class CachedEmbedder:
    """Wraps an embedder and keeps vectors by the SHA-256 of the text, so re-indexing skips unchanged chunks."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.cache: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def fit(self, texts: list[str]):
        self.inner.fit(texts)
        self.cache.clear()      # a refit changes the vector space, so old vectors are no longer valid
        return self

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        keys = [self.key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vecs = self.inner.encode([texts[i] for i in missing], batch_size=batch_size)
            for i, v in zip(missing, vecs, strict=True):
                self.cache[keys[i]] = v
        return np.stack([self.cache[k] for k in keys])
