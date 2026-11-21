"""Three vector indexes with the same interface: exact search, HNSW (approximate) and the Qdrant vector database."""
from __future__ import annotations

import numpy as np


class FlatIndex:
    """Exact cosine search: compares the query with every vector. Accurate, but slower as data grows."""

    name = "flat"

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self.vectors @ query.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]

    def __len__(self) -> int:
        return len(self.vectors)


class HNSWIndex:
    """Approximate nearest-neighbour search with an HNSW graph (hnswlib). Much faster on large collections."""

    name = "hnsw"

    def __init__(self, dim: int, max_elements: int = 100_000, m: int = 16, ef_construction: int = 200, ef_search: int = 64):
        import hnswlib
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=m, random_seed=0)
        self.index.set_ef(ef_search)
        self.count = 0

    def add(self, vectors: np.ndarray) -> None:
        ids = np.arange(self.count, self.count + len(vectors))
        self.index.add_items(vectors.astype(np.float32), ids)
        self.count += len(vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.count)
        labels, distances = self.index.knn_query(query.astype(np.float32), k=k)
        return labels[0].astype(int), 1.0 - distances[0]

    def __len__(self) -> int:
        return self.count


class QdrantIndex:
    """Qdrant vector database (local in-memory mode, the same API as a Qdrant server)."""

    name = "qdrant"

    def __init__(self, dim: int, collection: str = "manuals"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.client = QdrantClient(":memory:")
        self.collection = collection
        self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self.count = 0

    def add(self, vectors: np.ndarray) -> None:
        from qdrant_client.models import PointStruct
        points = [PointStruct(id=self.count + i, vector=v.tolist()) for i, v in enumerate(vectors)]
        self.client.upsert(self.collection, points=points)
        self.count += len(vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        hits = self.client.query_points(self.collection, query=query.tolist(), limit=k).points
        return np.array([h.id for h in hits], dtype=int), np.array([h.score for h in hits], dtype=np.float32)

    def __len__(self) -> int:
        return self.count


def make_index(kind: str, dim: int):
    return {"flat": FlatIndex, "hnsw": HNSWIndex, "qdrant": QdrantIndex}[kind](dim)
