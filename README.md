# RAG Pipeline with Performance Optimization

![CI](https://github.com/Snehachowdary56/rag-pipeline-optimization/actions/workflows/ci.yml/badge.svg)

A **retrieval-augmented generation (RAG)** question-answering pipeline over technical product manuals, with **measured performance optimizations**: chunking strategy, exact vs. approximate (HNSW) vector search, the Qdrant vector database, hybrid search, embedding caching and query caching. It has a **REST API** (FastAPI), a **Go client**, Docker, and CI on GitHub Actions.

> **Data note:** the six manuals (a production switcher, camera control unit, graphics engine, replay server, video router and remote production gateway) are **made up** for this demo under the fictional brand "NorthLight". They are not real products.

## How it works

```
manuals (.md) -> chunking -> embeddings -> vector index (flat / HNSW / Qdrant)
                                                   |
question -> embedding -> semantic search (top k) -> grounded answer with citations
                                                   -> "I could not find this in the manuals." if nothing matches
```

1. **Load** the manuals and split them into **chunks**: one per section, or fixed word windows with overlap.
2. **Embed** each chunk. The default embedder is TF-IDF + truncated SVD (latent semantic analysis), which runs offline. CI also benchmarks a neural model, **all-MiniLM-L6-v2** (sentence-transformers).
3. **Index** the vectors in one of three backends with the same interface:
   - `flat`: exact cosine search (numpy)
   - `hnsw`: approximate nearest neighbours with an HNSW graph (hnswlib)
   - `qdrant`: the **Qdrant vector database** (local mode)
4. **Retrieve** the top k chunks for a question (**semantic search**), optionally with **hybrid search** that re-scores candidates with keyword (TF-IDF) similarity.
5. **Answer only from what was retrieved**, with a citation like `[router#Control Protocol]`. Offline, the answer is the best chunk's text. If `ANTHROPIC_API_KEY` is set, an LLM writes the answer from the retrieved text only and must reply "I could not find this in the manuals." when the context does not contain it. If the best match is too weak, the pipeline refuses instead of guessing.
6. **Cache**: embeddings are cached by the SHA-256 of their text, so re-indexing skips unchanged chunks, and repeated questions are served from an LRU query cache.

## Cohere Embed and Rerank

`ragkit/cohere_models.py` adds Cohere's models through the v2 REST API:

- **Cohere Embed** (`embed-v4.0`): chunks are embedded with `input_type="search_document"` and questions with
  `input_type="search_query"`; batches of up to 96 texts; calls are paced to the trial key's per-minute limits, retries honour `Retry-After`
  on rate limits (429) and server errors, and repeated rerank requests are cached.
- **Cohere Rerank** (`rerank-v3.5`): a second stage. The first stage retrieves 20 candidates, then the reranker reads
  the question and each chunk together and re-orders them; the pipeline refuses when the best rerank score is below 0.1.

```bash
export COHERE_API_KEY=...            # free trial key from dashboard.cohere.com/api-keys; never commit it
python -m ragkit.cohere_eval         # compares LSA, LSA + Rerank, Cohere Embed, Cohere Embed + Rerank
```

Results go to `results/benchmark_cohere.md`. Tests use a fake Cohere API (`tests/test_cohere.py`, 7 tests), so CI needs no key;
a `cohere-eval` CI job runs the real comparison when the repository has a `COHERE_API_KEY` secret.

## Results

From `python -m ragkit.benchmark` (TF-IDF/LSA embeddings, run on a laptop-class Linux machine; exact timings vary by machine). Full tables: [results/benchmark_lsa.md](results/benchmark_lsa.md).

**Retrieval quality** on 30 questions written in different words from the manuals:

| Setup | recall@1 | recall@3 | MRR | Answer contains the fact |
|---|---|---|---|---|
| Section chunks, dense | **0.90** | **0.967** | **0.933** | **0.933** |
| 40-word windows (overlap 10) | 0.90 | 0.967 | 0.933 | 0.80 |
| 80-word windows (overlap 20) | 0.867 | 0.967 | 0.917 | 0.90 |
| Section chunks, hybrid (dense + keyword) | 0.867 | 0.967 | 0.917 | 0.90 |

**With a neural embedding model (all-MiniLM-L6-v2, run in GitHub Actions CI):** [results/benchmark_minilm.md](results/benchmark_minilm.md)

| Setup | recall@1 | recall@3 | MRR | Answer contains the fact |
|---|---|---|---|---|
| Section chunks, dense | 0.933 | 0.967 | 0.95 | 0.967 |
| 40-word windows (overlap 10) | 0.967 | **1.0** | 0.983 | **1.0** |
| Section chunks, hybrid (dense + keyword) | **0.967** | **1.0** | **0.983** | **1.0** |

With MiniLM, **hybrid search** lifted recall@1 from 0.933 to 0.967 and found the right section in the top 3 for every question, for about 1 ms of extra retrieval time. With the simpler LSA embeddings below it did not help, so whether hybrid search is worth it depends on the embedding model: measure, then choose.

With LSA embeddings, section-based chunking worked best: windows split facts across chunks and lowered answer quality. Hybrid search did not help on this small corpus with these embeddings, and it doubled retrieval time, so dense search stays the default.

**Search speed** on 50,000 simulated 384-dimension embeddings (top 10):

| Index | Query time (median) | Recall vs. exact |
|---|---|---|
| Flat (exact) | 3.59 ms | 1.000 |
| HNSW, ef_search = 16 | 0.06 ms | 0.625 |
| HNSW, ef_search = 64 | 0.15 ms | 0.880 |
| HNSW, ef_search = 256 | **0.40 ms** | **0.986** |

HNSW at ef_search = 256 is about **9 times faster** than exact search while finding 98.6% of the true top 10. Lower ef_search is faster but misses more neighbours, so it is a speed/accuracy setting to tune.

**Caching:** re-indexing unchanged chunks dropped from 1.85 ms to 0.05 ms (all 30 served from the embedding cache), and a repeated question is answered from the query cache in under 0.001 ms.

## Run it (Windows Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python -m ragkit.benchmark
uvicorn ragkit.api:app --port 8000
```

Then, in a second window:

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Which TCP port does router control use?\"}"
cd go
go run ./cmd/ask -url http://localhost:8000 "How are remote production streams encrypted?"
```

API endpoints: `GET /health`, `POST /search` (top k chunks with scores), `POST /ask` (answer, citations, grounded, cached, latency). Interactive docs: http://localhost:8000/docs

Docker: `docker build -t rag-api .` then `docker run -p 8000:8000 rag-api`

## Tests and CI

- **19 pytest tests:** loading, chunk size and overlap, unit-length embeddings, embedding cache hits, HNSW and Qdrant agree with exact search, retrieval quality targets (recall@1 >= 0.85, recall@3 >= 0.95), grounded and cited answers, refusal of off-topic questions, query cache, and the REST API (including input validation).
- **Go:** `go vet`, 2 unit tests for the client (request body, decoding, HTTP errors), and a build.
- **End to end:** CI builds the Docker image, starts the API and asks questions with the Go client, including one that must be refused.
- **Neural embeddings:** CI runs the full benchmark with all-MiniLM-L6-v2 and saves the results as a build artifact.

## Files

```
ragkit/loader.py      read manuals into sections
ragkit/chunking.py    section and sliding-window chunking
ragkit/embedders.py   LSA and MiniLM embedders, embedding cache
ragkit/index.py       flat, HNSW and Qdrant indexes
ragkit/pipeline.py    retrieval, hybrid search, grounded answers, query cache
ragkit/evaluate.py    recall@k, MRR, answer hit rate, latency
ragkit/benchmark.py   performance experiments
ragkit/api.py         FastAPI REST API
go/                   Go client library, CLI and tests
data/                 made-up manuals and 30 evaluation questions
results/              benchmark output
```
