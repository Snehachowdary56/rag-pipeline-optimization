# Benchmark results (lsa embeddings)

## Retrieval quality: chunking and hybrid search (30 questions, k = 3)

| setup | chunks | questions | recall@1 | recall@3 | mrr | answer_hit | retrieve_ms_p50 | retrieve_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| section chunks, dense | 30 | 30 | 0.9 | 0.967 | 0.933 | 0.933 | 0.484 | 0.573 |
| 40-word windows (overlap 10), dense | 34 | 30 | 0.9 | 0.967 | 0.933 | 0.8 | 0.502 | 0.693 |
| 80-word windows (overlap 20), dense | 18 | 30 | 0.867 | 0.967 | 0.917 | 0.9 | 0.527 | 1.284 |
| section chunks, hybrid (dense + keyword) | 30 | 30 | 0.867 | 0.967 | 0.917 | 0.9 | 0.957 | 1.091 |

## Index speed (50,000 simulated 384-dimension embeddings, top 10)

| index | vectors | build_s | query_ms_p50 | query_ms_p95 | recall@10_vs_exact |
|---|---|---|---|---|---|
| flat (exact) | 50000 | 0.04 | 3.594 | 4.211 | 1.0 |
| hnsw ef_search=16 | 50000 | 7.45 | 0.058 | 0.116 | 0.625 |
| hnsw ef_search=64 | 50000 | 7.45 | 0.149 | 0.245 | 0.88 |
| hnsw ef_search=256 | 50000 | 7.45 | 0.396 | 0.494 | 0.986 |

## Index speed (5,000 vectors, includes Qdrant)

| index | vectors | build_s | query_ms_p50 | query_ms_p95 | recall@10_vs_exact |
|---|---|---|---|---|---|
| flat (exact) | 5000 | 0.0 | 0.368 | 0.428 | 1.0 |
| hnsw ef_search=16 | 5000 | 0.56 | 0.034 | 0.073 | 0.85 |
| hnsw ef_search=64 | 5000 | 0.56 | 0.094 | 0.152 | 0.99 |
| hnsw ef_search=256 | 5000 | 0.56 | 0.273 | 0.332 | 1.0 |
| qdrant (local mode) | 5000 | 0.32 | 4.16 | 6.504 | 1.0 |

## Caching

| reindex_cold_ms | reindex_warm_ms | embedding_cache_hits_on_reindex | ask_first_ms_p50 | ask_repeat_ms_p50 |
|---|---|---|---|---|
| 1.851 | 0.048 | 30 | 0.598 | 0.0006 |
