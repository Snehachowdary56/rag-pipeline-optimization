# Benchmark results (minilm embeddings)

## Retrieval quality: chunking and hybrid search (30 questions, k = 3)

| setup | chunks | questions | recall@1 | recall@3 | mrr | answer_hit | retrieve_ms_p50 | retrieve_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| section chunks, dense | 30 | 30 | 0.933 | 0.967 | 0.95 | 0.967 | 6.945 | 8.151 |
| 40-word windows (overlap 10), dense | 34 | 30 | 0.967 | 1.0 | 0.983 | 1.0 | 6.755 | 7.37 |
| 80-word windows (overlap 20), dense | 18 | 30 | 0.9 | 1.0 | 0.95 | 0.933 | 6.949 | 7.724 |
| section chunks, hybrid (dense + keyword) | 30 | 30 | 0.967 | 1.0 | 0.983 | 1.0 | 7.755 | 8.433 |

## Index speed (50,000 simulated 384-dimension embeddings, top 10)

| index | vectors | build_s | query_ms_p50 | query_ms_p95 | recall@10_vs_exact |
|---|---|---|---|---|---|
| flat (exact) | 50000 | 0.02 | 1.724 | 2.176 | 1.0 |
| hnsw ef_search=16 | 50000 | 3.9 | 0.038 | 0.056 | 0.632 |
| hnsw ef_search=64 | 50000 | 3.9 | 0.099 | 0.138 | 0.885 |
| hnsw ef_search=256 | 50000 | 3.9 | 0.297 | 0.35 | 0.986 |

## Index speed (5,000 vectors, includes Qdrant)

| index | vectors | build_s | query_ms_p50 | query_ms_p95 | recall@10_vs_exact |
|---|---|---|---|---|---|
| flat (exact) | 5000 | 0.0 | 0.086 | 0.104 | 1.0 |
| hnsw ef_search=16 | 5000 | 0.27 | 0.018 | 0.022 | 0.845 |
| hnsw ef_search=64 | 5000 | 0.27 | 0.046 | 0.059 | 0.99 |
| hnsw ef_search=256 | 5000 | 0.27 | 0.135 | 0.145 | 1.0 |
| qdrant (local mode) | 5000 | 0.26 | 2.466 | 2.747 | 1.0 |

## Caching

| reindex_cold_ms | reindex_warm_ms | embedding_cache_hits_on_reindex | ask_first_ms_p50 | ask_repeat_ms_p50 |
|---|---|---|---|---|
| 271.549 | 0.059 | 30 | 6.738 | 0.0011 |
