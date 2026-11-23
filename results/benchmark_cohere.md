# Cohere Embed and Rerank (30 questions, section chunks, k = 3)

| setup | recall@1 | recall@3 | MRR |
| LSA embeddings (offline baseline) | 0.9 | 0.967 | 0.933 |
| LSA + Cohere rerank-v3.5 | 1.0 | 1.0 | 1.0 |
| Cohere embed-v4.0 | 0.967 | 1.0 | 0.983 |
| Cohere embed-v4.0 + rerank-v3.5 | 1.0 | 1.0 | 1.0 |
