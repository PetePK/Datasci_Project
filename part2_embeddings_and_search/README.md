# Part 2: Vector Search

Built semantic search using embeddings + ChromaDB.

## Why Vectors?

Traditional keyword search misses semantically similar papers. Vector embeddings capture meaning.

## Model Choice

Tested 3 models, picked **all-MiniLM-L6-v2**:
- 384 dimensions
- 80 MB size
- Fast (1000 docs/sec)
- Good quality for our dataset

## What We Built

1. Generated embeddings for 19,523 papers (title + abstract)
2. Indexed in ChromaDB with HNSW
3. Search returns results in <100ms

Output: `data/vector_db/` (263 MB)
