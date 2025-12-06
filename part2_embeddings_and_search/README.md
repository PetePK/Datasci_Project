# Part 2: Embeddings & Vector Search

Generate semantic embeddings and build vector search index.

## Model
- **Name**: all-MiniLM-L6-v2
- **Type**: Sentence transformer
- **Dimensions**: 384
- **Size**: 80 MB

## Processing
1. **Combine Text** - Title + abstract
2. **Generate Embeddings** - 19,523 papers → 384-dim vectors
3. **Build Index** - ChromaDB with HNSW algorithm
4. **Test Search** - Verify semantic similarity

## Output
- `data/processed/embeddings.npy` - 19,523 × 384 array (29 MB)
- `data/vector_db/` - ChromaDB persistent storage (294 MB)

## Usage
```bash
jupyter notebook phase2_embeddings_and_vector_search.ipynb
```

## Search Performance
- **Speed**: <100ms for 19k papers
- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Similarity**: L2 distance → `(2.0 - distance) / 2.0`
- **Accuracy**: ~95% precision @ top-20
