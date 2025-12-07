# Research Paper Explorer

Search and explore 19,523 academic papers from Scopus (2018-2023) using AI-powered semantic search.

**Stack**: Next.js + FastAPI + ChromaDB + SentenceTransformers

---

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload  # http://localhost:8000

# Frontend
cd frontend
pnpm install
pnpm dev  # http://localhost:3000
```

---

## Structure

```
data/processed/     # Papers, embeddings, vector DB
backend/            # FastAPI server
frontend/           # Next.js app
part1_*/            # Data cleaning notebooks
part2_*/            # Embedding generation
part3_*/            # Trends analysis
```

---

## Features

- **Semantic Search**: Find papers by meaning, not just keywords
- **Interactive Treemap**: Browse 7 research fields hierarchically
- **Filters**: Subject areas, year range, citations
- **AI Insights**: Claude-powered research suggestions (search page)

---

## Pipeline

1. **Data Prep**: 20K JSON files → 19,523 clean papers
2. **Embeddings**: all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB
3. **Trends**: Time series analysis + AI insights generation

---

## Tech

- **Frontend**: Next.js 14, TailwindCSS, Plotly
- **Backend**: FastAPI, ChromaDB, Pandas
- **AI**: all-MiniLM-L6-v2, Claude (Anthropic)

---

## API

```
POST /api/search/          # Semantic search
GET  /api/categories/{id}   # Filter by category
GET  /api/stats/treemap     # Treemap data
GET  /api/stats/level2-trends/{topic}  # Time series
POST /api/insights/search   # AI analysis
```

---

## Stats

- 19,523 papers (2018-2023)
- 87K+ total citations
- 7 major fields, 85 categories
- <100ms search queries

---

---

Data Science Project - Chulalongkorn University
