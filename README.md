# Research Paper Explorer

Platform for exploring 19,523 Scopus papers (2018-2023) with semantic search, interactive visualizations, and AI-powered insights.

Next.js + FastAPI + ChromaDB + Claude AI

## What It Does

- Semantic search across papers (vector embeddings, not keyword matching)
- Interactive treemap for browsing 7 research fields and 85 categories
- AI-generated research suggestions (Claude API)
- Time series analysis showing publication and citation trends
- Filter by subject, year range, citations

## Getting Started

Backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
# Runs on http://localhost:8000
```

Frontend:
```bash
cd frontend
pnpm install
pnpm dev
# Runs on http://localhost:3000
```

Create `backend/.env` with your API key:
```
ANTHROPIC_API_KEY=your_key_here
```

## Project Layout

```
data/processed/
  papers.parquet        # 19,523 papers (29 MB)
  embeddings/           # 384-dim vectors (30 MB)
  vector_db/            # ChromaDB index (263 MB)

backend/
  api/                  # REST endpoints
  services/             # Data loading, vector search, LLM
  main.py

frontend/
  app/                  # Browse + search pages
  components/           # UI components

part1_*/                # Data cleaning notebooks
part2_*/                # Embedding generation
part3_*/                # Trends analysis
```

## How It Works

**Data Processing**  
Raw Scopus JSON (20K files) → cleaned dataset (19,523 papers)

**Embeddings**  
Papers encoded with all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB

**Analysis**  
Time series + AI insights using Claude

## API Routes

```
POST /api/search/                      # Vector search
GET  /api/categories/{category}        # Category filter
GET  /api/stats/treemap                # Hierarchy data
GET  /api/stats/level2-trends/{topic}  # Time series
POST /api/insights/search              # AI analysis
```

## Numbers

- 19,523 papers (2018-2023)
- 87K+ citations
- 7 fields, 85 categories
- <100ms query time
- 384-dim embeddings

## Stack

Frontend: Next.js 14, TailwindCSS, Plotly  
Backend: FastAPI, ChromaDB, Pandas  
AI: SentenceTransformers, Claude API

---

Chulalongkorn University Data Science Project
