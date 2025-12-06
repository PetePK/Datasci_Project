# Research Paper Observatory

AI-powered research paper explorer with semantic search and interactive visualizations.

## Quick Start

### Backend (FastAPI)
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend (Next.js)
```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000

## Features

- **Semantic Search** - AI-powered search across 19,523 papers
- **Interactive Treemap** - Browse by research categories
- **Vector Search** - 384-dimensional embeddings with ChromaDB
- **Modern UI** - Next.js 14 with TailwindCSS

## Tech Stack

- **Backend**: FastAPI, ChromaDB, SentenceTransformers
- **Frontend**: Next.js 14, React, TypeScript, Plotly
- **ML**: all-MiniLM-L6-v2 embeddings, Claude 3.5 Haiku
- **Data**: 19,523 Scopus papers (2018-2023)

## Documentation

See [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) for full details.

## Project Structure

```
backend/          # FastAPI API server
frontend/         # Next.js web app
data/             # Papers, embeddings, vector DB
part1-4/          # Original pipeline notebooks
docs/             # Documentation
```
