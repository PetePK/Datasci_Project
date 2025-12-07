# Research Paper Observatory - Project Summary

## Overview

AI-powered platform for exploring 19,523 academic research papers from Scopus (2018-2023) using semantic search, interactive visualizations, and machine learning.

## Architecture

### Backend (FastAPI)
- **API Server**: FastAPI with async support
- **Vector Search**: ChromaDB with HNSW algorithm
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2, 384-dim)
- **AI Analysis**: Claude 3.5 Haiku for summaries
- **Data**: Parquet (29MB), NumPy embeddings (29MB), ChromaDB (294MB)

**Endpoints**:
- `POST /api/search/` - Semantic paper search
- `GET /api/categories/{name}` - Category-based filtering
- `POST /api/insights/search` - AI-powered search insights
- `GET /api/recommendations` - Research recommendations
- `GET /api/stats/` - Statistics
- `GET /api/stats/treemap` - Treemap data
- `GET /api/stats/trends/{topic}` - Publication trends (Level 1 topics)
- `GET /api/stats/citation-analysis/{topic}` - Citation analysis (Level 2 topics)
- `GET /api/papers/{id}` - Paper details

### Frontend (Next.js)
- **Framework**: Next.js 14 App Router
- **UI**: TailwindCSS + custom purple/pink gradient theme
- **Visualizations**: Plotly.js treemap
- **State**: React hooks, URL-based navigation
- **Search**: Embedded in navigation bar

**Pages**:

- `/` - Interactive treemap with drill-down navigation
  - Level 1 topics: Show publication trends over time
  - Level 2 topics: Show citation analysis
  - Papers load on demand when clicking topics
- `/search?q=...` - Search results with AI insights
  - Papers display immediately
  - AI insights load asynchronously (~5-7s)

## Data Pipeline

### Part 1: Data Preparation
- **Input**: 20,216 raw Scopus JSON files
- **Processing**: Extract, clean, deduplicate
- **Output**: 19,523 papers in Parquet format
- **Fields**: title, abstract, year, citations, authors, etc.

### Part 2: Embeddings & Search
- **Model**: all-MiniLM-L6-v2 (80MB)
- **Embeddings**: Title + abstract → 384-dim vectors
- **Storage**: ChromaDB with persistent storage
- **Search**: Cosine similarity, L2 distance conversion

### Part 3: Stance & Network
- **Stance Detection**: Claude 3.5 Haiku (support/contradict/neutral)
- **Network Analysis**: Citation graphs, community detection
- **Summaries**: AI-generated paper summaries

### Part 4: Dashboard (Legacy)
- **Original**: Streamlit app
- **Migrated to**: Next.js + FastAPI (current system)

## Key Features

### 1. Semantic Search with AI Insights

- Type query in nav bar → auto-redirect to `/search?q=...`
- Vector similarity search (threshold: 0.3)
- Results sorted by relevance (47-48% typical)
- Shows: title, abstract, year, citations, authors, match %
- **AI Insights** (powered by Claude 3.5 Haiku):
  - Relevance summary of search results
  - Key papers identification
  - Emerging research directions
  - Search refinement tips
  - Loads asynchronously (~5-7s) without blocking results

### 2. Interactive Treemap with Trend Visualizations

- Hierarchical visualization of research categories
- 7 main areas: Medicine, Life Sciences, CS/AI, Engineering, Materials, Physics, Environment
- 85 total nodes (including "Other" subcategories)
- **Smart drill-down navigation**:
  - Click topic → Treemap zooms in
  - **Level 1 topics**: Show publication trends over time (2018-2023 line chart)
  - **Level 2 topics**: Show citation analysis (top 20 papers bar chart)
  - Papers load on demand below visualizations
  - Back button to navigate up the hierarchy
- **Lazy loading**: Homepage loads only treemap initially for fast performance

### 3. Data Quality
- **Deduplication**: By title similarity
- **Cleaning**: Remove incomplete records
- **Validation**: 19,523 high-quality papers
- **Coverage**: 2018-2023, multiple disciplines

## ML Models

### Embedding Model: all-MiniLM-L6-v2
- **Type**: Sentence transformer (distilled BERT)
- **Dimensions**: 384
- **Speed**: ~1000 docs/sec
- **Quality**: Strong for English academic text
- **Size**: 80 MB

**Why this model?**
- Fast inference for real-time search
- Good semantic understanding
- Reasonable size for deployment
- Well-tested on academic papers

### LLM: Claude 3.5 Haiku
- **Use**: Summaries and stance detection
- **Optimization**: Short prompts for speed
- **Cost**: ~$0.01 per 1000 papers analyzed
- **Quality**: High accuracy for academic content

## Performance

### Vector Search
- **Index**: HNSW (Hierarchical Navigable Small World)
- **Search time**: <100ms for 19k papers
- **Precision**: ~95% for top-20 results
- **Relevance formula**: `(2.0 - L2_distance) / 2.0`

### API Response Times
- Search: 30-50ms
- Stats: 5-10ms
- Treemap: 2-5ms (cached)
- Paper details: 1-2ms

## Deployment

### Local Development
```bash
# Backend
cd backend && python -m uvicorn main:app --reload

# Frontend
cd frontend && pnpm dev
```

### Production (Suggested)
- **Backend**: Railway.app ($5/mo) or Render.com (free)
- **Frontend**: Vercel (free)
- **Total cost**: $0-5/month

## Statistics

- **Total Papers**: 19,523
- **Year Range**: 2018-2023
- **Total Citations**: 87,543
- **Research Fields**: 7 main categories, 85 subcategories
- **Embeddings**: 384 dimensions × 19,523 = 7.5M values
- **Vector DB Size**: 294 MB

## Color Scheme

### Purple-Pink Gradient Theme
- **Primary**: Purple (#7209b7) to Pink (#f72585)
- **Treemap**: Dark navy → Deep purple → Magenta → Cyan
- **UI**: Professional, modern, academic feel
- **Consistency**: Used across nav, buttons, headers, cards

## Recent Improvements (Dec 2024)

1. **Category Filtering** - Click treemap categories to filter papers (not semantic search)
   - Word-boundary regex matching prevents false positives ("Physics" won't match "Biophysics")
   - Recursive subcategory inclusion
   - Returns 644 papers for "Immunology" vs searching all papers

2. **Pagination** - Load 20 papers at a time with "Load More" button
   - Improves initial page load time
   - Smooth async loading (300ms delay for UX)
   - Shows remaining paper count

3. **Code Cleanup** - Removed 7 dead code files
   - Deleted test scripts (test_db.py, test_search.py, test_search_fixed.py)
   - Removed debug scripts (compare_data.py, test_dbscan.py, test_queries.py)
   - **Security fix**: Removed hardcoded API keys (create_env.py)

4. **Navigation Improvements** - Simplified UI
   - Search bar embedded in navigation
   - Removed unused menu items (Home, Analytics)
   - Added back button beside search results heading

5. **Treemap Enhancements** - Better visualization
   - Index-based color palette (40+ colors)
   - Zoom into categories while showing papers
   - Fixed schema validation errors

## Future Enhancements

1. **"Other" Subcategories** - Papers not in specific topics → "{Category} - Other"
2. **Paper Details Page** - Full metadata, citations, similar papers
3. **Advanced Filters** - Year range, citation count, subject area
4. **Export** - CSV/JSON download of search results
5. **Analytics Dashboard** - Trends, patterns, insights
6. **User Accounts** - Save searches, favorites, notes
7. **Citation Network** - Interactive graph visualization

## Technical Decisions

### Why Next.js over Streamlit?
- Better UX/UI flexibility
- Faster page loads
- Separate frontend/backend (scalable)
- Professional appearance
- Custom navigation and layouts

### Why FastAPI over Flask?
- Async support (faster concurrent requests)
- Automatic API docs (OpenAPI)
- Type validation (Pydantic)
- Modern Python features
- Better performance

### Why ChromaDB?
- Built for vector search
- Fast HNSW indexing
- Persistent storage
- Easy to use
- Good for 10k-100k documents

## Project Timeline

1. **Data Collection** - Scopus API scraping
2. **Preprocessing** - Cleaning, deduplication (Part 1)
3. **Embeddings** - Vector generation (Part 2)
4. **Analysis** - Stance, summaries (Part 3)
5. **Streamlit Dashboard** - First version (Part 4)
6. **Migration** - Next.js + FastAPI rebuild
7. **Current** - Production-ready web app

## Lessons Learned

1. **Start with quality data** - GIGO applies to ML
2. **Choose right model** - Balance speed vs accuracy
3. **Optimize early** - Caching, indexing, async
4. **User experience matters** - Pretty UI = more usage
5. **Document everything** - Future you will thank you

---

**Built with**: Python, TypeScript, FastAPI, Next.js, ChromaDB, SentenceTransformers, Claude AI

**License**: Academic use

**Contact**: Chulalongkorn University
