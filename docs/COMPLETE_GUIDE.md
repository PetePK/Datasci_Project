# AI Literature Review Assistant - Complete Technical Guide

**Search 19,523 research papers with semantic search, AI analysis, and interactive visualization**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technical Stack](#technical-stack)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Part 1: Data Preparation](#part-1-data-preparation)
5. [Part 2: Embeddings & Vector Search](#part-2-embeddings--vector-search)
6. [Part 3: Stance Detection & Network Analysis](#part-3-stance-detection--network-analysis)
7. [Part 4: Interactive Dashboard](#part-4-interactive-dashboard)
8. [Data Files Reference](#data-files-reference)
9. [Performance Optimizations](#performance-optimizations)
10. [Deployment Guide](#deployment-guide)

---

## Project Overview

### The Problem

Literature review is time-consuming and challenging:
- **10+ hours** to manually read and analyze papers
- **Keyword search** misses synonyms and related concepts
- **No visualization** of research landscape and paper relationships
- **Difficult** to identify which papers support or contradict a hypothesis

### Our Solution

AI-powered research assistant with three core capabilities:

**1. Semantic Search**
- Search 19,523 papers by meaning, not just keywords
- Vector-based search using sentence transformers
- Returns results in <1 second

**2. AI Analysis (Claude 3.5 Haiku)**
- Automatic one-sentence summaries for each paper
- Stance detection: SUPPORT ✓ / CONTRADICT ✗ / NEUTRAL ○
- Analyzes 20 papers in 10-12 seconds

**3. Interactive Visualization**
- Tree map showing research landscape across 7 subject categories
- Timeline analysis with publication trends
- Color-coded subject tags and stance badges
- Export results to CSV

### Real Impact

- Saves researchers **10+ hours** per project
- Visualizes **research landscape** across subject areas
- Identifies **contradicting evidence** (critical for research quality)
- Explores **19,523 papers** across 6 years (2018-2023)

---

## Technical Stack

### Backend
- **Python 3.9+**: Core language
- **Pandas**: Data processing (19K papers in 0.65 seconds)
- **NumPy**: Numerical operations and vector math
- **NetworkX**: Network analysis and community detection

### AI/ML Models

**1. Text Embeddings: sentence-transformers (all-MiniLM-L6-v2)**
- Converts text → 384-dimensional vectors
- Free, runs locally
- 2-5 minutes to embed 19K papers
- Used for semantic search

**2. Vector Database: ChromaDB**
- Fast similarity search with HNSW algorithm
- PersistentClient (saves to disk, no server needed)
- <0.1 second per query

**3. AI Analysis: Claude 3.5 Haiku (Anthropic)**
- One-sentence paper summaries
- Stance detection (support/contradict/neutral)
- Fast: 10-12 seconds for 20 papers
- Cost: ~$0.01-0.02 per search

### Frontend
- **Streamlit**: Interactive web dashboard framework
- **Plotly**: Tree map and chart visualizations
- **Custom CSS**: Modern UI with color-coded elements

### Data Format
- **Parquet**: Main format (50% smaller, 5-10x faster than CSV)
- **JSON**: Configuration and cache files
- **NumPy arrays**: Vector embeddings

### Cost
- **Embedding generation**: $0 (free local model)
- **Vector search**: $0 (free local database)
- **AI analysis**: ~$0.01-0.02 per 20 papers (Claude API)
- **Total for testing**: ~$1-2

---

## Pipeline Architecture

### Complete Flow

```
RAW JSON FILES (20,216)
    ↓
PARSE & CLEAN
    ↓
STRUCTURED DATA (19,523 papers)
    ↓
GENERATE EMBEDDINGS (384-dim vectors)
    ↓
BUILD VECTOR DATABASE (ChromaDB)
    ↓
SEMANTIC SEARCH (user query)
    ↓
AI ANALYSIS (summaries + stance)
    ↓
INTERACTIVE DASHBOARD (visualize results)
```

### Four-Part Pipeline

**Part 1: Data Preparation** (4-5 hours)
- Parse 20,216 JSON files from Scopus API
- Clean and validate data
- Extract subject areas and create hierarchy
- Generate tree map data (pre-computed)
- Output: `papers.parquet` (19,523 papers)

**Part 2: Embeddings & Vector Search** (3-4 hours)
- Generate 384-dim embeddings for all papers
- Build ChromaDB vector database
- Implement semantic search
- Output: `paper_embeddings.npy` + ChromaDB files

**Part 3: Stance Detection & Network** (4-5 hours)
- Integrate Claude AI for analysis
- Implement stance detection
- Build similarity network
- Timeline and community analysis
- Output: Analysis notebooks

**Part 4: Interactive Dashboard** (5-6 hours)
- Build Streamlit web application
- Create tree map visualization
- Add timeline analysis
- Implement filters and export
- Output: Production-ready web app

---

## Part 1: Data Preparation

### Input
- **20,216 JSON files** from Scopus API
- Raw research paper data from Chulalongkorn University

### Process

**Step 1: Exploratory Data Analysis (EDA)**
```python
# Examine JSON structure
sample_file = 'data/raw/00001.json'
# Key fields: title, abstract, authors, year, citations, subject areas
```

**Step 2: Parse JSON Files**
```python
# Extract core fields
- scopus_id, doi, title, abstract
- authors, affiliations
- year, citation_count
- subject_areas (ASJC codes)
- references (for network analysis)
```

**Step 3: Clean & Validate**
```python
# Remove duplicates (check title + first author)
# Filter out papers without abstracts
# Validate year range (2018-2023)
# Handle missing values
# Clean text (remove extra whitespace, special chars)
```

**Step 4: Extract Subject Areas**
```python
# Parse ASJC classification codes
# Map to human-readable subject names
# Create 7-category hierarchy:
subject_groups = {
    'Medicine & Health': ['Medicine', 'Nursing', 'Health Professions', ...],
    'Life Sciences': ['Biochemistry', 'Molecular Biology', ...],
    'Computer Science & AI': ['Artificial Intelligence', 'Computer Science', ...],
    'Engineering': ['Engineering', 'Chemical Engineering', ...],
    'Materials & Chemistry': ['Chemistry', 'Materials Science', ...],
    'Physics': ['Physics and Astronomy'],
    'Environmental Science': ['Environmental Science', 'Earth Sciences', ...]
}
```

**Step 5: Generate Tree Map Data**
```python
# Pre-compute tree map structure
# Count papers per subject area
# Create hierarchical labels/parents/values
# Assign colors based on categories
# Save to treemap_data.json (6KB)
```

### Output Files

**1. papers.parquet** (29MB)
- 19,523 papers after cleaning
- Columns: id, scopus_id, doi, title, abstract, year, citation_count, authors, affiliations, references, subject_areas, abstract_length, num_authors, num_references

**2. subject_hierarchy.json** (3KB)
- 7 subject categories mapping
- 100+ unique subject areas
- Used for color coding

**3. treemap_data.json** (6KB)
- Pre-computed tree map visualization
- Hierarchical structure with labels, parents, values, colors
- Instant loading (saves 1-2 seconds per dashboard load)

**4. metadata.json**
- Dataset statistics
- Paper counts by year
- Subject distribution

### Key Statistics

| Metric | Value |
|--------|-------|
| Total papers | 19,523 |
| Year range | 2018-2023 |
| Avg citations | 9.2 |
| Papers with abstracts | 100% |
| Papers with references | 49.1% |
| Unique subjects | 100+ |
| Avg subjects per paper | 2.5 |

### Tools Used
- `pandas`, `json`, `pathlib`
- `tqdm` (progress bars)
- `matplotlib`, `seaborn` (visualization)

---

## Part 2: Embeddings & Vector Search

### Step 1: Text Embeddings

**What are Embeddings?**

Embeddings convert text into numerical vectors that capture semantic meaning:

```
"machine learning for medical diagnosis"
    ↓ (embedding model)
[0.23, -0.15, 0.87, ..., 0.45]  (384 numbers)

"AI helps doctors detect diseases"
    ↓ (embedding model)
[0.25, -0.12, 0.89, ..., 0.42]  (similar numbers!)
```

Papers with similar meanings have similar vector representations.

**Why Embeddings Over Keyword Search?**

Keyword search problem:
- Query: "AI helps doctors"
- Misses: "machine learning improves diagnosis" (same meaning, different words)

Embedding solution:
- Both texts convert to similar vectors
- Cosine similarity measures how close vectors are
- Finds papers by meaning, not just exact keywords

**Model: all-MiniLM-L6-v2**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Combine title + abstract for better context
papers_df['combined_text'] = papers_df['title'] + ' ' + papers_df['abstract']

# Generate embeddings (batch processing)
embeddings = model.encode(
    papers_df['combined_text'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

# Save to disk
np.save('data/embeddings/paper_embeddings.npy', embeddings)
```

**Why this model?**
- Free and runs locally
- Fast: 2-5 minutes for 19K papers
- Good quality for semantic search
- 384 dimensions (good balance of quality vs speed)
- Widely used and well-tested

**Alternatives considered:**
- `all-mpnet-base-v2`: Better quality but 2x slower
- OpenAI embeddings: Best quality but costs $2-3

### Step 2: Vector Database (ChromaDB)

**What is a Vector Database?**

Naive approach:
- Compare query vector to all 19,523 paper vectors
- Slow: ~2 seconds per search

Vector database approach:
- Pre-build index using HNSW algorithm
- Check only ~100-200 papers instead of all 19,523
- Fast: <0.1 seconds per search

**How HNSW Works**

Hierarchical Navigable Small World (HNSW) is like finding a restaurant:
- **Naive**: Check all 20,000 restaurants in the city
- **HNSW**: Start in your neighborhood, jump to nearby popular areas, narrow down
- Result: Check only ~100-200 instead of all 20,000

**Implementation**

```python
import chromadb
from chromadb.config import Settings

# Create persistent client (saves to disk)
client = chromadb.PersistentClient(path="data/vector_db")

# Create collection
collection = client.create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"}
)

# Add papers to database
collection.add(
    ids=[str(i) for i in papers_df['id']],
    embeddings=embeddings.tolist(),
    metadatas=papers_df.to_dict('records')
)
```

**Search Implementation**

```python
def semantic_search(query, n_results=20):
    # Embed query
    query_embedding = model.encode([query])[0]

    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results
```

**Cosine Similarity**

Measures angle between vectors:
- **1.0**: Same direction (very similar)
- **0.5**: 60° angle (somewhat related)
- **0.0**: Perpendicular (unrelated)
- **-1.0**: Opposite direction (very different)

Example:
```
Query: "machine learning for diagnosis"
Paper 1: "ML improves medical detection" → 0.92 (very relevant)
Paper 2: "Chemistry lab procedures" → 0.15 (not relevant)
```

**Why ChromaDB?**
- Simple Python API
- PersistentClient (no server needed)
- Perfect for 20K papers
- Works well with Streamlit deployment
- Free and open source

**Alternatives considered:**
- FAISS: Faster for millions of papers (overkill for 20K)
- Pinecone: $70/month (too expensive)
- Weaviate: Requires server setup (more complex)

### Output Files

**1. paper_embeddings.npy** (29MB)
- 19,523 × 384 dimensional array
- Float32 precision
- Fast loading with NumPy

**2. data/vector_db/** (~294MB)
- ChromaDB index files
- HNSW graph structure
- Metadata and IDs

**3. metadata.json**
- Embedding model name
- Dimensions
- Creation timestamp

### Tools Used
- `sentence-transformers`
- `chromadb`
- `numpy`
- `scikit-learn` (cosine similarity)

---

## Part 3: Stance Detection & Network Analysis

### AI Analysis with Claude 3.5 Haiku

**Why Claude?**
- Fast: 10-12 seconds for 20 papers
- Accurate: Better than keyword-based methods
- Cost-effective: ~$0.01 per search
- API: Simple integration with Anthropic SDK

**Two AI Tasks**

### Task 1: Generate Summaries

**Goal**: Create concise one-sentence summaries for each paper

**Prompt Design** (optimized for speed):
```python
prompt = f"""Summarize in one sentence:

{paper['abstract'][:800]}

Summary:"""
```

**Implementation**:
```python
import anthropic
import asyncio

async def generate_summary(paper, client):
    """Generate one-sentence summary (cached)"""
    response = await client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=80,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()

# Process all papers concurrently
async def generate_all_summaries(papers):
    client = anthropic.AsyncAnthropic(api_key=api_key)
    tasks = [generate_summary(p, client) for p in papers]
    summaries = await asyncio.gather(*tasks)
    return summaries
```

**Caching Strategy**:
```python
# Save summaries to JSON
with open('data/abstract_summaries.json', 'w') as f:
    json.dump(summary_cache, f)

# Reuse across searches (saves API calls)
if paper_id in summary_cache:
    summary = summary_cache[paper_id]
else:
    summary = await generate_summary(paper)
    summary_cache[paper_id] = summary
```

### Task 2: Stance Detection

**Goal**: Classify if paper supports, contradicts, or is neutral to user's query

**Examples**:
- Query: "Machine learning improves medical diagnosis"
- Paper A: "Our ML model achieved 95% accuracy" → **SUPPORT** ✓
- Paper B: "Automated systems performed worse than humans" → **CONTRADICT** ✗
- Paper C: "We analyzed 10,000 patient records" → **NEUTRAL** ○

**Prompt Design** (optimized):
```python
prompt = f"""Query: {query}

Abstract: {paper['abstract'][:600]}

Does this paper SUPPORT, CONTRADICT, or is NEUTRAL to the query? One word only."""
```

**Implementation**:
```python
async def detect_stance(paper, query, client):
    """Detect paper stance relative to query"""
    response = await client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=5,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    stance_text = response.content[0].text.strip().upper()

    # Parse response
    if 'SUPPORT' in stance_text or 'ENTAIL' in stance_text:
        return 'SUPPORT'
    elif 'CONTRADICT' in stance_text:
        return 'CONTRADICT'
    else:
        return 'NEUTRAL'
```

**Performance Optimization**:
- Shortened prompts by ~30% → 30-40% faster
- Reduced max_tokens (summary: 80, stance: 5)
- Async processing (all papers in parallel)
- Caching (reuse summaries across searches)

**Result**: 20 papers analyzed in 10-12 seconds (was 15-20s before optimization)

### Network Analysis

**Similarity Network**:
```python
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Build similarity graph
def build_similarity_network(papers, embeddings, threshold=0.7):
    G = nx.Graph()

    # Add nodes
    for i, paper in enumerate(papers):
        G.add_node(paper['id'], **paper)

    # Add edges based on similarity
    similarities = cosine_similarity(embeddings)
    for i in range(len(papers)):
        for j in range(i+1, len(papers)):
            if similarities[i][j] > threshold:
                G.add_edge(papers[i]['id'], papers[j]['id'],
                          weight=similarities[i][j])

    return G
```

**Community Detection**:
```python
from networkx.algorithms import community

# Detect research clusters
communities = community.greedy_modularity_communities(G)

# Assign colors to communities
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i
```

### Timeline Analysis

**Publication Trends**:
```python
# Papers over time
yearly_counts = papers_df.groupby('year').size()

# Stance distribution by year
stance_by_year = papers_df.groupby(['year', 'stance']).size()

# Citation patterns
citation_trends = papers_df.groupby('year')['citation_count'].mean()
```

### Tools Used
- `anthropic` (Claude API)
- `asyncio` (concurrent processing)
- `networkx` (graph analysis)
- `scikit-learn` (similarity metrics)

---

## Part 4: Interactive Dashboard

### Framework: Streamlit

**Why Streamlit?**
- Pure Python (no HTML/CSS/JavaScript needed)
- Fast development (2-3 days for full dashboard)
- Built-in components (search, sliders, tabs)
- Easy deployment
- Perfect for data science projects

### Dashboard Structure (715 lines)

**File**: `part4_dashboard/app.py`

**Three Main Tabs**:

### Tab 1: Home - Research Landscape

**Features**:
1. **Key Metrics** (4 cards at top)
   - Total papers: 19,523
   - Subject categories: 7
   - Year range: 2018-2023
   - Avg citations: 9.2

2. **Interactive Tree Map** (Plotly)
   ```python
   @st.cache_data
   def load_treemap_data():
       """Load pre-computed tree map (instant)"""
       with open('../data/processed/treemap_data.json', 'r') as f:
           return json.load(f)

   fig = go.Figure(go.Treemap(
       labels=data['labels'],
       parents=data['parents'],
       values=data['values'],
       marker_colors=data['colors']
   ))
   ```

3. **Top 20 Subject Areas Table**
   - Subject name
   - Paper count
   - Percentage of total

**Color Scheme**:
```python
CATEGORY_COLORS = {
    'Medicine & Health': '#E91E63',        # Pink
    'Life Sciences': '#4CAF50',            # Green
    'Computer Science & AI': '#2196F3',    # Blue
    'Engineering': '#FF9800',              # Orange
    'Materials & Chemistry': '#9C27B0',    # Purple
    'Physics': '#00BCD4',                  # Cyan
    'Environmental Science': '#8BC34A',    # Light Green
    'Other': '#9E9E9E'                     # Gray
}
```

### Tab 2: Timeline - Temporal Analysis

**Features**:
1. **Year Range Slider**
   ```python
   year_range = st.slider(
       "Select Year Range",
       min_value=2018,
       max_value=2023,
       value=(2018, 2023)
   )
   ```

2. **Papers Over Time** (Bar chart)
   - X-axis: Year
   - Y-axis: Paper count
   - Color: Blue gradient

3. **Stance Distribution** (Pie chart)
   - SUPPORT: Green (#10b981)
   - CONTRADICT: Red (#ef4444)
   - NEUTRAL: Gray (#6b7280)

4. **Stance by Year** (Stacked bar chart)
   - Shows how stance distribution changes over time
   - Helps identify research trends

### Tab 3: Papers - Search & Analysis

**Sidebar - Search Controls**:
```python
# API Key input
api_key = st.text_input("Anthropic API Key", type="password")

# Search query
query = st.text_area("Research Question", height=100)

# Number of papers
n_papers = st.slider("Number of papers", 3, 50, 20)

# Search button
if st.button("Search & Analyze"):
    # Perform search and AI analysis
```

**Main Panel - Results**:

1. **Search Summary**
   - Total papers found
   - Search relevance score
   - Processing time

2. **Filter Controls**
   ```python
   # Filter by stance
   stance_filter = st.multiselect(
       "Filter by Stance",
       ['SUPPORT', 'CONTRADICT', 'NEUTRAL']
   )

   # Filter by subject
   subject_filter = st.selectbox(
       "Filter by Subject",
       ['All'] + list(subject_areas)
   )

   # Sort options
   sort_by = st.selectbox(
       "Sort by",
       ['Relevance', 'Citations', 'Year']
   )
   ```

3. **Paper Cards** (color-coded)
   ```python
   def show_paper_card(row, subject_groups):
       stance = row['stance']
       emoji = {'SUPPORT': '✓', 'CONTRADICT': '✗', 'NEUTRAL': '○'}[stance]

       with st.expander(f"**{emoji} {stance}** | {row['title']}", expanded=False):
           # Summary
           st.markdown(f"*{row['summary']}*")

           # Color-coded subject tags
           if 'subject_areas' in row:
               subjects = row['subject_areas'][:5]
               for subject in subjects:
                   bg_color = get_subject_color(subject, subject_groups)
                   # Display tag with background color

           # Metadata
           col1, col2, col3 = st.columns(3)
           col1.metric("Year", row['year'])
           col2.metric("Citations", row['citation_count'])
           col3.metric("Relevance", f"{row['relevance']:.1%}")

           # Abstract
           with st.expander("Full Abstract"):
               st.write(row['abstract'])
   ```

4. **Export to CSV**
   ```python
   csv = results_df.to_csv(index=False)
   st.download_button(
       "Download Results (CSV)",
       csv,
       "search_results.csv",
       "text/csv"
   )
   ```

### UI/UX Design

**Modern CSS** (blue/gray palette):
```python
st.markdown("""
<style>
    /* Typography */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #f8fafc;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }

    /* Cards */
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)
```

**Stance Badges**:
```python
STANCE_COLORS = {
    'SUPPORT': '#10b981',      # Green
    'CONTRADICT': '#ef4444',   # Red
    'NEUTRAL': '#6b7280'       # Gray
}

STANCE_EMOJIS = {
    'SUPPORT': '✓',
    'CONTRADICT': '✗',
    'NEUTRAL': '○'
}
```

### Caching Strategy

**1. Data Loading** (survives app reruns):
```python
@st.cache_data
def load_papers():
    return pd.read_parquet('../data/processed/papers.parquet')

@st.cache_data
def load_embeddings():
    return np.load('../data/embeddings/paper_embeddings.npy')

@st.cache_data
def load_treemap_data():
    with open('../data/processed/treemap_data.json', 'r') as f:
        return json.load(f)
```

**2. Model Loading** (survives app reruns, only one instance):
```python
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="../data/vector_db")
```

**3. Summary Caching** (persistent across sessions):
```python
# Load cache from disk
with open('../data/abstract_summaries.json', 'r') as f:
    summary_cache = json.load(f)

# Use cached summary if available
if paper_id in summary_cache:
    summary = summary_cache[paper_id]
else:
    summary = await generate_summary(paper)
    summary_cache[paper_id] = summary
    # Save back to disk
    with open('../data/abstract_summaries.json', 'w') as f:
        json.dump(summary_cache, f)
```

### Tools Used
- `streamlit`
- `plotly`
- `pandas`
- `asyncio` (for concurrent AI calls)

---

## Data Files Reference

### Essential Files (keep these)

**Core Data** (~352MB total):

1. **papers.parquet** (29MB)
   - Main dataset with 19,523 papers
   - Columns: id, scopus_id, doi, title, abstract, year, citation_count, authors, affiliations, references, subject_areas
   - Format: Parquet (fast loading, preserves types)

2. **paper_embeddings.npy** (29MB)
   - 19,523 × 384 dimensional array
   - Sentence transformer embeddings
   - Used for semantic search

3. **vector_db/** (~294MB)
   - ChromaDB index and files
   - HNSW graph structure
   - Enables fast similarity search

4. **subject_hierarchy.json** (3KB)
   - 7 subject category mapping
   - Used for color coding and filtering

5. **treemap_data.json** (6KB)
   - Pre-computed tree map structure
   - Instant loading (saves 1-2s)

6. **abstract_summaries.json** (24KB)
   - Cached AI summaries
   - Reused across searches
   - Saves API costs

7. **metadata.json** (few KB)
   - Dataset statistics
   - Embedding model info
   - Creation timestamps

### File Locations

```
data/
├── abstract_summaries.json          # AI summaries cache (24KB)
├── embeddings/
│   ├── paper_embeddings.npy         # Vector embeddings (29MB)
│   └── metadata.json                # Embedding metadata
├── processed/
│   ├── papers.parquet               # Main dataset (29MB)
│   ├── subject_hierarchy.json       # Category mapping (3KB)
│   ├── treemap_data.json           # Pre-computed tree map (6KB)
│   └── metadata.json                # Dataset statistics
└── vector_db/                       # ChromaDB files (294MB)
```

**Total**: ~352MB (after cleanup from original ~516MB)

---

## Performance Optimizations

### 1. Tree Map Pre-computation

**Problem**: Tree map computed on every dashboard load (1-2 seconds)

**Solution**: Pre-compute in Part 1, save to JSON, load instantly

**Implementation**:
```python
# In Part 1 notebook
treemap_data = build_treemap_data(papers_df)
with open('data/processed/treemap_data.json', 'w') as f:
    json.dump(treemap_data, f)

# In dashboard
@st.cache_data
def load_treemap_data():
    with open('../data/processed/treemap_data.json', 'r') as f:
        return json.load(f)
```

**Result**: Instant loading (was 1-2s)

### 2. LLM Prompt Optimization

**Problem**: AI analysis taking 15-20 seconds for 20 papers

**Changes**:
- Shortened prompts by ~30%
- Reduced max_tokens (summary: 100→80, stance: 10→5)
- More direct prompt format
- Truncate abstracts (summary: 800 chars, stance: 600 chars)

**Before**:
```python
prompt = f"""You are a research assistant. Please read the following abstract
and provide a comprehensive summary that captures the key points...

Abstract: {abstract}

Please provide a one-sentence summary:"""

max_tokens = 100
```

**After**:
```python
prompt = f"""Summarize in one sentence:

{abstract[:800]}

Summary:"""

max_tokens = 80
```

**Result**: 30-40% faster (now 10-12 seconds for 20 papers)

### 3. Summary Caching

**Problem**: Regenerating summaries for same papers across searches

**Solution**: Save summaries to JSON file, reuse across sessions

**Implementation**:
```python
# Load cache
if os.path.exists('data/abstract_summaries.json'):
    with open('data/abstract_summaries.json', 'r') as f:
        summary_cache = json.load(f)
else:
    summary_cache = {}

# Check cache before generating
if paper_id not in summary_cache:
    summary = await generate_summary(paper)
    summary_cache[paper_id] = summary
    # Save immediately
    with open('data/abstract_summaries.json', 'w') as f:
        json.dump(summary_cache, f)
```

**Result**: Instant summaries for previously analyzed papers

### 4. Streamlit Caching

**Data Caching** (survives reruns):
```python
@st.cache_data
def load_papers():
    return pd.read_parquet('../data/processed/papers.parquet')
```

**Resource Caching** (only one instance):
```python
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
```

**Result**: Dashboard loads in ~1 second (was ~3 seconds)

### Performance Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Tree map load | 1-2s | Instant | 100% faster |
| LLM analysis (20 papers) | 15-20s | 10-12s | 30-40% faster |
| Dashboard load | ~3s | ~1s | 66% faster |
| Code size | 930 lines | 715 lines | 23% smaller |
| Data folder | 516MB | 352MB | 164MB saved |

---

## Deployment Guide

### Local Development

**1. Install Dependencies**:
```bash
cd part4_dashboard
pip install -r requirements.txt
```

**2. Set API Key**:
```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY="your-key-here"

# Option 2: Enter in dashboard sidebar
```

**3. Run Dashboard**:
```bash
streamlit run app.py
```

**4. Open Browser**:
```
http://localhost:8501
```

### Streamlit Cloud Deployment (Free)

**1. Push to GitHub**:
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

**2. Deploy on Streamlit Cloud**:
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Click "New app"
- Select repository: `yourusername/Datasci_Project`
- Main file path: `part4_dashboard/app.py`
- Click "Deploy"

**3. Configure Secrets**:
- In Streamlit Cloud dashboard, go to Settings → Secrets
- Add:
```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

**4. Access Your App**:
```
https://your-app-name.streamlit.app
```

### Deployment Checklist

- [ ] All data files in correct locations
- [ ] requirements.txt up to date
- [ ] API keys in secrets (not in code)
- [ ] .gitignore configured (exclude API keys)
- [ ] ChromaDB files included in repo
- [ ] Streamlit caching configured
- [ ] App tested locally
- [ ] README updated

### Troubleshooting

**Issue**: ChromaDB not found after deploy
- **Solution**: Ensure `data/vector_db/` is not in `.gitignore`

**Issue**: App times out on load
- **Solution**: Use `@st.cache_resource` for model loading

**Issue**: Out of memory
- **Solution**: Use batch processing, limit number of papers

---

## Common Issues & Solutions

**Issue: "No module named 'sentence_transformers'"**
- Solution: `pip install sentence-transformers`

**Issue: "API key not found"**
- Solution: Set `ANTHROPIC_API_KEY` environment variable or enter in sidebar

**Issue: "ChromaDB collection not found"**
- Solution: Run Part 2 notebook to build vector database

**Issue: Dashboard loads slowly**
- Solution: Ensure caching is working (`@st.cache_data`, `@st.cache_resource`)

**Issue: Tree map not displaying**
- Solution: Check that `treemap_data.json` exists in `data/processed/`

**Issue: AI analysis fails**
- Solution: Check API key, check internet connection, check API quota

---

## Educational Value

### Learning Objectives

**Part 1: Data Engineering**
- Parse large JSON datasets
- Clean and validate data
- Handle missing values
- Export to efficient formats (Parquet)
- Create data hierarchies

**Part 2: NLP & Vector Search**
- Generate text embeddings
- Build vector databases
- Implement semantic search
- Understand similarity metrics

**Part 3: AI Integration**
- Use LLM APIs (Anthropic Claude)
- Implement stance detection
- Build similarity networks
- Analyze research communities

**Part 4: Full-Stack Development**
- Build interactive dashboards
- Create visualizations
- Optimize performance
- Deploy web applications

### Skills Demonstrated

**Technical Skills**:
- Python programming (advanced)
- Data processing (Pandas, NumPy)
- Machine learning (embeddings, transformers)
- API integration (Anthropic Claude)
- Web development (Streamlit)
- Database systems (ChromaDB)
- Graph analysis (NetworkX)

**Soft Skills**:
- Problem decomposition
- Performance optimization
- Documentation writing
- Code organization
- Project management

---

## References & Resources

### Documentation
- **Sentence Transformers**: https://www.sbert.net/
- **ChromaDB**: https://docs.trychroma.com/
- **Anthropic Claude**: https://docs.anthropic.com/
- **Streamlit**: https://docs.streamlit.io/
- **Plotly**: https://plotly.com/python/
- **NetworkX**: https://networkx.org/

### Similar Projects
- **Connected Papers**: https://www.connectedpapers.com/
- **Semantic Scholar**: https://www.semanticscholar.org/
- **Research Rabbit**: https://www.researchrabbit.ai/

### Academic Papers
- SBERT: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- HNSW: "Efficient and robust approximate nearest neighbor search"
- Transformers: "Attention Is All You Need"

---

## Project Timeline

**Day 1-2: Data Preparation** (8-10 hours)
- Parse JSON files
- Clean and validate
- Extract subject areas
- Create hierarchy
- Generate tree map data

**Day 3-4: Embeddings & Search** (6-8 hours)
- Generate embeddings
- Build vector database
- Test semantic search
- Optimize performance

**Day 5-6: AI & Network** (8-10 hours)
- Integrate Claude API
- Implement stance detection
- Build similarity network
- Timeline analysis

**Day 7-8: Dashboard** (8-10 hours)
- Build Streamlit app
- Create visualizations
- Add filters and export
- Polish UI/UX

**Day 9: Deploy & Document** (4-6 hours)
- Deploy to Streamlit Cloud
- Write documentation
- Final testing
- Prepare presentation

**Total**: 35-45 hours over 9 days

---

## Success Metrics

### Must Have ✅
- [x] All 19,523 papers parsed and searchable
- [x] Semantic search working (<1 second)
- [x] AI summaries and stance detection (10-12 seconds)
- [x] Interactive tree map visualization
- [x] Timeline analysis with charts
- [x] Color-coded subject classification
- [x] Export to CSV

### Achieved ✅
- [x] Pre-computed tree map (instant load)
- [x] Summary caching (persistent)
- [x] Optimized LLM prompts (30-40% faster)
- [x] Modern UI/UX design
- [x] Multi-filter system
- [x] Clean codebase (715 lines)
- [x] Comprehensive documentation

---

## Conclusion

This project demonstrates a complete AI-powered research assistant pipeline:

**Data Engineering**: Parsed 20K+ JSON files → cleaned dataset → efficient storage

**AI/ML**: Embeddings → vector search → LLM analysis → stance detection

**Visualization**: Tree map → timeline → subject classification → color coding

**Web Development**: Interactive dashboard → real-time search → export capabilities

**Production Ready**: Optimized performance → caching → deployment guide

**Educational**: Clear structure → comprehensive docs → reproducible workflow

The system successfully combines modern NLP, vector databases, LLM APIs, and interactive visualization to solve a real research problem.

---

**Built with ❤️ for the research community**

*Data Science Course Project - Chulalongkorn University 2024*
