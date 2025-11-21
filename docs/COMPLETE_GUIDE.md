# 📚 Complete Project Guide - Literature Review Assistant

**This document contains EVERYTHING about the project. Read this to understand every detail.**

---

## 📋 TABLE OF CONTENTS

1. [Quick Summary](#quick-summary)
2. [What We're Building](#what-were-building)
3. [Complete Pipeline](#complete-pipeline)
4. [Tech Stack Decisions](#tech-stack-decisions)
5. [Implementation Details](#implementation-details)
6. [Deployment Guide](#deployment-guide)
7. [Timeline](#timeline)

---

## 🎯 QUICK SUMMARY

### Project Goal
Build an AI-powered literature review assistant that searches 20,000 research papers and visualizes results as an interactive graph.

### Key Features
1. **Semantic Search** - Search by meaning, not keywords
2. **Interactive Graph** - Papers as nodes (size = citations), edges = citations/stance
3. **Relevance Scoring** - Color gradient (red = very relevant → blue = less)
4. **Stance Detection** - Does paper support/contradict your hypothesis?
5. **AI Summaries** - Context-aware summaries for your specific query
6. **Report Generation** - Combine multiple papers into one document

### Timeline
**1 week** (30-35 hours total)

### Cost
**$0-2** (free with Ollama, or $2 with OpenAI for better quality)

### Users
**5 testers** (prototype/demo)

---

## 🔬 WHAT WE'RE BUILDING

### The Problem
**Literature review is painful:**
- Takes 10+ hours to read papers manually
- Hard to find relevant papers (keyword search misses synonyms)
- Can't see connections between papers
- Don't know which papers support/contradict your hypothesis

### Our Solution
**AI-powered search + visualization:**

```
User enters: "Does machine learning improve medical diagnosis?"
                           ↓
System finds 50 relevant papers in 3 seconds
                           ↓
Shows interactive graph:
  - Bigger nodes = more cited papers
  - Red nodes = very relevant to query
  - Green edges = paper supports your idea
  - Red edges = paper contradicts
                           ↓
User clicks paper → Get AI summary in context of their query
                           ↓
User selects 10 papers → Generate combined literature review report
```

### Real Impact
- Saves researchers 10+ hours per project
- Helps find contradicting evidence (important for research)
- Visualizes research landscape
- Actually useful (not just academic exercise)

---

## 🗺️ COMPLETE PIPELINE

### Overview

```
RAW DATA (JSON) → PARSE → CLEAN → EMBEDDINGS → VECTOR DB → SEARCH → ANALYZE → VISUALIZE
```

### Phase-by-Phase Breakdown

---

### **PHASE 1: Data Inspection & Understanding** (4 hours)

**What**: Understand what data we have

**Input**: 20,216 JSON files (2018-2023)

**Process**:
1. Sample random files from each year
2. Inspect structure (what fields exist?)
3. Check data quality (missing values? errors?)
4. Calculate statistics (papers per year, citation distribution)

**Output**:
- Understanding of data structure
- Statistics report
- List of issues to fix

**Tools**: Python, pandas, json

**Code Location**: Already done (see archive folder)

---

### **PHASE 2: Data Cleaning & Preparation** (6 hours)

**What**: Parse all JSON files into clean, structured format

#### **Decision Point: Output Format**

| Option | Pros | Cons | Choice |
|--------|------|------|--------|
| **CSV** | Simple, easy to inspect, works in Excel | Takes disk space (~500MB) | ✅ CHOSEN |
| SQLite DB | Better for queries, normalized | More complex, overkill for 20K | ❌ |
| Keep JSON | No preprocessing | Very slow for searches | ❌ |

**Why CSV?** Simple, fast, easy to debug. Perfect for prototype.

#### **Implementation**

**Input**:
```
raw_data/
  2018/201800001  (JSON file)
  2018/201800002  (JSON file)
  ...
```

**Process**:
```python
# src/data/parser.py

For each JSON file:
  1. Read file
  2. Extract: title, abstract, authors, year, citations, references
  3. Handle missing data:
     - No abstract? Skip paper (can't search without text)
     - No citations? Set to 0
     - Missing year? Try to infer or skip
  4. Normalize:
     - Convert year to integer
     - Remove special characters in text
     - Deduplicate (same paper might appear twice)
```

**Output**:
```
data/processed/papers.csv

Columns:
- id: Unique identifier
- title: Paper title
- abstract: Full abstract text
- year: Publication year (2018-2023)
- citation_count: Number of times cited
- authors: Semicolon-separated author names
- affiliations: Semicolon-separated institutions
- references: Semicolon-separated reference IDs
```

**Data Cleaning Rules**:
- Remove papers without abstracts (can't search)
- Remove duplicates (same DOI/title)
- Remove papers with year < 2018 or > 2023
- Final count: ~19,000 papers (from 20,216)

**Tools**: pandas, json, tqdm (progress bar)

**Time**: ~20 minutes to parse all files

---

### **PHASE 3: Embedding Creation** (3 hours)

**What**: Convert text to numerical vectors (embeddings)

#### **Why Embeddings?**

**Problem with keyword search:**
```
Query: "AI helps doctors"
Keyword search finds: Papers with words "AI", "helps", "doctors"
Misses: "Machine learning improves diagnosis" (same meaning, different words!)
```

**Solution with embeddings:**
```
"AI helps doctors" → [0.23, -0.41, 0.88, ..., 0.56] (384 numbers)
"ML improves diagnosis" → [0.25, -0.38, 0.85, ..., 0.54] (very similar numbers!)

Search = find papers with similar numbers (cosine similarity)
```

#### **Decision Point: Embedding Model**

| Model | Dimensions | Speed | Quality | Cost | Choice |
|-------|-----------|-------|---------|------|--------|
| **all-MiniLM-L6-v2** | 384 | Fast (2 min) | Good | Free | ✅ CHOSEN |
| all-mpnet-base-v2 | 768 | Medium (5 min) | Better | Free | ❌ |
| OpenAI text-embedding-3-small | 1536 | Fast | Best | $2 for 20K | ❌ |

**Why all-MiniLM-L6-v2?**
- Free, runs locally
- Fast enough (2 minutes for 20K papers)
- Good quality (fine for prototype)
- Can upgrade later if needed

#### **Implementation**

```python
# src/data/embeddings.py

from sentence_transformers import SentenceTransformer

# Load model (downloads once, ~80MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# For each paper:
for paper in papers:
    # Combine title + abstract
    text = f"{paper['title']} {paper['abstract']}"

    # Convert to vector (384 numbers)
    embedding = model.encode(text)

    # embedding = [0.234, -0.412, 0.881, ..., 0.567]

    # Save
    embeddings.append(embedding)

# Save all embeddings
np.save('data/embeddings/paper_embeddings.npy', embeddings)
```

**Output**:
```
data/embeddings/paper_embeddings.npy

Shape: (19000, 384)
- 19,000 papers
- Each has 384-dimensional vector
- File size: ~30MB
```

**Tools**: sentence-transformers, numpy

**Time**: ~2-5 minutes

---

### **PHASE 4: Vector Database Setup** (2 hours)

**What**: Store embeddings in searchable database

#### **Why Vector Database?**

**Naive approach (slow):**
```python
# For each query:
for paper in 20000_papers:
    similarity = cosine_similarity(query_vector, paper_vector)
# 20,000 calculations = 2 seconds per search ❌ Slow!
```

**With Vector DB (fast):**
```python
# Pre-built index (like a map)
# Smart algorithms skip irrelevant areas
results = db.search(query_vector, top_k=50)
# 0.03 seconds ✅ Fast!
```

#### **Decision Point: Vector Database**

| Database | Speed | Setup | Cost | Scale | Choice |
|----------|-------|-------|------|-------|--------|
| **ChromaDB** | Fast | Easy | Free | Good for <100K | ✅ CHOSEN |
| FAISS | Very fast | Medium | Free | Good for millions | ❌ |
| Pinecone | Fast | Easy | $70/month | Unlimited | ❌ |

**Why ChromaDB?**
- Free, runs locally
- Simple API (3 lines of code)
- Fast enough for 20K papers (0.03 sec/search)
- Persistent (saves to disk, don't need to reload)
- Perfect for prototype

#### **How Vector Search Works**

**Cosine Similarity:**
```
Think of vectors as arrows in space:
- Same direction = Similar (score: 1.0)
- Perpendicular = Unrelated (score: 0.0)
- Opposite = Very different (score: -1.0)

Query: [0.8, 0.9, ...]
Paper A: [0.85, 0.92, ...] → Similarity = 0.95 (very similar!)
Paper B: [0.1, 0.2, ...]  → Similarity = 0.12 (not similar)
```

**HNSW Algorithm (How ChromaDB is fast):**
```
Like finding a restaurant:

Naive way:
- Check ALL 20,000 restaurants in city
- Very slow

HNSW way:
- Start in your neighborhood
- Jump to nearby popular areas (highways)
- Check only ~100 restaurants
- Much faster!
```

#### **Implementation**

```python
# src/search/load_vector_db.py

import chromadb
from sentence_transformers import SentenceTransformer

# Load embeddings
embeddings = np.load('data/embeddings/paper_embeddings.npy')
papers = pd.read_csv('data/processed/papers.csv')

# Create ChromaDB client (saves to disk)
client = chromadb.PersistentClient(path="data/vector_db")

# Create collection
collection = client.create_collection(
    name="research_papers",
    metadata={"description": "CU research papers 2018-2023"}
)

# Add papers (batch for speed)
collection.add(
    ids=[str(i) for i in range(len(papers))],
    embeddings=embeddings.tolist(),
    documents=papers['abstract'].tolist(),
    metadatas=papers[['title', 'year', 'citation_count']].to_dict('records')
)

print("✅ Loaded 19,000 papers into ChromaDB")
```

**Output**:
```
data/vector_db/
  chroma.sqlite3  (database file)
  (other ChromaDB files)
```

**Tools**: chromadb, sentence-transformers

**Time**: ~2 minutes to load

---

### **PHASE 5: Search & Relevance** (3 hours)

**What**: When user searches, find relevant papers and score them

#### **How Search Works**

```python
# User query
query = "Does machine learning improve medical diagnosis?"

# 1. Convert query to vector
model = SentenceTransformer('all-MiniLM-L6-v2')
query_vector = model.encode(query)  # [0.82, 0.91, ...]

# 2. Search in ChromaDB
results = collection.query(
    query_embeddings=[query_vector],
    n_results=50  # Get top 50 most similar
)

# 3. Results include similarity scores
for i, paper in enumerate(results['documents'][0]):
    distance = results['distances'][0][i]  # 0.0 (identical) to 2.0 (opposite)

    # Convert to 0-100 relevance score
    relevance_score = (2.0 - distance) / 2.0 * 100

    # Assign color based on score
    if relevance_score > 80:
        color = '#FF0000'  # Red (very relevant)
    elif relevance_score > 60:
        color = '#FFA500'  # Orange
    elif relevance_score > 40:
        color = '#FFFF00'  # Yellow
    else:
        color = '#0000FF'  # Blue (less relevant)
```

#### **No LLM Needed for Relevance!**

Just use the similarity score from ChromaDB directly. It's already meaningful:
- 90-100% = Very relevant (same topic, directly answers query)
- 70-90% = Related (same field, adjacent topic)
- 50-70% = Somewhat related (broader context)
- <50% = Less relevant (different topic)

**Tools**: chromadb, sentence-transformers

**Time**: 0.03 seconds per search

---

### **PHASE 6: Stance Detection** (4 hours)

**What**: Determine if paper supports/contradicts user's hypothesis

#### **What is Stance?**

```
User query: "Machine learning improves medical diagnosis"

Paper A: "Our ML model achieved 95% accuracy in cancer detection"
→ STANCE: SUPPORTS (entailment) → Green edge

Paper B: "Automated systems performed worse than human doctors"
→ STANCE: CONTRADICTS (contradiction) → Red edge

Paper C: "We analyzed 10,000 patient records"
→ STANCE: NEUTRAL (no clear support/contradict) → Gray edge
```

#### **Decision Point: Stance Detection Method**

| Method | Accuracy | Speed | Cost | Reasoning | Choice |
|--------|----------|-------|------|-----------|--------|
| **NLI Model (DeBERTa)** | 85% | Fast (5 sec/50 papers) | Free | Yes | ✅ CHOSEN |
| GPT-4o-mini batch | 90% | Medium (10 sec/50) | $0.01/search | Yes | ❌ |
| Simple keywords | 40% | Instant | Free | No | ❌ |

**Why NLI Model (DeBERTa)?**
- Free, runs locally
- Fast enough (0.1 sec per paper, or 5 sec for 50 papers)
- Good accuracy (85%)
- Can process all 50 papers (not just top 10)
- Trained specifically for entailment/contradiction detection

#### **How NLI Works**

**NLI = Natural Language Inference**

```python
# NLI model checks relationship between two texts

Premise (user query): "AI improves healthcare"
Hypothesis (paper): "Our AI model achieved 95% accuracy"

NLI Model outputs:
- ENTAILMENT (0.89 confidence) ✅ Paper supports query
- CONTRADICTION (0.05)
- NEUTRAL (0.06)

Result: ENTAILMENT → Green edge
```

#### **Implementation**

```python
# src/search/stance_detection.py

from transformers import pipeline

# Load NLI model (downloads once, ~500MB)
nli = pipeline(
    "text-classification",
    model="microsoft/deberta-v3-base-mnli"
)

# For each paper in results:
for paper in top_50_papers:
    # Create input (query [SEP] paper abstract)
    input_text = f"{query} [SEP] {paper['abstract']}"

    # Get stance
    result = nli(input_text)

    # result = {'label': 'ENTAILMENT', 'score': 0.89}

    if result['label'] == 'ENTAILMENT':
        paper['stance'] = 'supports'
        paper['edge_color'] = '#00FF00'  # Green
    elif result['label'] == 'CONTRADICTION':
        paper['stance'] = 'contradicts'
        paper['edge_color'] = '#FF0000'  # Red
    else:  # NEUTRAL
        paper['stance'] = 'neutral'
        paper['edge_color'] = '#808080'  # Gray
```

**Optimization (if slow on CPU):**
```python
# Option 1: Only analyze top 20 papers (not all 50)
top_papers = results[:20]

# Option 2: Use GPU if available
device = 0 if torch.cuda.is_available() else -1
nli = pipeline(..., device=device)

# Option 3: Batch processing
results = nli([f"{query} [SEP] {p['abstract']}" for p in papers])
```

**Tools**: transformers (HuggingFace), torch

**Time**: 5 seconds for 50 papers (CPU), or 1 second (GPU)

---

### **PHASE 7: Citation Network** (3 hours)

**What**: Build graph showing how papers cite each other

#### **Graph Structure**

```python
# Nodes = Papers
Node attributes:
  - id: Paper ID
  - title: Paper title
  - size: citation_count * 10 (for visualization)
  - color: relevance_color (from Phase 5)
  - relevance_score: 0-100

# Edges = Citations OR Stance
Edge type 1 - Citations:
  - From: Paper A
  - To: Paper B (if A cites B)
  - Color: Gray

Edge type 2 - Stance (to query):
  - From: Paper
  - To: User query (virtual node)
  - Color: stance_color (green/red/gray)
```

#### **Implementation**

```python
# src/graph/builder.py

import networkx as nx

# Create directed graph
G = nx.DiGraph()

# Add nodes (papers)
for paper in search_results:
    G.add_node(
        paper['id'],
        title=paper['title'],
        abstract=paper['abstract'],
        year=paper['year'],
        citations=paper['citation_count'],
        relevance_score=paper['relevance_score'],
        stance=paper['stance'],
        # Visual properties
        size=max(10, paper['citation_count'] * 10),  # Min size 10
        color=paper['color'],  # Relevance color
        border_color=paper['edge_color']  # Stance color as border
    )

# Add edges (citations between papers)
for paper in search_results:
    # Check if this paper cites any other papers in results
    for ref_id in paper['references']:
        if ref_id in result_ids:
            G.add_edge(
                paper['id'],
                ref_id,
                color='#808080',  # Gray for citations
                weight=1
            )

# Network stats
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Connected components: {nx.number_connected_components(G.to_undirected())}")
```

**Output**: NetworkX graph object

**Tools**: networkx

**Time**: Instant (<1 second)

---

### **PHASE 8: Summarization** (2 hours)

**What**: When user clicks paper, generate context-aware summary

#### **Decision Point: LLM Choice**

| LLM | Quality | Speed | Cost | Choice |
|-----|---------|-------|------|--------|
| **GPT-4o-mini** | Excellent | Fast (2 sec) | $0.001/summary | ✅ RECOMMENDED |
| Ollama (Llama 3.1) | Good | Slow (5-10 sec) | Free | ✅ FREE OPTION |
| GPT-4o | Best | Fast | $0.01/summary | ❌ Expensive |

**Recommendation**: Use GPT-4o-mini
- Only $0.001 per summary (~$1-2 total for testing)
- Fast (2 seconds)
- High quality
- Much better than free options

**Fallback**: If $0 budget, use Ollama (free but slower)

#### **Why Context-Aware?**

**Bad approach (generic summary):**
```
"This paper discusses machine learning applications in healthcare..."
(Doesn't relate to user's specific question)
```

**Good approach (context-aware):**
```
User query: "Does ML improve diagnosis accuracy?"

Summary: "This paper DIRECTLY SUPPORTS your hypothesis.
The authors achieved 95% accuracy using CNNs for lung cancer
detection, improving upon the 87% baseline of human radiologists.
Key finding: Deep learning models can identify subtle patterns
invisible to human experts."
```

#### **Implementation**

```python
# src/llm/summarizer.py

from openai import OpenAI

client = OpenAI(api_key="sk-...")

def summarize_paper(query, paper):
    """Generate context-aware summary"""

    prompt = f"""You are a research assistant helping with literature review.

USER'S RESEARCH QUESTION:
{query}

PAPER DETAILS:
Title: {paper['title']}
Abstract: {paper['abstract']}
Year: {paper['year']}
Citations: {paper['citation_count']}

TASK:
Summarize this paper in relation to the user's research question.
- Does it support, contradict, or provide neutral context?
- What are the key findings?
- What methodology was used?
- How does it relate to the user's question?

Keep summary under 150 words, focused and actionable.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content
```

**Cost Calculation:**
```
GPT-4o-mini pricing: $0.15 per 1M input tokens, $0.60 per 1M output tokens

Average summary:
- Input: ~300 tokens (prompt + paper)
- Output: ~150 tokens (summary)
- Cost per summary: ~$0.0001 (basically free)

For 5 users × 100 searches × 10 summaries:
= 5,000 summaries
= 5,000 × $0.0001 = $0.50

Realistic usage: ~$1-2 total
```

**Tools**: openai, python-dotenv (for API key)

**Time**: 2-3 seconds per summary

---

### **PHASE 9: Dashboard** (10 hours)

**What**: Build interactive web interface

#### **Decision Point: Framework**

| Framework | Speed to Build | Flexibility | Learning Curve | Choice |
|-----------|---------------|-------------|----------------|--------|
| **Streamlit** | Fast (2 days) | Medium | Easy | ✅ CHOSEN |
| Plotly Dash | Medium (3 days) | High | Medium | ❌ |
| React + FastAPI | Slow (1 week) | Very High | Hard | ❌ |

**Why Streamlit?**
- Fastest to build (pure Python, no HTML/CSS/JS)
- Perfect for prototypes and demos
- Built-in components (sliders, buttons, etc.)
- Easy deployment (Streamlit Cloud free)
- Good enough for 5 users

#### **Dashboard Features**

**Page Layout:**
```
┌─────────────────────────────────────────┐
│  Literature Review Assistant            │
├─────────────────────────────────────────┤
│  [Search Box: Enter research question]  │
│  [Search Button]                        │
├─────────────────────────────────────────┤
│  Results: 50 papers found               │
│  Relevance: 🔴 10 🟠 20 🟡 15 🔵 5      │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐   │
│  │   INTERACTIVE GRAPH             │   │
│  │   (Pyvis network)               │   │
│  │   - Hover: show title           │   │
│  │   - Click: show summary panel   │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│  Selected Paper Details:                │
│  Title: ...                             │
│  Abstract: ...                          │
│  [Summarize Button] [Add to Report]    │
└─────────────────────────────────────────┘
```

#### **Implementation**

```python
# dashboard/app.py

import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

# Configure page
st.set_page_config(
    page_title="Literature Review Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Literature Review Assistant")

# Search interface
query = st.text_input(
    "Enter your research question:",
    placeholder="e.g., Does machine learning improve medical diagnosis?"
)

if st.button("Search", type="primary"):
    with st.spinner("Searching 20,000 papers..."):
        # 1. Search
        results = search(query)  # From Phase 5

        # 2. Detect stance
        results = detect_stance(query, results)  # From Phase 6

        # 3. Build graph
        G = build_graph(results)  # From Phase 7

        # Store in session state
        st.session_state.results = results
        st.session_state.graph = G

# Display results
if 'results' in st.session_state:
    results = st.session_state.results

    # Stats
    st.write(f"**Found {len(results)} relevant papers**")

    # Relevance distribution
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Very Relevant 🔴", sum(1 for r in results if r['score'] > 80))
    col2.metric("Related 🟠", sum(1 for r in results if 60 < r['score'] <= 80))
    col3.metric("Somewhat 🟡", sum(1 for r in results if 40 < r['score'] <= 60))
    col4.metric("Less 🔵", sum(1 for r in results if r['score'] <= 40))

    # Interactive graph
    st.subheader("Citation Network")

    # Build Pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes
    for node_id, data in st.session_state.graph.nodes(data=True):
        net.add_node(
            node_id,
            label=data['title'][:50],  # Truncate long titles
            size=data['size'],
            color=data['color'],
            title=f"{data['title']}<br>Citations: {data['citations']}<br>Relevance: {data['relevance_score']:.0f}%"
        )

    # Add edges
    for source, target, data in st.session_state.graph.edges(data=True):
        net.add_edge(source, target, color=data['color'])

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "springLength": 100
            },
            "solver": "forceAtlas2Based"
        }
    }
    """)

    # Save and display
    net.save_graph("temp_graph.html")
    with open("temp_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=650)

    # Paper selection
    st.subheader("Paper Details")
    selected_paper = st.selectbox(
        "Select a paper to view details:",
        options=range(len(results)),
        format_func=lambda i: results[i]['title']
    )

    paper = results[selected_paper]

    # Display details
    st.write(f"**Title:** {paper['title']}")
    st.write(f"**Year:** {paper['year']} | **Citations:** {paper['citation_count']}")
    st.write(f"**Relevance:** {paper['relevance_score']:.0f}% | **Stance:** {paper['stance']}")

    with st.expander("Abstract"):
        st.write(paper['abstract'])

    # Summarize button
    if st.button("Generate AI Summary"):
        with st.spinner("Generating summary..."):
            summary = summarize_paper(query, paper)  # From Phase 8
            st.success(summary)

    # Add to report
    if 'report_papers' not in st.session_state:
        st.session_state.report_papers = []

    if st.button("Add to Report"):
        st.session_state.report_papers.append(paper)
        st.success(f"Added! ({len(st.session_state.report_papers)} papers in report)")

# Generate report
if 'report_papers' in st.session_state and len(st.session_state.report_papers) > 0:
    st.sidebar.subheader(f"Report ({len(st.session_state.report_papers)} papers)")

    if st.sidebar.button("Generate Literature Review"):
        with st.spinner("Generating report..."):
            report = generate_report(query, st.session_state.report_papers)
            st.download_button(
                "Download Report",
                report,
                file_name="literature_review.md",
                mime="text/markdown"
            )
```

**Tools**: streamlit, pyvis, plotly

**Time**: 1-2 days to build, 1 day to polish

---

### **PHASE 10: Deployment** (3 hours)

**What**: Host app for 5 users to test

#### **Decision Point: Deployment Platform**

| Platform | Cost | Setup | Speed | Uptime | Choice |
|----------|------|-------|-------|--------|--------|
| **ngrok (local)** | Free | 2 min | Fast | While laptop on | ✅ FOR TESTING |
| **Streamlit Cloud** | Free | 10 min | Medium | 24/7 | ✅ FOR PRESENTATION |
| Hugging Face Spaces | Free | 15 min | Slow | 24/7 | ❌ |
| Railway.app | $5/month | 10 min | Fast | 24/7 | ❌ |

**Decision Tree:**
```
Testing with 5 users for a few days?
→ Use ngrok (free, runs on your laptop)

Need it online 24/7 for presentation/demo?
→ Use Streamlit Cloud (free, permanent)

Need more RAM/GPU?
→ Use Hugging Face Spaces (free, more resources)
```

#### **Option 1: ngrok (Quick Testing)**

```bash
# 1. Run Streamlit locally
streamlit run dashboard/app.py

# 2. In another terminal, expose to internet
ngrok http 8501

# 3. Share the URL
# ngrok gives you: https://abc123.ngrok-free.app
# Send this to your 5 testers

# Note: Your laptop must stay on!
# Free tier: 40 connections/minute (enough for 5 users)
```

**Pros:**
- Free
- 2 minutes to setup
- Full power (your laptop's CPU/GPU)
- Easy to debug

**Cons:**
- Your laptop must stay on
- If laptop sleeps, app goes down
- URL changes each time you restart ngrok

#### **Option 2: Streamlit Cloud (Permanent)**

```bash
# 1. Push code to GitHub (already done!)
git push

# 2. Go to: share.streamlit.io

# 3. Sign in with GitHub

# 4. Click "New app"

# 5. Select:
#    - Repository: PetePK/Datasci_Project
#    - Branch: main
#    - Main file: dashboard/app.py

# 6. Click "Deploy"

# 7. Get permanent URL:
#    https://datasci-project.streamlit.app

# Auto-deploys on git push!
```

**Pros:**
- Free
- Always online (24/7)
- Auto-updates on git push
- Good for 5 users
- Easy to share (clean URL)

**Cons:**
- Slow cold start (first load takes 10 sec)
- Limited resources (1 CPU, 1GB RAM)
- Might need to reduce NLI model size

**Optimization for Streamlit Cloud:**
```python
# Only load models once (use st.cache_resource)
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    nli = pipeline("text-classification", model="microsoft/deberta-v3-base-mnli")
    return model, nli

# Load vector DB once
@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="data/vector_db")
    return client.get_collection("research_papers")
```

**Tools**: streamlit cloud, git, GitHub

**Time**: 10 minutes

---

## 💰 COMPLETE COST BREAKDOWN

### **Free Option ($0)**

| Component | Tool | Cost |
|-----------|------|------|
| Embeddings | sentence-transformers | $0 |
| Vector DB | ChromaDB (local) | $0 |
| Stance Detection | DeBERTa NLI | $0 |
| Summaries | Ollama (Llama 3.1) | $0 |
| Dashboard | Streamlit | $0 |
| Deployment | ngrok or Streamlit Cloud | $0 |
| **TOTAL** | | **$0** |

**Trade-off**: Slower summaries (5-10 sec with Ollama)

---

### **Recommended Option (~$2)**

| Component | Tool | Cost |
|-----------|------|------|
| Embeddings | sentence-transformers | $0 |
| Vector DB | ChromaDB (local) | $0 |
| Stance Detection | DeBERTa NLI | $0 |
| Summaries | **OpenAI GPT-4o-mini** | **~$1-2** |
| Dashboard | Streamlit | $0 |
| Deployment | Streamlit Cloud | $0 |
| **TOTAL** | | **~$1-2** |

**Cost Calculation:**
```
GPT-4o-mini:
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

Usage estimate (5 users testing):
- 5 users × 20 searches = 100 searches
- 100 searches × 10 summaries = 1000 summaries
- 1000 summaries × 450 tokens avg = 450K tokens
- Cost = (450K × $0.15/1M) + (150K × $0.60/1M) ≈ $0.16

Realistic with some experimentation: $1-2 total
```

**Worth it?** YES - Much better quality, faster summaries

---

## ⏰ COMPLETE TIMELINE

### **Day 1 (4 hours)**
- ✅ Morning: Data inspection (2 hours)
  - Sample files, understand structure
  - Check data quality, missing values
  - Calculate statistics
- ✅ Afternoon: Data parsing (2 hours)
  - Write parser.py
  - Parse all 20K JSON → CSV
  - Clean and validate

### **Day 2 (5 hours)**
- ✅ Morning: Create embeddings (3 hours)
  - Install sentence-transformers
  - Generate vectors for all papers
  - Save to numpy file
- ✅ Afternoon: Load to ChromaDB (2 hours)
  - Setup ChromaDB
  - Load embeddings + metadata
  - Test search functionality

### **Day 3 (5 hours)**
- ✅ Morning: Implement search (2 hours)
  - Write search function
  - Add relevance scoring
  - Test on sample queries
- ✅ Afternoon: Add stance detection (3 hours)
  - Setup DeBERTa NLI
  - Test on sample papers
  - Optimize for speed

### **Day 4 (5 hours)**
- ✅ Morning: Build citation network (3 hours)
  - Write graph builder
  - Add nodes and edges
  - Calculate network stats
- ✅ Afternoon: Setup LLM summarization (2 hours)
  - Get OpenAI API key
  - Write summarizer function
  - Test summaries

### **Day 5 (5 hours)**
- ✅ All day: Build Streamlit dashboard
  - Create search interface
  - Add graph visualization (Pyvis)
  - Test locally

### **Day 6 (5 hours)**
- ✅ Morning: Add features (3 hours)
  - Summary panel
  - Report generation
  - Export functionality
- ✅ Afternoon: Polish UI (2 hours)
  - Fix bugs
  - Improve layout
  - Add loading indicators

### **Day 7 (4 hours)**
- ✅ Morning: Deploy (2 hours)
  - Push to GitHub
  - Deploy to Streamlit Cloud
  - Test with 5 users
- ✅ Afternoon: Documentation & video (2 hours)
  - Record demo video
  - Write README
  - Prepare submission

**Total**: 33 hours over 7 days

---

## 🎯 DELIVERABLES CHECKLIST

### **Code**
- [ ] `src/data/parser.py` - Parse JSON files
- [ ] `src/data/embeddings.py` - Create embeddings
- [ ] `src/search/load_vector_db.py` - Load to ChromaDB
- [ ] `src/search/vector_search.py` - Semantic search function
- [ ] `src/search/stance_detection.py` - NLI stance detection
- [ ] `src/graph/builder.py` - Build citation network
- [ ] `src/llm/summarizer.py` - GPT summarization
- [ ] `dashboard/app.py` - Main Streamlit app
- [ ] `requirements.txt` - All dependencies

### **Data**
- [ ] `data/processed/papers.csv` - Cleaned papers
- [ ] `data/embeddings/paper_embeddings.npy` - Vectors
- [ ] `data/vector_db/` - ChromaDB files

### **Documentation**
- [ ] `README.md` - Project overview
- [ ] `docs/COMPLETE_GUIDE.md` - This file
- [ ] Code comments and docstrings

### **Deployment**
- [ ] Working app (local or deployed)
- [ ] Tested with 5 users
- [ ] No critical bugs

### **Presentation**
- [ ] 15-minute video
- [ ] Google Drive folder
- [ ] Posted in Discord

---

## 🚨 COMMON ISSUES & SOLUTIONS

### **Issue: Out of memory**
```python
# Solution: Process in batches
for i in range(0, len(papers), 100):
    batch = papers[i:i+100]
    embeddings = model.encode(batch)
```

### **Issue: NLI model is slow**
```python
# Solution 1: Only analyze top 20 papers
papers = papers[:20]

# Solution 2: Use GPU
device = 0 if torch.cuda.is_available() else -1
nli = pipeline(..., device=device)

# Solution 3: Reduce model size
nli = pipeline(..., model="distilbert-base-uncased-mnli")  # Smaller model
```

### **Issue: Streamlit Cloud times out**
```python
# Solution: Cache everything
@st.cache_resource
def load_everything():
    # Load models, db, etc.
    return models

# Use cached version
models = load_everything()
```

### **Issue: ChromaDB not found**
```python
# Make sure vector_db folder is included
# Check .gitignore doesn't exclude it
```

### **Issue: OpenAI API costs too much**
```python
# Solution: Switch to Ollama (free)
from ollama import chat

response = chat(model="llama3.1", messages=[...])
```

---

## 📚 REFERENCES & RESOURCES

### **Tools Documentation**
- **sentence-transformers**: https://www.sbert.net/
- **ChromaDB**: https://docs.trychroma.com/
- **DeBERTa NLI**: https://huggingface.co/microsoft/deberta-v3-base-mnli
- **OpenAI**: https://platform.openai.com/docs
- **Streamlit**: https://docs.streamlit.io/
- **NetworkX**: https://networkx.org/documentation/stable/
- **Pyvis**: https://pyvis.readthedocs.io/

### **Learning Resources**
- Vector Search: https://www.pinecone.io/learn/vector-search/
- NLI Explained: https://huggingface.co/tasks/text-classification#natural-language-inference
- RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/

### **Similar Projects**
- Connected Papers: https://www.connectedpapers.com/
- Semantic Scholar: https://www.semanticscholar.org/
- Research Rabbit: https://www.researchrabbit.ai/

---

## 🎓 KEY CONCEPTS EXPLAINED

### **What is an Embedding?**
A numerical representation of text that captures meaning:
```
"cat" → [0.2, 0.8, 0.1, ..., 0.5]
"dog" → [0.3, 0.7, 0.2, ..., 0.4]  (similar to cat!)
"car" → [0.9, 0.1, 0.8, ..., 0.2]  (very different)
```

### **What is Cosine Similarity?**
Measures angle between two vectors:
```
similarity = dot(A, B) / (||A|| × ||B||)

Range: -1 to 1
- 1.0 = Same direction (very similar)
- 0.0 = Perpendicular (unrelated)
- -1.0 = Opposite (very different)
```

### **What is NLI (Natural Language Inference)?**
Given two sentences, determine if first entails/contradicts/neutral to second:
```
Premise: "A dog is running"
Hypothesis: "An animal is moving" → ENTAILMENT ✅
Hypothesis: "A cat is sleeping" → CONTRADICTION ❌
Hypothesis: "The sky is blue" → NEUTRAL ⚪
```

### **What is a Vector Database?**
Database optimized for finding similar vectors fast:
```
Regular DB: "Find WHERE name = 'John'"
Vector DB: "Find 10 most similar vectors to [0.8, 0.9, ...]"

Uses HNSW (graph-based algorithm) to skip most vectors
```

---

## ✅ SUCCESS CRITERIA

### **Must Have (Required)**
- ✅ All 20K papers parsed and searchable
- ✅ Semantic search works (finds relevant papers)
- ✅ Interactive graph displays
- ✅ Can click nodes to see details
- ✅ Summaries work (with LLM)

### **Nice to Have (Bonus)**
- ⭐ Stance detection (green/red edges)
- ⭐ Report generation
- ⭐ Deployed online (not just localhost)
- ⭐ Fast (<5 sec total per search)
- ⭐ Professional UI

### **Grading Alignment**

**Completeness (Required):**
- ✅ Data Module: Parsing, cleaning, EDA
- ✅ AI Module: Embeddings, search, NLI, LLM
- ✅ Visualization: Dashboard, graph, charts

**Interestingness (Bonus Points):**
- ⭐ Effort: Processing 20K papers, multiple AI models
- ⭐ Creativity: Stance detection, context-aware summaries
- ⭐ Execution: Working deployment, polished UI
- ⭐ Technical Quality: Modern NLP, efficient search
- ⭐ Real Impact: Actually useful for researchers

---

## 🎬 END

**This document contains EVERYTHING you need to understand and build this project.**

When starting a new conversation with Claude, share this document and say:
> "Read COMPLETE_GUIDE.md - this explains our entire project. Help me implement [specific phase]."

**Good luck!** 🚀
