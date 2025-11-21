# 🎯 Literature Review Assistant - Complete Project Summary

**Goal**: AI-powered search engine that finds relevant research papers, visualizes connections, and provides intelligent summaries.

**Timeline**: 1 week
**Users**: Max 5 (prototype/demo)
**Budget**: Minimal cost

---

## 📊 What We're Building

### **User Experience:**
```
1. User enters research question
   "Does machine learning improve medical diagnosis?"

2. System searches 20,000 papers
   → Finds 50 most relevant papers in 3 seconds

3. Shows interactive graph:
   - Node size = citation count (bigger = more cited)
   - Node color = relevance (gradient: red=very relevant, yellow=somewhat, blue=less)
   - Edge color = stance (green=supports idea, red=contradicts, gray=neutral)

4. User hovers over node:
   → Shows title, abstract, relevance score

5. User clicks node:
   → AI summarizes paper in context of their question

6. User selects multiple nodes:
   → Generate combined literature review report
```

---

## 🗺️ Complete Pipeline

### **Phase 1: Data Inspection & Understanding** (Day 1 - 4 hours)

**What we're doing**: Explore the 20K JSON files to understand structure and quality

**Steps:**
1. Sample random files from each year (2018-2023)
2. Analyze structure: titles, abstracts, authors, citations
3. Check data quality: missing values, inconsistencies
4. Document findings

**Tools**: Python, pandas, json

**Output**:
- Data quality report
- Statistics (papers per year, citation distribution, etc.)
- Identified issues to fix

---

### **Phase 2: Data Cleaning & Preparation** (Day 1-2 - 6 hours)

**What we're doing**: Parse JSON files and create clean, structured dataset

#### **Option 1: Parse to CSV** ⭐ RECOMMENDED
**Pros:**
- ✅ Simple, easy to debug
- ✅ Can open in Excel to inspect
- ✅ Fast for 20K papers (~10 min)

**Cons:**
- ⚠️ Takes disk space (~500MB)

**Option 2: Parse to SQLite Database**
**Pros:**
- ✅ Better for queries
- ✅ Normalized structure

**Cons:**
- ❌ More complex code
- ❌ Overkill for 20K papers

**Option 3: Keep as JSON, parse on-the-fly**
**Pros:**
- ✅ No preprocessing needed

**Cons:**
- ❌ Very slow for searches
- ❌ Bad user experience

**CHOSEN: Option 1 (CSV)**
**Reason**: Simple, fast, easy to work with. Perfect for prototype.

**Implementation:**
```python
# src/data/parser.py
- Read all JSON files
- Extract: title, abstract, authors, year, citations, references
- Handle missing data (use empty string or None)
- Save to: data/processed/papers.csv
```

**Data Cleaning:**
- Remove papers without abstracts (can't search without text)
- Normalize years (convert to int)
- Remove special characters that break encoding
- Deduplicate (same paper might appear twice)

**Output**: `data/processed/papers.csv` (~20K rows)

---

### **Phase 3: Embedding Creation** (Day 2 - 3 hours)

**What we're doing**: Convert paper text (title + abstract) into numerical vectors

#### **Embedding Model Options:**

#### **Option 1: sentence-transformers (all-MiniLM-L6-v2)** ⭐ RECOMMENDED
**Specs**: 384 dimensions, fast
**Pros:**
- ✅ Free, local (no API)
- ✅ Fast (~2 min for 20K papers)
- ✅ Small size (80MB model)
- ✅ Good quality (fine for prototype)

**Cons:**
- ⚠️ Lower quality than larger models

**Option 2: sentence-transformers (all-mpnet-base-v2)**
**Specs**: 768 dimensions, better quality
**Pros:**
- ✅ Free, local
- ✅ Higher quality embeddings
- ✅ Still fast (~5 min for 20K)

**Cons:**
- ⚠️ Larger model (420MB)
- ⚠️ More compute needed

**Option 3: OpenAI Embeddings (text-embedding-3-small)**
**Specs**: 1536 dimensions, cloud
**Pros:**
- ✅ Best quality
- ✅ Very fast

**Cons:**
- ❌ Costs money (~$2 for 20K papers)
- ❌ Requires internet
- ❌ Not private

**CHOSEN: Option 1 (all-MiniLM-L6-v2)**
**Reason**: Free, fast, good enough quality. Can always upgrade later.

**Implementation:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

for paper in papers:
    text = f"{paper['title']} {paper['abstract']}"
    embedding = model.encode(text)
    # Save embedding
```

**Output**: `data/embeddings/paper_embeddings.npy` (20K vectors × 384 dimensions)

---

### **Phase 4: Vector Database Setup** (Day 2 - 2 hours)

**What we're doing**: Store embeddings in searchable database

#### **Vector DB Options:**

#### **Option 1: ChromaDB (Local)** ⭐ RECOMMENDED
**Pros:**
- ✅ FREE (runs locally)
- ✅ Easy setup (3 lines of code)
- ✅ Fast for 20K papers (~100ms search)
- ✅ Persistent (saves to disk)

**Cons:**
- ⚠️ Not for millions of papers

**Option 2: FAISS (Facebook)**
**Pros:**
- ✅ Free, very fast
- ✅ Best for large scale

**Cons:**
- ❌ More complex setup
- ❌ Harder to manage metadata
- ❌ Overkill for 20K

**Option 3: Pinecone (Cloud)**
**Pros:**
- ✅ Cloud-hosted (accessible anywhere)
- ✅ Scalable

**Cons:**
- ❌ Expensive ($70/month)
- ❌ Overkill for prototype

**CHOSEN: Option 1 (ChromaDB)**
**Reason**: Free, simple, perfect for 20K papers and 5 users.

**Implementation:**
```python
import chromadb

client = chromadb.PersistentClient(path="./data/vector_db")
collection = client.create_collection("papers")

collection.add(
    ids=paper_ids,
    embeddings=embeddings,
    documents=abstracts,
    metadatas=metadata
)
```

**Output**: `data/vector_db/` (persistent ChromaDB)

---

### **Phase 5: Search & Relevance** (Day 3 - 3 hours)

**What we're doing**: When user searches, find relevant papers and score them

#### **Search Strategy:**

**Step 1: Semantic Search**
```python
query = "Does ML improve diagnosis?"
query_embedding = model.encode(query)
results = collection.query(query_embedding, n_results=50)
# Get 50 most similar papers (by cosine similarity)
```

**Step 2: Relevance Scoring**

Use the similarity scores from ChromaDB directly!

```python
for paper in results:
    score = paper['distance']  # 0.0 (identical) to 2.0 (opposite)

    # Convert to 0-100 scale
    relevance_score = (2.0 - score) / 2.0 * 100

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

**No LLM needed for relevance! Just use the similarity score!**

---

### **Phase 6: Stance Detection** (Day 3-4 - 4 hours)

**What we're doing**: Determine if paper supports/contradicts user's query

#### **Stance Detection Options:**

#### **Option 1: NLI Model (DeBERTa)** ⭐ RECOMMENDED
**Pros:**
- ✅ FREE (runs locally)
- ✅ Fast (0.1 sec per paper)
- ✅ Accurate for stance detection
- ✅ 50 papers in ~5 seconds

**Cons:**
- ⚠️ Requires GPU for speed (CPU is slower ~0.5 sec/paper)

**Option 2: LLM (GPT-4o-mini batch)**
**Pros:**
- ✅ Most accurate
- ✅ Can provide reasoning

**Cons:**
- ❌ Costs money (~$0.01 per search)
- ❌ Slower (5-10 sec for 50 papers)
- ❌ Requires API key

**Option 3: Simple heuristic (keyword matching)**
**Pros:**
- ✅ Free, instant

**Cons:**
- ❌ Inaccurate
- ❌ Too simplistic

**CHOSEN: Option 1 (NLI Model)**
**Reason**: Free, fast enough, good quality. Perfect for prototype.

**Implementation:**
```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="microsoft/deberta-v3-base-mnli")

for paper in papers:
    result = nli(f"{query} [SEP] {paper['abstract']}")

    if result['label'] == 'ENTAILMENT':
        stance = 'supports'
        edge_color = '#00FF00'  # Green
    elif result['label'] == 'CONTRADICTION':
        stance = 'contradicts'
        edge_color = '#FF0000'  # Red
    else:
        stance = 'neutral'
        edge_color = '#808080'  # Gray
```

**Output**: Each paper gets `stance` and `edge_color` attributes

---

### **Phase 7: Citation Network** (Day 4 - 3 hours)

**What we're doing**: Build graph showing how papers cite each other

**Implementation:**
```python
import networkx as nx

G = nx.DiGraph()

# Add nodes (papers)
for paper in results:
    G.add_node(
        paper['id'],
        title=paper['title'],
        citations=paper['citation_count'],
        relevance_score=paper['relevance_score'],
        color=paper['color'],
        size=paper['citation_count'] * 10  # Scale for visualization
    )

# Add edges (citations)
for paper in results:
    for ref in paper['references']:
        if ref in result_ids:  # Only link papers in results
            G.add_edge(
                paper['id'],
                ref,
                color=paper['edge_color']  # Stance color
            )
```

**Output**: NetworkX graph object

---

### **Phase 8: Summarization** (Day 4 - 2 hours)

**What we're doing**: When user clicks paper, summarize in context of query

#### **LLM Options:**

#### **Option 1: Ollama (Llama 3.1 local)** ⭐ RECOMMENDED FOR FREE
**Pros:**
- ✅ FREE (no API costs)
- ✅ Private (runs locally)
- ✅ Decent quality
- ✅ Works offline

**Cons:**
- ⚠️ Slower (5-10 sec per summary)
- ⚠️ Needs good CPU/GPU
- ⚠️ Lower quality than GPT-4

**Option 2: OpenAI GPT-4o-mini** ⭐ RECOMMENDED FOR QUALITY
**Pros:**
- ✅ Fast (2-3 sec per summary)
- ✅ High quality
- ✅ Cheap ($0.15 per 1M tokens = ~$0.001 per summary)

**Cons:**
- ❌ Costs money (~$0.05 for 50 summaries)
- ❌ Requires API key
- ❌ Needs internet

**Option 3: OpenAI GPT-4o**
**Pros:**
- ✅ Best quality

**Cons:**
- ❌ 10x more expensive
- ❌ Overkill for this task

**CHOSEN: Option 2 (GPT-4o-mini)**
**Reason**: Only $0.001 per summary, fast, good quality. Total cost for 5 users testing = ~$1-2.

**Fallback**: If budget is $0, use Option 1 (Ollama).

**Implementation:**
```python
from openai import OpenAI

client = OpenAI(api_key="...")

def summarize_paper(query, paper):
    prompt = f"""
    User's research question: {query}

    Paper title: {paper['title']}
    Paper abstract: {paper['abstract']}

    Summarize how this paper relates to the user's question.
    Focus on: key findings, methodology, and relevance.
    Keep it under 150 words.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message.content
```

**Output**: Context-aware summary on-demand

---

### **Phase 9: Dashboard** (Day 5-6 - 10 hours)

**What we're doing**: Build interactive web interface

#### **Framework Options:**

#### **Option 1: Streamlit** ⭐ RECOMMENDED
**Pros:**
- ✅ Fastest to build (pure Python)
- ✅ Built-in components
- ✅ Easy deployment
- ✅ Perfect for prototypes

**Cons:**
- ⚠️ Less customizable UI
- ⚠️ Refreshes on interaction (not true SPA)

**Option 2: Plotly Dash**
**Pros:**
- ✅ More control over layout
- ✅ Good for complex dashboards

**Cons:**
- ❌ More code needed
- ❌ Steeper learning curve

**Option 3: React + FastAPI**
**Pros:**
- ✅ Full control
- ✅ Best UX

**Cons:**
- ❌ Need to know JavaScript
- ❌ 3x more work
- ❌ Not feasible in 1 week

**CHOSEN: Option 1 (Streamlit)**
**Reason**: Fastest to build, perfect for prototype, easy to deploy.

**Features:**
1. **Search Page**
   - Text input for query
   - Search button
   - Display stats (50 papers found, relevance distribution)

2. **Graph Visualization**
   - Interactive graph (Pyvis)
   - Zoom, pan, drag nodes
   - Hover shows title + score
   - Click shows summary panel

3. **Summary Panel**
   - Shows when node clicked
   - Title, abstract, full metadata
   - "Summarize" button → AI summary
   - "Add to report" checkbox

4. **Report Generator**
   - Select multiple papers
   - "Generate Report" button
   - Combines summaries into formatted document
   - Download as PDF/Markdown

**Implementation:**
```python
import streamlit as st
from pyvis.network import Network

st.title("Literature Review Assistant")

query = st.text_input("Enter your research question:")

if st.button("Search"):
    # Perform search
    results = search(query)

    st.write(f"Found {len(results)} relevant papers")

    # Build graph
    net = Network(height="700px", width="100%")
    for paper in results:
        net.add_node(
            paper['id'],
            label=paper['title'][:50],
            size=paper['size'],
            color=paper['color']
        )

    for edge in edges:
        net.add_edge(edge['from'], edge['to'], color=edge['color'])

    # Display
    net.save_graph("graph.html")
    st.components.v1.html(open("graph.html").read(), height=750)
```

**Output**: Fully functional web app

---

### **Phase 10: Deployment** (Day 7 - 3 hours)

**What we're doing**: Host the app for 5 users to test

#### **Deployment Options:**

#### **Option 1: Streamlit Community Cloud** ⭐ RECOMMENDED
**Specs**: 1 CPU, 1GB RAM, free
**Pros:**
- ✅ **100% FREE**
- ✅ Easiest (connect GitHub, click deploy)
- ✅ Auto-updates on git push
- ✅ Good for 5 users

**Cons:**
- ⚠️ Public URL (anyone can access)
- ⚠️ Slow for heavy compute
- ⚠️ 1GB RAM (might be tight with NLI model)

**Cost**: $0/month

---

#### **Option 2: Hugging Face Spaces (Free tier)**
**Specs**: 2 CPU, 16GB RAM, free
**Pros:**
- ✅ **FREE**
- ✅ More RAM (better for NLI model)
- ✅ Good for ML apps
- ✅ Easy deployment

**Cons:**
- ⚠️ Slower cold starts
- ⚠️ Public URL

**Cost**: $0/month

---

#### **Option 3: Local + ngrok** ⭐ BEST FOR TESTING
**Pros:**
- ✅ **FREE**
- ✅ Runs on your laptop
- ✅ Full power (GPU if you have it)
- ✅ Private (only people with link)
- ✅ Easy to debug

**Cons:**
- ⚠️ Your laptop must stay on
- ⚠️ Limited to 40 connections/minute (free tier)

**Cost**: $0/month

**Implementation:**
```bash
# Run Streamlit locally
streamlit run dashboard/app.py

# In another terminal, expose to internet
ngrok http 8501

# Share the ngrok URL with 5 testers
# URL looks like: https://abc123.ngrok.io
```

---

#### **Option 4: Railway.app (Paid)**
**Specs**: $5/month credit, ~500 hours
**Pros:**
- ✅ Easy deployment
- ✅ Good performance
- ✅ Private repos

**Cons:**
- ❌ Costs $5/month
- ⚠️ Credit runs out

**Cost**: ~$5 for testing period

---

#### **Option 5: AWS/Google Cloud**
**Pros:**
- ✅ Full control
- ✅ Scalable

**Cons:**
- ❌ Complex setup
- ❌ Expensive (~$20-50/month)
- ❌ Overkill for 5 users

**Cost**: $20-50/month

---

### **CHOSEN: Option 3 (Local + ngrok)** for testing, then **Option 1 (Streamlit Cloud)** if you want it public

**Reason**:
- **ngrok**: FREE, full power, easy debugging, perfect for 5 testers
- **Streamlit Cloud**: FREE, permanent hosting if needed

**Decision tree:**
```
Testing with 5 users for a few days?
→ Use ngrok (free, your laptop)

Need it hosted 24/7 for presentation?
→ Use Streamlit Cloud (free, always online)

Need privacy + power?
→ Use Hugging Face Spaces (free, more RAM)
```

---

### **Deployment Steps:**

**Option A: ngrok (Testing)**
```bash
# 1. Install ngrok
# Download from: https://ngrok.com/download

# 2. Run your app
streamlit run dashboard/app.py

# 3. In new terminal, expose it
ngrok http 8501

# 4. Share the URL (https://abc123.ngrok-free.app)
# Send to your 5 testers

# Your laptop must stay on!
```

**Option B: Streamlit Cloud (Permanent)**
```bash
# 1. Push code to GitHub
git add .
git commit -m "Deploy app"
git push

# 2. Go to: share.streamlit.io
# 3. Connect GitHub repo
# 4. Click "Deploy"
# 5. Get public URL: https://yourapp.streamlit.app

# Done! Always online, auto-updates on git push
```

---

## 💰 Total Cost Breakdown

### **100% Free Option:**
```
Embeddings: sentence-transformers (free)
Vector DB: ChromaDB local (free)
NLI: DeBERTa local (free)
LLM: Ollama local (free)
Deployment: ngrok or Streamlit Cloud (free)

Total: $0
```

### **Best Quality Option (Recommended):**
```
Embeddings: sentence-transformers (free)
Vector DB: ChromaDB local (free)
NLI: DeBERTa local (free)
LLM: OpenAI GPT-4o-mini ($1-2 for testing)
Deployment: ngrok/Streamlit Cloud (free)

Total: ~$2
```

### **Premium Option:**
```
Everything above, but:
LLM: OpenAI GPT-4o ($10-20)
Deployment: Railway.app ($5/month)

Total: ~$25
```

**RECOMMENDATION: Best Quality Option ($2 total)**
- Worth the $2 for faster, better summaries
- Still basically free
- Can show off high-quality AI in presentation

---

## 🗂️ Final Project Structure

```
Datasci_Project/
├── raw_data/                    # Original 20K JSON files
│   ├── 2018/
│   ├── 2019/
│   ├── 2020/
│   ├── 2021/
│   ├── 2022/
│   └── 2023/
│
├── data/
│   ├── processed/
│   │   └── papers.csv           # Cleaned data
│   ├── embeddings/
│   │   └── paper_embeddings.npy # Vectors
│   └── vector_db/               # ChromaDB
│
├── src/
│   ├── data/
│   │   ├── parser.py            # JSON → CSV
│   │   └── embeddings.py        # Create vectors
│   ├── search/
│   │   ├── vector_search.py     # Semantic search
│   │   └── stance_detection.py  # NLI
│   ├── graph/
│   │   └── builder.py           # Build network
│   └── llm/
│       └── summarizer.py        # GPT-4o-mini
│
├── dashboard/
│   └── app.py                   # Streamlit app
│
├── docs/                        # All documentation
│   ├── PROJECT_PLAN.md
│   ├── QUICK_START.md
│   ├── DATA_DICTIONARY.md
│   └── RAG_EXPLAINED.md
│
├── archive/                     # Old analysis scripts
│
├── requirements.txt
├── README.md
└── PROJECT_SUMMARY.md           # This file!
```

---

## ⏱️ 7-Day Timeline

### **Day 1 (4 hours)**
- Morning: Data inspection (explore JSONs, statistics)
- Afternoon: Parse JSON → CSV, clean data

### **Day 2 (5 hours)**
- Morning: Create embeddings (sentence-transformers)
- Afternoon: Load into ChromaDB, test search

### **Day 3 (5 hours)**
- Morning: Implement relevance scoring
- Afternoon: Add NLI stance detection

### **Day 4 (5 hours)**
- Morning: Build citation network graph
- Afternoon: Setup LLM summarization

### **Day 5 (5 hours)**
- All day: Build Streamlit dashboard (search + graph)

### **Day 6 (5 hours)**
- Morning: Add summary panel + report generator
- Afternoon: Polish UI, fix bugs

### **Day 7 (4 hours)**
- Morning: Deploy (ngrok or Streamlit Cloud)
- Afternoon: Record demo video, write README

**Total: ~33 hours over 7 days**

---

## ✅ Deliverables Checklist

### **Code & Data**
- [ ] `src/data/parser.py` - Parse JSON files
- [ ] `src/data/embeddings.py` - Create embeddings
- [ ] `src/search/vector_search.py` - Semantic search
- [ ] `src/search/stance_detection.py` - NLI
- [ ] `src/graph/builder.py` - Build network
- [ ] `src/llm/summarizer.py` - GPT summarization
- [ ] `dashboard/app.py` - Main Streamlit app
- [ ] `data/processed/papers.csv` - Clean data
- [ ] `data/vector_db/` - ChromaDB

### **Documentation**
- [ ] `README.md` - Overview and setup instructions
- [ ] `requirements.txt` - Dependencies
- [ ] Code comments and docstrings

### **Demo**
- [ ] Working deployed app (ngrok or Streamlit Cloud)
- [ ] 15-minute demo video
- [ ] Test with 5 users

### **Submission**
- [ ] YouTube video (unlisted)
- [ ] Google Drive folder (code + video)
- [ ] Discord post in #project-showroom

---

## 🎯 Success Criteria

### **Must Have (Required Components)**
✅ **Data Module**:
- Parse 20K papers
- Clean and prepare
- Create embeddings
- Load to vector DB

✅ **AI Module**:
- Semantic search (ChromaDB)
- Relevance scoring (similarity)
- Stance detection (NLI)
- Summarization (LLM)

✅ **Visualization Module**:
- Interactive graph (Pyvis)
- Search interface
- Summary panel
- Stats display

### **Bonus (WOW Factor)**
⭐ Color-coded relevance (gradient)
⭐ Stance-based edges (green/red/gray)
⭐ On-demand AI summaries
⭐ Report generation
⭐ Professional deployment

---

## 🚀 Next Steps

Ready to start? I can create:

1. **Data parsing script** (`src/data/parser.py`)
2. **Embedding pipeline** (`src/data/embeddings.py`)
3. **Vector search** (`src/search/vector_search.py`)
4. **Stance detection** (`src/search/stance_detection.py`)
5. **Streamlit dashboard** (`dashboard/app.py`)

Let me know which part to start with! 🎉
