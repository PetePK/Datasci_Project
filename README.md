# 🔬 Literature Review Assistant

**AI-powered research paper search engine with interactive visualization**

Data Science Course Project - Chulalongkorn University

---

## 📋 What We Do

Search 20,000 research papers and get instant insights:

1. **Semantic Search**: Enter your research question → Get 50 most relevant papers in 3 seconds
2. **Interactive Graph**: Visualize how papers connect (citations, relevance, stance)
3. **Stance Detection**: See which papers support/contradict your hypothesis
4. **AI Summaries**: Get context-aware summaries tailored to your question
5. **Report Generation**: Combine multiple papers into one literature review

**Problem Solved**: Literature review takes 10+ hours manually → Now done in 3 seconds

---

## 🏗️ System Architecture

```
User Query → Embeddings → Vector Search → Stance Detection → Graph Viz → Dashboard
              ↓              ↓              ↓                ↓           ↓
         (sentence-    (ChromaDB)    (DeBERTa NLI)    (NetworkX)  (Streamlit)
          transformers)
```

### Pipeline Flow

```
1. DATA MODULE
   JSON files (20K) → Parse → Clean → CSV

2. AI MODULE
   Text → Embeddings (384-dim vectors) → ChromaDB
   Query → Semantic Search → Top 50 papers
   Papers → NLI Model → Stance (support/contradict/neutral)

3. VISUALIZATION MODULE
   Papers + Citations → NetworkX Graph
   Graph → Pyvis (interactive)
   Display → Streamlit Dashboard
```

---

## 🛠️ Tech Stack

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **json** - Parse 20K JSON files

### AI/ML
- **sentence-transformers** (all-MiniLM-L6-v2) - Text embeddings
- **ChromaDB** - Vector database for semantic search
- **transformers** (DeBERTa) - NLI for stance detection
- **OpenAI GPT-4o-mini** - Context-aware summaries
- **NetworkX** - Citation network analysis

### Visualization
- **Streamlit** - Web dashboard
- **Pyvis** - Interactive graph
- **Plotly** - Charts

### Why These Choices?

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | sentence-transformers | Free, local, fast (5 min for 20K) |
| Vector DB | ChromaDB | Simple API, perfect for 20K papers |
| Stance | DeBERTa NLI | Free, accurate (85%), fast (5 sec/50 papers) |
| Summaries | GPT-4o-mini | Best quality/cost ($0.001 per summary) |
| Dashboard | Streamlit | Fastest to build, easy deployment |

---

## 📊 Dataset

**Source**: CU Office of Academic Resources (2018-2023)

- **Size**: 20,216 papers
- **Format**: JSON (Scopus metadata)
- **Fields**: Title, Abstract, Authors, Citations, References
- **Coverage**: 94% have abstracts

**Distribution**:
| Year | Papers | Avg Citations |
|------|--------|---------------|
| 2018 | 2,792 | 15.5 |
| 2019 | 3,082 | 13.7 |
| 2020 | 3,393 | 14.2 |
| 2021 | 3,815 | 7.8 |
| 2022 | 4,244 | 3.0 |
| 2023 | 2,890 | 1.0 |

---

## 🎯 Assignment Requirements

### ✅ 1. Data Module
- Parse 20K JSON files
- Clean & prepare data
- EDA with visualizations

### ✅ 2. AI Module
- Semantic search (embeddings + vector DB)
- Stance detection (NLI)
- Citation network analysis
- LLM summarization

### ✅ 3. Visualization Module
- Streamlit dashboard
- Interactive graph (Pyvis)
- Search & filter features

---

## ⚡ Quick Start

```bash
# Clone & install
git clone https://github.com/PetePK/Datasci_Project.git
cd Datasci_Project
pip install -r requirements.txt

# Run pipeline
python src/data/parser.py           # Parse JSON
python src/data/embeddings.py       # Create embeddings
python src/search/load_vector_db.py # Load to ChromaDB
streamlit run dashboard/app.py      # Run dashboard
```

---

## 📁 Project Structure

```
Datasci_Project/
├── raw_data/          # 20K JSON files
├── data/
│   ├── processed/     # Cleaned CSV
│   ├── embeddings/    # Vectors
│   └── vector_db/     # ChromaDB
├── src/
│   ├── data/          # Parser, embeddings
│   ├── search/        # Vector search, NLI
│   ├── graph/         # Network builder
│   └── llm/           # Summarizer
├── dashboard/         # Streamlit app
└── docs/              # Documentation
```

---

## 🎥 Demo

**Video**: [YouTube Link](YOUR_YOUTUBE_LINK)

**Live App**: [Streamlit Cloud](YOUR_STREAMLIT_LINK)

---

## 🤝 Team

- [Your Name] - Full-stack implementation

---

## 📧 Contact

**GitHub**: [@PetePK](https://github.com/PetePK)

**Email**: [your_email@student.chula.ac.th]

---

**CU Data Science Course - 2024**
