# 🚀 Getting Started - Literature Review Assistant

## Quick Summary

You're building a smart literature review tool that:
1. Searches 20K papers by meaning (not keywords)
2. Shows an interactive graph
3. Uses AI to summarize papers

**Timeline**: 1 week
**Cost**: ~$2 (or $0 with free options)

---

## 📋 What You Need

### Must Have
- Python 3.9+
- 4GB RAM
- Internet (for initial setup)

### Optional
- GPU (faster processing)
- OpenAI API key (better summaries, ~$2)

---

## ⚡ Quick Setup (15 minutes)

### Step 1: Install Python Packages

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install everything
pip install -r requirements.txt
```

### Step 2: Download AI Models

```bash
# These run locally (free)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from transformers import pipeline; pipeline('text-classification', model='microsoft/deberta-v3-base-mnli')"
```

### Step 3: (Optional) Setup LLM

**Option A: Free (Ollama)**
```bash
# Download from: https://ollama.com
ollama pull llama3.1
```

**Option B: Paid ($2 total for testing)**
```bash
# Get API key from: https://platform.openai.com
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## 📊 Process Your Data (1 hour)

### Parse JSON Files
```bash
python src/data/parser.py
# Output: data/processed/papers.csv
```

### Create Embeddings
```bash
python src/data/embeddings.py
# Output: data/embeddings/paper_embeddings.npy
# Takes ~5 minutes for 20K papers
```

### Load to Vector DB
```bash
python src/search/load_vector_db.py
# Output: data/vector_db/ (ChromaDB)
```

---

## 🎮 Run the App

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`

---

## 🌐 Share with Others

### Option 1: ngrok (Quick Test)
```bash
# Install ngrok from: https://ngrok.com

# Run app
streamlit run dashboard/app.py

# In new terminal
ngrok http 8501

# Share the URL: https://abc123.ngrok-free.app
```

### Option 2: Streamlit Cloud (Permanent)
```bash
# Push to GitHub
git add .
git commit -m "Deploy app"
git push

# Go to: share.streamlit.io
# Connect repo → Deploy
# Get URL: https://yourapp.streamlit.app
```

---

## 💰 Cost Breakdown

### Free Option ($0)
```
✓ sentence-transformers (embeddings)
✓ ChromaDB (vector search)
✓ DeBERTa NLI (stance detection)
✓ Ollama + Llama 3.1 (summaries)
✓ Streamlit Cloud (hosting)

Total: $0
Trade-off: Slower summaries (5-10 sec each)
```

### Recommended ($2)
```
✓ Everything above, BUT:
✓ OpenAI GPT-4o-mini (summaries)

Cost: $0.15 per 1M tokens
Usage: 5 users × 100 searches × 10 summaries = 5000 summaries
Cost: 5000 × $0.001 = $5 (max)
Realistic: ~$1-2

Trade-off: Fast (2 sec), better quality
```

---

## 📁 File Structure (After Setup)

```
Datasci_Project/
├── raw_data/                    # Your 20K JSON files
├── data/
│   ├── processed/
│   │   └── papers.csv           # ✓ After step 1
│   ├── embeddings/
│   │   └── paper_embeddings.npy # ✓ After step 2
│   └── vector_db/               # ✓ After step 3
├── src/
│   ├── data/
│   │   ├── parser.py            # Run this first
│   │   └── embeddings.py        # Run this second
│   ├── search/
│   │   ├── load_vector_db.py    # Run this third
│   │   ├── vector_search.py     # Used by dashboard
│   │   └── stance_detection.py  # Used by dashboard
│   └── llm/
│       └── summarizer.py        # Used by dashboard
└── dashboard/
    └── app.py                   # Run this to start
```

---

## 🎯 7-Day Plan

### Day 1 (4 hours)
- ✓ Setup environment
- ✓ Parse JSON → CSV
- ✓ Data exploration

### Day 2 (5 hours)
- ✓ Create embeddings
- ✓ Load to ChromaDB
- ✓ Test search

### Day 3 (5 hours)
- ✓ Add relevance scoring
- ✓ Add stance detection (NLI)
- ✓ Test both

### Day 4 (5 hours)
- ✓ Build citation network
- ✓ Setup LLM summarization
- ✓ Test graph

### Day 5 (5 hours)
- ✓ Build Streamlit dashboard
- ✓ Search interface
- ✓ Graph visualization

### Day 6 (5 hours)
- ✓ Add summary panel
- ✓ Add report generator
- ✓ Polish UI

### Day 7 (4 hours)
- ✓ Deploy (ngrok or Streamlit Cloud)
- ✓ Test with 5 users
- ✓ Record demo video

---

## 🐛 Troubleshooting

### "Out of memory" error
```bash
# Process in batches
# Edit src/data/embeddings.py
# Change batch_size from 32 to 16 or 8
```

### NLI model is slow
```bash
# Use GPU if available
# Or reduce number of papers analyzed
# Only analyze top 20 instead of 50
```

### Streamlit Cloud deployment fails
```bash
# Check requirements.txt has all dependencies
# Make sure data/vector_db/ is included
# Or regenerate it on first run
```

### OpenAI API costs too much
```bash
# Switch to Ollama (free)
# Edit src/llm/summarizer.py
# Use Ollama instead of OpenAI
```

---

## 📚 Key Files to Read

1. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete guide with all options
2. **[CONCEPT_SUMMARY.md](CONCEPT_SUMMARY.md)** - Quick concept overview
3. **[requirements.txt](requirements.txt)** - All dependencies
4. **[README.md](README.md)** - Project overview

---

## ✅ Checklist

Before you start:
- [ ] Python 3.9+ installed
- [ ] Git installed
- [ ] 5GB free disk space
- [ ] Internet connection

After setup:
- [ ] Virtual environment activated
- [ ] All packages installed
- [ ] Models downloaded
- [ ] papers.csv exists
- [ ] Embeddings created
- [ ] Vector DB loaded
- [ ] Dashboard runs locally

Ready to share:
- [ ] Dashboard works on localhost
- [ ] Can search and get results
- [ ] Graph displays correctly
- [ ] Summaries work
- [ ] Deployed (ngrok or Streamlit Cloud)
- [ ] 5 users can access

---

## 🆘 Need Help?

### Common Questions

**Q: Do I need GPU?**
A: No, but it's faster. CPU works fine (just slower).

**Q: Can I use free options only?**
A: Yes! Use Ollama instead of OpenAI.

**Q: How long does setup take?**
A: 15 min install + 1 hour data processing = ~1.5 hours

**Q: Will it work with more/fewer papers?**
A: Yes! Works with 1K to 100K papers.

**Q: Can I customize the graph colors?**
A: Yes! Edit `dashboard/app.py`

---

## 🎓 Learning Resources

- **Vector Search**: Read [docs/RAG_EXPLAINED.md](docs/RAG_EXPLAINED.md)
- **ChromaDB**: https://docs.trychroma.com/
- **Streamlit**: https://docs.streamlit.io/
- **sentence-transformers**: https://www.sbert.net/

---

## 🚀 Ready to Start?

```bash
# 1. Clone repo (if not done)
git clone https://github.com/PetePK/Datasci_Project.git
cd Datasci_Project

# 2. Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Process data
python src/data/parser.py
python src/data/embeddings.py
python src/search/load_vector_db.py

# 4. Run
streamlit run dashboard/app.py

# 5. Share
ngrok http 8501
```

**That's it! You're ready to go!** 🎉

---

**Questions? Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for detailed explanations of every step.**
