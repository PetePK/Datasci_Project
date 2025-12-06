# Part 3: Stance Detection & Network Analysis

AI-powered paper analysis and citation networks.

## Features

### 1. Stance Detection
- **Model**: Claude 3.5 Haiku
- **Task**: Classify paper stance (support/contradict/neutral)
- **Input**: Abstract + research question
- **Output**: Stance label + confidence

### 2. Paper Summaries
- **Model**: Claude 3.5 Haiku
- **Task**: Generate 1-sentence summaries
- **Input**: Abstract (first 800 chars)
- **Optimization**: Short prompts for speed

### 3. Network Analysis
- **Graph**: Citation relationships
- **Nodes**: Papers
- **Edges**: Citations/references
- **Analysis**: Community detection, centrality

## Processing
1. **AI Summaries** - Generate for all papers (async)
2. **Stance Detection** - Classify based on query
3. **Build Graph** - NetworkX citation network
4. **Detect Communities** - Modularity-based clustering

## Output
- Enhanced paper metadata with summaries
- Citation network graph
- Community assignments

## Usage
```bash
jupyter notebook phase3_stance_detection_and_network_analysis.ipynb
```

## Cost
- Claude API: ~$0.01 per 1000 papers
- Total for 19k papers: ~$0.20
