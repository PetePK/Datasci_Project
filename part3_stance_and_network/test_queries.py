"""
Test different query types to analyze:
1. Stance distribution (Support/Neutral/Contradict)
2. Network connectivity (number of edges)
3. Relevance scores
4. Performance
"""

import pandas as pd
import numpy as np
import asyncio
import time
from sentence_transformers import SentenceTransformer
import chromadb
from anthropic import AsyncAnthropic
import os
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Load data
print("Loading data...")
df = pd.read_parquet('../data/processed/papers.parquet')
embeddings_full = np.load('../data/embeddings/paper_embeddings.npy')

model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="../data/vector_db")
collection = client.get_collection("papers")

os.environ['ANTHROPIC_API_KEY'] = "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"
llm_client = AsyncAnthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

print(f"Loaded {len(df):,} papers\n")

# Test queries of different types
test_queries = [
    {
        "query": "machine learning improves medical diagnosis",
        "type": "Positive claim (expect SUPPORT papers)",
        "expected": "Papers showing ML helps diagnosis"
    },
    {
        "query": "social media harms mental health",
        "type": "Controversial claim (expect mixed stances)",
        "expected": "Mix of support/contradict/neutral"
    },
    {
        "query": "climate change causes extreme weather",
        "type": "Scientific consensus (expect SUPPORT)",
        "expected": "Mostly support papers"
    },
    {
        "query": "renewable energy storage solutions",
        "type": "Neutral topic (expect mostly NEUTRAL)",
        "expected": "Technical papers, neutral stance"
    },
    {
        "query": "cancer immunotherapy treatment effectiveness",
        "type": "Medical research (expect varied results)",
        "expected": "Mix based on study outcomes"
    }
]

# Helper functions
def smart_search(query, n=20):
    query_emb = model.encode(query)
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=100)

    papers_data = []
    for i, (meta, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        similarity = (2.0 - distance) / 2.0 * 100
        papers_data.append({
            'rank': i + 1,
            'title': meta['title'],
            'similarity': similarity
        })

    results_df = pd.DataFrame(papers_data)
    results_df = results_df.merge(
        df[['title', 'id', 'abstract', 'year', 'citation_count']],
        on='title', how='left'
    )

    title_to_idx = {title: idx for idx, title in enumerate(df['title'])}
    results_df['paper_idx'] = results_df['title'].map(title_to_idx)

    final_papers = results_df.head(n)
    paper_embeddings = embeddings_full[final_papers['paper_idx'].values]

    return final_papers.reset_index(drop=True), paper_embeddings

async def analyze_paper_llm(paper, query):
    prompt = f"""Analyze this research paper's relationship to the user's query.

Paper:
Title: {paper['title']}
Abstract: {paper['abstract'][:600]}

User Query: "{query}"

Task: Determine the paper's stance and provide a summary.

Respond EXACTLY in this format:
STANCE: [SUPPORT/CONTRADICT/NEUTRAL]
SUMMARY: [In 2-3 sentences, explain how this paper relates to the query]

Rules:
- SUPPORT: Paper's findings align with or support the query
- CONTRADICT: Paper's findings oppose or contradict the query
- NEUTRAL: Paper is relevant but doesn't clearly support or contradict
"""

    try:
        response = await llm_client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=150,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text
        lines = text.split('\n')
        stance_line = [l for l in lines if 'STANCE:' in l][0]
        summary_line = [l for l in lines if 'SUMMARY:' in l][0]

        stance = stance_line.split('STANCE:')[1].strip().upper()
        summary = summary_line.split('SUMMARY:')[1].strip()

        return {'id': paper['id'], 'stance': stance, 'summary': summary}
    except Exception as e:
        return {'id': paper['id'], 'stance': 'NEUTRAL', 'summary': f'Error: {str(e)}'}

async def analyze_all_papers(papers, query):
    tasks = [analyze_paper_llm(p, query) for p in papers.to_dict('records')]
    results = await asyncio.gather(*tasks)
    return {r['id']: r for r in results}

def build_network(paper_embeddings, threshold=0.60, max_edges=5):
    sim_matrix = cosine_similarity(paper_embeddings)
    edge_count = 0

    for i in range(len(sim_matrix)):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_k = np.argsort(sims)[::-1][:max_edges]

        for j in top_k:
            if sims[j] >= threshold and i < j:  # Avoid double counting
                edge_count += 1

    return edge_count

# Run tests
async def run_tests():
    results = []

    for test in test_queries:
        query = test['query']
        print("="*80)
        print(f"Query: {query}")
        print(f"Type: {test['type']}")
        print(f"Expected: {test['expected']}")
        print("="*80)

        # Search
        start = time.time()
        papers, embeddings = smart_search(query, n=20)
        search_time = time.time() - start

        # LLM analysis
        start = time.time()
        analysis = await analyze_all_papers(papers, query)
        llm_time = time.time() - start

        # Add stance to papers
        papers['stance'] = papers['id'].map(lambda x: analysis[x]['stance'])

        # Network analysis
        edge_count = build_network(embeddings, threshold=0.60, max_edges=5)

        # Statistics
        stance_dist = papers['stance'].value_counts().to_dict()
        avg_relevance = papers['similarity'].mean()
        min_relevance = papers['similarity'].min()

        result = {
            'Query': query,
            'Type': test['type'],
            'Support': stance_dist.get('SUPPORT', 0),
            'Contradict': stance_dist.get('CONTRADICT', 0),
            'Neutral': stance_dist.get('NEUTRAL', 0),
            'Edges': edge_count,
            'Avg Relevance': f"{avg_relevance:.1f}%",
            'Min Relevance': f"{min_relevance:.1f}%",
            'Search Time': f"{search_time:.2f}s",
            'LLM Time': f"{llm_time:.2f}s"
        }

        results.append(result)

        # Print summary
        print(f"\nStance Distribution:")
        print(f"  SUPPORT: {stance_dist.get('SUPPORT', 0)} ({stance_dist.get('SUPPORT', 0)/20*100:.0f}%)")
        print(f"  CONTRADICT: {stance_dist.get('CONTRADICT', 0)} ({stance_dist.get('CONTRADICT', 0)/20*100:.0f}%)")
        print(f"  NEUTRAL: {stance_dist.get('NEUTRAL', 0)} ({stance_dist.get('NEUTRAL', 0)/20*100:.0f}%)")
        print(f"\nNetwork Connectivity:")
        print(f"  Edges: {edge_count} (out of 190 possible)")
        print(f"  Density: {edge_count/190*100:.1f}%")
        print(f"\nRelevance:")
        print(f"  Average: {avg_relevance:.1f}%")
        print(f"  Minimum: {min_relevance:.1f}%")
        print(f"\nPerformance:")
        print(f"  Search: {search_time:.2f}s")
        print(f"  LLM Analysis: {llm_time:.2f}s")
        print(f"  Total: {search_time + llm_time:.2f}s")

        # Show example papers
        print(f"\nExample Papers:")
        for idx, row in papers.head(3).iterrows():
            print(f"  [{row['stance']}] {row['title'][:70]}... ({row['similarity']:.1f}%)")

        print("\n")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80 + "\n")

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    return summary_df

# Run
summary = asyncio.run(run_tests())
