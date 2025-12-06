"""
Network Analysis Service
Builds similarity networks and detects communities
Same logic as Streamlit app
"""

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def build_similarity_network(
    papers: List[dict],
    embeddings: np.ndarray,
    threshold: float = 0.6,
    max_edges_per_node: int = 5
) -> dict:
    """
    Build similarity network from papers and embeddings
    Same as Streamlit's network building logic

    Args:
        papers: List of paper dicts
        embeddings: Paper embeddings (subset)
        threshold: Similarity threshold (0.6 default)
        max_edges_per_node: Max connections per paper

    Returns:
        dict with 'nodes' and 'edges'
    """
    if len(papers) == 0:
        return {"nodes": [], "edges": []}

    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Build NetworkX graph
    G = nx.Graph()

    # Add nodes
    for i, paper in enumerate(papers):
        G.add_node(
            i,
            id=paper.get("id", ""),
            title=paper.get("title", ""),
            citations=paper.get("citation_count", 0),
            year=paper.get("year", 2020)
        )

    # Add edges (top-k similar papers per node)
    for i in range(len(papers)):
        # Get similarities for this paper
        sims = similarities[i].copy()
        sims[i] = -1  # Ignore self-similarity

        # Get top-k most similar
        top_indices = np.argsort(sims)[-max_edges_per_node:][::-1]

        for j in top_indices:
            if sims[j] >= threshold and i < j:  # Add edge only once
                G.add_edge(i, j, weight=float(sims[j]))

    logger.info(f"Built network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Detect communities
    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G))

        # Assign community IDs to nodes
        community_map = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                community_map[node] = comm_id

        logger.info(f"Detected {len(communities)} communities")

    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        community_map = {i: 0 for i in range(len(papers))}

    # Convert to JSON-serializable format
    nodes = []
    for node in G.nodes(data=True):
        node_id, attrs = node
        nodes.append({
            "id": node_id,
            "paperId": attrs.get("id", ""),
            "title": attrs.get("title", ""),
            "citations": attrs.get("citations", 0),
            "year": attrs.get("year", 2020),
            "community": community_map.get(node_id, 0)
        })

    edges = []
    for edge in G.edges(data=True):
        source, target, attrs = edge
        edges.append({
            "source": source,
            "target": target,
            "weight": attrs.get("weight", 0.5)
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "num_communities": len(set(community_map.values()))
    }
