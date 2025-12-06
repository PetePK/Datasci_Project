"""
Vector Database Service
Wraps ChromaDB operations from Streamlit app
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Path to vector DB
VECTOR_DB_PATH = Path(__file__).parent.parent.parent / "data" / "vector_db"

# Global model (loaded once)
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Load sentence transformer model (cached globally)
    Same model as used in Streamlit: all-MiniLM-L6-v2
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")

    return _embedding_model

def load_vector_db():
    """
    Initialize ChromaDB persistent client
    Same as Streamlit's get_chroma_client()
    """
    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(f"Vector DB not found: {VECTOR_DB_PATH}")

    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    # Get or create collection
    try:
        collection = client.get_collection(name="papers")
        logger.info(f"✅ ChromaDB collection loaded: {collection.count()} papers")
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        raise

    return {"client": client, "collection": collection}

def search_papers(query: str, n_results: int = 20, vector_db: dict = None) -> dict:
    """
    Semantic search using ChromaDB
    Same logic as Streamlit app

    Args:
        query: Search query text
        n_results: Number of results to return
        vector_db: Vector DB dict from load_vector_db()

    Returns:
        dict with ids, distances, metadatas
    """
    if vector_db is None:
        raise ValueError("vector_db not provided")

    collection = vector_db["collection"]
    model = get_embedding_model()

    # Embed query
    query_embedding = model.encode([query])[0]

    # Search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    logger.info(f"Search: '{query[:50]}...' -> {len(results['ids'][0])} results")

    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else []
    }

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    """
    from sklearn.metrics.pairwise import cosine_similarity

    return float(cosine_similarity(
        embedding1.reshape(1, -1),
        embedding2.reshape(1, -1)
    )[0][0])
