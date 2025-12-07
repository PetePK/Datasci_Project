"""
Data Loading Services
Loads papers, embeddings, and metadata
Reuses existing data files from Streamlit app
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Data paths (relative to backend/)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

def load_papers() -> pd.DataFrame:
    """
    Load papers.parquet - main dataset
    Same as Streamlit's load_papers()
    """
    parquet_path = PROCESSED_DIR / "papers.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Papers file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    logger.info(f"Loaded {len(df)} papers from {parquet_path}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df

def load_embeddings() -> np.ndarray:
    """
    Load paper_embeddings.npy
    Same as Streamlit's load_embeddings()
    """
    embeddings_path = EMBEDDINGS_DIR / "paper_embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)

    logger.info(f"Loaded embeddings: {embeddings.shape}")

    return embeddings

def load_metadata() -> dict:
    """
    Load metadata files (subject hierarchy, treemap data, etc.)
    """
    metadata = {}

    # Subject hierarchy
    subject_hierarchy_path = PROCESSED_DIR / "subject_hierarchy.json"
    if subject_hierarchy_path.exists():
        with open(subject_hierarchy_path, 'r', encoding='utf-8') as f:
            metadata['subject_hierarchy'] = json.load(f)
        logger.info("Loaded subject hierarchy")

    # Treemap data (pre-computed)
    treemap_path = PROCESSED_DIR / "treemap_data.json"
    if treemap_path.exists():
        with open(treemap_path, 'r', encoding='utf-8') as f:
            treemap_data = json.load(f)
        
        # Merge subject_groups from subject_hierarchy into treemap_data
        # This allows the categories API to know about subcategories
        if 'subject_hierarchy' in metadata:
            subject_groups = metadata['subject_hierarchy'].get('subject_groups', {})
            treemap_data['subject_groups'] = subject_groups
            logger.info(f"Added {len(subject_groups)} subject groups to treemap data")
        
        metadata['treemap_data'] = treemap_data
        logger.info("Loaded treemap data")

    # Dataset metadata
    metadata_path = PROCESSED_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata['dataset_metadata'] = json.load(f)
        logger.info("Loaded dataset metadata")

    return metadata

def load_summary_cache() -> dict:
    """
    Load AI summary cache (grows over time)
    """
    cache_path = DATA_DIR / "abstract_summaries.json"

    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        logger.info(f"Loaded {len(cache)} cached summaries")
        return cache

    return {}

def save_summary_cache(cache: dict):
    """
    Save updated summary cache
    """
    cache_path = DATA_DIR / "abstract_summaries.json"

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

    logger.info(f"Saved {len(cache)} cached summaries")
