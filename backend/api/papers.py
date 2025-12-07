"""
Papers API Endpoint
Get paper details, similar papers, etc.
"""

from fastapi import APIRouter, HTTPException, Query
from models.schemas import PaperDetail, PaperSummary, NetworkData
from typing import List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()

def get_app_state():
    """Get app state from main.py"""
    from main import get_app_state
    return get_app_state()

@router.get("", response_model=dict)
async def get_all_papers(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get all papers with pagination
    
    Args:
        limit: Number of papers to return (1-100)
        offset: Number of papers to skip
        
    Returns:
        Dict with results, total, and has_more
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')
        
        if papers_df is None:
            raise HTTPException(status_code=500, detail="Server not ready")
        
        # Get paginated subset
        total = len(papers_df)
        papers_subset = papers_df.iloc[offset:offset + limit]
        
        results = []
        for _, paper in papers_subset.iterrows():
            results.append({
                "paper": {
                    "id": str(paper.get("id", "")),
                    "title": str(paper.get("title", "")),
                    "abstract": str(paper.get("abstract", "")),
                    "year": int(paper.get("year", 2020)),
                    "citation_count": int(paper.get("citation_count", 0)),
                    "num_authors": int(paper.get("num_authors", 0)),
                    "doi": paper.get("doi"),
                    "subject_areas": paper.get("subject_areas", []) if isinstance(paper.get("subject_areas"), list) else []
                },
                "relevance": 1.0  # No relevance score for browse-all
            })
        
        return {
            "results": results,
            "total": total,
            "has_more": (offset + limit) < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get all papers failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{paper_id}", response_model=PaperDetail)
async def get_paper(paper_id: str):
    """
    Get full details for a single paper

    Args:
        paper_id: Paper ID

    Returns:
        PaperDetail with all fields
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')

        if papers_df is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Find paper
        paper_row = papers_df[papers_df["id"] == paper_id]

        if paper_row.empty:
            raise HTTPException(status_code=404, detail="Paper not found")

        paper_data = paper_row.iloc[0].to_dict()

        return PaperDetail(
            id=str(paper_data.get("id", "")),
            scopus_id=str(paper_data.get("scopus_id", "")),
            doi=paper_data.get("doi"),
            title=str(paper_data.get("title", "")),
            abstract=str(paper_data.get("abstract", "")),
            year=int(paper_data.get("year", 2020)),
            citation_count=int(paper_data.get("citation_count", 0)),
            authors=str(paper_data.get("authors", "")) if paper_data.get("authors") else None,
            affiliations=str(paper_data.get("affiliations", "")) if paper_data.get("affiliations") else None,
            subject_areas=paper_data.get("subject_areas", []) if isinstance(paper_data.get("subject_areas"), list) else [],
            num_authors=int(paper_data.get("num_authors", 0)),
            num_references=int(paper_data.get("num_references", 0)),
            abstract_length=int(paper_data.get("abstract_length", 0))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get paper failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{paper_id}/similar", response_model=List[PaperSummary])
async def get_similar_papers(
    paper_id: str,
    limit: int = Query(5, ge=1, le=20)
):
    """
    Get papers similar to this one

    Args:
        paper_id: Paper ID
        limit: Number of similar papers (1-20)

    Returns:
        List of PaperSummary
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')
        embeddings = state.get('embeddings')

        if papers_df is None or embeddings is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Find paper index
        paper_row = papers_df[papers_df["id"] == paper_id]

        if paper_row.empty:
            raise HTTPException(status_code=404, detail="Paper not found")

        paper_index = paper_row.index[0]
        paper_embedding = embeddings[paper_index]

        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity(
            paper_embedding.reshape(1, -1),
            embeddings
        )[0]

        # Get top-k similar (excluding self)
        similarities[paper_index] = -1  # Exclude self
        top_indices = np.argsort(similarities)[-limit:][::-1]

        # Build response
        similar_papers = []
        for idx in top_indices:
            if idx >= len(papers_df):
                continue

            paper_data = papers_df.iloc[idx].to_dict()

            similar_papers.append(PaperSummary(
                id=str(paper_data.get("id", "")),
                title=str(paper_data.get("title", "")),
                year=int(paper_data.get("year", 2020)),
                citations=int(paper_data.get("citation_count", 0)),
                subjects=paper_data.get("subject_areas", []) if isinstance(paper_data.get("subject_areas"), list) else [],
                authors=str(paper_data.get("authors", "")) if paper_data.get("authors") else None
            ))

        return similar_papers

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get similar papers failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/network", response_model=NetworkData)
async def get_network(
    paper_ids: List[str],
    threshold: float = Query(0.6, ge=0.0, le=1.0)
):
    """
    Build similarity network for given papers

    Args:
        paper_ids: List of paper IDs
        threshold: Similarity threshold

    Returns:
        NetworkData with nodes and edges
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')
        embeddings = state.get('embeddings')

        if papers_df is None or embeddings is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Get papers and embeddings
        papers = []
        paper_embeddings = []

        for paper_id in paper_ids:
            paper_row = papers_df[papers_df["id"] == paper_id]

            if not paper_row.empty:
                idx = paper_row.index[0]
                papers.append(paper_row.iloc[0].to_dict())
                paper_embeddings.append(embeddings[idx])

        if not papers:
            raise HTTPException(status_code=404, detail="No papers found")

        paper_embeddings = np.array(paper_embeddings)

        # Build network
        from services.network import build_similarity_network

        network_data = build_similarity_network(
            papers=papers,
            embeddings=paper_embeddings,
            threshold=threshold
        )

        return NetworkData(**network_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Network generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
