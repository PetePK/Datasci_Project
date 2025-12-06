"""
Search API Endpoint
Handles semantic search queries
"""

from fastapi import APIRouter, HTTPException, Depends
from models.schemas import SearchRequest, SearchResponse, SearchResult, PaperDetail
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def get_app_state():
    """Get app state from main.py"""
    from main import get_app_state
    return get_app_state()

@router.post("/", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """
    Semantic search for papers

    Args:
        request: SearchRequest with query, limit, threshold

    Returns:
        SearchResponse with list of relevant papers
    """
    start_time = time.time()

    try:
        # Get app state
        state = get_app_state()
        papers_df = state.get('papers_df')
        vector_db = state.get('vector_db')

        if papers_df is None or vector_db is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Perform vector search
        from services.vector_db import search_papers as search_fn

        search_results = search_fn(
            query=request.query,
            n_results=request.limit,
            vector_db=vector_db
        )

        # Build response
        results = []
        for i, paper_id in enumerate(search_results["ids"]):
            distance = search_results["distances"][i]
            # Convert L2 distance to similarity score (0-1 range)
            # ChromaDB uses L2 distance where smaller = more similar
            # Formula matches the original Streamlit implementation
            relevance = (2.0 - distance) / 2.0

            # Apply threshold
            if relevance < request.threshold:
                continue

            # Get full paper details from DataFrame
            paper_row = papers_df[papers_df["id"] == paper_id]

            if paper_row.empty:
                continue

            paper_data = paper_row.iloc[0].to_dict()

            # Convert to PaperDetail
            paper_detail = PaperDetail(
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

            results.append(SearchResult(
                paper=paper_detail,
                relevance=float(relevance),
                distance=float(distance)
            ))

        # Sort by relevance (descending)
        results.sort(key=lambda x: x.relevance, reverse=True)

        took_ms = (time.time() - start_time) * 1000

        logger.info(f"Search '{request.query[:50]}' -> {len(results)} results in {took_ms:.0f}ms")

        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
            took_ms=took_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
