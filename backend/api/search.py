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

        # DISABLED: logger.info(f"Starting search for query: '{request.query}'")

        # Perform vector search
        from services.vector_db import search_papers as search_fn

        try:
            search_results = search_fn(
                query=request.query,
                n_results=request.limit,
                vector_db=vector_db
            )
            # DISABLED: logger.info(f"Vector search completed: {len(search_results['ids'])} results")
        except Exception as ve:
            # DISABLED: logger.error(f"Vector search failed: {ve}")
            raise

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

            # Convert numpy array to list for subject_areas
            subject_areas_data = paper_data.get("subject_areas", [])
            if hasattr(subject_areas_data, '__iter__') and not isinstance(subject_areas_data, str):
                subject_areas_list = list(subject_areas_data)
            else:
                subject_areas_list = []

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
                subject_areas=subject_areas_list,
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

        # DISABLED: query_preview = request.query[:50] if len(request.query) > 50 else request.query
        # DISABLED: logger.info(f"Search '{query_preview}' -> {len(results)} results in {took_ms:.0f}ms")

        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
            took_ms=took_ms
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Write to file to avoid Windows console issues
        try:
            with open("search_error.log", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Error at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Query: {request.query}\n")
                f.write(f"Error: {e}\n")
                f.write(f"Traceback:\n{error_traceback}\n")
        except:
            pass
            
        # DISABLED: logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
