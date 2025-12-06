"""
Analyze API Endpoint
Handles AI analysis (summaries + stance detection)
"""

from fastapi import APIRouter, HTTPException
from models.schemas import AnalyzeRequest, AnalyzeResponse, AnalysisResult, StanceType
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def get_app_state():
    """Get app state from main.py"""
    from main import get_app_state
    return get_app_state()

@router.post("/", response_model=AnalyzeResponse)
async def analyze_papers(request: AnalyzeRequest):
    """
    AI analysis of papers (summaries + stance detection)

    Args:
        request: AnalyzeRequest with query, paper_ids, optional api_key

    Returns:
        AnalyzeResponse with summaries and stances
    """
    start_time = time.time()

    try:
        # Get app state
        state = get_app_state()
        papers_df = state.get('papers_df')

        if papers_df is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Get papers from IDs
        papers_to_analyze = []
        for paper_id in request.paper_ids:
            paper_row = papers_df[papers_df["id"] == paper_id]

            if not paper_row.empty:
                paper_data = paper_row.iloc[0].to_dict()
                papers_to_analyze.append({
                    "id": str(paper_data.get("id", "")),
                    "title": str(paper_data.get("title", "")),
                    "abstract": str(paper_data.get("abstract", ""))
                })

        if not papers_to_analyze:
            raise HTTPException(status_code=404, detail="No papers found")

        # Perform AI analysis
        from services.llm import analyze_papers as analyze_fn

        analyses = await analyze_fn(
            papers=papers_to_analyze,
            query=request.query,
            api_key=request.api_key
        )

        # Convert to response format
        results = [
            AnalysisResult(
                id=a["id"],
                summary=a["summary"],
                stance=StanceType(a["stance"])
            )
            for a in analyses
        ]

        took_ms = (time.time() - start_time) * 1000

        logger.info(f"Analyzed {len(results)} papers in {took_ms:.0f}ms")

        return AnalyzeResponse(
            query=request.query,
            analyses=results,
            count=len(results),
            took_ms=took_ms
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
