"""
API endpoints for LLM-powered insights

Endpoints:
- POST /api/insights/search - Generate insights for search results
- GET /api/insights/topic/{topic_id} - Get pre-computed topic insights
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from services.llm_service import get_llm_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/insights", tags=["insights"])


# Request/Response models
class SearchInsightRequest(BaseModel):
    """Request model for search insights"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    results: List[Dict[str, Any]] = Field(..., max_items=50, description="Search results")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning for medical diagnosis",
                "results": [
                    {"title": "Deep Learning for Medical Diagnosis", "year": 2023, "citations": 45},
                    {"title": "ML in Healthcare", "year": 2022, "citations": 78}
                ]
            }
        }


class InsightResponse(BaseModel):
    """Response model for insights"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/search", response_model=InsightResponse)
async def generate_search_insights(request: SearchInsightRequest):
    """
    Generate AI-powered insights for search results (Use Case 2)

    This endpoint runs asynchronously on each search query.
    Expected response time: ~3-7 seconds

    Returns:
        - relevance_summary: Main themes in results
        - key_papers: Most influential papers
        - research_directions: Emerging directions
        - search_tips: Tips to refine search
    """
    try:
        llm_service = get_llm_service()

        # Generate insights
        insights = await llm_service.generate_search_insights(
            query=request.query,
            search_results=request.results
        )

        return InsightResponse(
            success=True,
            data=insights
        )

    except Exception as e:
        logger.error(f"Error in generate_search_insights: {e}")
        return InsightResponse(
            success=False,
            error=str(e)
        )


@router.get("/topic/{topic_id}", response_model=InsightResponse)
async def get_topic_insights(topic_id: str):
    """
    Get pre-computed insights for a topic (Use Case 1)

    These insights are pre-generated and cached, so response is instant.
    Used when user clicks on treemap topics.

    Args:
        topic_id: ID of the topic

    Returns:
        - trend_summary: Summary of trends
        - key_themes: Emerging themes
        - recommendations: Research recommendations
        - momentum: rising/stable/declining
    """
    try:
        llm_service = get_llm_service()

        # Load pre-computed insights
        insights = llm_service.load_topic_insights(topic_id)

        if insights is None:
            raise HTTPException(status_code=404, detail=f"Insights not found for topic: {topic_id}")

        return InsightResponse(
            success=True,
            data=insights
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_topic_insights: {e}")
        return InsightResponse(
            success=False,
            error=str(e)
        )
