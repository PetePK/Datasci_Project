"""
API endpoint for research recommendations

Endpoint:
- GET /api/recommendations - Get AI-generated research opportunities
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from services.llm_service import get_llm_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


# Response models
class Recommendation(BaseModel):
    """Single research recommendation"""
    topic: str
    rationale: str
    potential_impact: str
    difficulty: str


class RecommendationsResponse(BaseModel):
    """Response model for recommendations"""
    success: bool
    recommendations: Optional[List[Recommendation]] = None
    generated_at: Optional[str] = None
    valid_until: Optional[str] = None
    error: Optional[str] = None


@router.get("", response_model=RecommendationsResponse)
@router.get("/", response_model=RecommendationsResponse)
async def get_research_recommendations():
    """
    Get AI-generated research opportunities (Use Case 3)

    Returns 5 emerging research topics that would be valuable to pursue
    based on current research landscape and global trends.

    These are pre-generated monthly, so response is instant.

    Returns:
        - recommendations: List of 5 research opportunities
        - generated_at: When recommendations were generated
        - valid_until: When to regenerate (30 days)
    """
    try:
        llm_service = get_llm_service()

        # Load pre-generated recommendations
        data = llm_service.load_research_recommendations()

        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Research recommendations not found. Please run generation script."
            )

        # Parse recommendations
        recommendations = [
            Recommendation(**rec)
            for rec in data.get('recommendations', [])
        ]

        return RecommendationsResponse(
            success=True,
            recommendations=recommendations,
            generated_at=data.get('generated_at'),
            valid_until=data.get('valid_until')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_research_recommendations: {e}")
        return RecommendationsResponse(
            success=False,
            error=str(e)
        )
