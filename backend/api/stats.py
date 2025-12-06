"""
Stats API Endpoint
Provides dashboard statistics and visualizations
"""

from fastapi import APIRouter, HTTPException
from models.schemas import StatsResponse, PaperSummary, TreeMapData
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

def get_app_state():
    """Get app state from main.py"""
    from main import get_app_state
    return get_app_state()

@router.get("/", response_model=StatsResponse)
async def get_stats():
    """
    Get overall dataset statistics

    Returns:
        StatsResponse with counts, distributions, top papers
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')
        metadata = state.get('metadata', {})

        if papers_df is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Basic stats
        total_papers = len(papers_df)
        year_range = [int(papers_df["year"].min()), int(papers_df["year"].max())]
        total_citations = int(papers_df["citation_count"].sum())

        # Count unique subjects
        unique_subjects = set()
        for subjects in papers_df["subject_areas"]:
            if isinstance(subjects, list):
                unique_subjects.update(subjects)
        unique_subjects_count = len(unique_subjects)

        # Papers by year
        papers_by_year = papers_df.groupby("year").size().to_dict()
        papers_by_year = {int(k): int(v) for k, v in papers_by_year.items()}

        # Papers by subject (top 20)
        subject_counts = {}
        for subjects in papers_df["subject_areas"]:
            if isinstance(subjects, list):
                for subject in subjects:
                    subject_counts[subject] = subject_counts.get(subject, 0) + 1

        # Sort and take top 20
        top_subjects = dict(sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:20])

        # Top cited papers (top 10)
        top_papers_df = papers_df.nlargest(10, "citation_count")
        top_cited_papers = []

        for _, row in top_papers_df.iterrows():
            top_cited_papers.append(PaperSummary(
                id=str(row["id"]),
                title=str(row["title"]),
                year=int(row["year"]),
                citations=int(row["citation_count"]),
                subjects=row["subject_areas"] if isinstance(row["subject_areas"], list) else [],
                authors=str(row["authors"]) if row.get("authors") else None
            ))

        return StatsResponse(
            total_papers=total_papers,
            year_range=year_range,
            total_citations=total_citations,
            unique_subjects=unique_subjects_count,
            papers_by_year=papers_by_year,
            papers_by_subject=top_subjects,
            top_cited_papers=top_cited_papers
        )

    except Exception as e:
        logger.error(f"Stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/treemap", response_model=TreeMapData)
async def get_treemap():
    """
    Get pre-computed tree map data

    Returns:
        TreeMapData with labels, parents, values, colors
    """
    try:
        state = get_app_state()
        metadata = state.get('metadata', {})

        treemap_data = metadata.get('treemap_data')

        if not treemap_data:
            raise HTTPException(status_code=404, detail="Treemap data not found")

        return TreeMapData(**treemap_data)

    except Exception as e:
        logger.error(f"Treemap failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
