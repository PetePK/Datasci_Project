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

@router.get("/trends/{topic}")
async def get_topic_trends(topic: str):
    """
    Get publication trends over time for a specific topic (Level 1)

    Args:
        topic: Topic name from treemap

    Returns:
        Trend data with years and paper counts
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')

        if papers_df is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Filter papers by topic in subject_areas
        filtered_papers = papers_df[
            papers_df['subject_areas'].apply(
                lambda subjects: isinstance(subjects, list) and
                any(topic.lower() in str(s).lower() for s in subjects)
            )
        ]

        if len(filtered_papers) == 0:
            return {"years": [], "counts": [], "total": 0}

        # Group by year and count
        trend_data = filtered_papers.groupby('year').size().reset_index(name='count')

        return {
            "years": trend_data['year'].tolist(),
            "counts": trend_data['count'].tolist(),
            "total": len(filtered_papers)
        }

    except Exception as e:
        logger.error(f"Trends failed for topic {topic}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/level2-trends/{topic}")
async def get_level2_trends(topic: str):
    """
    Get Level 2 trends: dual line chart (paper count + citations over time)

    Args:
        topic: Level 2 topic name (e.g., 'Analytical Chemistry')

    Returns:
        Time series data with years, paper counts, and citation counts
    """
    try:
        state = get_app_state()
        papers_df = state.get('papers_df')

        if papers_df is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Filter papers using same logic as categories endpoint
        topic_lower = topic.lower()
        matching_indices = []

        for idx, paper_row in papers_df.iterrows():
            subject_areas = paper_row.get('subject_areas', [])
            if not hasattr(subject_areas, '__iter__') or isinstance(subject_areas, str):
                continue

            subject_list = list(subject_areas) if hasattr(subject_areas, '__iter__') else []

            matches = False
            for subject in subject_list:
                subject_lower = str(subject).lower()
                if topic_lower in subject_lower or subject_lower in topic_lower:
                    matches = True
                    break

            if matches:
                matching_indices.append(idx)

        if len(matching_indices) == 0:
            logger.warning(f"No papers found for topic: {topic}")
            return {"years": [], "paper_counts": [], "citation_counts": [], "total": 0}

        # Get filtered papers
        filtered_papers = papers_df.loc[matching_indices]

        # Group by year and aggregate
        yearly_data = filtered_papers.groupby('year').agg({
            'id': 'count',  # Paper count
            'citation_count': 'sum'  # Total citations
        }).reset_index()
        yearly_data.columns = ['year', 'paper_count', 'citation_count']

        logger.info(f"Level 2 trends for '{topic}': {len(filtered_papers)} total papers across {len(yearly_data)} years")

        return {
            "years": yearly_data['year'].tolist(),
            "paper_counts": yearly_data['paper_count'].tolist(),
            "citation_counts": yearly_data['citation_count'].tolist(),
            "total": len(filtered_papers),
            "avg_citations": float(filtered_papers['citation_count'].mean())
        }

    except Exception as e:
        logger.error(f"Level 2 trends failed for topic {topic}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
