"""
Category Filter API Endpoint
Handles category-based paper filtering (not semantic search)
"""

from fastapi import APIRouter, HTTPException
from models.schemas import PaperDetail, SearchResult
from typing import List
import logging
import re

logger = logging.getLogger(__name__)

router = APIRouter()

def get_app_state():
    """Get app state from main.py"""
    from main import get_app_state
    return get_app_state()

@router.get("/{category_name}")
async def get_papers_by_category(
    category_name: str,
    limit: int = 20,
    offset: int = 0
):
    """
    Get papers in the category with pagination

    Args:
        category_name: Category name from treemap
        limit: Number of papers to return (default 20)
        offset: Number of papers to skip (default 0)

    Returns:
        Paginated list of papers with full details, sorted by citation count
    """
    try:
        # Get app state
        state = get_app_state()
        papers_df = state.get('papers_df')
        treemap_data = state.get('treemap_data')

        if papers_df is None or treemap_data is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Special case: "All Papers" returns all papers
        if category_name == "All Papers":
            logger.info(f"Fetching all papers (paginated)")

            # Sort all papers by citation count
            sorted_df = papers_df.sort_values('citation_count', ascending=False)
            total_count = len(sorted_df)

            # Get paginated slice
            paginated_df = sorted_df.iloc[offset:offset + limit]

            matching_papers = []
            for idx, paper_row in paginated_df.iterrows():
                paper_data = paper_row.to_dict()
                subject_areas_data = paper_data.get('subject_areas', [])
                subject_list = list(subject_areas_data) if hasattr(subject_areas_data, '__iter__') and not isinstance(subject_areas_data, str) else []

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
                    subject_areas=subject_list,
                    num_authors=int(paper_data.get("num_authors", 0)),
                    num_references=int(paper_data.get("num_references", 0)),
                    abstract_length=int(paper_data.get("abstract_length", 0))
                )

                matching_papers.append(SearchResult(
                    paper=paper_detail,
                    relevance=1.0,
                    distance=0.0
                ))

            logger.info(f"All Papers -> {total_count} total papers (returning {len(matching_papers)} from offset {offset})")

            return {
                "category": category_name,
                "results": matching_papers,
                "count": len(matching_papers),
                "total": total_count,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count
            }

        # Get the exact category (no recursive subcategories)
        subcategories = get_all_subcategories(category_name, treemap_data)
        logger.info(f"Filtering papers for category '{category_name}'")

        # Filter papers by subject_areas matching any subcategory
        matching_papers = []

        for idx, paper_row in papers_df.iterrows():
            subject_areas = paper_row.get('subject_areas', [])
            if not hasattr(subject_areas, '__iter__') or isinstance(subject_areas, str):
                continue

            # Convert to list if it's a numpy array
            subject_list = list(subject_areas) if hasattr(subject_areas, '__iter__') else []

            # Check if any subcategory name appears in any subject area
            matches = False
            for subcat in subcategories:
                subcat_lower = subcat.lower()
                for subject in subject_list:
                    subject_lower = str(subject).lower()

                    # Try exact match first
                    if subcat_lower == subject_lower:
                        matches = True
                        break

                    # Try substring match (case-insensitive)
                    # This handles cases like "Pharmacology (medical)" in subject areas
                    if subcat_lower in subject_lower:
                        matches = True
                        break

                if matches:
                    break

            if matches:
                paper_data = paper_row.to_dict()

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
                    subject_areas=subject_list,
                    num_authors=int(paper_data.get("num_authors", 0)),
                    num_references=int(paper_data.get("num_references", 0)),
                    abstract_length=int(paper_data.get("abstract_length", 0))
                )

                # Create SearchResult with relevance=1.0 for category matches
                matching_papers.append(SearchResult(
                    paper=paper_detail,
                    relevance=1.0,
                    distance=0.0
                ))

        # Sort by citations (highest first)
        matching_papers.sort(key=lambda x: x.paper.citation_count, reverse=True)

        total_count = len(matching_papers)

        # Apply pagination
        paginated_papers = matching_papers[offset:offset + limit]

        logger.info(f"Category '{category_name}' -> {total_count} papers (returning {len(paginated_papers)} from offset {offset})")

        return {
            "category": category_name,
            "results": paginated_papers,
            "count": len(paginated_papers),
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "has_more": offset + limit < total_count
        }

    except Exception as e:
        logger.error(f"Category filter failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_all_subcategories(category_name: str, treemap_data: dict) -> List[str]:
    """
    Get all subcategories for a given category
    
    For Level 1 topics (e.g., "Medicine & Health"), returns all Level 2 subcategories
    For Level 2 topics (e.g., "Cardiology"), returns just that topic
    For "All Papers", returns empty list (handled separately)

    Args:
        category_name: Category name
        treemap_data: Treemap hierarchy data (must contain subject_groups)

    Returns:
        List of category names to match against paper subject_areas
    """
    # Special case: All Papers
    if category_name == "All Papers":
        return []
    
    # Get subject hierarchy
    subject_groups = treemap_data.get('subject_groups', {})
    
    # Check if this is a Level 1 topic (key in subject_groups)
    if category_name in subject_groups:
        # Return all subcategories for this Level 1 topic
        subcategories = subject_groups[category_name]
        logger.info(f"Level 1 topic '{category_name}' has {len(subcategories)} subcategories")
        return subcategories
    
    # Check if this is a Level 2 topic (value in any subject_groups list)
    for level1, subtopics in subject_groups.items():
        if category_name in subtopics:
            logger.info(f"Level 2 topic '{category_name}' (under '{level1}')")
            return [category_name]
    
    # If not found in hierarchy, return the category itself
    logger.warning(f"Category '{category_name}' not found in hierarchy, using exact match")
    return [category_name]
