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
async def get_papers_by_category(category_name: str):
    """
    Get all papers in a category and its subcategories

    Args:
        category_name: Category name from treemap

    Returns:
        List of papers with full details
    """
    try:
        # Get app state
        state = get_app_state()
        papers_df = state.get('papers_df')
        treemap_data = state.get('treemap_data')

        if papers_df is None or treemap_data is None:
            raise HTTPException(status_code=500, detail="Server not ready")

        # Get all subcategories for this category
        subcategories = get_all_subcategories(category_name, treemap_data)
        logger.info(f"Category '{category_name}' has {len(subcategories)} subcategories")

        # Filter papers by subject_areas matching any subcategory (substring match)
        matching_papers = []

        for _, paper_row in papers_df.iterrows():
            subject_areas = paper_row.get('subject_areas', [])
            if not hasattr(subject_areas, '__iter__'):
                continue

            # Convert to list if it's a numpy array
            subject_list = list(subject_areas) if hasattr(subject_areas, '__iter__') else []

            # Check if any subcategory name appears as a word in any subject area
            matches = False
            for subcat in subcategories:
                for subject in subject_list:
                    # Use word boundary matching to avoid "Physics" matching "Biophysics"
                    pattern = r'\b' + re.escape(subcat.lower()) + r'\b'
                    if re.search(pattern, str(subject).lower()):
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

        logger.info(f"Category '{category_name}' -> {len(matching_papers)} papers")

        return {
            "category": category_name,
            "results": matching_papers,
            "count": len(matching_papers)
        }

    except Exception as e:
        logger.error(f"Category filter failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_all_subcategories(category_name: str, treemap_data: dict) -> List[str]:
    """
    Recursively get all subcategories for a given category

    Args:
        category_name: Parent category name
        treemap_data: Treemap hierarchy data

    Returns:
        List of category names including the parent and all descendants
    """
    labels = treemap_data.get('labels', [])
    parents = treemap_data.get('parents', [])

    # Start with the category itself
    all_categories = [category_name]

    # Find all direct children
    children = [labels[i] for i, parent in enumerate(parents) if parent == category_name]

    # Recursively get subcategories for each child
    for child in children:
        all_categories.extend(get_all_subcategories(child, treemap_data))

    return list(set(all_categories))  # Remove duplicates
