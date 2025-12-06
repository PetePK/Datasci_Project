"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# ===== Enums =====

class StanceType(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"

class SortBy(str, Enum):
    RELEVANCE = "relevance"
    CITATIONS = "citations"
    YEAR = "year"

# ===== Request Models =====

class SearchRequest(BaseModel):
    """Request for semantic search"""
    query: str = Field(..., description="Research question or search query")
    limit: int = Field(20, ge=1, le=100, description="Number of results (1-100)")
    threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Relevance threshold")

class AnalyzeRequest(BaseModel):
    """Request for AI analysis (summaries + stance)"""
    query: str = Field(..., description="Research question for stance detection")
    paper_ids: List[str] = Field(..., description="List of paper IDs to analyze")
    api_key: Optional[str] = Field(None, description="Anthropic API key (optional)")

class FilterRequest(BaseModel):
    """Request for filtered paper list"""
    stances: Optional[List[StanceType]] = Field(None, description="Filter by stance")
    subjects: Optional[List[str]] = Field(None, description="Filter by subject areas")
    year_min: Optional[int] = Field(None, ge=2018, le=2023)
    year_max: Optional[int] = Field(None, ge=2018, le=2023)
    citations_min: Optional[int] = Field(None, ge=0)
    citations_max: Optional[int] = Field(None)
    sort_by: SortBy = Field(SortBy.RELEVANCE, description="Sort order")
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)

# ===== Response Models =====

class PaperSummary(BaseModel):
    """Minimal paper info for lists"""
    id: str
    title: str
    year: int
    citations: int
    subjects: List[str] = []
    authors: Optional[str] = None

class PaperDetail(BaseModel):
    """Full paper information"""
    id: str
    scopus_id: str
    doi: Optional[str]
    title: str
    abstract: str
    year: int
    citation_count: int
    authors: Optional[str]
    affiliations: Optional[str]
    subject_areas: List[str] = []
    num_authors: int
    num_references: int
    abstract_length: int

class SearchResult(BaseModel):
    """Single search result with relevance"""
    paper: PaperDetail
    relevance: float = Field(..., ge=0.0, le=1.0)
    distance: float

class AnalysisResult(BaseModel):
    """AI analysis result for one paper"""
    id: str
    summary: str
    stance: StanceType

class SearchResponse(BaseModel):
    """Response from search endpoint"""
    query: str
    results: List[SearchResult]
    count: int
    took_ms: float

class AnalyzeResponse(BaseModel):
    """Response from analyze endpoint"""
    query: str
    analyses: List[AnalysisResult]
    count: int
    took_ms: float

class StatsResponse(BaseModel):
    """Dashboard statistics"""
    total_papers: int
    year_range: List[int]
    total_citations: int
    unique_subjects: int
    papers_by_year: Dict[int, int]
    papers_by_subject: Dict[str, int]
    top_cited_papers: List[PaperSummary]

class TreeMapData(BaseModel):
    """Pre-computed tree map data"""
    labels: List[str]
    parents: List[str]
    values: List[int]
    colors: Optional[List[str]] = None

class NetworkData(BaseModel):
    """Network graph data"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    num_communities: int

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
