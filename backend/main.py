"""
FastAPI Backend for Research Paper Explorer
Wraps existing Streamlit logic into REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import API routers (will create these next)
from api import search, analyze, stats, papers, categories

# Global state for models and data
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - loads models/data on startup
    Similar to Streamlit's @st.cache_resource pattern
    """
    logger.info("🚀 Starting up backend server...")

    # Load heavy resources once at startup
    from services.vector_db import load_vector_db
    from services.data_loader import load_papers, load_embeddings, load_metadata

    try:
        logger.info("📊 Loading papers dataset...")
        app_state['papers_df'] = load_papers()
        logger.info(f"✅ Loaded {len(app_state['papers_df'])} papers")

        logger.info("🧠 Loading embeddings...")
        app_state['embeddings'] = load_embeddings()
        logger.info(f"✅ Loaded embeddings: {app_state['embeddings'].shape}")

        logger.info("🔍 Initializing vector database...")
        app_state['vector_db'] = load_vector_db()
        logger.info("✅ Vector DB ready")

        logger.info("📈 Loading metadata...")
        metadata = load_metadata()
        app_state['metadata'] = metadata
        app_state['treemap_data'] = metadata.get('treemap_data')
        logger.info("✅ Metadata loaded")

        logger.info("🎉 Server ready!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise

    yield  # Server runs here

    # Cleanup on shutdown
    logger.info("👋 Shutting down...")
    app_state.clear()

# Create FastAPI app
app = FastAPI(
    title="Research Paper Explorer API",
    description="Backend API for semantic search and AI analysis of 19,523 academic papers",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow Next.js frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "https://*.vercel.app",   # Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(analyze.router, prefix="/api/analyze", tags=["analyze"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(categories.router, prefix="/api/categories", tags=["categories"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Research Paper Explorer API",
        "version": "1.0.0",
        "papers_loaded": len(app_state.get('papers_df', [])),
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "papers_loaded": app_state.get('papers_df') is not None,
        "embeddings_loaded": app_state.get('embeddings') is not None,
        "vector_db_ready": app_state.get('vector_db') is not None,
        "metadata_loaded": app_state.get('metadata') is not None,
    }

# Make app_state accessible to routers
def get_app_state():
    return app_state

# Export for use in API routes
__all__ = ['app', 'get_app_state']
