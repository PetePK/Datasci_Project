# Part 4: Dashboard (Legacy Streamlit)

Original Streamlit dashboard - **migrated to Next.js + FastAPI**.

## Original Features
- Search papers with semantic similarity
- Interactive treemap of research categories
- AI-powered paper analysis
- Network visualization
- Statistical dashboard

## Migration
This Streamlit app has been **completely rebuilt** as:
- **Backend**: FastAPI (see `/backend`)
- **Frontend**: Next.js 14 (see `/frontend`)

## Why Migrate?
- Better UX/UI flexibility
- Faster performance
- Professional appearance
- Easier deployment
- Separation of concerns

## Current Status
This folder contains the **original Streamlit code** for reference. The production system uses the new Next.js + FastAPI architecture.

To run the modern version:
```bash
# Backend
cd backend && python -m uvicorn main:app --reload

# Frontend
cd frontend && pnpm dev
```

## Legacy Usage (if needed)
```bash
streamlit run app.py
```

**Note**: The new system is recommended for production use.
