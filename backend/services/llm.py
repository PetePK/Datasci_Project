"""
LLM Service - Claude AI for summaries and stance detection
Wraps Anthropic API calls from Streamlit app
"""

import anthropic
import asyncio
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Get API key from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

async def generate_summary(paper: dict, client: anthropic.AsyncAnthropic) -> dict:
    """
    Generate one-sentence summary for a paper
    Same prompt as Streamlit app (optimized version)

    Args:
        paper: Dict with 'id', 'title', 'abstract'
        client: AsyncAnthropic client

    Returns:
        dict with 'id' and 'summary'
    """
    prompt = f"""Summarize in one sentence:

{paper.get('abstract', '')[:800]}

Summary:"""

    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=80,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        summary = response.content[0].text.strip()

        return {
            "id": paper["id"],
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Summary generation failed for paper {paper['id']}: {e}")
        return {
            "id": paper["id"],
            "summary": "Summary unavailable"
        }

async def detect_stance(paper: dict, query: str, client: anthropic.AsyncAnthropic) -> str:
    """
    Detect paper stance relative to query
    Same prompt as Streamlit app

    Args:
        paper: Dict with 'abstract'
        query: User's research question
        client: AsyncAnthropic client

    Returns:
        'SUPPORT', 'CONTRADICT', or 'NEUTRAL'
    """
    prompt = f"""Query: {query}

Abstract: {paper.get('abstract', '')[:600]}

Does this paper SUPPORT, CONTRADICT, or is NEUTRAL to the query? One word only."""

    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        stance_text = response.content[0].text.strip().upper()

        # Parse response
        if 'SUPPORT' in stance_text or 'ENTAIL' in stance_text:
            return 'SUPPORT'
        elif 'CONTRADICT' in stance_text:
            return 'CONTRADICT'
        else:
            return 'NEUTRAL'

    except Exception as e:
        logger.error(f"Stance detection failed: {e}")
        return 'NEUTRAL'

async def analyze_papers(papers: List[dict], query: str, api_key: str = None) -> List[dict]:
    """
    Analyze multiple papers in parallel (summaries + stance)
    Same async logic as Streamlit app

    Args:
        papers: List of paper dicts
        query: Research question
        api_key: Anthropic API key (optional, uses env var if not provided)

    Returns:
        List of dicts with 'id', 'summary', 'stance'
    """
    if not api_key:
        api_key = ANTHROPIC_API_KEY

    if not api_key:
        logger.warning("No API key provided, analysis skipped")
        return [
            {
                "id": p["id"],
                "summary": "Analysis unavailable (no API key)",
                "stance": "NEUTRAL"
            }
            for p in papers
        ]

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Generate summaries and stances in parallel
    summary_tasks = [generate_summary(p, client) for p in papers]
    stance_tasks = [detect_stance(p, query, client) for p in papers]

    summaries = await asyncio.gather(*summary_tasks)
    stances = await asyncio.gather(*stance_tasks)

    # Combine results
    results = []
    for i, paper in enumerate(papers):
        results.append({
            "id": paper["id"],
            "summary": summaries[i]["summary"],
            "stance": stances[i]
        })

    logger.info(f"Analyzed {len(results)} papers")

    return results
