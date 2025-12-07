"""
LLM Service for AI-powered insights and recommendations

This service provides:
1. Search insights generation (on-demand)
2. Topic insights (pre-computed, loaded from file)
3. Research recommendations (monthly updates)

Uses Claude 3.5 Haiku for fast, cost-effective generation.
"""

import os
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Model configuration
HAIKU_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.3


class LLMService:
    """Service for AI-powered insights using Claude 3.5 Haiku"""

    def __init__(self):
        self.client = client
        self.model = HAIKU_MODEL

    async def generate_search_insights(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate real-time insights for search queries (Use Case 2)

        Args:
            query: User's search query
            search_results: List of papers from search (with title, year, citations)

        Returns:
            Dict with insights (relevance_summary, key_papers, research_directions, search_tips)

        Raises:
            Exception: If LLM API call fails
        """
        try:
            # Extract key information
            titles = [paper.get('title', '') for paper in search_results[:15]]
            years = [paper.get('year', '') for paper in search_results]
            citations = [paper.get('citations', 0) for paper in search_results]

            # Compute stats
            year_dist = pd.Series(years).value_counts().sort_index()
            avg_citations = np.mean([c for c in citations if c > 0]) if any(citations) else 0

            # Build context
            context = f"""
Search Query: "{query}"

Found {len(search_results)} papers

Top Papers:
{chr(10).join([f"- {title}" for title in titles[:8]])}

Distribution:
{chr(10).join([f"{year}: {count} papers" for year, count in year_dist.items()])}

Average Citations: {avg_citations:.1f}
"""

            prompt = f"""Analyze these search results and provide insights for researchers.

{context}

Provide a JSON response with:
1. "relevance_summary": What are the main themes in these results?
2. "key_papers": 3 papers that seem most influential (from the titles)
3. "research_directions": 2-3 emerging directions based on these papers
4. "search_tips": 1-2 tips to refine the search or explore related topics

Be specific and actionable. Help the researcher understand what they found."""

            # Call Claude Haiku
            message = self.client.messages.create(
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = message.content[0].text

            # Try to parse as JSON, fallback to structured text
            try:
                insights = json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, create structured response
                insights = {
                    "relevance_summary": response_text[:500],
                    "key_papers": [],
                    "research_directions": [],
                    "search_tips": [],
                    "raw_response": response_text
                }

            # Add metadata
            insights['generated_at'] = datetime.now().isoformat()
            insights['query'] = query
            insights['result_count'] = len(search_results)

            return insights

        except Exception as e:
            logger.error(f"Error generating search insights: {e}")
            raise Exception(f"Failed to generate insights: {str(e)}")

    def load_topic_insights(self, topic_id: str) -> Optional[Dict[str, Any]]:
        """
        Load pre-computed topic insights from file (Use Case 1)

        Args:
            topic_id: ID of the topic

        Returns:
            Dict with insights or None if not found
        """
        try:
            # Load from pre-generated file
            insights_path = Path(__file__).parent.parent.parent / 'data' / 'topic_insights.json'

            if not insights_path.exists():
                logger.warning(f"Topic insights file not found: {insights_path}")
                return None

            with open(insights_path, 'r') as f:
                all_insights = json.load(f)

            # Find insights for this topic
            for topic in all_insights:
                if topic.get('topic_id') == topic_id:
                    return topic

            return None

        except Exception as e:
            logger.error(f"Error loading topic insights: {e}")
            return None

    def load_research_recommendations(self) -> Optional[Dict[str, Any]]:
        """
        Load pre-generated research recommendations (Use Case 3)

        Returns:
            Dict with recommendations or None if not found
        """
        try:
            recommendations_path = Path(__file__).parent.parent.parent / 'data' / 'research_recommendations.json'

            if not recommendations_path.exists():
                logger.warning(f"Recommendations file not found: {recommendations_path}")
                return None

            with open(recommendations_path, 'r') as f:
                recommendations = json.load(f)

            return recommendations

        except Exception as e:
            logger.error(f"Error loading research recommendations: {e}")
            return None


# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService:
    """Get or create LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
