"""
Generate Forward-Looking Research Recommendations

This script analyzes existing research and current world trends to suggest
5 potential research topics that would be valuable to pursue right now.

Use Case: Homepage feature showing "Emerging Research Opportunities"
- Run once per month
- Save to database
- Display on main page
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from anthropic import Anthropic
import json
from datetime import datetime

# Load API key from environment
if 'ANTHROPIC_API_KEY' not in os.environ:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

print("="*80)
print("GENERATING RESEARCH RECOMMENDATIONS")
print("="*80)
print()

# Load data
data_path = Path('../../data/processed')
papers_df = pd.read_parquet(data_path / 'papers.parquet')
print(f"[OK] Loaded {len(papers_df):,} papers")
print(f"    Years: {papers_df['year'].min()} - {papers_df['year'].max()}")
print()

# Initialize Anthropic client
client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
print(f"[OK] Anthropic API configured")
print()


def analyze_research_landscape(papers_df):
    """
    Analyze the research dataset to understand what's been done
    """

    # Get statistics
    total_papers = len(papers_df)
    year_range = f"{papers_df['year'].min()}-{papers_df['year'].max()}"

    # Recent trends (last 2 years)
    recent_papers = papers_df[papers_df['year'] >= papers_df['year'].max() - 1]

    # Top subjects in recent years
    all_subjects = []
    for subjects in recent_papers['subject_areas']:
        if isinstance(subjects, np.ndarray):
            all_subjects.extend(subjects.tolist())
        elif isinstance(subjects, list):
            all_subjects.extend(subjects)

    from collections import Counter
    top_subjects = Counter(all_subjects).most_common(15)

    # Growth analysis
    papers_by_year = papers_df.groupby('year').size()
    recent_growth = ((papers_by_year.iloc[-1] / papers_by_year.iloc[-2]) - 1) * 100 if len(papers_by_year) > 1 else 0

    # High-impact recent papers (recent + high citations)
    high_impact = recent_papers[recent_papers['citation_count'] > recent_papers['citation_count'].quantile(0.75)]
    high_impact_titles = high_impact.nlargest(10, 'citation_count')['title'].tolist()

    return {
        'total_papers': total_papers,
        'year_range': year_range,
        'recent_papers_count': len(recent_papers),
        'recent_growth_rate': recent_growth,
        'top_subjects': top_subjects,
        'high_impact_titles': high_impact_titles
    }


def generate_research_recommendations(landscape_data):
    """
    Use Claude 3.5 Haiku to generate forward-looking research recommendations

    Returns:
        JSON with 5 research recommendations
    """

    # Prepare context
    context = f"""
RESEARCH LANDSCAPE ANALYSIS ({landscape_data['year_range']})

Dataset Overview:
- Total papers: {landscape_data['total_papers']:,}
- Recent papers (last 2 years): {landscape_data['recent_papers_count']:,}
- Recent growth rate: {landscape_data['recent_growth_rate']:.1f}%

Top Research Areas (Recent):
{chr(10).join([f"- {subject}: {count} papers" for subject, count in landscape_data['top_subjects'][:10]])}

High-Impact Recent Papers:
{chr(10).join([f"- {title}" for title in landscape_data['high_impact_titles'][:8]])}

Current Date: {datetime.now().strftime('%B %Y')}
"""

    prompt = f"""You are a research strategist analyzing current research trends and global developments.

{context}

Based on this research landscape and current world situation (2025), identify 5 VERY SPECIFIC, NARROW research opportunities.

REQUIREMENTS:
1. Topics must be NICHE and SPECIFIC (not broad categories)
2. Must have clear connection to current world events/situation
3. Rationale must start with "Based on [current situation/trend]..." and be concise (2-3 sentences max)
4. Each topic should be unique and actionable

For each opportunity, provide:
- topic: A SPECIFIC, NARROW research question or topic (be precise, not generic)
- rationale: Start with "Based on [current world situation]..." then explain why this specific topic is interesting NOW. Keep it concise and direct.
- potential_impact: What could this research achieve (1-2 sentences)
- difficulty: "Beginner-friendly", "Intermediate", or "Advanced"

EXAMPLES OF GOOD vs BAD:
❌ BAD (too broad): "AI in Healthcare"
✅ GOOD (specific): "Machine Learning for Early Detection of Antibiotic-Resistant Infections in ICU Patients"

❌ BAD (vague connection): "Climate research is important"
✅ GOOD (clear connection): "Based on the increasing frequency of extreme weather events and recent floods in Southeast Asia, developing predictive models for urban flooding using satellite imagery and social media data is a very interesting topic that could save lives."

Current global situations to consider (December 2025):
- Ongoing energy crisis and nuclear concerns
- Extreme weather events and climate impacts
- AI boom and concerns about AI safety/alignment
- Post-pandemic healthcare system challenges
- Microplastic pollution awareness
- Space exploration advancement
- Geopolitical tensions affecting supply chains
- Mental health crisis in younger generations

Respond in JSON format:
{{
  "recommendations": [
    {{
      "topic": "...",
      "rationale": "Based on [current situation]...",
      "potential_impact": "...",
      "difficulty": "..."
    }}
  ],
  "generated_at": "{datetime.now().isoformat()}",
  "valid_until": "30 days"
}}

Make each topic NARROW and SPECIFIC, not broad categories. Connect clearly to current world events."""

    print("[1/2] Generating recommendations...")
    print(f"      Using Claude 3.5 Haiku...")

    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=2500,  # Increased for more detailed niche topics
        temperature=0.5,  # Balanced creativity and specificity
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return message.content[0].text


# Main execution
print("="*80)
print("STEP 1: Analyzing Research Landscape")
print("="*80)
print()

landscape = analyze_research_landscape(papers_df)

print(f"Analysis complete:")
print(f"  - {landscape['total_papers']:,} total papers analyzed")
print(f"  - {landscape['recent_papers_count']:,} recent papers (last 2 years)")
print(f"  - Top research areas identified: {len(landscape['top_subjects'])}")
print(f"  - High-impact papers: {len(landscape['high_impact_titles'])}")
print()

print("="*80)
print("STEP 2: Generating Research Recommendations")
print("="*80)
print()

recommendations_json = generate_research_recommendations(landscape)

print("[OK] Recommendations generated!")
print()

print("="*80)
print("RESEARCH RECOMMENDATIONS")
print("="*80)
print()
print(recommendations_json)
print()

# Save to file
output_path = Path('../../data/research_recommendations.json')
with open(output_path, 'w') as f:
    f.write(recommendations_json)

print("="*80)
print(f"[OK] Saved to {output_path}")
print("="*80)
print()

print("INTEGRATION GUIDE:")
print()
print("1. HOMEPAGE DISPLAY")
print("   - Load from research_recommendations.json")
print("   - Show in 'Emerging Research Opportunities' section")
print("   - Display all 5 recommendations with expand/collapse")
print()
print("2. UPDATE FREQUENCY")
print("   - Run this script monthly")
print("   - Or set up cron job / scheduled task")
print()
print("3. FRONTEND COMPONENT")
print("   - ResearchOpportunities.tsx")
print("   - Card-based layout")
print("   - Filter by difficulty level")
print()
print("="*80)
