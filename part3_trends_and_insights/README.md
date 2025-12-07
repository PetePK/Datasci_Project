# Part 3: Trends & AI Insights

Analyzed research trends and built AI-powered insights.

## What We Built

1. **Time Series Analysis**: Track paper counts and citations per year for each topic
2. **Level 2 Trends**: Dual-axis charts (papers + citations) for subtopics
3. **AI Insights**: Claude generates research suggestions based on search results

## Integrated Features

- `/api/stats/level2-trends/{topic}` - Returns yearly trends data
- `/api/insights/search` - LLM analyzes search results and suggests opportunities
- Frontend displays trend charts on treemap drill-down
- Search page shows AI-generated insights sidebar

Output: Trend visualizations + AI recommendations in the web app.
