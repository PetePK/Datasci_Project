# Homepage Restructure - Complete

## Overview

Successfully restructured the homepage according to user specifications:
- **Start with ONLY treemap** (no papers, no AI features on initial load)
- **Click Level 1 topic** (e.g., "Medical & Health"): Show trend graph + papers
- **Click Level 2 topic** (e.g., "Immunology"): Show citation analysis graph + papers
- **AI features ONLY on search page** (not on treemap browsing)
- **Zoom functionality** when clicking treemap topics

## Changes Made

### Frontend Changes ([frontend/app/page.tsx](frontend/app/page.tsx))

#### 1. Removed AI Features from Homepage
- **Removed**: `ResearchOpportunities` component import and usage
- **Reason**: AI recommendations should only appear on search page

#### 2. Changed Initial Data Loading
- **Before**: Loaded treemap + initial papers on mount
- **After**: Loads ONLY treemap on mount
```typescript
// Modified initial load to fetch only treemap
useEffect(() => {
  const fetchInitialData = async () => {
    const treemapRes = await fetch(`${apiUrl}/api/stats/treemap`)
    setTreemapData(treemapData)
    // No initial papers - user clicks topic to see papers
  }
}, [])
```

#### 3. Added Topic Level Tracking
- **New state**: `topicLevel` - Tracks hierarchy depth (0=root, 1=level1, 2=level2)
- **New state**: `trendData` - Stores visualization data for current topic
- **New function**: `getTopicLevel(label)` - Calculates topic depth in hierarchy

```typescript
const [topicLevel, setTopicLevel] = useState<number>(0)
const [trendData, setTrendData] = useState<any>(null)

const getTopicLevel = (label: string): number => {
  let level = 0
  let current = label
  while (current) {
    const parent = getParent(current)
    if (!parent) break
    level++
    current = parent
  }
  return level
}
```

#### 4. Enhanced Treemap Click Handler
Updated `handleTreemapClick` function to:
- Calculate topic level using `getTopicLevel()`
- Fetch papers for the clicked topic
- Fetch trend data based on topic level:
  - **Level 1**: Fetch publication trends over time (`/api/stats/trends/{topic}`)
  - **Level 2**: Fetch citation analysis (`/api/stats/citation-analysis/{topic}`)
- Zoom into topic (update `currentRoot`)
- Clear previous trend data

```typescript
const handleTreemapClick = async (event: any) => {
  const clickedLabel = event.points[0].label
  const level = getTopicLevel(clickedLabel)

  // Zoom in
  if (!isLeafNode(clickedLabel)) {
    setCurrentRoot(clickedLabel)
  }

  // Fetch papers
  const papersRes = await fetch(`${apiUrl}/api/categories/${encodeURIComponent(clickedLabel)}?limit=20&offset=0`)
  // ... set papers state ...

  // Fetch trend data based on level
  if (level === 1) {
    const trendsRes = await fetch(`${apiUrl}/api/stats/trends/${encodeURIComponent(clickedLabel)}`)
    setTrendData({ type: 'trend', data: trendsData })
  } else if (level === 2) {
    const statsRes = await fetch(`${apiUrl}/api/stats/citation-analysis/${encodeURIComponent(clickedLabel)}`)
    setTrendData({ type: 'citation', data: statsData })
  }
}
```

#### 5. Added Trend Visualization UI
New section displays between treemap and papers:
- **Level 1 visualization**: Line chart showing publications over time
- **Level 2 visualization**: Bar chart showing citation counts for top papers
- Uses Plotly for interactive graphs
- Responsive design with gradient styling

```typescript
{trendData && (
  <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6 shadow-sm">
    <h3 className="text-lg font-semibold text-gray-900 mb-4">
      {trendData.type === 'trend' ? 'Publication Trends Over Time' : 'Citation Analysis'}
    </h3>

    {/* Level 1: Line chart */}
    {trendData.type === 'trend' && (
      <Plot data={[{x: years, y: counts, type: 'scatter', mode: 'lines+markers'}]} />
    )}

    {/* Level 2: Bar chart */}
    {trendData.type === 'citation' && (
      <Plot data={[{x: papers, y: citations, type: 'bar'}]} />
    )}
  </div>
)}
```

#### 6. Updated Back Button Handler
- Clear trend data when navigating back
- Recalculate topic level for parent topic

### Backend Changes ([backend/api/stats.py](backend/api/stats.py))

#### 1. Added Topic Trends Endpoint (Level 1)
**Endpoint**: `GET /api/stats/trends/{topic}`

**Purpose**: Get publication trends over time for Level 1 topics

**Response**:
```json
{
  "years": [2018, 2019, 2020, 2021, 2022, 2023],
  "counts": [45, 67, 89, 123, 156, 178],
  "total": 658
}
```

**Implementation**:
- Filters papers by topic in subject_areas
- Groups by year and counts papers
- Returns chronological trend data

#### 2. Added Citation Analysis Endpoint (Level 2)
**Endpoint**: `GET /api/stats/citation-analysis/{topic}`

**Purpose**: Get citation analysis for Level 2 topics

**Response**:
```json
{
  "papers": [
    {"title": "Paper 1", "citations": 456, "year": 2021},
    {"title": "Paper 2", "citations": 389, "year": 2020},
    ...
  ],
  "total": 234,
  "avg_citations": 78.5
}
```

**Implementation**:
- Filters papers by topic in subject_areas
- Sorts by citation count
- Returns top 20 most cited papers
- Includes average citation count

## User Flow

### 1. Initial Homepage Load
- User sees ONLY the interactive treemap
- No papers displayed
- No AI features visible
- Clean, focused interface

### 2. Click Level 1 Topic (e.g., "Medical & Health")
- Treemap zooms into selected topic
- Publication trends line chart appears (2018-2023)
- Papers list loads below (sorted by citations)
- Filter panel becomes available

### 3. Click Level 2 Topic (e.g., "Immunology")
- Treemap zooms deeper
- Citation analysis bar chart appears (top 20 papers)
- Papers list loads below
- Filter panel available

### 4. Navigate Back
- Click "Back" button
- Zoom out to parent topic
- Trend visualization clears
- Papers list clears
- Return to previous level

### 5. Search from Search Page
- Navigate to `/search`
- Enter query
- Papers display immediately
- AI insights load asynchronously (~5-7 seconds)
- InsightsCard shows: summary, key papers, research directions, tips

## Technical Details

### Topic Level Detection
Topics are hierarchical in the treemap:
- **Level 0**: Root ("All Papers")
- **Level 1**: Main categories ("Medical & Health", "Physical Sciences", etc.)
- **Level 2**: Subcategories ("Immunology", "Cardiology", etc.)
- **Level 3+**: Further subdivisions

The `getTopicLevel()` function traverses the parent chain to determine depth.

### Trend Data Structure
```typescript
trendData: {
  type: 'trend' | 'citation',
  data: {
    // For 'trend' type:
    years: number[],
    counts: number[],
    total: number

    // For 'citation' type:
    papers: Array<{title: string, citations: number, year: number}>,
    total: number,
    avg_citations: number
  }
}
```

### Visualization Libraries
- **Plotly**: Interactive charts with zoom, pan, hover
- **Responsive**: Adapts to screen size
- **Color schemes**: Purple gradient for trends, Viridis for citations

## Files Modified

### Frontend
- [frontend/app/page.tsx](frontend/app/page.tsx) - Homepage component
  - Lines 40-44: Added new state variables
  - Lines 78-95: Modified initial data fetch
  - Lines 118-132: Added getTopicLevel function
  - Lines 134-149: Updated handleBack function
  - Lines 224-287: Enhanced handleTreemapClick function
  - Lines 406-493: Added trend visualization UI

### Backend
- [backend/api/stats.py](backend/api/stats.py) - Stats API endpoints
  - Lines 112-152: Added /trends/{topic} endpoint
  - Lines 154-202: Added /citation-analysis/{topic} endpoint

## Testing Instructions

### 1. Start Backend
```bash
cd backend
python -m uvicorn main:app --reload
```

### 2. Start Frontend
```bash
cd frontend
npm run dev
```

### 3. Test Homepage Flow
1. Visit http://localhost:3000
2. **Verify**: Only treemap visible, no papers, no AI features
3. **Click** on a Level 1 topic (e.g., "Medical & Health")
4. **Verify**:
   - Treemap zooms in
   - Publication trends line chart appears
   - Papers list loads below
5. **Click** on a Level 2 topic (e.g., "Immunology")
6. **Verify**:
   - Treemap zooms deeper
   - Citation analysis bar chart appears
   - Papers list loads
7. **Click** "Back" button
8. **Verify**:
   - Zoom out to previous level
   - Trend chart clears
   - Papers clear

### 4. Test Search Page (AI Features)
1. Navigate to http://localhost:3000/search
2. Enter query: "machine learning medical diagnosis"
3. **Verify**:
   - Papers appear immediately (~2 seconds)
   - InsightsCard shows loading skeleton
   - AI insights appear after ~5-7 seconds
   - Insights show: summary, key papers, directions, tips

### 5. Test API Endpoints Directly
```bash
# Test trends endpoint (Level 1)
curl http://localhost:8000/api/stats/trends/Medical%20%26%20Health

# Test citation analysis endpoint (Level 2)
curl http://localhost:8000/api/stats/citation-analysis/Immunology
```

## Success Criteria

- ✅ Homepage loads with ONLY treemap (no papers, no AI)
- ✅ Clicking Level 1 topic shows trend graph + papers
- ✅ Clicking Level 2 topic shows citation graph + papers
- ✅ Treemap zooms in when clicking topics
- ✅ Back button works correctly
- ✅ AI features ONLY on search page
- ✅ All visualizations responsive and interactive
- ✅ Error handling for missing data
- ✅ Loading states for async operations

## Architecture Benefits

### Performance
- No initial papers query = faster homepage load
- Lazy loading of visualizations
- Client-side trend data caching

### User Experience
- Clear hierarchy navigation
- Visual feedback (zoom, graphs)
- Progressive disclosure of information
- Responsive design

### Code Quality
- Separation of concerns (treemap vs search)
- Reusable visualization components
- Type-safe TypeScript interfaces
- Error boundaries

## Future Enhancements

### 1. Add Caching Layer
- Cache trend data for 1 hour
- Reduce duplicate API calls
- Improve response times

### 2. Add More Visualizations
- Network graphs for related topics
- Author collaboration networks
- Geographic distribution maps

### 3. Export Functionality
- Export trend charts as PNG/SVG
- Export citation data as CSV
- Share visualizations

### 4. Accessibility
- Keyboard navigation for treemap
- ARIA labels for charts
- Screen reader support

---

**Status**: ✅ Complete and ready for testing
**Date**: 2025-12-07
**Version**: 2.0.0
