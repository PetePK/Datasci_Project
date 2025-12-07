'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'
import PaperCard from '@/components/PaperCard'
import FilterPanel from '@/components/FilterPanel'
// ResearchOpportunities removed - AI only on search page

// Dynamically import Plotly to avoid SSR issues
// Remove ref - Plot component doesn't need it
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface TreemapData {
  labels: string[]
  parents: string[]
  values: number[]
}

interface SearchResult {
  paper: {
    id: string
    title: string
    abstract: string
    year: number
    citation_count: number
    num_authors: number
    doi: string | null
    subject_areas?: string[]
  }
  relevance: number
}

type SortOption = 'citations' | 'year' | 'relevance'

export default function BrowsePage() {
  const [treemapData, setTreemapData] = useState<TreemapData | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [currentRoot, setCurrentRoot] = useState<string>('')
  const [treemapHistory, setTreemapHistory] = useState<string[]>(['']) // Track navigation history
  const [topicLevel, setTopicLevel] = useState<number>(0) // Track depth: 0=root, 1=level1, 2=level2
  const [papers, setPapers] = useState<SearchResult[]>([])
  const [allPapersCache, setAllPapersCache] = useState<SearchResult[]>([]) // Cache all loaded papers
  const [displayedPapers, setDisplayedPapers] = useState<SearchResult[]>([])
  const [trendData, setTrendData] = useState<any>(null) // Trend visualization data
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [currentOffset, setCurrentOffset] = useState(0)
  const [totalPapers, setTotalPapers] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [error, setError] = useState('')

  // Filter & Sort State
  const [selectedSubjects, setSelectedSubjects] = useState<string[]>([])
  const [yearRange, setYearRange] = useState<[number, number]>([2018, 2023])
  const [sortBy, setSortBy] = useState<SortOption>('citations')
  const [showFilters, setShowFilters] = useState(true)
  const [availableSubjects, setAvailableSubjects] = useState<string[]>([])

  // Simple color palette
  const getColorForIndex = (index: number): string => {
    const colors = [
      '#6B7280', // Gray for root
      '#E91E63', '#EC407A', '#F06292', '#F48FB1', // Pinks
      '#9C27B0', '#AB47BC', '#BA68C8', '#CE93D8', // Purples
      '#2196F3', '#42A5F5', '#64B5F6', '#90CAF9', // Blues
      '#00BCD4', '#26C6DA', '#4DD0E1', '#80DEEA', // Cyans
      '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', // Greens
      '#8BC34A', '#9CCC65', '#AED581', '#C5E1A5', // Light Greens
      '#FF9800', '#FFA726', '#FFB74D', '#FFCC80', // Oranges
      '#FF5722', '#FF7043', '#FF8A65', '#FFAB91', // Deep Oranges
      '#795548', '#8D6E63', '#A1887F', '#BCAAA4', // Browns
      '#607D8B', '#78909C', '#90A4AE', '#B0BEC5', // Blue Grays
    ]
    return colors[index % colors.length]
  }

  // Load treemap data and initial papers (20 at a time)
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

        // Load treemap
        const treemapRes = await fetch(`${apiUrl}/api/stats/treemap`)
        if (!treemapRes.ok) throw new Error('Failed to load treemap data')
        const treemapData = await treemapRes.json()
        setTreemapData(treemapData)

        // Load initial 20 papers (all papers, no category filter)
        setLoading(true)
        const papersRes = await fetch(`${apiUrl}/api/papers?limit=20&offset=0`)
        if (!papersRes.ok) throw new Error('Failed to load papers')
        const papersData = await papersRes.json()
        
        setPapers(papersData.results || [])
        setAllPapersCache(papersData.results || [])
        setTotalPapers(papersData.total || 0)
        setHasMore(papersData.has_more || false)
        setCurrentOffset(20)
        setSelectedCategory('All Papers') // Show "All Papers" initially
        setLoading(false)
      } catch (err: any) {
        setError(err.message)
        setLoading(false)
      }
    }
    fetchInitialData()
  }, [])

  // Generate colors for treemap
  const generateColors = () => {
    if (!treemapData) return []
    return treemapData.labels.map((_, i) => getColorForIndex(i))
  }

  // Check if a category is a leaf node (has no children)
  const isLeafNode = (label: string): boolean => {
    if (!treemapData) return false
    return !treemapData.parents.includes(label)
  }

  // Get parent of a category
  const getParent = (label: string): string => {
    if (!treemapData) return ''
    const index = treemapData.labels.indexOf(label)
    if (index === -1) return ''
    return treemapData.parents[index]
  }

  // Determine topic level (depth in hierarchy)
  const getTopicLevel = (label: string): number => {
    if (!treemapData || !label) return 0

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

  // Handle back button click - use history-based navigation
  const handleBack = () => {
    if (treemapHistory.length <= 1) {
      // Already at root
      return
    }
    
    // Remove current state and go back to previous
    const newHistory = treemapHistory.slice(0, -1)
    setTreemapHistory(newHistory)
    const previousRoot = newHistory[newHistory.length - 1]
    setCurrentRoot(previousRoot)
    
    // Clear papers if going back from Level 2
    setSelectedCategory(null)
    setPapers([])
    setAllPapersCache([])
    setDisplayedPapers([])
    setTrendData(null)
    setTopicLevel(previousRoot ? getTopicLevel(previousRoot) : 0)
    setCurrentOffset(0)
    setTotalPapers(0)
    setHasMore(false)
  }

  // Extract unique subjects from papers
  useEffect(() => {
    if (papers.length > 0) {
      const subjects = new Set<string>()
      papers.forEach(result => {
        const areas = result.paper.subject_areas || []
        areas.forEach(area => subjects.add(area))
      })
      setAvailableSubjects(Array.from(subjects).sort())
    } else {
      setAvailableSubjects([])
    }
  }, [papers])

  // Update displayed papers when filters or sort changes (using cache for fast filtering)
  useEffect(() => {
    if (papers.length === 0) {
      setDisplayedPapers([])
      return
    }

    let filtered = [...papers]

    // Subject filters - AND logic (paper must have ALL selected subjects)
    if (selectedSubjects.length > 0) {
      filtered = filtered.filter(result => {
        const paperSubjects = result.paper.subject_areas || []
        // Check if paper has ALL selected subjects
        return selectedSubjects.every(selectedSubject =>
          paperSubjects.some(paperSubject =>
            paperSubject.toLowerCase().includes(selectedSubject.toLowerCase())
          )
        )
      })
    }

    // Year range filter
    filtered = filtered.filter(result =>
      result.paper.year >= yearRange[0] && result.paper.year <= yearRange[1]
    )

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'citations':
          return b.paper.citation_count - a.paper.citation_count
        case 'year':
          return b.paper.year - a.paper.year
        case 'relevance':
          return (b.relevance || 0) - (a.relevance || 0)
        default:
          return 0
      }
    })

    setDisplayedPapers(filtered)
  }, [papers, selectedSubjects, yearRange, sortBy])

  // Toggle subject filter
  const toggleSubject = (subject: string) => {
    setSelectedSubjects(prev =>
      prev.includes(subject)
        ? prev.filter(s => s !== subject)
        : [...prev, subject]
    )
  }

  // Clear all filters
  const clearFilters = () => {
    setSelectedSubjects([])
    setYearRange([2018, 2023])
  }

  // Handle treemap click
  const handleTreemapClick = async (event: any) => {
    console.log('🖱️ Treemap clicked!', event)
    if (!event.points || event.points.length === 0) {
      console.log('⚠️ No points in click event')
      return
    }

    const clickedLabel = event.points[0].label
    const level = getTopicLevel(clickedLabel)
    console.log(`📍 Clicked: "${clickedLabel}" (Level ${level})`)

    // Update current root for navigation - zoom in for ALL levels
    setCurrentRoot(clickedLabel)
    // Add to history for back navigation
    setTreemapHistory(prev => [...prev, clickedLabel])

    // Fetch papers for ALL levels (0, 1, and 2)
    setSelectedCategory(clickedLabel)
    setTopicLevel(level)
    setLoading(true)
    setError('')
    setCurrentOffset(20) // Reset to 20 after first load
    setTrendData(null) // Clear previous trend data

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      console.log(`🌐 Fetching from: ${apiUrl}/api/categories/${encodeURIComponent(clickedLabel)}`)

      // Fetch papers
      const papersRes = await fetch(`${apiUrl}/api/categories/${encodeURIComponent(clickedLabel)}?limit=20&offset=0`)
      console.log(`📡 Papers response status: ${papersRes.status}`)
      if (!papersRes.ok) throw new Error('Failed to load papers')
      const papersData = await papersRes.json()
      console.log(`✅ Loaded ${papersData.results?.length || 0} papers (total: ${papersData.total})`)

      // Cache the loaded papers
      setAllPapersCache(papersData.results || [])
      setPapers(papersData.results || [])
      setTotalPapers(papersData.total || 0)
      setHasMore(papersData.has_more || false)

      // Fetch trend data only for Level 2 topics
      if (level === 2) {
        try {
          const statsRes = await fetch(`${apiUrl}/api/stats/level2-trends/${encodeURIComponent(clickedLabel)}`)
          if (statsRes.ok) {
            const statsData = await statsRes.json()
            setTrendData({ type: 'level2', data: statsData })
          }
        } catch (err) {
          console.error('Failed to load level 2 trends:', err)
        }
      } else {
        // Clear trend data for Level 0 and 1
        setTrendData(null)
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Load more papers from server
  const loadMorePapers = async () => {
    if (loadingMore || !hasMore) return

    setLoadingMore(true)

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      
      // If selectedCategory is 'All Papers', fetch all papers without category filter
      // Otherwise fetch by specific category
      let url
      if (!selectedCategory || selectedCategory === 'All Papers') {
        url = `${apiUrl}/api/papers?limit=20&offset=${currentOffset}`
      } else {
        url = `${apiUrl}/api/categories/${encodeURIComponent(selectedCategory)}?limit=20&offset=${currentOffset}`
      }
      
      const res = await fetch(url)
      if (!res.ok) throw new Error('Failed to load more papers')
      const data = await res.json()

      // Append new papers to cache
      const newPapers = [...allPapersCache, ...(data.results || [])]
      setAllPapersCache(newPapers)
      setPapers(newPapers)
      setCurrentOffset(currentOffset + 20)
      setHasMore(data.has_more || false)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoadingMore(false)
    }
  }

  if (!treemapData) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p className="text-muted-foreground">Loading research map...</p>
          </div>
        </div>
      </div>
    )
  }

  // Debug: Log when treemap is ready
  console.log('🗺️ Treemap data loaded:', {
    labels: treemapData.labels.length,
    currentRoot,
    hasClickHandler: typeof handleTreemapClick === 'function'
  })

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Back Button */}
      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* No AI features on homepage - AI only on search page */}

      {/* Treemap Visualization */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6 shadow-sm">
        <Plot
          data={[
            {
              type: 'treemap',
              labels: treemapData.labels,
              parents: treemapData.parents,
              values: treemapData.values,
              root: currentRoot,
              marker: {
                colors: generateColors(),
                line: {
                  width: 3,
                  color: 'white'
                }
              },
              text: treemapData.values.map((v, i) =>
                treemapData.labels[i] === 'All Papers' ? '' : `${v.toLocaleString()} papers`
              ),
              textposition: 'middle center',
              texttemplate: '<b>%{label}</b><br>%{text}',
              textfont: {
                size: 12,
                color: 'white',
                family: 'Inter, system-ui, sans-serif'
              },
              hovertemplate: '<b>%{label}</b><br>Papers: %{value:,}<br><extra></extra>',
            }
          ]}
          layout={{
            height: 650,
            margin: { t: 10, l: 10, r: 10, b: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
              family: 'Inter, system-ui, sans-serif',
              size: 12
            }
          }}
          config={{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud', 'lasso2d', 'select2d'],
            displaylogo: false,
          }}
          onClick={handleTreemapClick}
          className="w-full"
        />
      </div>

      {/* Trend Visualization - Shows only for Level 2 topics */}
      {trendData && trendData.type === 'level2' && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            📊 Papers & Citations Trends
          </h3>

          {/* Level 2: Dual line chart (papers + citations) */}
          {trendData.data?.years && (
            <>
              {/* Summary Stats */}
              <div className="grid grid-cols-3 gap-4 mb-4 p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
                <div className="text-center">
                  <p className="text-sm text-gray-600">Total Papers</p>
                  <p className="text-2xl font-bold text-purple-600">{trendData.data.total?.toLocaleString() || 0}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Avg Citations</p>
                  <p className="text-2xl font-bold text-pink-600">{trendData.data.avg_citations?.toFixed(1) || '0.0'}</p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-600">Year Range</p>
                  <p className="text-2xl font-bold text-indigo-600">
                    {trendData.data.years[0]} - {trendData.data.years[trendData.data.years.length - 1]}
                  </p>
                </div>
              </div>

              <Plot
                data={[
                  {
                    x: trendData.data.years,
                    y: trendData.data.paper_counts,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Papers Published',
                    yaxis: 'y',
                    marker: { color: '#8B5CF6', size: 10 },
                    line: { color: '#8B5CF6', width: 4 },
                    hovertemplate: '<b>Papers:</b> %{y}<br>Year: %{x}<extra></extra>'
                  },
                  {
                    x: trendData.data.years,
                    y: trendData.data.citation_counts,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Total Citations',
                    yaxis: 'y2',
                    marker: { color: '#EC4899', size: 10 },
                    line: { color: '#EC4899', width: 4, dash: 'dash' },
                    hovertemplate: '<b>Citations:</b> %{y}<br>Year: %{x}<extra></extra>'
                  }
                ]}
                layout={{
                  height: 400,
                  margin: { t: 20, l: 60, r: 60, b: 60 },
                  xaxis: {
                    title: 'Year',
                    gridcolor: '#E5E7EB',
                    dtick: 1
                  },
                  yaxis: {
                    title: 'Number of Papers',
                    titlefont: { color: '#8B5CF6' },
                    tickfont: { color: '#8B5CF6' },
                    gridcolor: '#E5E7EB',
                    rangemode: 'tozero'
                  },
                  yaxis2: {
                    title: 'Total Citations',
                    titlefont: { color: '#EC4899' },
                    tickfont: { color: '#EC4899' },
                    overlaying: 'y',
                    side: 'right',
                    rangemode: 'tozero'
                  },
                  plot_bgcolor: '#FAFAFA',
                  paper_bgcolor: 'white',
                  font: {
                    family: 'Inter, system-ui, sans-serif'
                  },
                  hovermode: 'x unified',
                  legend: {
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'center',
                    x: 0.5
                  }
                }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  displaylogo: false
                }}
                className="w-full"
              />
            </>
          )}
        </div>
      )}

      {/* Filter Panel */}
      <FilterPanel
        selectedSubjects={selectedSubjects}
        yearRange={yearRange}
        sortBy={sortBy}
        showFilters={showFilters}
        availableSubjects={availableSubjects}
        onToggleSubject={toggleSubject}
        onYearRangeChange={setYearRange}
        onSortByChange={setSortBy}
        onToggleShowFilters={() => setShowFilters(!showFilters)}
        onClearFilters={clearFilters}
      />

      {/* Current Path Breadcrumb */}
      {currentRoot && (
        <div className="mb-6 flex items-center gap-2 text-sm">
          <span className="text-gray-500">Current view:</span>
          <span className="font-semibold text-purple-600">{currentRoot}</span>
        </div>
      )}

      {/* Papers Section - Shows when a category is selected */}
      {selectedCategory && (
        <div id="papers-section" className="mt-8 scroll-mt-8">
          {/* Header showing current selection */}
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-6 rounded-lg mb-6">
            <h2 className="text-2xl font-bold mb-2">
              📚 {selectedCategory}
            </h2>
            {!loading && papers.length > 0 && (
              <p className="text-purple-100">
                Showing {displayedPapers.length} of {totalPapers.toLocaleString()} papers
                {' • '}
                Sorted by {sortBy === 'citations' ? '⭐ Citations' : sortBy === 'year' ? '📅 Year' : '🎯 Relevance'}
                {selectedSubjects.length > 0 && (
                  <> • Filtered by: {selectedSubjects.join(', ')}</>
                )}
              </p>
            )}
          </div>

          {/* Loading State */}
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
              <p className="mt-4 text-muted-foreground">Loading papers...</p>
            </div>
          )}

        {/* Papers List */}
        {!loading && displayedPapers.length > 0 && (
          <div className="space-y-4">
            {displayedPapers.map((result, index) => (
              <PaperCard
                key={result.paper.id}
                paper={result.paper}
                relevance={result.relevance}
                index={index}
                showRelevance={false}  // Hide relevance for treemap category results
              />
            ))}

            {/* Load More Button */}
            {hasMore && displayedPapers.length === papers.length && (
              <div className="text-center py-6">
                <button
                  type="button"
                  onClick={loadMorePapers}
                  disabled={loadingMore}
                  className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed font-medium shadow-md transition-opacity"
                >
                  {loadingMore ? (
                    <span className="flex items-center gap-2">
                      <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Loading...
                    </span>
                  ) : (
                    'Load More'
                  )}
                </button>
              </div>
            )}

            {/* Showing filtered subset message */}
            {papers.length > displayedPapers.length && (
              <div className="text-center py-4 text-sm text-gray-600">
                Showing {displayedPapers.length} of {papers.length} loaded papers (filters applied)
                {hasMore && <span className="block mt-1 text-gray-500">Load more papers to see additional results</span>}
              </div>
            )}
          </div>
        )}

        {/* No filtered results */}
        {!loading && papers.length > 0 && displayedPapers.length === 0 && (
          <div className="text-center py-12 bg-yellow-50 rounded-lg border-2 border-yellow-200">
            <p className="text-yellow-800 font-medium mb-2">
              No papers match your current filters
            </p>
            <p className="text-sm text-yellow-600">
              Try adjusting your filters or clearing them to see more results
            </p>
          </div>
        )}

        {/* No papers loaded yet */}
        {!loading && papers.length === 0 && (
          <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-gray-200">
            <p className="text-gray-600 mb-2">
              Click on a topic in the treemap above to view papers
            </p>
          </div>
        )}
        </div>
      )}

    </div>
  )
}
