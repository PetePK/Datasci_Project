'use client'

import { useState, useEffect, useRef } from 'react'
import dynamic from 'next/dynamic'
import { ArrowLeft } from 'lucide-react'

// Dynamically import Plotly to avoid SSR issues
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
  }
  relevance: number
}

export default function BrowsePage() {
  const [treemapData, setTreemapData] = useState<TreemapData | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [currentRoot, setCurrentRoot] = useState<string>('')
  const [papers, setPapers] = useState<SearchResult[]>([])
  const [displayedPapers, setDisplayedPapers] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [currentOffset, setCurrentOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [error, setError] = useState('')
  const plotRef = useRef<any>(null)

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

  // Load treemap data on mount
  useEffect(() => {
    const fetchTreemap = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        const res = await fetch(`${apiUrl}/api/stats/treemap`)
        if (!res.ok) throw new Error('Failed to load treemap data')
        const data = await res.json()
        setTreemapData(data)
      } catch (err: any) {
        setError(err.message)
      }
    }
    fetchTreemap()
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

  // Handle back button click
  const handleBack = () => {
    if (!currentRoot) return

    const parent = getParent(currentRoot)
    setCurrentRoot(parent)
    setSelectedCategory(null)
    setPapers([])
    setDisplayedPapers([])
    setCurrentOffset(0)
    setHasMore(false)

    // Update the treemap root
    if (plotRef.current && plotRef.current.el) {
      const update = {
        'root': parent || ''
      }
      // @ts-ignore
      window.Plotly.restyle(plotRef.current.el, update, 0)
    }
  }

  // Handle treemap click
  const handleTreemapClick = async (event: any) => {
    if (!event.points || event.points.length === 0) return

    const clickedLabel = event.points[0].label

    // Update current root for navigation (zoom in for non-leaf nodes)
    if (!isLeafNode(clickedLabel)) {
      setCurrentRoot(clickedLabel)
    }

    // Get papers by category (includes all subcategories)
    setSelectedCategory(clickedLabel)
    setLoading(true)
    setError('')
    setCurrentOffset(0)

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      const res = await fetch(`${apiUrl}/api/categories/${encodeURIComponent(clickedLabel)}`)

      if (!res.ok) throw new Error('Failed to load papers')
      const data = await res.json()

      // Sort by citations (highest first)
      const sortedResults = (data.results || []).sort((a: any, b: any) =>
        b.paper.citation_count - a.paper.citation_count
      )

      setPapers(sortedResults)
      // Display first 20 papers
      setDisplayedPapers(sortedResults.slice(0, 20))
      setHasMore(sortedResults.length > 20)

      // Scroll to results
      setTimeout(() => {
        document.getElementById('papers-section')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Load more papers
  const loadMorePapers = () => {
    const nextOffset = currentOffset + 20
    const nextBatch = papers.slice(nextOffset, nextOffset + 20)

    setLoadingMore(true)

    // Simulate async loading for smooth UX
    setTimeout(() => {
      setDisplayedPapers(prev => [...prev, ...nextBatch])
      setCurrentOffset(nextOffset)
      setHasMore(nextOffset + 20 < papers.length)
      setLoadingMore(false)
    }, 300)
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

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">Browse Research by Subject</h1>
            <p className="text-muted-foreground">
              Click to zoom into categories. Click on the smallest boxes to see papers.
            </p>
          </div>

          {/* Back Button */}
          {currentRoot && (
            <button
              type="button"
              onClick={handleBack}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:opacity-90 transition-opacity shadow-md"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Treemap Visualization */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-8 shadow-sm">
        <Plot
          ref={plotRef}
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

      {/* Current Path Breadcrumb */}
      {currentRoot && (
        <div className="mb-6 flex items-center gap-2 text-sm">
          <span className="text-gray-500">Current view:</span>
          <span className="font-semibold text-purple-600">{currentRoot}</span>
        </div>
      )}

      {/* Selected Category Papers */}
      {selectedCategory && (
        <div id="papers-section" className="mt-8 scroll-mt-8">
          <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-6 rounded-lg mb-6">
            <h2 className="text-2xl font-bold mb-2">
              📚 {selectedCategory}
            </h2>
            {!loading && (
              <p className="text-purple-100">
                Found {papers.length} papers • Showing {displayedPapers.length} • Sorted by citations (highest first)
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
                <div
                  key={result.paper.id}
                  className="p-6 bg-white rounded-lg border-2 border-gray-200 hover:border-purple-500 hover:shadow-lg transition-all group"
                >
                  <div className="flex items-start justify-between gap-4 mb-3">
                    <div className="flex items-start gap-3 flex-1">
                      <span className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 text-white rounded-full flex items-center justify-center font-bold text-sm">
                        {index + 1}
                      </span>
                      <h3 className="text-lg font-semibold text-gray-900 group-hover:text-purple-600 transition-colors flex-1">
                        {result.paper.title}
                      </h3>
                    </div>
                    <span className="flex-shrink-0 text-sm font-bold text-white bg-gradient-to-r from-green-500 to-emerald-500 px-3 py-1 rounded-full shadow-md">
                      {(result.relevance * 100).toFixed(0)}% match
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-4 line-clamp-3 leading-relaxed">
                    {result.paper.abstract}
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="flex items-center gap-1 text-gray-700 font-medium">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      {result.paper.year}
                    </span>
                    <span className="flex items-center gap-1 text-amber-600 font-medium">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                      </svg>
                      {result.paper.citation_count} citations
                    </span>
                    <span className="flex items-center gap-1 text-gray-600">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                      </svg>
                      {result.paper.num_authors} authors
                    </span>
                  </div>
                </div>
              ))}

              {/* Load More Button */}
              {hasMore && (
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
                      `Load More Papers (${papers.length - displayedPapers.length} remaining)`
                    )}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* No Results */}
          {!loading && papers.length === 0 && (
            <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
              <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-gray-600 font-medium">
                No papers found for "{selectedCategory}"
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Try clicking another category in the treemap above
              </p>
            </div>
          )}
        </div>
      )}

      {/* Instructions */}
      {!selectedCategory && !currentRoot && (
        <div className="text-center py-12 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border-2 border-blue-200">
          <svg className="mx-auto h-16 w-16 text-blue-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
          </svg>
          <h3 className="text-xl font-bold text-gray-900 mb-2">Interactive Research Explorer</h3>
          <div className="max-w-2xl mx-auto text-gray-700 space-y-3">
            <p className="text-base">
              🔍 <strong>Click</strong> on any colored box to zoom into that research area
            </p>
            <p className="text-base">
              🎯 <strong>Keep clicking</strong> to drill down into subcategories
            </p>
            <p className="text-base">
              📄 <strong>Click on the smallest boxes</strong> (leaf categories) to see papers
            </p>
            <p className="text-sm text-gray-500 mt-4">
              Larger boxes = more papers | Hover to see details | Use Back button to navigate up
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
