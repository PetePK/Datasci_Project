'use client'

import { useState, useEffect } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import PaperCard from '@/components/PaperCard'
import FilterPanel from '@/components/FilterPanel'
import InsightsCard, { SearchInsights } from '@/components/InsightsCard'

type SortOption = 'citations' | 'year' | 'relevance'

export default function SearchPage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const initialQuery = searchParams.get('q') || ''

  const [query, setQuery] = useState(initialQuery)
  const [results, setResults] = useState<any[]>([])
  const [displayedResults, setDisplayedResults] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // LLM Insights State
  const [insights, setInsights] = useState<SearchInsights | null>(null)
  const [insightsLoading, setInsightsLoading] = useState(false)
  const [insightsError, setInsightsError] = useState<string | null>(null)

  // Filter & Sort State
  const [selectedSubjects, setSelectedSubjects] = useState<string[]>([])
  const [yearRange, setYearRange] = useState<[number, number]>([2018, 2023])
  const [sortBy, setSortBy] = useState<SortOption>('relevance')
  const [showFilters, setShowFilters] = useState(true)
  const [availableSubjects, setAvailableSubjects] = useState<string[]>([])

  // Auto-search when URL has query parameter
  useEffect(() => {
    if (initialQuery) {
      performSearch(initialQuery)
    }
  }, [initialQuery])

  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return

    setLoading(true)
    setError(null)
    setInsights(null) // Reset insights

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

      // Perform search
      const response = await fetch(`${apiUrl}/api/search/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          limit: 50,
          threshold: 0.3,
        }),
      })

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      const data = await response.json()
      setResults(data.results || [])

      // Generate insights asynchronously (don't wait)
      if (data.results && data.results.length > 0) {
        generateInsights(searchQuery, data.results)
      }

    } catch (err: any) {
      console.error('Search error:', err)
      setError(err.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  // Generate AI insights asynchronously
  const generateInsights = async (searchQuery: string, searchResults: any[]) => {
    setInsightsLoading(true)
    setInsightsError(null)

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

      // Prepare results for API (only send necessary fields)
      const resultsForAPI = searchResults.slice(0, 15).map(r => ({
        title: r.paper?.title || '',
        year: r.paper?.year || 0,
        citations: r.paper?.citation_count || 0,
      }))

      const response = await fetch(`${apiUrl}/api/insights/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          results: resultsForAPI,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate insights')
      }

      const data = await response.json()

      if (data.success && data.data) {
        setInsights(data.data)
      } else {
        throw new Error(data.error || 'Unknown error')
      }

    } catch (err: any) {
      console.error('Insights error:', err)
      setInsightsError(err.message || 'Failed to generate insights')
    } finally {
      setInsightsLoading(false)
    }
  }

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    performSearch(query)
  }

  // Extract unique subjects from results
  useEffect(() => {
    if (results.length > 0) {
      const subjects = new Set<string>()
      results.forEach(result => {
        const areas = result.paper.subject_areas || []
        areas.forEach((area: string) => subjects.add(area))
      })
      setAvailableSubjects(Array.from(subjects).sort())
    }
  }, [results])

  // Update displayed results when filters or sort changes
  useEffect(() => {
    if (results.length === 0) {
      setDisplayedResults([])
      return
    }

    let filtered = [...results]

    // Subject filters - AND logic
    if (selectedSubjects.length > 0) {
      filtered = filtered.filter(result => {
        const paperSubjects = result.paper.subject_areas || []
        return selectedSubjects.every(selectedSubject =>
          paperSubjects.some((paperSubject: string) =>
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

    setDisplayedResults(filtered)
  }, [results, selectedSubjects, yearRange, sortBy])

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

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-2">
          {/* Back Button */}
          <button
            type="button"
            onClick={() => router.push('/')}
            className="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:opacity-90 transition-opacity shadow-md"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>

          <h1 className="text-3xl font-bold">Search Results</h1>
        </div>

        {initialQuery && (
          <p className="text-muted-foreground ml-24">
            Showing results for: <span className="font-semibold">&quot;{initialQuery}&quot;</span>
          </p>
        )}
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Refine your search..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </form>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">
            <strong>Error:</strong> {error}
          </p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center py-12">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
            <p className="mt-4 text-muted-foreground">Searching through 19,523 papers...</p>
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && results.length > 0 && (
        <div>
          {/* Filter Panel and AI Insights - Side by Side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Filter Panel - Left Side */}
            <div>
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
            </div>

            {/* AI Insights - Right Side */}
            <div>
              <InsightsCard
                insights={insights}
                loading={insightsLoading}
                error={insightsError}
              />
            </div>
          </div>

          <div className="mb-4 flex items-center justify-between">
            <p className="text-lg font-semibold">
              Sorted by {sortBy === 'citations' ? '⭐ Citations' : sortBy === 'year' ? '📅 Year' : '🎯 Relevance'}
              {selectedSubjects.length > 0 && (
                <> • Filtered by: {selectedSubjects.join(', ')}</>
              )}
            </p>
          </div>

          <div className="space-y-4">
            {displayedResults.map((result, idx) => (
              <PaperCard
                key={result.paper.id}
                paper={result.paper}
                relevance={result.relevance}
                index={idx}
                showRelevance={true}  // Show relevance for search results
              />
            ))}
          </div>

          {/* No results after filtering */}
          {displayedResults.length === 0 && results.length > 0 && (
            <div className="text-center py-12 bg-yellow-50 rounded-lg border-2 border-yellow-200">
              <p className="text-yellow-800 font-medium mb-2">
                No papers match your current filters
              </p>
              <p className="text-sm text-yellow-600">
                Try adjusting your filters or clearing them to see more results
              </p>
            </div>
          )}
        </div>
      )}

      {/* No Results */}
      {!loading && results.length === 0 && initialQuery && !error && (
        <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-gray-600 font-medium">
            No results found for &quot;{initialQuery}&quot;
          </p>
          <p className="text-sm text-gray-500 mt-2">
            Try different keywords or check the spelling
          </p>
        </div>
      )}

      {/* Initial State */}
      {!loading && results.length === 0 && !initialQuery && !error && (
        <div className="text-center py-12 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border-2 border-blue-200">
          <svg className="mx-auto h-16 w-16 text-blue-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <p className="text-gray-700 text-lg font-medium mb-2">Start Your Research Journey</p>
          <p className="text-gray-600">
            Search through 19,523 academic papers
          </p>
          <p className="text-sm text-gray-500 mt-3">
            💡 Try: &quot;machine learning in healthcare&quot; or &quot;climate change solutions&quot;
          </p>
        </div>
      )}
    </div>
  )
}
