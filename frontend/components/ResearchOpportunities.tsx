'use client';

import React, { useState, useEffect } from 'react';

/**
 * Research recommendation structure
 */
export interface Recommendation {
  topic: string;
  rationale: string;
  potential_impact: string;
  difficulty: 'Beginner-friendly' | 'Intermediate' | 'Advanced';
}

interface RecommendationsData {
  success: boolean;
  recommendations: Recommendation[];
  generated_at?: string;
  valid_until?: string;
  error?: string;
}

/**
 * ResearchOpportunities Component
 *
 * Displays AI-generated emerging research opportunities on homepage
 * Data is pre-generated monthly for instant display
 *
 * Best Practices:
 * - Client-side data fetching with caching
 * - Expandable cards for better UX
 * - Filter by difficulty level
 * - Responsive grid layout
 * - Error boundaries
 */
export default function ResearchOpportunities() {
  const [data, setData] = useState<RecommendationsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('All');
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  // Fetch recommendations on mount
  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/recommendations');

      if (!response.ok) {
        throw new Error(`Failed to load recommendations: ${response.status}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Unknown error');
      }

      setData(result);
      setError(null);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err instanceof Error ? err.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  // Filter recommendations
  const filteredRecommendations = data?.recommendations.filter(
    rec => filter === 'All' || rec.difficulty === filter
  ) || [];

  // Difficulty badge color
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner-friendly':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'Intermediate':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Advanced':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-8">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-200 rounded w-64 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-96 mb-8"></div>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="border border-gray-200 rounded-lg p-6">
                <div className="h-5 bg-gray-200 rounded w-3/4 mb-3"></div>
                <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
                <div className="h-4 bg-gray-200 rounded w-5/6"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-start">
          <svg className="h-5 w-5 text-red-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Unable to load research opportunities</h3>
            <p className="mt-1 text-sm text-red-700">{error}</p>
            <button
              onClick={fetchRecommendations}
              className="mt-3 text-sm font-medium text-red-800 hover:text-red-900"
            >
              Try again →
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <svg className="h-6 w-6 text-indigo-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Emerging Research Opportunities
          </h2>
          {data?.generated_at && (
            <span className="text-sm text-gray-500">
              Updated {new Date(data.generated_at).toLocaleDateString()}
            </span>
          )}
        </div>
        <p className="text-gray-600">
          AI-identified promising research directions based on current trends and existing work
        </p>
      </div>

      {/* Filter buttons */}
      <div className="mb-6 flex flex-wrap gap-2">
        {['All', 'Beginner-friendly', 'Intermediate', 'Advanced'].map((level) => (
          <button
            key={level}
            onClick={() => setFilter(level)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === level
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {level}
          </button>
        ))}
      </div>

      {/* Recommendations grid */}
      <div className="space-y-4">
        {filteredRecommendations.length === 0 ? (
          <p className="text-center text-gray-500 py-8">
            No recommendations match the selected filter
          </p>
        ) : (
          filteredRecommendations.map((rec, index) => (
            <div
              key={index}
              className="border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
            >
              {/* Card header */}
              <button
                onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                className="w-full px-6 py-4 flex items-start justify-between text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex-1 pr-4">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getDifficultyColor(rec.difficulty)}`}>
                      {rec.difficulty}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 leading-tight">
                    {rec.topic}
                  </h3>
                </div>
                <svg
                  className={`h-5 w-5 text-gray-400 transition-transform flex-shrink-0 ${
                    expandedIndex === index ? 'transform rotate-180' : ''
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Expanded content */}
              {expandedIndex === index && (
                <div className="px-6 pb-6 space-y-4 border-t border-gray-100">
                  <div className="pt-4">
                    <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                      <svg className="h-4 w-4 mr-2 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      Why Now?
                    </h4>
                    <p className="text-gray-700 leading-relaxed">{rec.rationale}</p>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                      <svg className="h-4 w-4 mr-2 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Potential Impact
                    </h4>
                    <p className="text-gray-700 leading-relaxed">{rec.potential_impact}</p>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Footer note */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-sm text-gray-500 text-center">
          These recommendations are generated monthly using AI analysis of research trends and global developments
        </p>
      </div>
    </div>
  );
}
