'use client';

import React from 'react';

/**
 * Insights data structure from API
 */
export interface SearchInsights {
  relevance_summary?: string;
  key_papers?: string[];
  research_directions?: string[];
  search_tips?: string[];
  generated_at?: string;
  query?: string;
  result_count?: number;
}

interface InsightsCardProps {
  insights: SearchInsights | null;
  loading?: boolean;
  error?: string | null;
}

/**
 * InsightsCard Component
 *
 * Displays AI-generated insights for search results
 * Shows loading state while generating (~3-7 seconds)
 *
 * Best Practices:
 * - Skeleton loader for smooth UX
 * - Error handling
 * - Responsive design
 * - Accessible markup
 */
export default function InsightsCard({ insights, loading = false, error = null }: InsightsCardProps) {
  // Error state
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-start">
          <svg className="h-5 w-5 text-red-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Unable to generate insights</h3>
            <p className="mt-1 text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  // Loading state
  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 animate-pulse">
        <div className="flex items-center mb-4">
          <div className="h-5 w-5 bg-blue-200 rounded-full mr-3"></div>
          <div className="h-5 bg-gray-200 rounded w-48"></div>
        </div>
        <div className="space-y-3">
          <div className="h-4 bg-gray-200 rounded w-full"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          <div className="h-4 bg-gray-200 rounded w-4/6"></div>
        </div>
        <div className="mt-6 space-y-2">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-2/3"></div>
        </div>
        <p className="mt-4 text-xs text-gray-400">Generating AI insights...</p>
      </div>
    );
  }

  // No insights yet
  if (!insights) {
    return null;
  }

  return (
    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6 shadow-sm">
      {/* Header */}
      <div className="flex items-center mb-4">
        <svg className="h-5 w-5 text-blue-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <h3 className="text-lg font-semibold text-gray-900">AI Insights</h3>
        <span className="ml-auto text-xs text-gray-500">Powered by Claude</span>
      </div>

      {/* Summary */}
      {insights.relevance_summary && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Research Overview</h4>
          <p className="text-gray-700 leading-relaxed">{insights.relevance_summary}</p>
        </div>
      )}

      {/* Key Papers */}
      {insights.key_papers && insights.key_papers.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Most Influential Papers</h4>
          <ul className="space-y-2">
            {insights.key_papers.map((paper, index) => (
              <li key={index} className="flex items-start">
                <span className="flex-shrink-0 h-6 w-6 flex items-center justify-center bg-blue-100 text-blue-700 rounded-full text-xs font-medium mr-3">
                  {index + 1}
                </span>
                <span className="text-gray-700 text-sm">{paper}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Research Directions */}
      {insights.research_directions && insights.research_directions.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Emerging Research Directions</h4>
          <ul className="space-y-2">
            {insights.research_directions.map((direction, index) => (
              <li key={index} className="flex items-start">
                <svg className="flex-shrink-0 h-5 w-5 text-indigo-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                <span className="text-gray-700 text-sm">{direction}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Search Tips */}
      {insights.search_tips && insights.search_tips.length > 0 && (
        <div className="bg-white bg-opacity-60 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
            <svg className="h-4 w-4 mr-2 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
              <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1h4v1a2 2 0 11-4 0zM12 14c.015-.34.208-.646.477-.859a4 4 0 10-4.954 0c.27.213.462.519.476.859h4.002z" />
            </svg>
            Tips to Refine Your Search
          </h4>
          <ul className="space-y-1">
            {insights.search_tips.map((tip, index) => (
              <li key={index} className="text-sm text-gray-600 pl-6">• {tip}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Metadata */}
      {insights.generated_at && (
        <div className="mt-4 pt-4 border-t border-blue-200">
          <p className="text-xs text-gray-500">
            Generated {new Date(insights.generated_at).toLocaleTimeString()}
            {insights.result_count && ` • Analyzed ${insights.result_count} papers`}
          </p>
        </div>
      )}
    </div>
  );
}
