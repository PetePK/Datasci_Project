interface PaperCardProps {
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
  relevance?: number
  index: number
  showRelevance?: boolean  // Show relevance badge for search results only
}

export default function PaperCard({ paper, relevance, index, showRelevance = false }: PaperCardProps) {
  return (
    <div className="p-6 bg-white rounded-lg border-2 border-gray-200 hover:border-purple-500 hover:shadow-lg transition-all group">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex items-start gap-3 flex-1">
          <span className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 text-white rounded-full flex items-center justify-center font-bold text-sm">
            {index + 1}
          </span>
          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-purple-600 transition-colors flex-1">
            {paper.title}
          </h3>
        </div>

        {/* Relevance Badge - Only show for search results */}
        {showRelevance && relevance !== undefined && (
          <span className="flex-shrink-0 text-sm font-bold text-white bg-gradient-to-r from-green-500 to-emerald-500 px-3 py-1 rounded-full shadow-md">
            {(relevance * 100).toFixed(0)}% match
          </span>
        )}
      </div>

      <p className="text-sm text-gray-600 mb-4 line-clamp-3 leading-relaxed">
        {paper.abstract}
      </p>

      {/* Subject Area Tags */}
      {paper.subject_areas && paper.subject_areas.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {paper.subject_areas.slice(0, 5).map((subject, idx) => (
            <span
              key={idx}
              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 border border-purple-200"
            >
              {subject}
            </span>
          ))}
          {paper.subject_areas.length > 5 && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
              +{paper.subject_areas.length - 5} more
            </span>
          )}
        </div>
      )}

      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4 text-sm">
          <span className="flex items-center gap-1 text-gray-700 font-medium">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            {paper.year}
          </span>
          <span className="flex items-center gap-1 text-amber-600 font-medium">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
            </svg>
            {paper.citation_count} citations
          </span>
          <span className="flex items-center gap-1 text-gray-600">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            {paper.num_authors} authors
          </span>
        </div>

        {/* DOI Link Button */}
        {paper.doi && paper.doi.trim() !== '' && (
          <a
            href={`https://doi.org/${paper.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-sm font-medium rounded-lg hover:opacity-90 transition-opacity shadow-sm"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            View Paper
          </a>
        )}
      </div>
    </div>
  )
}
