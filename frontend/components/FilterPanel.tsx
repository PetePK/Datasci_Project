import { Filter, X, ChevronDown, ChevronUp } from 'lucide-react'

type SortOption = 'citations' | 'year' | 'relevance'

interface FilterPanelProps {
  // State
  selectedSubjects: string[]
  yearRange: [number, number]
  sortBy: SortOption
  showFilters: boolean
  availableSubjects: string[]

  // Actions
  onToggleSubject: (subject: string) => void
  onYearRangeChange: (range: [number, number]) => void
  onSortByChange: (sort: SortOption) => void
  onToggleShowFilters: () => void
  onClearFilters: () => void
}

export default function FilterPanel({
  selectedSubjects,
  yearRange,
  sortBy,
  showFilters,
  availableSubjects,
  onToggleSubject,
  onYearRangeChange,
  onSortByChange,
  onToggleShowFilters,
  onClearFilters
}: FilterPanelProps) {
  const hasActiveFilters = selectedSubjects.length > 0 ||
    yearRange[0] !== 2018 ||
    yearRange[1] !== 2023

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <button
          type="button"
          onClick={onToggleShowFilters}
          className="flex items-center gap-2 text-lg font-semibold text-gray-900 hover:text-purple-600 transition-colors"
        >
          <Filter className="w-5 h-5" />
          Filters & Sort
          {showFilters ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </button>

        {hasActiveFilters && (
          <button
            type="button"
            onClick={onClearFilters}
            className="text-sm text-red-600 hover:text-red-700 font-medium flex items-center gap-1"
          >
            <X className="w-4 h-4" />
            Clear All Filters
          </button>
        )}
      </div>

      {/* Active Filter Tags */}
      {selectedSubjects.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {selectedSubjects.map(subject => (
            <span
              key={subject}
              className="inline-flex items-center gap-1 px-3 py-1 bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 rounded-full text-sm font-medium border border-purple-200"
            >
              {subject}
              <button
                type="button"
                onClick={() => onToggleSubject(subject)}
                className="hover:bg-purple-200 rounded-full p-0.5 transition-colors"
                aria-label={`Remove ${subject} filter`}
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Filter Controls - Two Column Layout */}
      {showFilters && (
        <div className="pt-4 border-t border-gray-200">
          <div className="space-y-6">
            {/* Subject Area Checkboxes */}
            {availableSubjects.length > 0 && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Subject Areas (papers must have ALL selected)
                </label>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-60 overflow-y-auto p-2 bg-gray-50 rounded-lg">
                  {availableSubjects.map(subject => (
                    <label
                      key={subject}
                      className="flex items-center gap-2 p-2 hover:bg-white rounded cursor-pointer transition-colors"
                    >
                      <input
                        type="checkbox"
                        checked={selectedSubjects.includes(subject)}
                        onChange={() => onToggleSubject(subject)}
                        className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
                      />
                      <span className="text-sm text-gray-700">{subject}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {/* Sort Options */}
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                Sort By
              </label>
              <div className="flex gap-2 flex-wrap">
                {[
                  { value: 'citations' as SortOption, label: '⭐ Citations' },
                  { value: 'year' as SortOption, label: '📅 Year' },
                  { value: 'relevance' as SortOption, label: '🎯 Relevance' }
                ].map(option => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => onSortByChange(option.value)}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      sortBy === option.value
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-md'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Year Range */}
            <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Year: <span className="text-purple-600 font-bold">{yearRange[0]}</span> - <span className="text-pink-600 font-bold">{yearRange[1]}</span>
                </label>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-gray-600 font-medium">2018</span>
                  <div className="relative flex-1">
                    {/* Track */}
                    <div className="relative h-2 bg-gray-200 rounded-lg">
                      {/* Active range highlight */}
                      <div
                        className="absolute h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg"
                        style={{
                          left: `${((yearRange[0] - 2018) / (2023 - 2018)) * 100}%`,
                          right: `${100 - ((yearRange[1] - 2018) / (2023 - 2018)) * 100}%`
                        }}
                      />
                    </div>

                    {/* Min slider */}
                    <input
                      type="range"
                      min="2018"
                      max="2023"
                      value={yearRange[0]}
                      onChange={(e) => {
                        const newMin = parseInt(e.target.value)
                        if (newMin <= yearRange[1]) {
                          onYearRangeChange([newMin, yearRange[1]])
                        }
                      }}
                      className="absolute top-0 w-full h-2 bg-transparent appearance-none cursor-pointer pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-600 [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-purple-600 [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-white [&::-moz-range-thumb]:shadow-md [&::-moz-range-thumb]:cursor-pointer"
                      aria-label="Minimum publication year"
                    />

                    {/* Max slider */}
                    <input
                      type="range"
                      min="2018"
                      max="2023"
                      value={yearRange[1]}
                      onChange={(e) => {
                        const newMax = parseInt(e.target.value)
                        if (newMax >= yearRange[0]) {
                          onYearRangeChange([yearRange[0], newMax])
                        }
                      }}
                      className="absolute top-0 w-full h-2 bg-transparent appearance-none cursor-pointer pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-pink-600 [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-pink-600 [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-white [&::-moz-range-thumb]:shadow-md [&::-moz-range-thumb]:cursor-pointer"
                      aria-label="Maximum publication year"
                    />
                  </div>
                  <span className="text-xs text-gray-600 font-medium">2023</span>
                </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
