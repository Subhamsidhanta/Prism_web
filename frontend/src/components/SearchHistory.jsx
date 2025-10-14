import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { History, Search, MessageCircle, Image, Mic, Calendar, Clock, Filter, Trash2, FileText } from 'lucide-react'
import toast from 'react-hot-toast'

const SearchHistory = () => {
  const [searchHistory, setSearchHistory] = useState([])
  const [filteredHistory, setFilteredHistory] = useState([])
  const [activeFilter, setActiveFilter] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')

  // Load search history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('prism_search_history')
    if (saved) {
      try {
        const history = JSON.parse(saved)
        setSearchHistory(history)
        setFilteredHistory(history)
      } catch (error) {
        console.error('Failed to load search history:', error)
      }
    }
  }, [])

  // Filter history based on type and search query
  useEffect(() => {
    let filtered = searchHistory

    // Filter by type
    if (activeFilter !== 'all') {
      filtered = filtered.filter(item => item.type === activeFilter)
    }

    // Filter by search query
    if (searchQuery) {
      filtered = filtered.filter(item => 
        item.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.response.toLowerCase().includes(searchQuery.toLowerCase())
      )
    }

    // Sort by date (newest first)
    filtered = filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

    setFilteredHistory(filtered)
  }, [searchHistory, activeFilter, searchQuery])

  const clearHistory = () => {
    if (window.confirm('Are you sure you want to clear all search history?')) {
      setSearchHistory([])
      setFilteredHistory([])
      localStorage.removeItem('prism_search_history')
      toast.success('Search history cleared')
    }
  }

  const deleteItem = (id) => {
    const updated = searchHistory.filter(item => item.id !== id)
    setSearchHistory(updated)
    localStorage.setItem('prism_search_history', JSON.stringify(updated))
    toast.success('Item removed from history')
  }

  const getTypeIcon = (type) => {
    switch (type) {
      case 'document':
        return <FileText className="w-4 h-4 text-red-500" />
      case 'image':
        return <Image className="w-4 h-4 text-blue-500" />
      case 'audio':
        return <Mic className="w-4 h-4 text-green-500" />
      case 'search':
        return <Search className="w-4 h-4 text-purple-500" />
      default:
        return <MessageCircle className="w-4 h-4 text-gray-500" />
    }
  }

  const getTypeLabel = (type) => {
    switch (type) {
      case 'document':
        return 'Document Q&A'
      case 'image':
        return 'Image Q&A'
      case 'audio':
        return 'Audio Q&A'
      case 'search':
        return 'General Search'
      default:
        return 'Unknown'
    }
  }

  const formatDate = (timestamp) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffTime = Math.abs(now - date)
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))

    if (diffDays === 1) {
      return 'Today'
    } else if (diffDays === 2) {
      return 'Yesterday'
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const filters = [
    { id: 'all', label: 'All Types', icon: History },
    { id: 'document', label: 'Documents', icon: FileText },
    { id: 'image', label: 'Images', icon: Image },
    { id: 'audio', label: 'Audio', icon: Mic },
    { id: 'search', label: 'Search', icon: Search },
  ]

  return (
    <div className="search-container">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-6 md:mb-8"
      >
        <h2 className="text-2xl md:text-3xl font-bold gradient-text mb-3 md:mb-4 px-2">
          Search History
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto text-sm md:text-base px-2">
          View and manage all your previous queries and conversations across documents, images, and audio.
        </p>
      </motion.div>

      {/* Filters and Search */}
      <div className="mb-6 space-y-4">
        {/* Search bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search in history..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Filter tabs */}
        <div className="flex flex-wrap gap-2">
          {filters.map((filter) => {
            const Icon = filter.icon
            return (
              <button
                key={filter.id}
                onClick={() => setActiveFilter(filter.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                  activeFilter === filter.id
                    ? 'bg-primary-500 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium">{filter.label}</span>
              </button>
            )
          })}
        </div>

        {/* Clear history button */}
        {searchHistory.length > 0 && (
          <div className="flex justify-end">
            <button
              onClick={clearHistory}
              className="flex items-center space-x-2 text-red-600 hover:text-red-700 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              <span className="text-sm">Clear All History</span>
            </button>
          </div>
        )}
      </div>

      {/* History Items */}
      <div className="space-y-4">
        {filteredHistory.length === 0 ? (
          <div className="text-center py-12">
            <History className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {searchHistory.length === 0 ? 'No Search History' : 'No Results Found'}
            </h3>
            <p className="text-gray-500">
              {searchHistory.length === 0 
                ? 'Your search and conversation history will appear here.'
                : 'Try adjusting your filters or search query.'
              }
            </p>
          </div>
        ) : (
          <AnimatePresence>
            {filteredHistory.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ delay: index * 0.05 }}
                className="bg-white rounded-xl border border-gray-200 p-6 hover:shadow-lg transition-all duration-200"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    {getTypeIcon(item.type)}
                    <div>
                      <h3 className="font-medium text-gray-900">
                        {getTypeLabel(item.type)}
                      </h3>
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <Calendar className="w-3 h-3" />
                        <span>{formatDate(item.timestamp)}</span>
                        <Clock className="w-3 h-3 ml-2" />
                        <span>{new Date(item.timestamp).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => deleteItem(item.id)}
                    className="text-gray-400 hover:text-red-500 transition-colors"
                    title="Remove from history"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>

                <div className="space-y-3">
                  {/* Query */}
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <MessageCircle className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Query:</span>
                    </div>
                    <p className="text-gray-800 text-sm leading-relaxed">
                      {item.query}
                    </p>
                  </div>

                  {/* Response preview */}
                  <div className="bg-blue-50 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <MessageCircle className="w-4 h-4 text-blue-500" />
                      <span className="text-sm font-medium text-blue-700">Response:</span>
                    </div>
                    <p className="text-blue-800 text-sm leading-relaxed">
                      {item.response.length > 200 
                        ? `${item.response.substring(0, 200)}...` 
                        : item.response
                      }
                    </p>
                  </div>

                  {/* Sources (if any) */}
                  {item.sources && item.sources.length > 0 && (
                    <div className="bg-green-50 rounded-lg p-3">
                      <div className="flex items-center space-x-2 mb-2">
                        <FileText className="w-4 h-4 text-green-500" />
                        <span className="text-sm font-medium text-green-700">Sources:</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {item.sources.map((source, idx) => (
                          <span
                            key={idx}
                            className="inline-block bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full"
                          >
                            {source.file_name || source}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Context (if any) */}
                  {item.context && (
                    <div className="text-xs text-gray-500 border-t pt-3">
                      <span className="font-medium">Context:</span> {item.context}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>

      {/* Load more button (for future pagination) */}
      {filteredHistory.length > 0 && filteredHistory.length % 10 === 0 && (
        <div className="text-center mt-8">
          <button className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-6 py-3 rounded-lg transition-colors">
            Load More
          </button>
        </div>
      )}
    </div>
  )
}

// Helper function to add items to search history
export const addToSearchHistory = (type, query, response, sources = [], context = '') => {
  const historyItem = {
    id: Date.now() + Math.random(),
    type,
    query,
    response,
    sources,
    context,
    timestamp: new Date().toISOString()
  }

  try {
    const existing = localStorage.getItem('prism_search_history')
    const history = existing ? JSON.parse(existing) : []
    
    // Add new item to the beginning
    history.unshift(historyItem)
    
    // Keep only last 100 items
    const trimmed = history.slice(0, 100)
    
    localStorage.setItem('prism_search_history', JSON.stringify(trimmed))
  } catch (error) {
    console.error('Failed to save search history:', error)
  }
}

export default SearchHistory