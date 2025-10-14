import { motion } from 'framer-motion'
import { Search, Upload, FileText, MessageCircle, History, Image, Mic } from 'lucide-react'
import { useState, useEffect } from 'react'

const Sidebar = ({ activeTab, setActiveTab, isOpen, onClose }) => {
  const [fileCounts, setFileCounts] = useState({
    documents: 0,
    images: 0,
    audio: 0
  })

  // Fetch file counts from API
  const fetchFileCounts = async () => {
    try {
      // Fetch documents count
      const documentsResponse = await fetch('http://localhost:8000/api/documents')
      const documentsData = await documentsResponse.json()
      const documentsCount = documentsData.success ? documentsData.count || 0 : 0

      // Note: Images and Audio endpoints would need to be implemented in backend
      // For now, we'll get counts from localStorage if available
      const searchHistory = localStorage.getItem('prism_search_history')
      let imageCount = 0
      let audioCount = 0
      
      if (searchHistory) {
        try {
          const history = JSON.parse(searchHistory)
          imageCount = history.filter(item => item.type === 'image').length
          audioCount = history.filter(item => item.type === 'audio').length
        } catch (error) {
          console.error('Error parsing search history:', error)
        }
      }

      setFileCounts({
        documents: documentsCount,
        images: imageCount,
        audio: audioCount
      })
    } catch (error) {
      console.error('Error fetching file counts:', error)
      // Keep counts at 0 if there's an error
    }
  }

  useEffect(() => {
    fetchFileCounts()
    // Refresh counts every 10 seconds
    const interval = setInterval(fetchFileCounts, 10000)
    return () => clearInterval(interval)
  }, [])
  const menuItems = [
    { id: 'search', label: 'Search', icon: Search },
    { id: 'qa', label: 'Document Q&A', icon: MessageCircle },
    { id: 'upload', label: 'Upload Files', icon: Upload },
    { id: 'images', label: 'Image Q&A', icon: Image },
    { id: 'audio', label: 'Audio Q&A', icon: Mic },
    { id: 'history', label: 'Search History', icon: History },
  ]

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      <motion.aside 
        initial={{ x: -300 }}
        animate={{ 
          x: typeof window !== 'undefined' && window.innerWidth >= 1024 ? 0 : (isOpen ? 0 : -300)
        }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-white/95 backdrop-blur-lg border-r border-gray-200 z-50 lg:z-40 lg:translate-x-0"
      >
      <div className="p-6">
        <nav className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActiveTab(item.id)
                  // Close sidebar on mobile when item is selected
                  if (typeof window !== 'undefined' && window.innerWidth < 1024) {
                    onClose()
                  }
                }}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-left transition-all duration-200 ${
                  activeTab === item.id
                    ? 'bg-primary-500 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
                style={activeTab === item.id ? { backgroundColor: '#0ea5e9', color: 'white' } : {}}
              >
                <Icon 
                  className="w-5 h-5" 
                  style={activeTab === item.id ? { color: 'white' } : {}}
                />
                <span className="font-medium">{item.label}</span>
              </button>
            )
          })}
        </nav>

        <div className="mt-8 p-4 bg-gradient-to-r from-primary-50 to-purple-50 rounded-xl">
          <h3 className="font-semibold text-gray-900 mb-2">Quick Stats</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Documents</span>
              <span className="font-medium">{fileCounts.documents}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Images</span>
              <span className="font-medium">{fileCounts.images}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">Audio Files</span>
              <span className="font-medium">{fileCounts.audio}</span>
            </div>
          </div>
        </div>
      </div>
    </motion.aside>
    </>
  )
}

export default Sidebar