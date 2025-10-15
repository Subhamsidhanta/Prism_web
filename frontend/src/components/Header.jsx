import { motion } from 'framer-motion'
import { Brain, Menu, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import { useState, useEffect } from 'react'

const Header = ({ onMenuToggle }) => {
  const [modelStatus, setModelStatus] = useState({
    model_loaded: false,
    model_path: '',
    loading: true
  })

  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
  const base = import.meta.env.VITE_API_BASE || ''
  const response = await fetch(`${base}/api/model/status`)
        const data = await response.json()
        setModelStatus({
          model_loaded: data.model_loaded,
          model_path: data.model_path,
          loading: false
        })
      } catch (error) {
        console.error('Failed to fetch model status:', error)
        setModelStatus(prev => ({ ...prev, loading: false }))
      }
    }

    fetchModelStatus()
    // Check status every 10 seconds
    const interval = setInterval(fetchModelStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  const getModelName = (path) => {
    if (!path) return 'Unknown'
    const filename = path.split('\\').pop() || path.split('/').pop()
    // Remove file extension for cleaner display
    return filename.replace('.gguf', '')
  }
  return (
    <motion.header 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white border-b border-gray-200 px-4 md:px-6 py-4 relative z-50"
    > 
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {/* Mobile menu button */}
          <button
            onClick={onMenuToggle}
            className="lg:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Toggle menu"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>
          
          <div className="w-8 h-8 md:w-10 md:h-10 bg-gradient-to-r from-primary-600 to-purple-600 rounded-xl flex items-center justify-center">
            <Brain className="w-4 h-4 md:w-6 md:h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg md:text-xl font-bold gradient-text">Prism</h1>
            <p className="text-xs md:text-sm text-gray-500 hidden sm:block">Multimodal RAG System</p>
          </div>
        </div>
        {/* Responsive prototype notice */}
        <h4 className='text-center font-medium text-red-500 tracking-tight 
                       text-xs sm:text-sm md:text-base lg:text-lg 
                       px-2 md:px-4'>
          {/* <span className="md:hidden">Prototype build – final version coming soon.</span> */}
          {/* <span className="hidden md:inline">This is our prototype not the final project, the final project must have offline mode...</span> */}
        </h4>
        <div className="flex items-center space-x-2 md:space-x-4">
          {/* Model Status */}
          <div className="flex items-center space-x-2 text-xs md:text-sm">
            {modelStatus.loading ? (
              <div className="flex items-center space-x-1 text-gray-500">
                <Loader className="w-3 h-3 md:w-4 md:h-4 animate-spin" />
                <span className="hidden sm:inline">Loading...</span>
              </div>
            ) : modelStatus.model_loaded ? (
              <div className="flex items-center space-x-1 text-green-600">
                <CheckCircle className="w-3 h-3 md:w-4 md:h-4" />
                {/* <span className="hidden sm:inline">{getModelName(modelStatus.model_path)} </span> */}
                <span className="hidden sm:inline">Model Loaded</span>
                <span className="sm:hidden">✓</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1 text-red-600">
                <AlertCircle className="w-3 h-3 md:w-4 md:h-4" />
                <span className="hidden sm:inline">Model Offline</span>
                <span className="sm:hidden">✗</span>
              </div>
            )}
          </div>
          
          <div className="text-xs md:text-sm text-gray-600">
            <span className="font-medium hidden sm:inline">Status:</span>
            <span className="ml-0 sm:ml-1 text-green-600">Online</span>
          </div>
        </div>
      </div>
    </motion.header>
  )
}

export default Header