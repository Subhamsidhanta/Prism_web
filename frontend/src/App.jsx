import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Toaster } from 'react-hot-toast'
import Header from './components/Header'
import SearchInterface from './components/SearchInterface'
import FileUpload from './components/FileUpload'
import ResultsDisplay from './components/ResultsDisplay'
import Sidebar from './components/Sidebar'
import DocumentQA from './components/DocumentQA'
import ImageQA from './components/ImageQA'
import AudioQA from './components/AudioQA'
import SearchHistory from './components/SearchHistory'
import ErrorBoundary from './components/ErrorBoundary'
import { Search, Upload, FileText, Image, Mic, Brain } from 'lucide-react'

function App() {
  const [searchResults, setSearchResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [activeTab, setActiveTab] = useState('search')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const handleSearch = async (query) => {
    // Remove dummy data - actual search functionality would go here
    setIsLoading(true)
    setSearchResults([])
    
    setTimeout(() => {
      setIsLoading(false)
      // No dummy results - real search would happen here
    }, 1000)
  }

  const handleFileUpload = (files) => {
    setUploadedFiles(prev => [...prev, ...files])
  }

  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-indigo-100">
          <Header onMenuToggle={() => setSidebarOpen(!sidebarOpen)} />
          
          <div className="flex">
            <Sidebar 
              activeTab={activeTab} 
              setActiveTab={setActiveTab}
              isOpen={sidebarOpen}
              onClose={() => setSidebarOpen(false)}
            />
            
            <main className="flex-1 p-4 md:p-6 lg:ml-64 transition-all duration-300">
              <div className="max-w-7xl mx-auto">
                {/* Hero Section */}
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="text-center mb-8 md:mb-12"
                >
                  <div className="flex justify-center mb-4 md:mb-6">
                    <div className="relative">
                      <div className="w-16 h-16 md:w-20 md:h-20 bg-gradient-to-r from-primary-600 to-purple-600 rounded-2xl flex items-center justify-center animate-float">
                        <Brain className="w-8 h-8 md:w-10 md:h-10 text-white" />
                      </div>
                      <div className="absolute -top-2 -right-2 w-4 h-4 md:w-6 md:h-6 bg-green-500 rounded-full animate-pulse"></div>
                    </div>
                  </div>
                  <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold gradient-text mb-3 md:mb-4 px-4">
                    Welcome to Prism
                  </h1>
                  <p className="text-base md:text-lg lg:text-xl text-gray-600 max-w-3xl mx-auto px-4">
                    Your intelligent multimodal search companion. Find insights across documents, images, and audio with natural language queries.
                  </p>
                </motion.div>

                {/* Tab Content */}
                <ErrorBoundary>
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    {activeTab === 'search' && (
                      <div className="space-y-8">
                        <SearchInterface onSearch={handleSearch} isLoading={isLoading} />
                        {searchResults.length > 0 && (
                          <ResultsDisplay results={searchResults} />
                        )}
                      </div>
                    )}

                    {activeTab === 'qa' && (
                      <DocumentQA />
                    )}

                    {activeTab === 'upload' && (
                      <FileUpload onFileUpload={handleFileUpload} uploadedFiles={uploadedFiles} />
                    )}

                    {activeTab === 'images' && (
                      <ImageQA />
                    )}

                    {activeTab === 'audio' && (
                      <AudioQA />
                    )}

                    {activeTab === 'history' && (
                      <SearchHistory />
                    )}
                  </motion.div>
                </ErrorBoundary>
              </div>
            </main>
          </div>
          
          {/* Toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                style: {
                  background: '#10B981',
                },
              },
              error: {
                duration: 5000,
                style: {
                  background: '#EF4444',
                },
              },
            }}
          />
        </div>
      </Router>
    </ErrorBoundary>
  )
}

export default App