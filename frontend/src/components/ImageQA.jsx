import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Image, Send, Upload, Bot, User, Copy, ThumbsUp, ThumbsDown, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'

const ImageQA = () => {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedImages, setUploadedImages] = useState([])
  const [selectedImage, setSelectedImage] = useState(null)
  const [uploadProgress, setUploadProgress] = useState({})
  const messagesEndRef = useRef(null)
  const fileInputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const loadImages = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/images')
      if (response.ok) {
        const data = await response.json()
        if (data.success) {
          setUploadedImages(data.images || [])
        }
      }
    } catch (error) {
      console.error('Failed to load images:', error)
    }
  }

  useEffect(() => {
    loadImages()
  }, [])

  const handleFileUpload = async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    try {
      // Start progress tracking
      setUploadProgress(prev => ({
        ...prev,
        [file.name]: { status: 'uploading', percentage: 0 }
      }))

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const result = await response.json()
      
      if (result.success) {
        setUploadProgress(prev => ({
          ...prev,
          [file.name]: { status: 'completed', percentage: 100 }
        }))
        
        toast.success(`${file.name} uploaded successfully!`)
        await loadImages()
      }
    } catch (error) {
      setUploadProgress(prev => ({
        ...prev,
        [file.name]: { status: 'error', percentage: 0 }
      }))
      toast.error(`Upload failed: ${error.message}`)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputValue.trim()) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString(),
      image: selectedImage
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      // Note: This would need backend implementation for image Q&A
      const response = await fetch('http://localhost:8000/api/image-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          image_id: selectedImage?.file_id
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()

      if (data.success) {
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.answer,
          timestamp: new Date().toISOString(),
          sources: data.sources || []
        }
        setMessages(prev => [...prev, botMessage])
      } else {
        throw new Error(data.error || 'Failed to get answer')
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error processing your question about the image. This feature requires additional backend implementation for image analysis.',
        timestamp: new Date().toISOString(),
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
      toast.error('Failed to process image question')
    } finally {
      setIsLoading(false)
    }
  }

  const copyMessage = (content) => {
    navigator.clipboard.writeText(content)
    toast.success('Copied to clipboard!')
  }

  return (
    <div className="h-full flex flex-col lg:flex-row bg-white rounded-2xl shadow-xl overflow-hidden">
      {/* Image Library Sidebar */}
      <div className="w-full lg:w-80 border-b lg:border-b-0 lg:border-r border-gray-200 bg-gray-50 flex flex-col">
        <div className="p-4 border-b border-gray-200 bg-white">
          <h3 className="font-semibold text-gray-900 flex items-center space-x-2">
            <Image className="w-5 h-5 text-blue-500" />
            <span>Image Library</span>
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Upload and analyze images
          </p>
        </div>

        {/* Upload Area */}
        <div className="p-4 border-b border-gray-200">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={(e) => {
              Array.from(e.target.files).forEach(handleFileUpload)
            }}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full p-3 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 transition-colors"
          >
            <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2" />
            <p className="text-sm text-gray-600">Click to upload images</p>
          </button>
        </div>

        {/* Images List */}
        <div className="flex-1 overflow-y-auto p-4">
          {uploadedImages.length === 0 ? (
            <div className="text-center py-8">
              <Image className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500 text-sm">No images uploaded yet</p>
            </div>
          ) : (
            <div className="space-y-2">
              {uploadedImages.map((image, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  onClick={() => setSelectedImage(image)}
                  className={`p-3 rounded-lg cursor-pointer transition-all ${
                    selectedImage?.file_id === image.file_id
                      ? 'bg-blue-100 border-2 border-blue-300'
                      : 'bg-white hover:bg-gray-50 border border-gray-200'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                      <Image className="w-6 h-6 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-gray-900 truncate">
                        {image.file_name || image.name || 'Unknown Image'}
                      </h4>
                      <p className="text-sm text-gray-500">
                        {image.size ? `${(image.size / 1024 / 1024).toFixed(2)} MB` : 'Image file'}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <AnimatePresence>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                  <div className={`flex items-start space-x-3 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      message.type === 'user' ? 'bg-blue-500' : 'bg-gray-100'
                    }`}>
                      {message.type === 'user' ? 
                        <User className="w-4 h-4 text-white" /> : 
                        <Bot className="w-4 h-4 text-gray-600" />
                      }
                    </div>
                    <div className={`flex-1 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                      <div className={`inline-block p-4 rounded-2xl ${
                        message.type === 'user'
                          ? 'bg-blue-500 text-white'
                          : message.isError
                          ? 'bg-red-50 text-red-800 border border-red-200'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        <p className="text-sm leading-relaxed">{message.content}</p>
                        
                        {message.image && (
                          <div className="mt-2 text-xs opacity-75">
                            📷 Analyzing: {message.image.file_name || 'Image'}
                          </div>
                        )}
                        
                        {message.sources && message.sources.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-gray-200">
                            <p className="text-xs font-medium mb-2">Sources:</p>
                            {message.sources.map((source, idx) => (
                              <div key={idx} className="text-xs bg-white/20 rounded-lg p-2 mb-1">
                                📄 {source.file_name || 'Unknown'} {source.page && `(Page ${source.page})`}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      
                      {message.type === 'bot' && !message.isError && (
                        <div className="flex items-center space-x-2 mt-2">
                          <button
                            onClick={() => copyMessage(message.content)}
                            className="text-gray-400 hover:text-gray-600 transition-colors"
                            title="Copy message"
                          >
                            <Copy className="w-4 h-4" />
                          </button>
                          <button className="text-gray-400 hover:text-green-600 transition-colors" title="Helpful">
                            <ThumbsUp className="w-4 h-4" />
                          </button>
                          <button className="text-gray-400 hover:text-red-600 transition-colors" title="Not helpful">
                            <ThumbsDown className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                      
                      <p className="text-xs text-gray-500 mt-2">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-gray-600" />
                </div>
                <div className="bg-gray-100 rounded-2xl p-4">
                  <div className="flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
                    <span className="text-sm text-gray-600">Analyzing image...</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 p-4">
          {selectedImage && (
            <div className="mb-3 p-2 bg-blue-50 rounded-lg border border-blue-200">
              <div className="flex items-center space-x-2 text-sm text-blue-700">
                <Image className="w-4 h-4" />
                <span>Selected: {selectedImage.file_name || 'Image'}</span>
                <button
                  onClick={() => setSelectedImage(null)}
                  className="ml-auto text-blue-500 hover:text-blue-700"
                >
                  ✕
                </button>
              </div>
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={selectedImage ? "Ask about the selected image..." : "Select an image first, then ask questions..."}
              disabled={!selectedImage || isLoading}
              className="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:text-gray-500"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || !selectedImage || isLoading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl transition-colors flex items-center space-x-2"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default ImageQA