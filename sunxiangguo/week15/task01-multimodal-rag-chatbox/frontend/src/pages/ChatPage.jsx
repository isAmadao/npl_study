import { useState, useRef, useEffect } from 'react'
import { Send, Paperclip, Image as ImageIcon, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { qaAPI } from '../services/api'

function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const question = input
    setInput('')
    setIsLoading(true)

    try {
      const response = await qaAPI.ask(question)
      const data = response.data.data

      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || '抱歉，我暂时无法回答这个问题。',
        references: data.references || [],
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '抱歉，发生了错误，请稍后重试。',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto scrollbar-thin p-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-primary-700 rounded-2xl flex items-center justify-center mb-6">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-slate-900 mb-2">开始聊天</h2>
            <p className="text-slate-500 max-w-md">
              上传文档后，我可以帮你分析内容，回答关于文档中的问题。
            </p>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-6">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div
                    className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      message.role === 'user'
                        ? 'bg-primary-600 text-white'
                        : 'bg-slate-200 text-slate-700'
                    }`}
                  >
                    {message.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div className={`flex-1 ${message.role === 'user' ? 'text-right' : ''}`}>
                    <div
                      className={`inline-block px-4 py-3 rounded-2xl max-w-[80%] ${
                        message.role === 'user'
                          ? 'bg-primary-600 text-white rounded-tr-sm'
                          : 'bg-white border border-slate-200 rounded-tl-sm'
                      }`}
                    >
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {message.content}
                      </p>
                    </div>
                    
                    {message.references && message.references.length > 0 && (
                      <div className="mt-3 grid grid-cols-3 gap-2 max-w-[80%]">
                        {message.references.map((ref, idx) => (
                          ref.image_url && (
                            <div key={idx} className="relative aspect-video bg-slate-100 rounded-lg overflow-hidden">
                              <img
                                src={ref.image_url}
                                alt={`reference-${idx}`}
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute top-1 left-1 px-1.5 py-0.5 bg-black/50 text-white text-xs rounded">
                                {idx + 1}
                              </div>
                            </div>
                          )
                        ))}
                      </div>
                    )}

                    <div className="mt-1 text-xs text-slate-400">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {isLoading && (
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 bg-slate-200 rounded-full flex items-center justify-center">
                  <Loader2 className="w-4 h-4 text-slate-600 animate-spin" />
                </div>
                <div className="bg-white border border-slate-200 px-4 py-3 rounded-2xl rounded-tl-sm">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="p-6 border-t border-slate-200 bg-white">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSend} className="flex gap-3">
            <button
              type="button"
              className="p-3 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-xl transition-colors"
            >
              <Paperclip className="w-5 h-5" />
            </button>
            <div className="flex-1 relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="输入你的问题..."
                className="w-full px-5 py-3 bg-slate-100 border border-transparent rounded-2xl focus:outline-none focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 transition-all"
                disabled={isLoading}
              />
            </div>
            <button
              type="button"
              className="p-3 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-xl transition-colors"
            >
              <ImageIcon className="w-5 h-5" />
            </button>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="p-3 bg-primary-600 text-white rounded-xl hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <p className="text-center text-xs text-slate-400 mt-3">
            按 Enter 发送，Shift+Enter 换行
          </p>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
