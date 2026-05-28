import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import ChatPage from './pages/ChatPage'
import DocumentPage from './pages/DocumentPage'
import Sidebar from './components/Sidebar'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <Router>
      <div className="flex h-screen bg-slate-50">
        {sidebarOpen && (
          <Sidebar onClose={() => setSidebarOpen(false)} />
        )}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
          <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-slate-200">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <h1 className="text-xl font-bold text-slate-900">多模态RAG聊天框</h1>
            </div>
          </header>
          <div className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/documents" element={<DocumentPage />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  )
}

export default App
