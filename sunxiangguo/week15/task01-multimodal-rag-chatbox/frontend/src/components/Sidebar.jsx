import { Link, useLocation } from 'react-router-dom'
import { MessageSquare, FileText, Plus } from 'lucide-react'

function Sidebar({ onClose }) {
  const location = useLocation()

  const navItems = [
    {
      path: '/',
      name: '聊天',
      icon: MessageSquare
    },
    {
      path: '/documents',
      name: '文档管理',
      icon: FileText
    }
  ]

  return (
    <div className="w-64 bg-white border-r border-slate-200 flex flex-col">
      <div className="p-4 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <span className="text-lg font-bold text-slate-900">MultiModal RAG</span>
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${
                isActive
                  ? 'bg-primary-50 text-primary-700 font-medium'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span>{item.name}</span>
            </Link>
          )
        })}
      </nav>

      <div className="p-4 border-t border-slate-200">
        <div className="text-sm text-slate-500 mb-2">历史对话</div>
        <div className="space-y-1">
          <div className="px-3 py-2 rounded-lg text-slate-600 hover:bg-slate-100 cursor-pointer text-sm">
            Q4财报分析
          </div>
          <div className="px-3 py-2 rounded-lg text-slate-600 hover:bg-slate-100 cursor-pointer text-sm">
            产品文档问答
          </div>
        </div>
        <button className="w-full mt-4 flex items-center justify-center gap-2 px-3 py-2 border border-dashed border-slate-300 rounded-lg text-slate-500 hover:border-primary-500 hover:text-primary-600 transition-colors">
          <Plus className="w-4 h-4" />
          <span className="text-sm">新建对话</span>
        </button>
      </div>
    </div>
  )
}

export default Sidebar
