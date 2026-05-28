import { useState, useEffect } from 'react'
import { Upload, File, Trash2, CheckCircle2, Clock, XCircle } from 'lucide-react'
import { motion } from 'framer-motion'
import { documentsAPI } from '../services/api'

function DocumentPage() {
  const [documents, setDocuments] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  useEffect(() => {
    loadDocuments()
  }, [])

  const loadDocuments = async () => {
    try {
      const response = await documentsAPI.list()
      setDocuments(response.data.data?.items || [])
    } catch (error) {
      console.error('Error loading documents:', error)
      // 使用示例数据
      setDocuments([
        {
          id: '1',
          name: 'Q4_Finance_Report.pdf',
          page_count: 24,
          status: 'completed',
          created_at: new Date(Date.now() - 3600000).toISOString(),
        },
        {
          id: '2',
          name: 'Product_Specs.pdf',
          page_count: 12,
          status: 'processing',
          created_at: new Date(Date.now() - 7200000).toISOString(),
        },
        {
          id: '3',
          name: 'Meeting_Notes.pdf',
          page_count: 5,
          status: 'failed',
          created_at: new Date(Date.now() - 86400000).toISOString(),
        },
      ])
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('请上传PDF文件')
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90))
      }, 300)

      await documentsAPI.upload(file)
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      
      setTimeout(() => {
        setIsUploading(false)
        setUploadProgress(0)
        loadDocuments()
      }, 500)
    } catch (error) {
      console.error('Upload error:', error)
      setIsUploading(false)
      setUploadProgress(0)
      alert('上传失败，请重试')
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm('确定要删除这个文档吗？')) return

    try {
      await documentsAPI.delete(id)
      loadDocuments()
    } catch (error) {
      console.error('Delete error:', error)
      setDocuments((prev) => prev.filter((d) => d.id !== id))
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />
      case 'processing':
        return <Clock className="w-5 h-5 text-yellow-500 animate-pulse" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      default:
        return <Clock className="w-5 h-5 text-slate-400" />
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'completed':
        return '已完成'
      case 'processing':
        return '处理中'
      case 'failed':
        return '失败'
      default:
        return '待处理'
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 border-b border-slate-200 bg-white">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">文档管理</h1>
              <p className="text-slate-500 mt-1">上传和管理你的文档</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-6">
        <div className="max-w-5xl mx-auto space-y-6">
          <label className="block cursor-pointer">
            <div className="border-2 border-dashed border-slate-300 rounded-2xl p-8 text-center hover:border-primary-500 hover:bg-primary-50/50 transition-all">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={isUploading}
                className="hidden"
              />
              {isUploading ? (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto bg-slate-100 rounded-full flex items-center justify-center">
                    <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
                  </div>
                  <div>
                    <p className="text-slate-700 font-medium">上传中...</p>
                    <p className="text-slate-500 text-sm">{uploadProgress}%</p>
                  </div>
                  <div className="w-48 mx-auto bg-slate-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full transition-all"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="w-16 h-16 mx-auto bg-primary-100 rounded-full flex items-center justify-center">
                    <Upload className="w-8 h-8 text-primary-600" />
                  </div>
                  <div>
                    <p className="text-slate-700 font-medium">点击上传文档</p>
                    <p className="text-slate-500 text-sm">支持 PDF 格式文件</p>
                  </div>
                </div>
              )}
            </div>
          </label>

          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-slate-900">已上传文档</h2>
            
            {documents.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 mx-auto bg-slate-100 rounded-full flex items-center justify-center mb-4">
                  <File className="w-8 h-8 text-slate-400" />
                </div>
                <p className="text-slate-500">还没有上传任何文档</p>
              </div>
            ) : (
              <div className="grid gap-3">
                {documents.map((doc, index) => (
                  <motion.div
                    key={doc.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center gap-4 p-4 bg-white rounded-xl border border-slate-200 hover:border-slate-300 transition-colors"
                  >
                    <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center flex-shrink-0">
                      <File className="w-6 h-6 text-red-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-slate-900 truncate">{doc.name}</h3>
                      <div className="flex items-center gap-3 text-sm text-slate-500 mt-1">
                        <span>{doc.page_count || 0} 页</span>
                        <span>•</span>
                        <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-100">
                        {getStatusIcon(doc.status)}
                        <span className="text-sm text-slate-700">{getStatusText(doc.status)}</span>
                      </div>
                      <button
                        onClick={() => handleDelete(doc.id)}
                        className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DocumentPage
