import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

export const documentsAPI = {
  upload: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  list: (page = 1, size = 10) => api.get(`/documents?page=${page}&size=${size}`),
  delete: (id) => api.delete(`/documents/${id}`),
}

export const searchAPI = {
  search: (query, topK = 5, documentIds = null) => 
    api.post('/search', { query, top_k: topK, document_ids: documentIds }),
}

export const qaAPI = {
  ask: (question, conversationId = null, documentIds = null) => 
    api.post('/qa/ask', { question, conversation_id: conversationId, document_ids: documentIds }),
  conversations: (page = 1, size = 10) => 
    api.get(`/qa/conversations?page=${page}&size=${size}`),
  conversation: (id) => api.get(`/qa/conversations/${id}`),
  deleteConversation: (id) => api.delete(`/qa/conversations/${id}`),
}

export default api
