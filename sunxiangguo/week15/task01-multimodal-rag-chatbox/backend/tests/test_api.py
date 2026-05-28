import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

class TestDocuments:
    def test_upload_document(self):
        test_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n199\n%%EOF"
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", test_pdf, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "id" in data["data"]
        assert data["data"]["status"] == "pending"

    def test_get_document_list(self):
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "items" in data["data"]
        assert "total" in data["data"]

class TestSearch:
    def test_search(self):
        response = client.post(
            "/api/v1/search",
            json={"query": "test query", "top_k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["query"] == "test query"
        assert "results" in data["data"]

class TestQA:
    def test_ask_question(self):
        response = client.post(
            "/api/v1/qa/ask",
            json={"question": "What is multimodal RAG?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "conversation_id" in data["data"]
        assert "answer" in data["data"]

    def test_get_conversations(self):
        response = client.get("/api/v1/qa/conversations")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "items" in data["data"]

class TestModels:
    def test_get_model_status(self):
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "clip" in data["data"]
        assert "qwen-vl" in data["data"]
        assert "llm" in data["data"]

class TestVectorDB:
    def test_get_vector_db_status(self):
        response = client.get("/api/v1/vector/db/status")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200

class TestHealth:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])