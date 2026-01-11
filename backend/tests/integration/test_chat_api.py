import pytest
from fastapi.testclient import TestClient
from src.main import app


def test_health_endpoint():
    """
    Test the health endpoint
    """
    client = TestClient(app)

    response = client.get("/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data
    assert data["service"] == "rag-chatbot-api"


def test_root_endpoint():
    """
    Test the root endpoint
    """
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert data["message"] == "RAG Chatbot API"
    assert "status" in data
    assert data["status"] == "running"


def test_invalid_endpoint():
    """
    Test that invalid endpoints return 404
    """
    client = TestClient(app)

    response = client.get("/nonexistent")
    assert response.status_code == 404