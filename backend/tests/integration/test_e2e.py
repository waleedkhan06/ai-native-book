import pytest
from fastapi.testclient import TestClient
from src.main import app


def test_e2e_basic_flow():
    """
    Basic end-to-end test to verify the API is running and accessible
    """
    client = TestClient(app)

    # Test that the API is running
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "RAG Chatbot API"
    assert data["status"] == "running"

    # Test health endpoint
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "rag-chatbot-api"

    # Test that API docs are available
    response = client.get("/docs")
    assert response.status_code == 200

    # Test that OpenAPI schema is available
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert data["info"]["title"] == "RAG Chatbot API"


def test_api_routes_exist():
    """
    Test that main API routes exist (even if they return errors due to missing dependencies)
    """
    client = TestClient(app)

    # These routes should exist even if they return 500 due to missing DB/config
    routes_to_test = [
        "/v1/health",
        "/v1/ready",
        "/v1/live",
        "/v1/conversations",
        "/v1/documents"
    ]

    for route in routes_to_test:
        response = client.get(route)
        # Should return either success (200) or server error (500) due to missing dependencies
        # but not 404 (route not found)
        assert response.status_code in [200, 405, 500], f"Route {route} should exist"