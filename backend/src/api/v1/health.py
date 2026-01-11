from fastapi import APIRouter
from typing import Dict
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint to verify the API is running
    """
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "service": "rag-chatbot-api"
    }


@router.get("/ready")
async def readiness_check() -> Dict:
    """
    Readiness check to verify the API is ready to serve traffic
    """
    # In a real implementation, you'd check connectivity to databases, external services, etc.
    # For now, just return that the service is ready
    return {
        "status": "ready",
        "timestamp": int(time.time()),
        "service": "rag-chatbot-api"
    }


@router.get("/live")
async def liveness_check() -> Dict:
    """
    Liveness check to verify the API is alive
    """
    return {
        "status": "alive",
        "timestamp": int(time.time()),
        "service": "rag-chatbot-api"
    }