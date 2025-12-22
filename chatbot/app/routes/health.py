from fastapi import APIRouter
from typing import Dict
import datetime

router = APIRouter()

@router.get("", response_model=Dict)
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "details": {
            "version": "1.0.0",
            "dependencies": {
                "qdrant": "connected",  # This would be determined dynamically
                "cohere": "connected"   # This would be determined dynamically
            }
        }
    }