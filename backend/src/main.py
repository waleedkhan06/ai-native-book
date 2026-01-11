from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .api.v1 import chat, documents, health
from .middleware.rate_limiter import rate_limit_middleware
from .database.init import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with documentation
app = FastAPI(
    title="RAG Chatbot API",
    description="API for the Retrieval-Augmented Generation Chatbot System",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI available at /docs
    redoc_url="/redoc"  # ReDoc available at /redoc
)

# Add middleware in order (last added is first executed)
app.middleware("http")(rate_limit_middleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
try:
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize database: {str(e)}. Running in limited mode.")

# Include API routes
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(documents.router, prefix="/v1", tags=["documents"])
app.include_router(health.router, prefix="/v1", tags=["health"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API", "status": "running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)