import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_cluster_id: str = os.getenv("QDRANT_CLUSTER_ID", "")

    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost/rag_chatbot")

    # OpenRouter Configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # Application Settings
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "your-default-secret-key-change-in-production")

    # Model Configuration
    primary_model: str = "xiaomi/mimo-v2-flash:free"
    fallback_model_1: str = "mistralai/devstral-2512:free"
    fallback_model_2: str = "tngtech/deepseek-r1t2-chimera:free"

    # Performance Configuration
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9

    # Document Processing
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens

    # Vector Database
    collection_name: str = "documents"

    class Config:
        env_file = ".env"


# Create a single instance of settings
settings = Settings()