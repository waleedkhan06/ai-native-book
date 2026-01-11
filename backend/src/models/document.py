from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class Document(BaseModel):
    """
    Represents a document in the knowledge base
    """
    id: Optional[str] = None
    title: str = Field(..., description="Title of the document")
    content: str = Field(..., description="Full text content of the document")
    source_url: str = Field(..., description="Original URL or path of the document")
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    meta_data: Optional[Dict[str, Any]] = {}
    status: Optional[str] = "PROCESSING"  # PROCESSING, INGESTED, FAILED
    chunk_count: Optional[int] = 0


class DocumentChunk(BaseModel):
    """
    Represents a chunk of a document for vector storage
    """
    id: Optional[str] = None
    document_id: str = Field(..., description="Reference to the parent document")
    content: str = Field(..., description="Text content of the chunk (typically 512-1024 tokens)")
    chunk_order: int = Field(..., description="Order of this chunk in the original document")
    vector_id: Optional[str] = Field(None, description="ID in the vector database (Qdrant)")
    embedding_model: Optional[str] = Field(None, description="Model used to generate the embedding")
    created_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """
    Response model for listing documents
    """
    documents: List[Document]
    total: int
    limit: int
    offset: int