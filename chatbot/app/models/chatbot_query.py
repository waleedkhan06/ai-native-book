from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class ChatbotQueryBase(BaseModel):
    user_id: str
    query_text: str
    context: Optional[Dict] = None

class ChatbotQueryCreate(ChatbotQueryBase):
    pass

class ChatbotQuery(ChatbotQueryBase):
    id: str
    query_embedding: Optional[List[float]] = None
    context: Optional[Dict] = None
    response: str
    timestamp: datetime
    accuracy: float
    feedback: Optional[str] = None

    class Config:
        from_attributes = True

class ChatHistoryEntry(BaseModel):
    query_id: str
    query: str
    response: str
    timestamp: datetime
    confidence: float

class ContentEmbedRequest(BaseModel):
    content_id: str
    content: str
    metadata: Dict

class ContentEmbedResponse(BaseModel):
    success: bool
    content_id: str
    chunks_embedded: int

class ChatQueryRequest(BaseModel):
    query: str
    user_id: str
    context: Optional[Dict] = None

class ChatQueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    query_id: str