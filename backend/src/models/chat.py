from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class Message(BaseModel):
    """
    Represents a message in a conversation
    """
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = None
    sources: Optional[List[str]] = []


class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completion endpoint
    """
    messages: List[Message] = Field(..., description="The conversation history including the user's query")
    model: Optional[str] = Field(default="xiaomi/mimo-v2-flash:free", description="The model to use for generation")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=1, description="Controls randomness in generation")
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1, description="Controls diversity via nucleus sampling")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID to continue a conversation")
    context_sources: Optional[List[str]] = Field(None, description="Sources to limit the RAG retrieval to")


class Source(BaseModel):
    """
    Represents a source used in the RAG response
    """
    document_id: str
    document_title: str
    content: str
    score: float


class ChatCompletionChoice(BaseModel):
    """
    Represents a choice in the chat completion response
    """
    index: int
    message: Message
    finish_reason: str  # stop, length, content_filter


class ChatCompletionUsage(BaseModel):
    """
    Represents token usage in the chat completion response
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """
    Response model for chat completion endpoint
    """
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    conversation_id: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
    sources: Optional[List[Source]] = []