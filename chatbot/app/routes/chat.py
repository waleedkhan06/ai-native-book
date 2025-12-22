from fastapi import APIRouter, HTTPException
from typing import Optional
from app.models.chatbot_query import ChatQueryRequest, ChatQueryResponse, ContentEmbedRequest, ContentEmbedResponse
from app.services.chat_service import chat_service

router = APIRouter()

@router.post("/query", response_model=ChatQueryResponse)
async def query_chatbot(request: ChatQueryRequest):
    """
    Submit a question to the RAG chatbot
    """
    try:
        response = await chat_service.process_query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/history/{user_id}")
async def get_chat_history(user_id: str, limit: Optional[int] = 10):
    """
    Get chat history for a user
    """
    try:
        history = await chat_service.get_chat_history(user_id, limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@router.post("/content/embed", response_model=ContentEmbedResponse)
async def embed_content(request: ContentEmbedRequest):
    """
    Embed textbook content for RAG retrieval
    """
    try:
        response = await chat_service.embed_content(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error embedding content: {str(e)}")