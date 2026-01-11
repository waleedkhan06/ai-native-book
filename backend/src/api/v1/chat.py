from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import logging
from ...models.chat import ChatCompletionRequest, ChatCompletionResponse
from ...api.deps import get_db_session
from ...services.chat_service import chat_service
from ...services.conversation_service import conversation_service
from ...utils.helpers import setup_logger
from ...database.models import Conversation as ConversationModel, Message as MessageDB

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db_session)
):
    """
    Generate chat completion with RAG
    """
    try:
        # Process the chat request with RAG
        response = await chat_service.process_chat_request(
            request=request,
            db=db,
            conversation_id=request.conversation_id
        )

        return response
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db_session)
):
    """
    Get conversation history
    """
    try:
        # Get the conversation
        conversation = await conversation_service.get_conversation(conversation_id, db)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        # Get messages in the conversation
        messages = await conversation_service.get_conversation_messages(conversation_id, db)

        return {
            "id": str(conversation.id),
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "sources": msg.sources
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving conversation: {str(e)}"
        )


@router.get("/conversations")
async def list_conversations(
    db: Session = Depends(get_db_session),
    limit: int = 20,
    offset: int = 0
):
    """
    List conversations for the authenticated user
    """
    try:
        # For now, return all conversations (in a real app, filter by user)
        conversations = await conversation_service.list_conversations(
            user_id=None,  # Would come from auth in real implementation
            db=db,
            limit=limit,
            offset=offset
        )

        # Get total count separately for pagination
        total_count = db.query(ConversationModel).count()

        return {
            "conversations": [
                {
                    "id": str(conv.id),
                    "title": conv.title,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    # Count messages for this conversation
                    "message_count": db.query(MessageDB).filter(MessageDB.conversation_id == str(conv.id)).count()
                }
                for conv in conversations
            ],
            "total": total_count,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing conversations: {str(e)}"
        )


@router.post("/conversations")
async def create_conversation(
    title: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """
    Create a new conversation
    """
    try:
        conversation = await conversation_service.create_conversation(
            title=title,
            db=db
        )

        return {
            "id": str(conversation.id),
            "title": conversation.title,
            "created_at": conversation.created_at
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating conversation: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db_session)
):
    """
    Delete a conversation
    """
    try:
        success = await conversation_service.delete_conversation(conversation_id, db)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting conversation: {str(e)}"
        )