import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime
from ..models.chat import Message as MessageModel
from ..database.models import Conversation as ConversationModel, Message as MessageDB
from ..utils.helpers import generate_uuid

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Service for conversation management
    """

    def __init__(self):
        pass

    async def create_conversation(
        self,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        db: Session = None
    ) -> ConversationModel:
        """
        Create a new conversation
        """
        try:
            conversation_id = generate_uuid()

            # If title is not provided, generate a default title
            if not title:
                title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            conversation = ConversationModel(
                id=conversation_id,
                user_id=user_id,
                title=title,
                meta_data={}
            )

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            logger.info(f"Created new conversation: {conversation_id}")
            return conversation

        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise

    async def get_conversation(
        self,
        conversation_id: str,
        db: Session
    ) -> Optional[ConversationModel]:
        """
        Get conversation by ID
        """
        try:
            conversation = db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).first()

            return conversation
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {str(e)}")
            raise

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        db: Session = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[ConversationModel]:
        """
        List conversations for a user
        """
        try:
            query = db.query(ConversationModel)

            if user_id:
                query = query.filter(ConversationModel.user_id == user_id)

            conversations = query.offset(offset).limit(limit).all()
            return conversations
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            raise

    async def update_conversation_title(
        self,
        conversation_id: str,
        title: str,
        db: Session
    ):
        """
        Update conversation title
        """
        try:
            conversation = db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).first()

            if conversation:
                conversation.title = title
                conversation.updated_at = datetime.now()
                db.commit()
        except Exception as e:
            logger.error(f"Error updating conversation title: {str(e)}")
            raise

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
        db: Session = None
    ) -> MessageDB:
        """
        Add a message to a conversation
        """
        try:
            message = MessageDB(
                id=generate_uuid(),
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources=sources or []
            )

            db.add(message)
            db.commit()
            db.refresh(message)

            # Update conversation timestamp
            conversation = db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).first()
            if conversation:
                conversation.updated_at = datetime.now()
                db.commit()

            return message
        except Exception as e:
            logger.error(f"Error adding message to conversation: {str(e)}")
            raise

    async def get_conversation_messages(
        self,
        conversation_id: str,
        db: Session
    ) -> List[MessageDB]:
        """
        Get all messages in a conversation
        """
        try:
            messages = db.query(MessageDB).filter(
                MessageDB.conversation_id == conversation_id
            ).order_by(MessageDB.timestamp).all()

            return messages
        except Exception as e:
            logger.error(f"Error getting conversation messages: {str(e)}")
            raise

    async def delete_conversation(
        self,
        conversation_id: str,
        db: Session
    ) -> bool:
        """
        Delete a conversation and all its messages
        """
        try:
            # Delete messages first (due to foreign key constraint)
            db.query(MessageDB).filter(
                MessageDB.conversation_id == conversation_id
            ).delete()

            # Delete conversation
            result = db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).delete()

            db.commit()

            return result > 0
        except Exception as e:
            logger.error(f"Error deleting conversation: {str(e)}")
            raise


# Global instance
conversation_service = ConversationService()