import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from ..models.chat import Message, ChatCompletionRequest, ChatCompletionResponse, Source
from ..models.document import DocumentChunk
from ..database.models import Conversation, Message as MessageModel, QueryLog
from ..services.vector_store import vector_store_service
from ..services.llm_service import llm_service
from ..config import settings
from ..utils.helpers import generate_uuid

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service to orchestrate the RAG flow
    """

    def __init__(self):
        pass

    async def process_chat_request(
        self,
        request: ChatCompletionRequest,
        db: Session,
        conversation_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """
        Process a chat completion request with RAG
        """
        start_time = datetime.now()

        try:
            # Create or get conversation
            if conversation_id:
                conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            else:
                conversation = Conversation(id=generate_uuid(), title="New Conversation")
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

            # Create message in database
            user_message = MessageModel(
                id=generate_uuid(),
                conversation_id=conversation.id,
                role="user",
                content=request.messages[-1].content,  # Last message is the user's query
                timestamp=datetime.now()
            )
            db.add(user_message)
            db.commit()

            # Perform similarity search to get relevant context
            query_embedding = await vector_store_service.generate_embedding(request.messages[-1].content)
            search_results = await vector_store_service.search_similar(
                query_embedding=query_embedding,
                limit=5  # Get top 5 most relevant chunks
            )

            # Build context from search results
            context_chunks = []
            sources = []
            for result in search_results:
                if result["payload"] and "content" in result["payload"]:
                    context_chunks.append(result["payload"]["content"])
                    sources.append(Source(
                        document_id=result["payload"].get("document_id", ""),
                        document_title="",  # Would need to fetch from document table
                        content=result["payload"]["content"][:200] + "...",  # Truncate for response
                        score=result["score"]
                    ))

            # Construct augmented prompt with context
            context_text = "\n\n".join(context_chunks)
            if context_text.strip():
                # Add system message with context
                system_message = Message(
                    role="system",
                    content=f"Use the following context to answer the user's question:\n\n{context_text}\n\nIf the context doesn't contain relevant information, respond based on your general knowledge but indicate that the response is not grounded in the provided documents."
                )
                augmented_messages = [system_message] + request.messages
            else:
                # No context found, proceed with original messages
                augmented_messages = request.messages

            # Get completion from LLM
            response = await llm_service.get_completion(
                messages=augmented_messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )

            # Update response with conversation ID
            response.conversation_id = str(conversation.id)

            # Update response with sources
            response.sources = sources

            # Create assistant message in database
            assistant_message = MessageModel(
                id=generate_uuid(),
                conversation_id=conversation.id,
                role="assistant",
                content=response.choices[0].message.content,
                timestamp=datetime.now(),
                sources=[source.document_id for source in sources]
            )
            db.add(assistant_message)
            db.commit()

            # Log the query for analytics
            response_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to ms
            query_log = QueryLog(
                id=generate_uuid(),
                conversation_id=conversation.id,
                query=request.messages[-1].content,
                response=response.choices[0].message.content,
                retrieved_chunks=len(context_chunks),
                response_time_ms=int(response_time),
                model_used=response.model
            )
            db.add(query_log)
            db.commit()

            logger.info(f"Processed chat request in {response_time:.2f}ms for conversation {conversation.id}")
            return response

        except Exception as e:
            logger.error(f"Error processing chat request: {str(e)}")
            raise

    def get_conversation_history(self, conversation_id: str, db: Session) -> List[Message]:
        """
        Retrieve conversation history
        """
        try:
            messages = db.query(MessageModel).filter(
                MessageModel.conversation_id == conversation_id
            ).order_by(MessageModel.timestamp).all()

            return [
                Message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    sources=msg.sources or []
                )
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            raise


# Global instance
chat_service = ChatService()