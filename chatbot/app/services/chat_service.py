from typing import List, Dict, Optional
from app.models.chatbot_query import ChatQueryRequest, ChatQueryResponse, ContentEmbedRequest, ContentEmbedResponse
from app.utils.database import qdrant_service
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

class ChatService:
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

    async def process_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """
        Process a user query against textbook content using RAG
        """
        # Generate embedding for the query using Cohere
        response = self.cohere_client.embed(
            texts=[request.query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = response.embeddings[0]

        # Search in Qdrant for relevant content
        search_results = qdrant_service.search(query_embedding, limit=5)

        if not search_results:
            return ChatQueryResponse(
                answer="I couldn't find relevant information in the textbook to answer your question.",
                confidence=0.0,
                sources=[],
                query_id="temp-id"
            )

        # Prepare context from search results
        context_text = "\n".join([result["content"] for result in search_results])

        # Generate response using Cohere
        response = self.cohere_client.chat(
            message=request.query,
            preamble=f"Answer the question based on the following context from the Physical AI & Humanoid Robotics textbook: {context_text}",
            temperature=0.3
        )

        # Extract confidence based on response quality
        confidence = 0.8  # This would be calculated based on actual response quality metrics

        return ChatQueryResponse(
            answer=response.text,
            confidence=confidence,
            sources=[result["id"] for result in search_results],
            query_id="temp-id"  # In a real implementation, this would be generated
        )

    async def embed_content(self, request: ContentEmbedRequest) -> ContentEmbedResponse:
        """
        Embed textbook content for RAG retrieval
        """
        # Split content into chunks for better retrieval
        chunks = self._chunk_text(request.content)
        embedded_count = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"{request.content_id}_chunk_{i}"

            # Generate embedding for the chunk
            response = self.cohere_client.embed(
                texts=[chunk],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embedding = response.embeddings[0]

            # Add to Qdrant
            qdrant_service.add_document(
                doc_id=chunk_id,
                content=chunk,
                metadata={
                    **request.metadata,
                    "original_content_id": request.content_id
                }
            )
            embedded_count += 1

        return ContentEmbedResponse(
            success=True,
            content_id=request.content_id,
            chunks_embedded=embedded_count
        )

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks of specified size
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            if current_size + len(word) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get chat history for a user (placeholder implementation)
        """
        # In a real implementation, this would fetch from a database
        return []

# Global instance
chat_service = ChatService()