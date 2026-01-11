from fastapi import HTTPException, status
from typing import Optional


class RAGException(HTTPException):
    """Base exception for RAG chatbot system"""

    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[dict] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class DocumentNotFoundException(RAGException):
    """Raised when a requested document is not found"""

    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )


class ConversationNotFoundException(RAGException):
    """Raised when a requested conversation is not found"""

    def __init__(self, conversation_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )


class VectorStoreException(RAGException):
    """Raised when there's an error with the vector store"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store error: {detail}"
        )


class LLMException(RAGException):
    """Raised when there's an error with the LLM service"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM service error: {detail}"
        )


class DocumentProcessingException(RAGException):
    """Raised when there's an error processing a document"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document processing error: {detail}"
        )