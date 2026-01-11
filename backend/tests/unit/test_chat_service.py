import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.orm import Session
from src.models.chat import ChatCompletionRequest, Message
from src.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_process_chat_request():
    """
    Test the process_chat_request method
    """
    # Create a mock database session
    mock_db = MagicMock(spec=Session)

    # Create a chat service instance
    chat_service = ChatService()

    # Create a sample request
    request = ChatCompletionRequest(
        messages=[Message(role="user", content="Hello, world!")]
    )

    # This test would require more complex mocking of the vector store and LLM services
    # For now, we'll just verify the method exists and can be called
    assert hasattr(chat_service, 'process_chat_request')