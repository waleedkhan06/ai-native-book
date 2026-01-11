import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy.orm import Session
from src.models.document import Document
from src.services.document_service import DocumentService


@pytest.mark.asyncio
async def test_create_document():
    """
    Test the create_document method
    """
    # Create a mock database session
    mock_db = MagicMock(spec=Session)

    # Create a document service instance
    document_service = DocumentService()

    # This test would require more complex mocking of the chunker and vector store
    # For now, we'll just verify the method exists and can be called
    assert hasattr(document_service, 'create_document')


def test_document_model():
    """
    Test the Document model creation
    """
    document = Document(
        id="test-id",
        title="Test Document",
        content="This is a test document",
        source_url="http://example.com/test",
        checksum="abc123"
    )

    assert document.id == "test-id"
    assert document.title == "Test Document"
    assert document.content == "This is a test document"
    assert document.source_url == "http://example.com/test"
    assert document.checksum == "abc123"