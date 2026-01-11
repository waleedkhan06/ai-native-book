from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from ...models.document import Document, DocumentListResponse
from ...api.deps import get_db_session
from ...services.document_service import document_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    db: Session = Depends(get_db_session),
    limit: int = 20,
    offset: int = 0
):
    """
    List documents in the knowledge base
    """
    try:
        documents = await document_service.list_documents(db, limit=limit, offset=offset)
        total = len(documents)  # In a real implementation, this would be the actual total count

        return DocumentListResponse(
            documents=documents,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.post("/documents", response_model=Document)
async def add_document(
    title: str,
    content: str,
    source_url: str,
    metadata: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """
    Add a new document to the knowledge base
    """
    try:
        # Parse metadata if provided
        import json
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata JSON format"
                )

        document = await document_service.create_document(
            title=title,
            content=content,
            source_url=source_url,
            db=db,
            metadata=metadata_dict
        )

        return document
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding document: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    db: Session = Depends(get_db_session)
):
    """
    Get document details
    """
    try:
        document = await document_service.get_document(document_id, db)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        return document
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db_session)
):
    """
    Delete a document
    """
    try:
        success = await document_service.delete_document(document_id, db)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    source_url: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """
    Upload a document file to the knowledge base
    """
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')

        # Generate title if not provided
        if not title:
            title = file.filename

        # Generate source URL if not provided
        if not source_url:
            source_url = f"upload://{file.filename}"

        document = await document_service.create_document(
            title=title,
            content=content_str,
            source_url=source_url,
            db=db
        )

        return document
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )