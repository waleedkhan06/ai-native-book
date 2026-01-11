import asyncio
import hashlib
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from ..models.document import Document, DocumentChunk
from ..database.models import Document as DocumentModel, DocumentChunk as DocumentChunkModel
from ..ingestion.chunker import document_chunker
from ..services.vector_store import vector_store_service
from ..utils.helpers import generate_uuid, generate_checksum

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for document management
    """

    def __init__(self):
        pass

    async def create_document(
        self,
        title: str,
        content: str,
        source_url: str,
        db: Session,
        metadata: Optional[dict] = None
    ) -> Document:
        """
        Create a new document and process it for RAG
        """
        try:
            # Calculate checksum to detect changes
            checksum = generate_checksum(content)

            # Check if document already exists
            existing_doc = db.query(DocumentModel).filter(
                DocumentModel.source_url == source_url,
                DocumentModel.checksum == checksum
            ).first()

            if existing_doc:
                # Document hasn't changed, return existing
                return Document(
                    id=str(existing_doc.id),
                    title=existing_doc.title,
                    content=existing_doc.content,
                    source_url=existing_doc.source_url,
                    checksum=existing_doc.checksum,
                    created_at=existing_doc.created_at,
                    updated_at=existing_doc.updated_at,
                    metadata=existing_doc.meta_data,
                    status=existing_doc.status
                )

            # Create new document
            doc_id = generate_uuid()
            db_document = DocumentModel(
                id=doc_id,
                title=title,
                content=content,
                source_url=source_url,
                checksum=checksum,
                metadata=metadata or {},
                status="PROCESSING"
            )

            db.add(db_document)
            db.commit()
            db.refresh(db_document)

            # Update status to processing
            db_document.status = "PROCESSING"
            db.commit()

            # Chunk the document
            chunks = document_chunker.chunk_document(content, str(doc_id))

            # Add chunks to database and commit them
            for chunk in chunks:
                db_chunk = DocumentChunkModel(
                    id=generate_uuid(),
                    document_id=doc_id,
                    content=chunk.content,
                    chunk_order=chunk.chunk_order,
                    embedding_model=chunk.embedding_model
                )
                db.add(db_chunk)

            # Commit chunks to database before vector store processing
            db.commit()

            # Generate embeddings and add to vector store
            try:
                chunk_ids = await vector_store_service.add_embeddings(chunks)

                # Update document status to ingested
                db_document.status = "INGESTED"
                db.commit()

                logger.info(f"Successfully processed document {doc_id} with {len(chunks)} chunks")

                return Document(
                    id=str(db_document.id),
                    title=db_document.title,
                    content=db_document.content,
                    source_url=db_document.source_url,
                    checksum=db_document.checksum,
                    created_at=db_document.created_at,
                    updated_at=db_document.updated_at,
                    metadata=db_document.metadata,
                    status=db_document.status,
                    chunk_count=len(chunks)
                )
            except Exception as e:
                logger.error(f"Error adding embeddings for document {doc_id}: {str(e)}")
                db_document.status = "FAILED"
                db.commit()
                raise

        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            if 'db_document' in locals():
                db_document.status = "FAILED"
                db.commit()
            raise

    async def get_document(self, document_id: str, db: Session) -> Optional[Document]:
        """
        Get a document by ID
        """
        try:
            db_document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
            if not db_document:
                return None

            # Get chunk count
            chunk_count = db.query(DocumentChunkModel).filter(
                DocumentChunkModel.document_id == document_id
            ).count()

            return Document(
                id=str(db_document.id),
                title=db_document.title,
                content=db_document.content,
                source_url=db_document.source_url,
                checksum=db_document.checksum,
                created_at=db_document.created_at,
                updated_at=db_document.updated_at,
                metadata=db_document.meta_data,
                status=db_document.status,
                chunk_count=chunk_count
            )
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {str(e)}")
            raise

    async def list_documents(self, db: Session, limit: int = 20, offset: int = 0) -> List[Document]:
        """
        List documents with pagination
        """
        try:
            db_documents = db.query(DocumentModel).offset(offset).limit(limit).all()

            documents = []
            for db_doc in db_documents:
                # Get chunk count
                chunk_count = db.query(DocumentChunkModel).filter(
                    DocumentChunkModel.document_id == db_doc.id
                ).count()

                documents.append(
                    Document(
                        id=str(db_doc.id),
                        title=db_doc.title,
                        content=db_doc.content,
                        source_url=db_doc.source_url,
                        checksum=db_doc.checksum,
                        created_at=db_doc.created_at,
                        updated_at=db_doc.updated_at,
                        metadata=db_doc.meta_data,
                        status=db_doc.status,
                        chunk_count=chunk_count
                    )
                )

            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise

    async def delete_document(self, document_id: str, db: Session) -> bool:
        """
        Delete a document and its associated chunks and vectors
        """
        try:
            # Delete from vector store
            await vector_store_service.delete_by_document_id(document_id)

            # Delete document chunks from database
            db.query(DocumentChunkModel).filter(
                DocumentChunkModel.document_id == document_id
            ).delete()

            # Delete document from database
            result = db.query(DocumentModel).filter(
                DocumentModel.id == document_id
            ).delete()

            db.commit()

            return result > 0
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise

    async def update_document_status(self, document_id: str, status: str, db: Session):
        """
        Update document processing status
        """
        try:
            db_document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
            if db_document:
                db_document.status = status
                db.commit()
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            raise


# Global instance
document_service = DocumentService()