from sqlalchemy import Column, Integer, String, DateTime, Text, UUID, Boolean, ForeignKey, JSON, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_url = Column(String, nullable=False)
    checksum = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    meta_data = Column(JSON)
    status = Column(String, default="PROCESSING")  # PROCESSING, INGESTED, FAILED

    # Add indexes for frequently queried columns
    __table_args__ = (
        Index('idx_document_source_url', 'source_url'),
        Index('idx_document_checksum', 'checksum'),
        Index('idx_document_status', 'status'),
        Index('idx_document_created_at', 'created_at'),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_order = Column(Integer, nullable=False)
    vector_id = Column(String, nullable=True)  # ID in the vector database
    embedding_model = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Add indexes for frequently queried columns
    __table_args__ = (
        Index('idx_document_chunk_document_id', 'document_id'),
        Index('idx_document_chunk_vector_id', 'vector_id'),
        Index('idx_document_chunk_created_at', 'created_at'),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    preferences = Column(JSON)

    # Add indexes for frequently queried columns
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_created_at', 'created_at'),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    meta_data = Column(JSON)

    # Add indexes for frequently queried columns
    __table_args__ = (
        Index('idx_conversation_user_id', 'user_id'),
        Index('idx_conversation_created_at', 'created_at'),
        Index('idx_conversation_updated_at', 'updated_at'),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    token_count = Column(Integer, nullable=True)
    sources = Column(JSON)  # Sources referenced in the response (for RAG)

    # Add indexes for frequently queried columns
    __table_args__ = (
        Index('idx_message_conversation_id', 'conversation_id'),
        Index('idx_message_role', 'role'),
        Index('idx_message_timestamp', 'timestamp'),
    )


class VectorIndex(Base):
    __tablename__ = "vector_indexes"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_name = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    dimension = Column(Integer, nullable=False)
    document_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    retrieved_chunks = Column(Integer, default=0)
    response_time_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_used = Column(String, nullable=True)
    feedback_score = Column(Integer, nullable=True)  # 1-5 rating