import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, models
from uuid import UUID
import logging
import uuid
from ..config import settings
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service to interface with Qdrant vector database
    """

    def __init__(self):
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=True
        )
        self.collection_name = settings.collection_name

    async def create_collection(self):
        """
        Create a collection in Qdrant if it doesn't exist
        """
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI ada-002 embedding size
                        distance=models.Distance.COSINE
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000,
                        indexing_threshold=20000
                    )
                )
                logger.info(f"Collection '{self.collection_name}' created successfully")

                # Create payload index for document_id to enable filtering
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("Created payload index for document_id")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")

                # Try to create the payload index if it doesn't exist
                try:
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_id",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    logger.info("Created payload index for document_id")
                except Exception as idx_error:
                    logger.warning(f"Could not create payload index (may already exist): {idx_error}")

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    async def add_embeddings(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Add document chunks with their embeddings to the vector store
        """
        try:
            points = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Generate a unique ID for this chunk in the vector store
                # Qdrant requires valid UUIDs, so we need to generate a proper UUID
                point_id = str(uuid.uuid4())
                ids.append(point_id)

                # Generate actual embeddings for the chunk content
                # Using the same embedding generation method as in search
                embedding = await self.generate_embedding(chunk.content, chunk.embedding_model or "text-embedding-ada-002")

                points.append(models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": str(chunk.document_id),
                        "content": chunk.content,
                        "chunk_order": chunk.chunk_order,
                        "embedding_model": chunk.embedding_model or "text-embedding-ada-002"
                    }
                ))

            # Upload points to Qdrant
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Added {len(points)} embeddings to collection '{self.collection_name}'")
            return ids

        except Exception as e:
            logger.error(f"Error adding embeddings: {str(e)}")
            raise

    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection
        """
        try:
            # Perform similarity search
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })

            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching for similar vectors: {str(e)}")
            raise

    async def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Generate embedding for text using OpenAI-compatible approach
        In a real implementation, this would call an embedding API
        """
        try:
            import httpx
            from ..config import settings

            # First try to use OpenAI's embedding API via OpenRouter if available
            if settings.openrouter_api_key:
                # Attempt to call the embedding API
                headers = {
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": model,  # Use the specified model
                    "input": text
                }

                timeout = httpx.Timeout(30.0, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/embeddings",
                        headers=headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        data = response.json()
                        embedding = data['data'][0]['embedding']
                        logger.info(f"Successfully generated embedding via OpenRouter")
                        return embedding
                    else:
                        logger.warning(f"Failed to generate embedding via OpenRouter: {response.status_code} - {response.text}")

            # If OpenRouter fails, try Cohere API (which is mentioned in the project setup)
            import os
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key:
                headers = {
                    "Authorization": f"Bearer {cohere_api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "texts": [text],
                    "model": "embed-english-v3.0",  # Cohere embedding model
                    "input_type": "search_document"
                }

                timeout = httpx.Timeout(30.0, connect=10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        "https://api.cohere.ai/v1/embed",
                        headers=headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        data = response.json()
                        embedding = data['embeddings'][0]
                        logger.info(f"Successfully generated embedding via Cohere")
                        return embedding
                    else:
                        logger.warning(f"Failed to generate embedding via Cohere: {response.status_code} - {response.text}")

            # Fallback to a local solution if all API calls fail
            # Using a simple approach that creates embeddings based on text content
            import hashlib
            import struct

            # Generate a deterministic embedding based on the text
            hash_input = text.encode('utf-8')
            hash_value = hashlib.sha256(hash_input).digest()

            # Convert the hash to a list of floats that sum to a reasonable range
            embedding = []
            for i in range(0, len(hash_value), 4):
                chunk = hash_value[i:i+4]
                if len(chunk) < 4:
                    chunk = chunk + b'\x00' * (4 - len(chunk))
                # Convert 4 bytes to a float value between -1 and 1
                int_val = struct.unpack('<I', chunk)[0]
                float_val = (int_val % 200000) / 100000.0 - 1.0
                embedding.append(float_val)

            # Pad or truncate to 1536 dimensions (OpenAI ada-002 size)
            while len(embedding) < 1536:
                embedding.append(0.0)
            embedding = embedding[:1536]

            logger.warning("Using deterministic embedding generation as fallback - for production, configure OpenRouter or Cohere API keys")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536

    async def delete_by_document_id(self, document_id: str):
        """
        Delete all vectors associated with a document
        """
        try:
            # Find all points with this document_id in payload
            scroll_result = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=10000  # Assuming max documents won't exceed this
            )

            point_ids = [point.id for point in scroll_result.points]

            if point_ids:
                await self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                logger.info(f"Deleted {len(point_ids)} vectors for document {document_id}")
            else:
                logger.info(f"No vectors found for document {document_id}")

        except Exception as e:
            logger.error(f"Error deleting vectors for document {document_id}: {str(e)}")
            raise

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        """
        try:
            info = await self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "points_count": info.points_count,
                "segments_count": info.segments_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise

    async def close(self):
        """
        Close the connection to Qdrant
        """
        await self.client.close()


# Global instance
vector_store_service = VectorStoreService()