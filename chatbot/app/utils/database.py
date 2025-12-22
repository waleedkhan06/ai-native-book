from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = "textbook_content"
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize the Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Using 384 for Cohere embeddings
            )

    def add_document(self, doc_id: str, content: str, metadata: Dict):
        """Add a document to the collection"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=[0.0] * 384,  # Placeholder - actual vector will be computed by Cohere
                    payload={
                        "content": content,
                        "metadata": metadata
                    }
                )
            ]
        )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """Search for similar documents"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        return [
            {
                "id": result.id,
                "content": result.payload.get("content"),
                "metadata": result.payload.get("metadata"),
                "score": result.score
            }
            for result in results
        ]

    def get_all_collections(self):
        """Get list of all collections"""
        return self.client.get_collections()

# Global instance
qdrant_service = QdrantService()