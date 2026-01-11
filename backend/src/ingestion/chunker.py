import re
from typing import List
from ..models.document import DocumentChunk
from ..utils.helpers import generate_uuid


class DocumentChunker:
    """
    Service to chunk documents for vector storage
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # in tokens (approximate)
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Split text into chunks of approximately chunk_size tokens
        This is a simple implementation that splits by sentences
        In a real implementation, you'd use a proper tokenization method
        """
        # Simple approach: split by sentences and estimate token count
        # In a real implementation, use tiktoken or similar for accurate tokenization

        # First, split text into sentences
        sentences = re.split(r'[.!?]+\s+', text)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_order = 0

        for sentence in sentences:
            # Estimate token count (roughly 1 token = 4 characters for English text)
            sentence_tokens = len(sentence) // 4

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create a chunk
                chunk = DocumentChunk(
                    id=generate_uuid(),
                    document_id=document_id,
                    content=current_chunk.strip(),
                    chunk_order=chunk_order,
                    embedding_model="text-embedding-ada-002"  # Default model
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Get last few sentences for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    temp_sentences = current_chunk.split('. ')

                    # Add sentences until we reach overlap size
                    for sent in reversed(temp_sentences):
                        sent_tokens = len(sent) // 4
                        if overlap_tokens + sent_tokens > self.chunk_overlap:
                            break
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens

                    current_chunk = '. '.join(overlap_sentences) + ' ' + sentence
                    current_tokens = overlap_tokens + sentence_tokens
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens

                chunk_order += 1
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=generate_uuid(),
                document_id=document_id,
                content=current_chunk.strip(),
                chunk_order=chunk_order,
                embedding_model="text-embedding-ada-002"
            )
            chunks.append(chunk)

        return chunks

    def chunk_document(self, document_content: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces for vector storage
        """
        return self.chunk_text(document_content, document_id)


# Global instance
document_chunker = DocumentChunker()