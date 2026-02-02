import chromadb
from typing import List, Optional, Dict, Any
from ..models import DocSegment

class VectorStoreService:
    """Service for semantic storage and retrieval of document segments using ChromaDB."""

    def __init__(self, path: str = "./chroma_db", collection_name: str = "sred_segments"):
        """Initialize the vector store service.
        
        Args:
            path: Path to the persistent ChromaDB storage.
            collection_name: Name of the collection to use.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_segment(self, segment: DocSegment):
        """Add a segment to the vector store.
        
        Args:
            segment: The DocSegment to add.
        """
        # Ensure metadata values are strings or other supported types
        metadata = {
            "document_id": str(segment.document_id),
            "workspace_id": str(segment.document.workspace_id) if segment.document else "unknown",
            "processing_state": str(segment.processing_state.value) if segment.processing_state else "unknown"
        }
        
        self.collection.add(
            documents=[segment.content],
            metadatas=[metadata],
            ids=[str(segment.id)]
        )

    def search_similar(self, query: str, n_results: int = 3) -> List[str]:
        """Search for semantically similar segments.
        
        Args:
            query: The text to search for.
            n_results: Number of results to return.
            
        Returns:
            List of segment contents found.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Chroma returns a dict with lists. 'documents' is List[List[str]]
        if results and results.get("documents"):
            return results["documents"][0]
        return []
