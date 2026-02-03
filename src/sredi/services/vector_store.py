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

    def add_segment(self, segment: DocSegment, workspace_id: Optional[Any] = None):
        """Add a segment to the vector store.
        
        Args:
            segment: The DocSegment to add.
            workspace_id: Optional workspace ID to override segment.document.workspace_id.
        """
        # Resolve workspace_id: passed > segment.document > unknown
        if workspace_id is None:
            if segment.document:
                workspace_id = segment.document.workspace_id
            else:
                workspace_id = "unknown"

        # Ensure metadata values are strings or other supported types
        metadata = {
            "document_id": str(segment.document_id),
            "workspace_id": str(workspace_id),
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

    def get_embeddings(self, segment_ids: List[str]) -> Dict[str, List[float]]:
        """Retrieve embeddings for specific segments to support clustering.
        
        Args:
            segment_ids: List of segment IDs (strings).
            
        Returns:
            Dictionary mapping segment_id -> embedding vector.
        """
        if not segment_ids:
            return {}
            
        # Fetch embeddings using get()
        results = self.collection.get(
            ids=segment_ids,
            include=["embeddings"]
        )
        
        embeddings_map = {}
        # Explicit check for None to avoid numpy ambiguity if returned as array
        if results and results.get("ids") is not None and results.get("embeddings") is not None:
            ids = results["ids"]
            embeddings = results["embeddings"]
            
            for i, seg_id in enumerate(ids):
                embeddings_map[seg_id] = embeddings[i]
                
        return embeddings_map
