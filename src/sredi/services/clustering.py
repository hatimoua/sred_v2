from typing import List, Optional, Dict
import numpy as np
import uuid
from sqlmodel import Session, select
from sklearn.cluster import AgglomerativeClustering

from ..models import DocSegment, WorkCluster, Workspace, ProcessingState, ClassificationLabel
from ..services.vector_store import VectorStoreService

class ClusteringService:
    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        self.vector_store = vector_store or VectorStoreService()

    def cluster_workspace(self, session: Session, workspace_id: uuid.UUID) -> int:
        """Clusters technical segments in a workspace into WorkClusters.
        
        Args:
            session: DB Session.
            workspace_id: ID of the workspace.
            
        Returns:
            Number of clusters created.
        """
        print(f"🧩 Clustering workspace {workspace_id}...")
        
        # 1. Fetch TECHNICAL segments
        # We look for segments that are labeled TECHNICAL or INDEX_READY
        # Since we are running this after routing, we assume state is updated.
        # But wait, in phase 6 we saw that LLM sets TECHNICAL label but final state might be REVIEW if no proof.
        # The prompt says "Fetch all TECHNICAL segments".
        # Let's target segments with classification_label='TECHNICAL' OR processing_state='INDEX_READY'.
        # Actually, let's stick to 'INDEX_READY' as "High Quality Technical" segments are what we want to group.
        # If we group 'REVIEW' segments, we might group noise.
        # However, the user said "groups TECHNICAL segments".
        # Let's check the models. ClassificationLabel.TECHNICAL.
        
        # Let's fetch segments that have classification_label = TECHNICAL
        # This covers both INDEX_READY and REVIEW (Technical but unproven)
        # But maybe we only want strong ones. The prompt says "high-quality TECHNICAL segments" in Phase 6 context, 
        # but Phase 7 prompt says "groups TECHNICAL segments into 'Work Items'".
        # If we only group INDEX_READY, we might have very few (0 in the last run).
        # But we are re-ingesting everything.
        # Let's filter by classification_label == ClassificationLabel.TECHNICAL.
        
        from ..models import Document

        query = select(DocSegment).join(Document).where(
            Document.workspace_id == workspace_id,
            DocSegment.classification_label == ClassificationLabel.TECHNICAL
        )
        segments = session.exec(query).all()
        
        if not segments:
            print("No TECHNICAL segments found to cluster.")
            return 0
            
        print(f"Found {len(segments)} technical segments.")
        
        if len(segments) < 3:
            print("Not enough segments to cluster (min 3). Skipping.")
            return 0
            
        # 2. Fetch Embeddings
        segment_ids = [str(s.id) for s in segments]
        embeddings_map = self.vector_store.get_embeddings(segment_ids)
        
        # Filter segments that actually have embeddings
        valid_segments = []
        valid_vectors = []
        
        for seg in segments:
            emb = embeddings_map.get(str(seg.id))
            if emb is not None:
                valid_segments.append(seg)
                valid_vectors.append(emb)
                
        if len(valid_segments) < 3:
             print("Not enough segments with embeddings. Skipping.")
             return 0
             
        X = np.array(valid_vectors)
        
        # 3. Perform Clustering
        # Distance threshold 1.0 (cosine distance is 1 - cosine_similarity).
        # Chroma embeddings are usually normalized, so cosine distance is appropriate.
        # Scikit-learn AgglomerativeClustering with 'cosine' metric.
        # Note: 'cosine' metric requires sklearn 1.2+
        # If distance_threshold is set, n_clusters must be None.
        
        print(f"Running AgglomerativeClustering on {len(X)} vectors...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0, # Adjust as needed. 1.0 allows for somewhat loose clusters.
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(X)
        
        # 4. Group and Persist
        unique_labels = set(labels)
        n_clusters = len(unique_labels)
        print(f"Created {n_clusters} clusters.")
        
        # Map label -> WorkCluster
        label_to_cluster = {}
        
        for label_id in unique_labels:
            # Create a new WorkCluster
            cluster = WorkCluster(
                workspace_id=workspace_id,
                title=f"Cluster {label_id} (Pending Title)"
            )
            session.add(cluster)
            session.flush() # get ID
            label_to_cluster[label_id] = cluster
            
        # Assign segments
        for i, label_id in enumerate(labels):
            segment = valid_segments[i]
            cluster = label_to_cluster[label_id]
            segment.cluster_id = cluster.id
            session.add(segment)
            
        session.commit()
        return n_clusters
