import os
import shutil
import asyncio
from pathlib import Path
from sqlmodel import Session, select
from sredi.db import engine
from sredi.main import reset_db
from sredi.services import ingestion, segmentation, router
from sredi.services.clustering import ClusteringService
from sredi.services.narrative import NarrativeService
from sredi.models import WorkCluster, DocSegment, Workspace

# Configuration
SOURCE_DIR = Path("./src")

def clean_environment():
    """Wipe DB and Vector Store for a fresh run."""
    from sredi.config import settings
    print(f"🧹 Cleaning environment (DB: {settings.DATABASE_URL})...")
    
    # 1. Reset Postgres/SQLite
    reset_db()
    
    # 2. Reset ChromaDB (Delete the folder)
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    print("✨ Environment clean.")

async def run_pipeline():
    print(f"🚀 Starting Phase 7 Full Run on {SOURCE_DIR}...")
    
    # 1. Ingestion
    print("\n📦 Ingestion...")
    scanned, new_docs, skipped = ingestion.ingest_directory(SOURCE_DIR, "default")
    print(f"Ingested: {new_docs} new documents.")

    # 2. Segmentation
    print("\n✂️ Segmentation...")
    num_segments = segmentation.segment_documents("default")
    print(f"Segmented: {num_segments} segments.")

    # 3. Routing
    print("\n🧠 Routing (LLM)...")
    # Using 'llm' router to ensure we get TECHNICAL labels
    processed = await router.route_segments_async(
        limit=1000,
        router_type="llm",
        shadow_mode=False, # We commit the LLM decision directly
        concurrency=5
    )
    print(f"Routed: {processed} segments.")

    # 4. Clustering
    print("\n🧩 Clustering...")
    with Session(engine) as session:
        # Get workspace ID
        ws = session.exec(select(Workspace).where(Workspace.name == "default")).first()
        if not ws:
            print("Error: Default workspace not found.")
            return
            
        clustering_service = ClusteringService()
        n_clusters = clustering_service.cluster_workspace(session, ws.id)
        print(f"Clusters Created: {n_clusters}")

    # 5. Narrative
    print("\n📝 Narrative Generation...")
    with Session(engine) as session:
        ws = session.exec(select(Workspace).where(Workspace.name == "default")).first()
        narrative_service = NarrativeService()
        await narrative_service.generate_cluster_titles(session, ws.id)

    # 6. Report
    print("\n🏆 PHASE 7 REPORT 🏆")
    with Session(engine) as session:
        ws = session.exec(select(Workspace).where(Workspace.name == "default")).first()
        clusters = session.exec(select(WorkCluster).where(WorkCluster.workspace_id == ws.id)).all()
        
        # Sort by number of segments desc
        clusters.sort(key=lambda c: len(c.segments), reverse=True)
        
        for i, cluster in enumerate(clusters, 1):
            print(f"\nCluster {i}: {cluster.title} ({len(cluster.segments)} segments)")
            # Print top 3 segment previews
            for seg in cluster.segments[:3]:
                preview = seg.content[:100].replace("\n", " ") + "..."
                print(f"  - {preview}")

if __name__ == "__main__":
    clean_environment()
    asyncio.run(run_pipeline())
