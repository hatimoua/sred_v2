"""
Full End-to-End Pipeline Test
Processes ./test_data and outputs audit_reports/TEST_DATA_CLAIM.md
"""
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
from sredi.services.compliance import ComplianceService
from sredi.models import WorkCluster, DocSegment, Workspace, Document

# Configuration
SOURCE_DIR = Path("./test_data")
REPORT_DIR = Path("./audit_reports")
REPORT_FILE = REPORT_DIR / "TEST_DATA_CLAIM.md"

# Force LLM routing (shadow_mode=False to persist decisions)
ROUTER_TYPE = "llm"
SHADOW_MODE = False


def clean_environment():
    """Wipe DB and Vector Store for a fresh run."""
    from sredi.config import settings
    print(f"🧹 Cleaning environment (DB: {settings.DATABASE_URL})...")
    
    # 1. Reset Postgres/SQLite
    reset_db()
    
    # 2. Reset ChromaDB (Delete the folder)
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    print("✨ Environment clean.\n")


async def run_pipeline():
    print(f"🚀 Starting Full E2E Pipeline Test on {SOURCE_DIR}...\n")
    print("=" * 60)
    
    # === Step 1: Ingestion ===
    print("\n📦 STEP 1: Ingestion")
    print("-" * 40)
    scanned, new_docs, skipped = ingestion.ingest_directory(SOURCE_DIR, "default")
    print(f"✅ Ingestion complete. Processed {new_docs} documents.\n")

    # === Step 2: Segmentation ===
    print("\n✂️ STEP 2: Segmentation")
    print("-" * 40)
    num_segments = segmentation.segment_documents("default")
    print(f"✅ Segmentation complete. Created {num_segments} segments.\n")

    # === Step 3: Routing (LLM) ===
    print("\n🧠 STEP 3: Routing (LLM)")
    print("-" * 40)
    print(f"Router Type: {ROUTER_TYPE}, Shadow Mode: {SHADOW_MODE}")
    processed = await router.route_segments_async(
        limit=1000,
        router_type=ROUTER_TYPE,
        shadow_mode=SHADOW_MODE,
        concurrency=5
    )
    print(f"✅ Routing complete. Processed {processed} segments.\n")

    # Get workspace
    with Session(engine) as session:
        ws = session.exec(select(Workspace).where(Workspace.name == "default")).first()
        if not ws:
            print("❌ Error: Default workspace not found.")
            return
        workspace_id = ws.id

    # === Step 4: Clustering ===
    print("\n🧩 STEP 4: Clustering")
    print("-" * 40)
    with Session(engine) as session:
        clustering_service = ClusteringService()
        n_clusters = clustering_service.cluster_workspace(session, workspace_id)
        print(f"✅ Clustering complete. Created {n_clusters} clusters.\n")

    # === Step 5: Narrative Generation ===
    print("\n📝 STEP 5: Narrative Generation")
    print("-" * 40)
    with Session(engine) as session:
        narrative_service = NarrativeService()
        await narrative_service.generate_cluster_titles(session, workspace_id)
        print("✅ Narrative generation complete.\n")

    # === Step 6: Compliance Mapping ===
    print("\n⚖️ STEP 6: Compliance Mapping")
    print("-" * 40)
    with Session(engine) as session:
        compliance_service = ComplianceService()
        await compliance_service.generate_compliance_narrative(session, workspace_id)
        print("✅ Compliance mapping complete.\n")

    # === Step 7: Report Generation ===
    print("\n📄 STEP 7: Report Generation")
    print("-" * 40)
    generate_markdown_report(workspace_id)
    
    print("\n" + "=" * 60)
    print("🏆 PIPELINE COMPLETE 🏆")
    print(f"📄 Output: {REPORT_FILE}")
    print("=" * 60)


def generate_markdown_report(workspace_id):
    """Generates the final Markdown claim document for test data."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    with Session(engine) as session:
        clusters = session.exec(
            select(WorkCluster).where(WorkCluster.workspace_id == workspace_id)
        ).all()
        
        if not clusters:
            print("No clusters found.")
            return
        
        # Sort by number of segments (largest first)
        clusters = list(clusters)
        clusters.sort(key=lambda c: len(c.segments), reverse=True)
        
        markdown_lines = [
            "# SR&ED Technical Narrative Report",
            "",
            "*Full E2E Pipeline Test - test_data*",
            "",
            f"**Total Clusters:** {len(clusters)}",
            "",
            "---",
            ""
        ]
        
        for cluster in clusters:
            # Fetch segments with document info for file references
            segments = session.exec(
                select(DocSegment)
                .where(DocSegment.cluster_id == cluster.id)
                .limit(10)
            ).all()
            
            markdown_lines.append(f"# Project: {cluster.title}")
            markdown_lines.append("")
            markdown_lines.append(f"*{len(cluster.segments)} evidence segments*")
            markdown_lines.append("")
            
            # Uncertainty
            markdown_lines.append("## 1. Technological Uncertainty")
            markdown_lines.append("")
            uncertainty = cluster.tech_uncertainty or "*Not yet generated*"
            markdown_lines.append(uncertainty)
            markdown_lines.append("")
            
            # Advancement
            markdown_lines.append("## 2. Technological Advancement")
            markdown_lines.append("")
            advancement = cluster.tech_advancement or "*Not yet generated*"
            markdown_lines.append(advancement)
            markdown_lines.append("")
            
            # Investigation
            markdown_lines.append("## 3. Systematic Investigation")
            markdown_lines.append("")
            investigation = cluster.systematic_investigation or "*Not yet generated*"
            markdown_lines.append(investigation)
            markdown_lines.append("")
            
            # Evidence Used with file references
            markdown_lines.append("### Evidence Used")
            markdown_lines.append("")
            for seg in segments:
                # Get the source file
                doc = session.get(Document, seg.document_id)
                file_ref = doc.file_path if doc else "unknown"
                # Make path relative
                if file_ref.startswith(str(SOURCE_DIR)):
                    file_ref = file_ref[len(str(SOURCE_DIR)):]
                
                preview = seg.content[:120].replace("\n", " ").strip()
                if len(seg.content) > 120:
                    preview += "..."
                markdown_lines.append(f"- **[{file_ref}]**: {preview}")
            markdown_lines.append("")
            markdown_lines.append("---")
            markdown_lines.append("")
        
        # Write file
        report_content = "\n".join(markdown_lines)
        with open(REPORT_FILE, "w") as f:
            f.write(report_content)
        
        print(f"✅ Report saved to: {REPORT_FILE}")
        print(f"   Clusters documented: {len(clusters)}")


if __name__ == "__main__":
    clean_environment()
    asyncio.run(run_pipeline())
