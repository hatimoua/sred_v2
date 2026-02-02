import os
import shutil
import asyncio
import csv
from collections import defaultdict
from pathlib import Path
from sqlmodel import Session, select
from sredi.main import reset_db
from sredi.db import engine
from sredi.services import ingestion, segmentation, router
from sredi.models.models import SegmentDecisionLog, DocSegment
from sredi.services.router_contract import RouterLabel

# Configuration
SOURCE_DIR = Path("./src")
REPORT_DIR = "./audit_reports"
REPORT_FILE = os.path.join(REPORT_DIR, "victory_lap.csv")

def clean_environment():
    """Wipe DB and Vector Store for a fresh run."""
    from sredi.config import settings
    print(f"🧹 Cleaning environment (DB: {settings.DATABASE_URL})...")
    
    # Debug: Check counts before
    with Session(engine) as session:
        doc_count = session.exec(select(DocSegment)).all()
        print(f"DEBUG: Pre-reset segment count: {len(doc_count)}")

    # 1. Reset Postgres/SQLite
    reset_db()
    
    # Debug: Check counts after
    with Session(engine) as session:
        from sredi.models.models import Document, Workspace
        docs = session.exec(select(Document)).all()
        workspaces = session.exec(select(Workspace)).all()
        print(f"DEBUG: Post-reset Document count: {len(docs)}")
        print(f"DEBUG: Post-reset Workspace count: {len(workspaces)}")

    # 2. Reset ChromaDB (Delete the folder)
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    print("✨ Environment clean.")

async def run_pipeline():
    """Run the full pipeline on our own source code."""
    print(f"🚀 Starting Ingestion on {SOURCE_DIR}...")
    
    # 1. Ingestion (Sync)
    # The user provided snippet used a service class, but existing code is a module-level function
    # that returns a tuple.
    # ingest_directory(directory: Path, workspace_name: str = "default", session: Session = None)
    scanned, new_docs, skipped = ingestion.ingest_directory(SOURCE_DIR, "default")
    print(f"✅ Ingestion Complete. Scanned: {scanned}, New: {new_docs}, Skipped: {skipped}")

    # 2. Segmentation (Sync)
    print("✂️ Starting Segmentation...")
    num_segments = segmentation.segment_documents("default")
    print(f"✅ Segmentation Complete. Created {num_segments} segments.")

    # 3. Routing (Async)
    print("🧠 Starting Shadow Mode Routing...")
    # In shadow mode, we route with the LLM (as configured in config.py or passed here)
    # but the logs will record both LLM and Stub because of the shadow_mode=True flag handled in the router service.
    # However, the user snippet calls `router_service.route_segments`.
    # We need to make sure we trigger the routing.
    # route_segments is sync, route_segments_async is async.
    
    # We want to force "llm" router type and shadow_mode=True explicitly here to match the tournament intent,
    # although config.py also defaults to this now.
    processed = await router.route_segments_async(
        limit=1000, # Large limit for tournament
        router_type="llm",
        shadow_mode=True,
        concurrency=5
    )
    print(f"✅ Routing Complete. Processed {processed} segments.")

def analyze_results():
    """Find segments where Stub failed but LLM succeeded."""
    print("📊 Analyzing Tournament Results...")
    
    with Session(engine) as session:
        # Fetch all decision logs
        logs = session.exec(select(SegmentDecisionLog)).all()
        
        # Find Divergence
        wins = [] # Recall Wins: Stub(No) -> LLM(Yes)
        precision_wins = [] # Precision Wins: Stub(Yes) -> LLM(No)
        agreements = 0
        context_wins = 0
        
        # Confusion Matrix
        # Rows: Stub, Cols: LLM
        # We use states here since that's what we have in the tournament payload
        states = ["INDEX_READY", "REVIEW", "NOISE", "QUARANTINE"]
        matrix = {s1: {s2: 0 for s2 in states} for s1 in states}
        
        total_analyzed = 0
        
        for log in logs:
            reason = log.reason
            # Handle potential string serialization of JSON
            if isinstance(reason, str):
                import json
                try:
                    reason = json.loads(reason)
                except:
                    continue
            
            if not isinstance(reason, dict):
                continue
                
            tournament = reason.get("tournament")
            if not tournament:
                # Skip logs that aren't tournament comparisons
                continue
            
            total_analyzed += 1
            
            stub_state = tournament.get("stub_final_state", "REVIEW")
            llm_state = tournament.get("llm_final_state", "REVIEW")
            
            # Update Matrix
            # Normalize to avoid key errors if unexpected strings appear
            s_key = stub_state if stub_state in states else "REVIEW"
            l_key = llm_state if llm_state in states else "REVIEW"
            
            matrix[s_key][l_key] += 1
            
            if s_key == l_key:
                agreements += 1

            # Get detailed LLM info from the main payload
            router_suggested = reason.get("router_suggested", {})
            llm_label = router_suggested.get("label", "AMBIGUOUS")
            llm_reasoning = router_suggested.get("reasoning") or ""
            
            # Infer Stub Label from State (Approximate)
            state_to_label = {
                "INDEX_READY": "TECHNICAL",
                "REVIEW": "AMBIGUOUS",
                "NOISE": "NOISE",
                "QUARANTINE": "AMBIGUOUS"
            }
            stub_label = state_to_label.get(stub_state, "AMBIGUOUS")

            # CRITERIA: Stub = NOISE/REVIEW vs LLM = INDEX_READY (Technical)
            stub_fail = stub_state in ["NOISE", "REVIEW"]
            llm_win = llm_state == "INDEX_READY"
            
            # CRITERIA: Stub = INDEX_READY vs LLM = NOISE/REVIEW (Precision Win)
            stub_overconfident = stub_state == "INDEX_READY"
            llm_correction = llm_state in ["NOISE", "REVIEW"]
            
            if stub_fail and llm_win:
                # We found a victory!
                # Check if it was due to "Memory"
                reasoning_upper = llm_reasoning.upper()
                has_context = "RECALLED MEMORY" in reasoning_upper or "STRUCTURAL CONTEXT" in reasoning_upper
                if has_context:
                    context_wins += 1
                
                wins.append({
                    "segment_id": str(log.segment_id),
                    "stub_label": stub_label,
                    "llm_label": llm_label,
                    "type": "RECALL_WIN",
                    "has_context": has_context,
                    "reasoning_snippet": llm_reasoning[:200].replace("\n", " ")
                })
            
            if stub_overconfident and llm_correction:
                precision_wins.append({
                    "segment_id": str(log.segment_id),
                    "stub_label": stub_label,
                    "llm_label": llm_label,
                    "type": "PRECISION_WIN",
                    "has_context": False,
                    "reasoning_snippet": llm_reasoning[:200].replace("\n", " ")
                })

    # Export Report
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "stub_label", "llm_label", "type", "has_context", "reasoning_snippet"])
        writer.writeheader()
        writer.writerows(wins + precision_wins)
        
    print(f"\n🏆 TOURNAMENT RESULTS 🏆")
    print(f"Total Analyzed Decisions: {total_analyzed}")
    print(f"Agreements: {agreements}")
    print(f"Recall Wins (Stub->No, LLM->Yes): {len(wins)}")
    print(f"Precision Wins (Stub->Yes, LLM->No): {len(precision_wins)}")
    print(f"Context-Assisted Wins: {context_wins}")
    
    print("\n📉 Confusion Matrix (Row=Stub State, Col=LLM State):")
    headers = [s for s in states if s != "QUARANTINE"] # Hide Quarantine from matrix output
    print(f"{'':<12} | {' | '.join([f'{h:<11}' for h in headers])}")
    print("-" * 55)
    for row_state in headers:
        counts = [matrix[row_state][col_state] for col_state in headers]
        print(f"{row_state:<12} | {' | '.join([f'{c:<11}' for c in counts])}")

    print(f"\n📄 Report saved to: {REPORT_FILE}")

if __name__ == "__main__":
    clean_environment()
    asyncio.run(run_pipeline())
    analyze_results()
