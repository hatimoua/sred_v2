import json
import csv
from datetime import datetime
from pathlib import Path
from sqlmodel import select, col, desc
from ..db import get_session
from ..models import SegmentDecisionLog, DocSegment

def run_audit(output_dir: str = "audit_reports"):
    """Runs the audit and generates both a CLI report and CSV files."""
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_csv_path = out_path / f"tournament_diff_{timestamp}.csv"
    summary_csv_path = out_path / f"tournament_summary_{timestamp}.csv"

    session = next(get_session())
    
    # 1. Fetch Shadow Logs
    all_logs = session.exec(
        select(SegmentDecisionLog)
        .order_by(desc(SegmentDecisionLog.timestamp))
    ).all()
    
    # Filter in Python to avoid SQLAlchemy JSON extraction issues
    logs = [log for log in all_logs if log.reason.get("shadow_mode") is True]
    
    print(f"=== AUDIT REPORT: {len(logs)} Shadow Decisions ===\n")
    
    stats = {
        "TOTAL": len(logs),
        "LLM_INDEX_READY": 0,
        "STUB_INDEX_READY": 0,
        "DISAGREEMENTS": 0,
        "PROOF_FAILURES": 0,
        "AGREEMENTS": 0
    }

    diff_records = []

    for log in logs:
        payload = log.reason
        
        # Extract Tournament Data
        tourney = payload.get("tournament", {})
        llm_state = tourney.get("llm_final_state", "UNKNOWN")
        stub_state = tourney.get("stub_final_state", "UNKNOWN")
        disagreement = tourney.get("disagreement", False)
        
        # Extract Validation Data
        validation = payload.get("validation", {})
        is_tainted = validation.get("tainted", False)
        errors = validation.get("validation_errors", [])
        
        # Stats
        if llm_state == "INDEX_READY": stats["LLM_INDEX_READY"] += 1
        if stub_state == "INDEX_READY": stats["STUB_INDEX_READY"] += 1
        if disagreement: 
            stats["DISAGREEMENTS"] += 1
        else:
            stats["AGREEMENTS"] += 1
            
        if is_tainted: stats["PROOF_FAILURES"] += 1

        # Record for CSV (all records for full analysis)
        diff_records.append({
            "segment_id": str(log.segment_id),
            "llm_state": llm_state,
            "stub_state": stub_state,
            "disagreement": disagreement,
            "is_tainted": is_tainted,
            "validation_errors": "|".join(errors) if errors else "",
            "content_snippet": log.segment.content[:200].replace("\n", " "),
            "llm_reasoning": payload.get("router_suggested", {}).get("reasoning", "")[:500].replace("\n", " "),
            "timestamp": log.timestamp.isoformat()
        })

        # CLI SPOT CHECK: Print Disagreements (Interesting cases)
        if disagreement and llm_state == "INDEX_READY":
            print(f"[DISAGREEMENT] Segment {str(log.segment_id)[:8]}")
            print(f"  LLM: {llm_state} | Stub: {stub_state}")
            print(f"  Content Snippet: {log.segment.content[:60]}...")
            print("-" * 50)
            
        if is_tainted:
            print(f"[PROOF FAILURE] Segment {str(log.segment_id)[:8]}")
            print(f"  Errors: {errors}")
            print("-" * 50)

    # Write Diff CSV
    if diff_records:
        keys = diff_records[0].keys()
        with open(diff_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(diff_records)

    # Write Summary CSV
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            writer.writerow([k, v])

    print("\n=== SUMMARY STATISTICS ===")
    for k, v in stats.items():
        print(f"{k:25}: {v}")
    
    print(f"\nCSV Reports Generated:")
    print(f"  - Diff:    {diff_csv_path}")
    print(f"  - Summary: {summary_csv_path}")

if __name__ == "__main__":
    run_audit()