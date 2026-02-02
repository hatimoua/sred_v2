import csv
from pathlib import Path
from typing import List, Optional
from sqlmodel import Session, select
import uuid

from ..models import (
    Project, 
    DocSegment, 
    ProjectSegmentLink, 
    LinkType,
    ProcessingState,
    SegmentDecisionLog
)
from sqlmodel import Session, select, desc
from ..db import get_session
from ..services.ingestion import get_or_create_workspace

def load_anchors_from_csv(file_path: Path, workspace_name: str = "default") -> int:
    """Loads project anchors from a CSV file into a workspace.

    Expected CSV format: name,source_anchor,description

    Args:
        file_path: Path to the CSV file to load.
        workspace_name: Name of the target workspace. Defaults to "default".

    Returns:
        The number of projects successfully loaded or updated.
    """
    session_gen = get_session()
    session = next(session_gen)
    workspace = get_or_create_workspace(session, workspace_name)
    
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name")
                source_anchor = row.get("source_anchor")
                description = row.get("description", "")
                
                if not name or not source_anchor:
                    print(f"Skipping invalid row: {row}")
                    continue
                
                # Check exist
                statement = select(Project).where(
                    Project.source_anchor == source_anchor, 
                    Project.workspace_id == workspace.id
                )
                existing = session.exec(statement).first()
                if existing:
                    # Update? For MVP, just skip or update fields
                    existing.name = name
                    existing.description = description
                    session.add(existing)
                else:
                    proj = Project(
                        workspace_id=workspace.id,
                        name=name,
                        source_anchor=source_anchor,
                        description=description
                    )
                    session.add(proj)
                    count += 1
            session.commit()
    except Exception as e:
        print(f"Error loading anchors: {e}")
        session.rollback()
    finally:
        session.close()

    return count

def group_segments(workspace_name: str = "default", use_tournament: bool = True, session: Optional[Session] = None) -> int:
    """Links segments to projects based on strong anchor matches or tournament results.

    Args:
        workspace_name: Name of the workspace to process. Defaults to "default".
        use_tournament: If True, leverages shadow-mode tournament results for discovery.
        session: Optional database session.
    """
    if session is None:
        session_gen = get_session()
        session = next(session_gen)
        should_close = True
    else:
        should_close = False
    workspace = get_or_create_workspace(session, workspace_name)
    
    new_links = 0
    try:
        # 1. Fetch all projects
        projects = session.exec(select(Project).where(Project.workspace_id == workspace.id)).all()
        if not projects:
            print("No projects defined.")
            return 0
        
        # 2. Fetch segments to process
        segments = session.exec(
            select(DocSegment).where(
                (DocSegment.processing_state == ProcessingState.INDEX_READY) |
                (DocSegment.processing_state == ProcessingState.REVIEW)
            )
        ).all()
        
        for seg in segments:
            # Check for existing anchors first (legacy path)
            matched_project_ids = set()
            for proj in projects:
                if proj.source_anchor and proj.source_anchor in seg.content:
                    matched_project_ids.add(proj.id)

            # 3. Leverage Shadow-Mode Tournament results
            if use_tournament:
                # Fetch latest decision log for this segment
                log_stmt = select(SegmentDecisionLog).where(
                    SegmentDecisionLog.segment_id == seg.id
                ).order_by(desc(SegmentDecisionLog.timestamp)).limit(1)
                latest_log = session.exec(log_stmt).first()

                if latest_log and latest_log.reason.get("shadow_mode"):
                    # Extract findings from the shadow log
                    # This is where we would expand to use LLM-discovered project associations
                    # For now, we look for 'signals' or 'reasoning' hints if applicable
                    pass

            for project_id in matched_project_ids:
                # Create Link if it doesn't exist
                link_stmt = select(ProjectSegmentLink).where(
                    ProjectSegmentLink.project_id == project_id,
                    ProjectSegmentLink.segment_id == seg.id
                )
                existing_link = session.exec(link_stmt).first()
                
                if not existing_link:
                    link = ProjectSegmentLink(
                        project_id=project_id,
                        segment_id=seg.id,
                        confidence=1.0,
                        link_type=LinkType.STRONG_ANCHOR
                    )
                    session.add(link)
                    new_links += 1
        
        session.commit()
    except Exception as e:
        print(f"Error grouping segments: {e}")
        session.rollback()
    finally:
        if should_close:
            session.close()
        
    return new_links
