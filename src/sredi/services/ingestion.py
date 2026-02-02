import hashlib
import os
from pathlib import Path
from typing import List, Tuple
from sqlmodel import Session, select
import uuid

from ..models import Document, Workspace, PipelineRun
from ..db import get_session

CHUNK_SIZE = 8192

def compute_sha256(file_path: Path) -> str:
    """Computes the SHA256 hash of a file's content.

    Args:
        file_path: The path to the file to hash.

    Returns:
        The hex-encoded SHA256 hash string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(CHUNK_SIZE), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_or_create_workspace(session: Session, name: str = "default") -> Workspace:
    """Retrieves an existing workspace or creates a new one.

    Args:
        session: The active database session.
        name: The name of the workspace. Defaults to "default".

    Returns:
        The retrieved or newly created Workspace object.
    """
    statement = select(Workspace).where(Workspace.name == name)
    workspace = session.exec(statement).first()
    if not workspace:
        workspace = Workspace(name=name)
        session.add(workspace)
        session.commit()
        session.refresh(workspace)
    return workspace

def ingest_directory(directory: Path, workspace_name: str = "default", session: Session = None) -> Tuple[int, int, int]:
    """Recursively scans and ingests a directory of files into a workspace.

    Args:
        directory: The path to the directory to scan.
        workspace_name: Name of the target workspace. Defaults to "default".
        session: Optional database session. If None, a new one is created and closed.

    Returns:
        A tuple containing (num_scanned, num_new_docs, num_skipped_docs).
    """
    # Using a sync logic for MVP
    if session is None:
        session_gen = get_session()
        session = next(session_gen)
        should_close = True
    else:
        should_close = False
    
    workspace = get_or_create_workspace(session, workspace_name)
    
    new_docs = 0
    skipped_docs = 0
    scanned_docs = 0
    ingested_doc_ids = []

    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                
                # Check extension (optional, but good for MVP)
                if file_path.name.startswith("."):
                    continue

                scanned_docs += 1
                try:
                    content_hash = compute_sha256(file_path)
                    file_size = file_path.stat().st_size
                    
                    # Check dedup
                    statement = select(Document).where(
                        Document.content_hash == content_hash,
                        Document.workspace_id == workspace.id
                    )
                    existing_doc = session.exec(statement).first()
                    
                    if existing_doc:
                        skipped_docs += 1
                        # We could optionally update file path if it moved, but MVP invariants say avoid processing duplicates.
                        continue
                    
                    # Create Document
                    doc = Document(
                        workspace_id=workspace.id,
                        content_hash=content_hash,
                        filename=file_path.name,
                        file_path=str(file_path.absolute()),
                        file_size_bytes=file_size
                    )
                    session.add(doc)
                    session.commit()
                    session.refresh(doc)
                    
                    ingested_doc_ids.append(doc.id)
                    new_docs += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    # In a real system, we'd log this properly.
                    continue
        
        # Create PipelineRun and link documents to it
        run = PipelineRun(
            workspace_id=workspace.id,
            command="ingest",
            parameters={
                "directory": str(directory),
                "scanned": scanned_docs,
                "new": new_docs,
                "skipped": skipped_docs
            }
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        # Update newly ingested documents with the run_id
        if ingested_doc_ids:
            for doc_id in ingested_doc_ids:
                doc = session.get(Document, doc_id)
                if doc:
                    doc.ingestion_run_id = run.id
                    session.add(doc)
            session.commit()
        
    finally:
        if should_close:
            session.close()

    return scanned_docs, new_docs, skipped_docs
