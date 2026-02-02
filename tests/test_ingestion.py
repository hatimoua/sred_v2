from sqlmodel import Session, select
import pytest
from pathlib import Path
import tempfile
import os
from sredi.services import ingestion
from sredi.models.models import Document, Workspace

def test_ingest_directory(session):
    # Setup test workspace
    workspace_name = "test_ingest"
    
    # Create a temporary directory with files
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "test1.txt"
        file1.write_text("Technical content about database and latency.")
        
        file2 = Path(tmpdir) / "test2.md"
        file2.write_text("Architecture review notes.")
        
        # Run ingestion
        scanned, new_docs, skipped = ingestion.ingest_directory(Path(tmpdir), workspace_name, session=session)
        
        assert scanned == 2
        assert new_docs == 2
        assert skipped == 0
        
        # Verify database records
        docs = session.exec(select(Document)).all()
        assert len(docs) == 2
        assert {d.filename for d in docs} == {"test1.txt", "test2.md"}

def test_deduplication(session):
    workspace_name = "test_dedup"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "dup.txt"
        file1.write_text("Same content")
        
        # First ingestion
        scanned, new_docs, skipped = ingestion.ingest_directory(Path(tmpdir), workspace_name, session=session)
        assert scanned == 1
        assert new_docs == 1
        assert skipped == 0
        
        # Second ingestion (same file)
        scanned, new_docs, skipped = ingestion.ingest_directory(Path(tmpdir), workspace_name, session=session)
        assert scanned == 1
        assert new_docs == 0
        assert skipped == 1
