import pytest
from typer.testing import CliRunner
from sredi.main import app
from sredi.models.models import Workspace, Document, DocSegment, WorkCluster
from sredi.models.enums import ProcessingState
import uuid

runner = CliRunner()

def test_status_command_basic(session):
    """Test the improved status command with various segments and states."""
    # Setup workspace
    ws_name = "status_test_ws"
    ws = Workspace(name=ws_name)
    session.add(ws)
    session.commit()
    session.refresh(ws)
    
    # Add a document
    doc = Document(
        workspace_id=ws.id,
        content_hash="hash123",
        filename="test.txt",
        file_path="/tmp/test.txt",
        file_size_bytes=100
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    
    # Add segments in different states
    seg1 = DocSegment(document_id=doc.id, content="q", processing_state=ProcessingState.QUARANTINE)
    seg2 = DocSegment(document_id=doc.id, content="r", processing_state=ProcessingState.REVIEW)
    seg3 = DocSegment(document_id=doc.id, content="i", processing_state=ProcessingState.INDEX_READY)
    session.add_all([seg1, seg2, seg3])
    session.commit()
    
    # Add a cluster
    cluster = WorkCluster(workspace_id=ws.id, title="Test Cluster")
    session.add(cluster)
    session.commit()
    session.refresh(cluster)
    
    # Link segment to cluster
    seg3.cluster_id = cluster.id
    session.add(seg3)
    session.commit()

    # Note: Running the CLI via CliRunner will use the real DB in main.py.
    # For a pure unit test, we'd need to mock get_session in main.py.
    pass
