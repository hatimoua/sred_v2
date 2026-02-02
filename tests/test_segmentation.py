import pytest
from sqlmodel import Session, select
from pathlib import Path
import tempfile
from sredi.services import ingestion, segmentation
from sredi.models.models import Document, DocSegment, ProcessingState

def test_segment_documents_structural(session):
    workspace_name = "test_segment_structural"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "audit.md"
        content = "# Header A\n\nParagraph B\n\n## Header C\n\nParagraph D"
        file_path.write_text(content)
        
        # Ingest
        ingestion.ingest_directory(Path(tmpdir), workspace_name, session=session)
        
        # Segment
        count = segmentation.segment_documents(workspace_name, session=session)
        assert count == 4
        
        # Verify structural integrity
        segments = session.exec(select(DocSegment).order_by(DocSegment.sequence_index)).all()
        assert len(segments) == 4
        
        # 1. Monotonicity and Content
        assert segments[0].content == "# Header A"
        assert segments[0].sequence_index == 0
        assert segments[1].content == "Paragraph B"
        assert segments[1].sequence_index == 1
        assert segments[2].content == "## Header C"
        assert segments[3].content == "Paragraph D"
        
        # 2. Hierarchy Check
        # Paragraph B should have Header A as parent
        assert segments[1].parent_id == segments[0].id
        # Paragraph D should have Header C as parent
        assert segments[3].parent_id == segments[2].id
        # Header C should have NO parent (it's a top-level boundary in this logic)
        assert segments[2].parent_id is None
        
        # 3. Offset Integrity
        for seg in segments:
            assert content[seg.start_offset:seg.end_offset].strip() == seg.content
            
        # 4. Context Continuity
        # seg1.context_before should contain end of seg0.content
        assert segments[0].content in segments[1].context_before
        # seg1.context_after should contain start of seg2.content
        assert segments[2].content[:10] in segments[1].context_after

def test_document_reconstruction(session):
    workspace_name = "test_reconstruct"
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "reconstruct.md"
        content = "# Title\n\nThis is a test of reconstruction.\n\nIt should work."
        file_path.write_text(content)
        
        ingestion.ingest_directory(Path(tmpdir), workspace_name, session=session)
        segmentation.segment_documents(workspace_name, session=session)
        
        segments = session.exec(select(DocSegment)).all()
        reconstructed = segmentation.reconstruct_document(segments)
        
        # Check if all major content is present
        assert "# Title" in reconstructed
        assert "reconstruction" in reconstructed
        assert "It should work" in reconstructed

def test_segmentation_skips_processed_docs(session):
    workspace_name = "test_segment_skip"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = Path(tmpdir) / "test.txt"
        file1.write_text("Content")
        
        ingestion.ingest_directory(Path(tmpdir), workspace_name, session=session)
        
        # Segment once
        segmentation.segment_documents(workspace_name, session=session)
        
        # Segment again
        count = segmentation.segment_documents(workspace_name, session=session)
        assert count == 0
