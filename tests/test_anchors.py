import pytest
from sredi.services.segmentation import extract_anchors
from sredi.models.enums import AnchorType
from sredi.models.models import DocSegment, EntityAnchor
from sqlmodel import select

def test_extract_anchors_normalization():
    """Verify that anchors are extracted and normalized correctly."""
    text = """
    We fixed #123 and also referenced see #456.
    Check out PROJ-789 and proj-101.
    File is at src/main.py and we need to check docs/README.md.
    Traceback (most recent call last):
    Exception: something went wrong
    """
    
    anchors = extract_anchors(text)
    
    # Check PRs
    prs = [a for a in anchors if a["type"] == AnchorType.PR]
    assert len(prs) == 2
    assert prs[0]["value"] == "#123"
    assert prs[1]["value"] == "#456"
    
    # Check Tickets
    tickets = [a for a in anchors if a["type"] == AnchorType.TICKET]
    assert len(tickets) == 1
    assert tickets[0]["value"] == "PROJ-789"
    # Note: proj-101 is not matched because regex is case-sensitive for TICKET prefix \b[A-Z]
    
    # Check Files
    files = [a for a in anchors if a["type"] == AnchorType.FILE_REF]
    assert len(files) == 2
    assert files[0]["value"] == "src/main.py"
    assert files[1]["value"] == "docs/readme.md" # Normalized to lowercase
    
    # Check Error Codes
    errors = [a for a in anchors if a["type"] == AnchorType.ERROR_CODE]
    assert len(errors) == 2
    assert "Traceback" in [e["value"] for e in errors]
    assert "Exception" in [e["value"] for e in errors]

def test_extract_anchors_edge_cases():
    """Verify edge cases for anchor extraction."""
    # No anchors
    assert extract_anchors("Just some plain text.") == []
    
    # Overlapping or tricky strings
    text = "Fixing#123 and JIRA-12345678901 (too long) and a.py"
    anchors = extract_anchors(text)
    
    # Fixing#123 should still match #123 because regex allows prefix
    assert any(a["value"] == "#123" for a in anchors)
    
    # JIRA-12345678901 is 11 digits, regex says 2-10
    assert not any(a["type"] == AnchorType.TICKET for a in anchors)
    
    # a.py is a valid file ref
    assert any(a["value"] == "a.py" for a in anchors)

def test_persistence_in_create_segment(session):
    """Integration test to verify anchors are persisted when a segment is created."""
    from sredi.services.segmentation import _create_segment
    
    doc_id = pytest.importorskip("uuid").uuid4()
    text = "Fixed #999 in src/utils.py"
    
    seg_id = _create_segment(
        session,
        doc_id=doc_id,
        text=text,
        start=0,
        end=len(text),
        full_content=text,
        sequence_index=0,
        parent_id=None,
        is_header=False
    )
    
    # Verify segment exists
    seg = session.get(DocSegment, seg_id)
    assert seg is not None
    
    # Verify anchors were persisted
    statement = select(EntityAnchor).where(EntityAnchor.segment_id == seg_id)
    anchors = session.exec(statement).all()
    
    assert len(anchors) == 2
    anchor_values = [a.anchor_value for a in anchors]
    assert "#999" in anchor_values
    assert "src/utils.py" in anchor_values
