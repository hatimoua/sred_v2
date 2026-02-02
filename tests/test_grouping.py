import pytest
import uuid
from sredi.services import grouping
from sredi.models.models import DocSegment, Project, ProjectSegmentLink
from sredi.models.enums import ProcessingState, LinkType
from sqlmodel import Session, select, col

def test_group_segments_strong_anchor(session, test_workspace):
    # Create a project with an anchor
    project = Project(
        workspace_id=test_workspace.id,
        name="Project Alpha",
        source_anchor="ALPHA-123",
        description="Test Project"
    )
    session.add(project)
    
    # Create an INDEX_READY segment containing the anchor
    seg1 = DocSegment(
        document_id=uuid.uuid4(),
        content="This evidence belongs to ALPHA-123 project.",
        processing_state=ProcessingState.INDEX_READY
    )
    
    # Create a REVIEW segment containing the anchor
    seg2 = DocSegment(
        document_id=uuid.uuid4(),
        content="Maybe this is ALPHA-123 related?",
        processing_state=ProcessingState.REVIEW
    )
    
    # Create a NOISE segment containing the anchor (should NOT be linked)
    seg3 = DocSegment(
        document_id=uuid.uuid4(),
        content="Ignore this ALPHA-123 mention.",
        processing_state=ProcessingState.NOISE
    )
    
    session.add_all([seg1, seg2, seg3])
    session.commit()
    
    count = grouping.group_segments(workspace_name=test_workspace.name, session=session)
    
    # Should link seg1 and seg2, but not seg3
    assert count == 2
    
    links = session.exec(select(ProjectSegmentLink)).all()
    assert len(links) == 2
    linked_segment_ids = {link.segment_id for link in links}
    assert seg1.id in linked_segment_ids
    assert seg2.id in linked_segment_ids
    assert seg3.id not in linked_segment_ids
    assert all(link.link_type == LinkType.STRONG_ANCHOR for link in links)

def test_containment_invariant_grouping(session, test_workspace):
    """
    Containment Invariant: Only INDEX_READY and REVIEW segments are eligible for grouping.
    Rule: QUARANTINE or NOISE segments should NEVER be linked to projects.
    """
    project = Project(
        workspace_id=test_workspace.id,
        name="Project Beta",
        source_anchor="BETA-456",
        description="Containment Test"
    )
    session.add(project)
    
    # Segment in QUARANTINE with anchor
    seg_quarantine = DocSegment(
        document_id=uuid.uuid4(),
        content="This is BETA-456 in quarantine.",
        processing_state=ProcessingState.QUARANTINE
    )
    
    # Segment in NOISE with anchor
    seg_noise = DocSegment(
        document_id=uuid.uuid4(),
        content="This is BETA-456 noise.",
        processing_state=ProcessingState.NOISE
    )
    
    session.add_all([seg_quarantine, seg_noise])
    session.commit()
    
    # Run grouping
    grouping.group_segments(workspace_name=test_workspace.name, session=session)
    
    # Assert no links were created for these segments
    links = session.exec(
        select(ProjectSegmentLink).where(col(ProjectSegmentLink.segment_id).in_([seg_quarantine.id, seg_noise.id]))
    ).all()
    assert len(links) == 0
