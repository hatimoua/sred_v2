import pytest
import uuid
from sredi.services import router
from sredi.models.models import DocSegment, SegmentDecisionLog
from sqlmodel import Session, select
from sredi.models.enums import ProcessingState, ClassificationLabel

def test_route_segments_technical(session):
    # Create a technical segment with multiple signals to pass the promotion gate
    # Signals: 'Traceback' (STACK_TRACE - Strong)
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="This is a technical error: Traceback found.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()
    
    count = router.route_segments(session=session)
    assert count == 1
    
    session.refresh(seg)
    assert seg.processing_state == ProcessingState.INDEX_READY
    assert seg.classification_label == ClassificationLabel.TECHNICAL

    # Verify DecisionLog
    logs = session.exec(select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == seg.id)).all()
    assert len(logs) == 1
    assert logs[0].old_state == ProcessingState.QUARANTINE
    assert logs[0].new_state == ProcessingState.INDEX_READY
    assert logs[0].actor == "RouterStub"
    payload = logs[0].reason
    assert payload["log_schema_version"] == "decision_log_v1"
    assert payload["router_suggested"]["confidence"] == 0.90
    assert "Strong evidence" in payload["router_suggested"]["reasoning"]

def test_route_segments_ambiguous(session):
    # Create an ambiguous segment
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="This is some random text with no keywords.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()
    
    count = router.route_segments(session=session)
    assert count == 1
    
    session.refresh(seg)
    assert seg.processing_state == ProcessingState.REVIEW
    assert seg.classification_label == ClassificationLabel.AMBIGUOUS

    # Verify DecisionLog
    logs = session.exec(select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == seg.id)).all()
    assert len(logs) == 1
    assert logs[0].old_state == ProcessingState.QUARANTINE
    assert logs[0].new_state == ProcessingState.REVIEW
    assert logs[0].actor == "RouterStub"
    payload = logs[0].reason
    assert payload["log_schema_version"] == "decision_log_v1"
    assert payload["router_suggested"]["confidence"] == 0.50
    assert payload["policy"]["final_state"] == "REVIEW"

def test_router_stub_redundancy(session):
    """Assert that 'database' remains REVIEW (1 distinct weak signal type)."""
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="Our database needs update.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()
    
    router.route_segments(session=session)
    session.refresh(seg)
    
    assert seg.processing_state == ProcessingState.REVIEW
    assert seg.classification_label == ClassificationLabel.AMBIGUOUS
    
    # Check logs for reasoning
    log = session.exec(select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == seg.id)).one()
    assert "Insufficient evidence (0 distinct types)" in log.reason["router_suggested"]["reasoning"]

def test_router_stub_multi_signal(session):
    """Assert that 'architecture' + 'experimental' promotes to INDEX_READY (2 distinct signal types)."""
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="The architecture supports experimental test cases.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()
    
    router.route_segments(session=session)
    session.refresh(seg)
    
    assert seg.processing_state == ProcessingState.INDEX_READY
    assert seg.classification_label == ClassificationLabel.TECHNICAL
    
    log = session.exec(select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == seg.id)).one()
    assert "Compound evidence (2 distinct types)" in log.reason["router_suggested"]["reasoning"]

def test_router_stub_strong_signal(session):
    """Assert that 'Traceback' promotes to INDEX_READY immediately."""
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="We found a Traceback in the logs.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()
    
    router.route_segments(session=session)
    session.refresh(seg)
    
    assert seg.processing_state == ProcessingState.INDEX_READY
    assert seg.classification_label == ClassificationLabel.TECHNICAL
    
    log = session.exec(select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == seg.id)).one()
    assert "Strong evidence" in log.reason["router_suggested"]["reasoning"]

def test_router_stub_log_hygiene(session):
    """Assert length of proof_spans is capped at unique patterns matched."""
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content="error error error architecture architecture",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()
    
    router.route_segments(session=session)
    session.refresh(seg)
    
    log = session.exec(select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == seg.id)).one()
    # "error" is one pattern, "architecture" is another. Both match once each in proof_spans.
    assert len(log.reason["router_suggested"]["proof_spans"]) == 2


