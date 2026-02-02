import uuid
import pytest
from sqlmodel import Session, select
from datetime import datetime, UTC
from unittest.mock import AsyncMock, patch

from sredi.models.models import DocSegment, SegmentDecisionLog, ProcessingState, ClassificationLabel
from sredi.services.router_contract import RouterResult, RouterLabel, RecommendedState
from sredi.services.router import route_segments
from sredi.services.validation import apply_router_decision

@pytest.fixture
def sample_segment(session: Session):
    segment = DocSegment(
        content="This is a test architectural component with hexagonal architecture.",
        document_id=uuid.uuid4(),
        sequence_index=1,
        start_offset=0,
        end_offset=67,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(segment)
    session.commit()
    session.refresh(segment)
    return segment

def test_shadow_mode_no_mutation(session: Session, sample_segment: DocSegment):
    """Test that shadow mode does not mutate DocSegment state."""
    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.9,
        signals=[],
        proof_spans=[],
        recommended_state=RecommendedState.INDEX_READY,
        reasoning="Test reasoning",
        model_id="TestRouter"
    )
    
    # Run apply_router_decision in shadow mode
    apply_router_decision(session, sample_segment, result, shadow_mode=True)
    session.commit()
    session.refresh(sample_segment)
    
    # Assert state is still QUARANTINE
    assert sample_segment.processing_state == ProcessingState.QUARANTINE
    
    # Verify log entry exists with shadow_mode: true
    log = session.exec(
        select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == sample_segment.id)
    ).one()
    assert log.reason["shadow_mode"] is True
    assert log.new_state == ProcessingState.QUARANTINE

def test_tournament_disagreement(session: Session, sample_segment: DocSegment):
    """Test that tournament disagreement is correctly computed and logged."""
    # LLM result would be INDEX_READY but fails validation because no proof spans
    llm_result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.9,
        signals=[],
        proof_spans=[],
        recommended_state=RecommendedState.INDEX_READY,
        reasoning="LLM suggested",
        model_id="openai:gpt-4o"
    )
    
    # Stub result would be REVIEW
    stub_result = RouterResult(
        label=RouterLabel.AMBIGUOUS,
        confidence=0.5,
        signals=[],
        proof_spans=[],
        recommended_state=RecommendedState.REVIEW,
        reasoning="Stub suggested",
        model_id="RouterStub"
    )
    
    apply_router_decision(
        session, 
        sample_segment, 
        llm_result, 
        shadow_mode=True, 
        tournament_stub_result=stub_result
    )
    session.commit()
    
    log = session.exec(
        select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == sample_segment.id)
    ).one()
    
    tournament = log.reason.get("tournament")
    assert tournament is not None
    assert tournament["stub_final_state"] == "REVIEW"
    assert tournament["llm_final_state"] == "REVIEW"
    assert tournament["disagreement"] is False

def test_tournament_disagreement_actual(session: Session, sample_segment: DocSegment):
    """Test that tournament disagreement is True when states actually differ."""
    from sredi.services.router_contract import HardSignal, ProofSpan
    
    # Mocked LLM result that passes validation for INDEX_READY
    llm_result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.9,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        proof_spans=[ProofSpan(start=0, end=10, excerpt=sample_segment.content[0:10], kind="test")],
        recommended_state=RecommendedState.INDEX_READY,
        reasoning="LLM suggested with proof",
        model_id="openai:gpt-4o"
    )
    
    # Stub result that results in REVIEW
    stub_result = RouterResult(
        label=RouterLabel.AMBIGUOUS,
        confidence=0.5,
        signals=[],
        proof_spans=[],
        recommended_state=RecommendedState.REVIEW,
        reasoning="Stub suggested no proof",
        model_id="RouterStub"
    )
    
    apply_router_decision(
        session, 
        sample_segment, 
        llm_result, 
        shadow_mode=True, 
        tournament_stub_result=stub_result
    )
    session.commit()
    
    log = session.exec(
        select(SegmentDecisionLog).where(SegmentDecisionLog.segment_id == sample_segment.id)
    ).one()
    
    tournament = log.reason.get("tournament")
    assert tournament["stub_final_state"] == "REVIEW"
    assert tournament["llm_final_state"] == "INDEX_READY"
    assert tournament["disagreement"] is True

@pytest.mark.asyncio
async def test_async_batch_routing_shadow(session: Session):
    """Test that the async routing pipeline handles shadow mode correctly."""
    from sredi.services.router import route_segments_async
    # Create 3 segments
    for i in range(3):
        seg = DocSegment(
            content=f"Segment {i} with architecture keyword",
            document_id=uuid.uuid4(),
            sequence_index=i,
            start_offset=0,
            end_offset=10,
            processing_state=ProcessingState.QUARANTINE
        )
        session.add(seg)
    session.commit()
    
    mock_router_result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.9,
        signals=[],
        proof_spans=[],
        recommended_state=RecommendedState.INDEX_READY,
        reasoning="Mocked LLM",
        model_id="openai:gpt-4o"
    )
    
    with patch("sredi.services.router.llm_route_segment", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_router_result
        
        # Run route_segments_async in shadow mode directly to avoid nested asyncio.run
        count = await route_segments_async(limit=10, router_type="llm", shadow_mode=True, session=session)
        
        assert count == 3
        # Check that segments are still QUARANTINE
        segments = session.exec(select(DocSegment)).all()
        for seg in segments:
            assert seg.processing_state == ProcessingState.QUARANTINE
            
        # Check logs
        logs = session.exec(select(SegmentDecisionLog)).all()
        assert len(logs) == 3
        for log in logs:
            assert log.reason["shadow_mode"] is True
            assert log.reason["tournament"] is not None
