import pytest
import uuid
import hashlib
import json
from sqlmodel import Session
from sredi.models.models import DocSegment, SegmentDecisionLog, ProcessingState, ClassificationLabel
from sredi.services.router_contract import RouterResult, RouterLabel, RecommendedState, ProofSpan, HardSignal, RiskFlag
from sredi.services.validation import apply_router_decision, generate_decision_fingerprint

def test_validation_payload_structure(session: Session):
    text = "Typed payload test."
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Typed payload test.")],
        reasoning="Test reasoning",
        model_id="gpt-4",
        prompt_version="v2",
        policy_version="router_policy_v1"
    )

    _, log, written = apply_router_decision(session, seg, result)
    assert written is True
    
    payload = log.reason
    assert payload["log_schema_version"] == "decision_log_v1"
    assert "fingerprinting" in payload
    assert payload["fingerprinting"]["model_id"] == "gpt-4"
    assert payload["router_suggested"]["label"] == "TECHNICAL"
    assert payload["validation"]["proof_valid"] is True
    assert payload["policy"]["final_state"] == "INDEX_READY"

def test_validation_suggested_vs_applied(session: Session):
    text = "Marketing speak for architecture."
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        risk_flags=[RiskFlag.MARKETING_LANGUAGE],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="architecture", kind="architecture")],
        reasoning="Test reasoning"
    )

    _, log, written = apply_router_decision(session, seg, result)
    assert written is True
    
    payload = log.reason
    # LLM suggested INDEX_READY
    assert payload["router_suggested"]["recommended_state"] == "INDEX_READY"
    # Policy forced REVIEW
    assert payload["policy"]["final_state"] == "REVIEW"
    assert "marketing_language" in payload["policy"]["blocking_flags_present"]

def test_validation_idempotency_extended(session: Session):
    text = "Idempotent content."
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Idempotent", kind="architecture")],
        reasoning="Test reasoning"
    )

    # First evaluation
    _, log1, written1 = apply_router_decision(session, seg, result)
    assert written1 is True
    assert log1 is not None
    
    # Second evaluation (identical)
    _, log2, written2 = apply_router_decision(session, seg, result)
    assert written2 is False
    assert log2.id == log1.id

def test_quote_first_populates_offsets_and_validates(session: Session):
    """Verify that a quote-only ProofSpan gets anchored and validated."""
    text = "Detailed technical implementation."
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="technical implementation")],
        reasoning="Test reasoning"
    )

    _, log, _ = apply_router_decision(session, seg, result)
    payload = log.reason
    assert payload["validation"]["proof_valid"] is True
    # Verify offsets were populated in the log's saved version of the span
    spans = payload["router_suggested"]["proof_spans"]
    assert spans[0]["start"] == 9
    assert spans[0]["end"] == 33

def test_quote_not_found_marks_proof_invalid(session: Session):
    """Verify that a hallucinated quote triggers PROOF_INVALID."""
    text = "Actual source text."
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="non-existent text")],
        reasoning="Test reasoning"
    )

    _, log, _ = apply_router_decision(session, seg, result)
    payload = log.reason
    assert payload["validation"]["proof_valid"] is False
    assert "proof_invalid" in payload["validation"]["added_risk_flags"]
    assert payload["policy"]["final_state"] == "REVIEW"

def test_quote_multiple_matches_marks_ambiguous(session: Session):
    """Verify that repeated quotes trigger PROOF_AMBIGUOUS."""
    text = "Repeated line. Repeated line."
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Repeated line.")],
        reasoning="Test reasoning"
    )

    _, log, _ = apply_router_decision(session, seg, result)
    payload = log.reason
    assert payload["validation"]["proof_valid"] is True # Valid, but ambiguous
    assert "proof_ambiguous" in payload["validation"]["added_risk_flags"]
    # Should still promote if valid, just with a warning flag
    assert payload["policy"]["final_state"] == "INDEX_READY"

def test_markdown_comment_and_whitespace_quote_matches_exactly(session: Session):
    """Verify that complex Markdown and whitespace are matched verbatim."""
    text = "<!-- KEP Header -->\n- [ ] Task list  "
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=text,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.commit()

    # First evaluation (Success path)
    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="- [ ] Task list  ")],
        reasoning="Test reasoning"
    )

    _, log, _ = apply_router_decision(session, seg, result)
    payload = log.reason
    assert payload["validation"]["proof_valid"] is True
    
    # Second evaluation (different result object, but same segment)
    # We must use a different model_id or prompt_version to bypass idempotency check in tests
    result_fail = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="THIS QUOTE DOES NOT EXIST")],
        reasoning="Test reasoning",
        model_id="gpt-4-fail-test"
    )
    _, log_fail, _ = apply_router_decision(session, seg, result_fail)
    
    # Debug: Print the log reason
    print(f"DEBUG log_fail.reason: {json.dumps(log_fail.reason, indent=2)}")
    
    assert log_fail.reason["validation"]["proof_valid"] is False
