import pytest
from sredi.services.router_contract import RouterResult, RouterLabel, RecommendedState, ProofSpan, HardSignal, RiskFlag
from sredi.services.router_validation import validate_proof_spans, decide_final_state

def test_validate_proof_spans_success():
    text = "The system architecture uses a database with 10ms latency."
    # Correct offsets:
    # "system architecture" starts at 4, length 19 -> end at 23
    # "database" starts at 31, length 8 -> end at 39
    # Note: "database" is only 8 chars, so it needs a marker OR we should use a longer excerpt to pass min 10 chars.
    # Let's use "uses a database" (index 24 to 39, length 15)
    spans = [
        ProofSpan(quote="system architecture", start=4, end=23, excerpt="system architecture", kind="architecture"),
        ProofSpan(quote="uses a database", start=24, end=39, excerpt="uses a database", kind="constraint")
    ]
    valid, errors, additional_flags = validate_proof_spans(text, spans)
    assert valid is True, f"Validation failed with errors: {errors}"
    assert not errors

def test_validate_proof_spans_offset_mismatch():
    text = "Error in the system."
    spans = [ProofSpan(quote="Fixed", start=0, end=5, excerpt="Fixed", kind="error")]
    valid, errors, additional_flags = validate_proof_spans(text, spans)
    assert valid is False
    assert "verbatim mismatch" in errors[0]

def test_validate_proof_spans_out_of_bounds():
    text = "Short."
    spans = [ProofSpan(quote="Too long", start=0, end=10, excerpt="Too long", kind="error")]
    valid, errors, additional_flags = validate_proof_spans(text, spans)
    assert valid is False
    assert "verbatim mismatch" in errors[0]

def test_validate_proof_spans_too_short_no_marker():
    text = "A tiny bit of text."
    # "tiny" is at index 2, end at 6.
    spans = [ProofSpan(quote="tiny", start=2, end=6, excerpt="tiny", kind="other")]
    valid, errors, additional_flags = validate_proof_spans(text, spans)
    assert valid is False
    assert any("too short" in err for err in errors)

def test_validate_proof_spans_short_with_marker():
    text = "Check ERROR log."
    spans = [ProofSpan(quote="ERROR", start=6, end=11, excerpt="ERROR", kind="error")]
    valid, errors, additional_flags = validate_proof_spans(text, spans)
    assert valid is True

def test_decide_final_state_noise():
    result = RouterResult(label=RouterLabel.NOISE, confidence=1.0, reasoning="Noise")
    state, risks = decide_final_state(result, "some text")
    assert state == RecommendedState.NOISE

def test_decide_final_state_financial():
    result = RouterResult(label=RouterLabel.FINANCIAL, confidence=0.9, reasoning="Finance")
    state, risks = decide_final_state(result, "money talk")
    assert state == RecommendedState.REVIEW

def test_decide_final_state_technical_promotion_success():
    text = "Traceback: NullPointerException at line 10"
    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.STACK_TRACE],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Traceback", start=0, end=9, excerpt="Traceback", kind="error")],
        reasoning="Technical"
    )
    state, risks = decide_final_state(result, text)
    assert state == RecommendedState.INDEX_READY

def test_decide_final_state_technical_invalid_proof():
    text = "Architecture review."
    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Wrong", start=0, end=5, excerpt="Wrong", kind="architecture")],
        reasoning="Technical"
    )
    state, risks = decide_final_state(result, text)
    assert state == RecommendedState.REVIEW
    assert RiskFlag.PROOF_INVALID in risks

def test_decide_final_state_technical_no_signals():
    text = "Generic technical discussion with no markers."
    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.95,
        signals=[],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Generic te", start=0, end=10, excerpt="Generic te", kind="other")],
        reasoning="Technical"
    )
    state, risks = decide_final_state(result, text)
    assert state == RecommendedState.REVIEW

def test_decide_final_state_technical_low_confidence():
    text = "Architecture match."
    result = RouterResult(
        label=RouterLabel.TECHNICAL,
        confidence=0.85,
        signals=[HardSignal.ARCHITECTURE_COMPONENT],
        recommended_state=RecommendedState.INDEX_READY,
        proof_spans=[ProofSpan(quote="Architecture", start=0, end=12, excerpt="Architecture", kind="architecture")],
        reasoning="Technical"
    )
    state, risks = decide_final_state(result, text)
    assert state == RecommendedState.REVIEW
