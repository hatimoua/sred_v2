import pytest
import asyncio
from sredi.services.router_validation import validate_proof_spans
from sredi.services.router_contract import ProofSpan, RiskFlag

def test_validate_quote_first_markdown():
    """Verify that complex Markdown quotes are correctly matched."""
    text = "Some text with <!-- a comment --> and `code`."
    # LLM provides verbatim quotes without offsets
    spans = [
        ProofSpan(quote="<!-- a comment -->"),
        ProofSpan(quote="`code`")
    ]
    is_valid, errors, flags = validate_proof_spans(text, spans)
    
    if not is_valid:
        print(f"Validation failed with errors: {errors}")
    
    assert is_valid
    assert len(errors) == 0
    # Python should have anchored them
    assert spans[0].start == 15
    assert spans[0].end == 33
    assert spans[1].start == 38
    assert spans[1].end == 44

def test_validate_quote_ambiguity():
    """Verify that repeated quotes trigger PROOF_AMBIGUOUS."""
    text = "The system is slow. The system is slow."
    spans = [ProofSpan(quote="The system is slow.")]
    is_valid, errors, flags = validate_proof_spans(text, spans)
    
    assert is_valid
    assert RiskFlag.PROOF_AMBIGUOUS in flags
    # Anchors to first occurrence
    assert spans[0].start == 0
    assert spans[0].end == 19

def test_validate_quote_whitespace_edge():
    """Verify that exact whitespace/wraps are respected or rejected if mismatched."""
    text = "Line 1  \nLine 2"
    # Exact match including trailing spaces
    spans_valid = [ProofSpan(quote="Line 1  \nLine 2")]
    is_valid, errors, flags = validate_proof_spans(text, spans_valid)
    assert is_valid
    
    # Paraphrased / stripped mismatch
    spans_invalid = [ProofSpan(quote="Line 1\nLine 2")]
    is_valid, errors, flags = validate_proof_spans(text, spans_invalid)
    assert not is_valid
    assert "verbatim mismatch" in errors[0]

def test_validate_quote_hallucination():
    """Verify that hallucinated quotes trigger PROOF_INVALID."""
    text = "The database is down."
    spans = [ProofSpan(quote="The server is up.")]
    is_valid, errors, flags = validate_proof_spans(text, spans)
    
    assert not is_valid
    assert "verbatim mismatch" in errors[0]
