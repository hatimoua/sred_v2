import re
import logging
from typing import List, Tuple
from .router_contract import ProofSpan, RouterResult, RecommendedState, RouterLabel, RiskFlag

logger = logging.getLogger(__name__)

def validate_proof_spans(segment_text: str, spans: List[ProofSpan]) -> Tuple[bool, List[str], List[RiskFlag]]:
    """Validates and anchors proof spans using the Quote-First approach.
    
    Instead of relying on LLM offsets, we search for the verbatim quote in the text.
    
    Returns:
        (is_valid, error_messages, additional_risk_flags)
    """
    errors = []
    additional_flags = []
    if not spans:
        return True, [], []

    for i, span in enumerate(spans):
        quote = span.quote
        if not quote:
            errors.append(f"Span {i} is missing a required quote string")
            continue

        # Find all occurrences of the quote
        # Using re.escape to handle Markdown special characters in the quote
        matches = list(re.finditer(re.escape(quote), segment_text))
        num_matches = len(matches)

        if num_matches == 0:
            errors.append(f"Span {i} verbatim mismatch. Quote not found in segment: '{quote[:50]}...' (len: {len(quote)})")
            continue
        
        if num_matches > 1:
            # Mark as ambiguous but keep the first occurrence as the anchor
            additional_flags.append(RiskFlag.PROOF_AMBIGUOUS)
            match_positions = [f"[{m.start()}:{m.end()}]" for m in matches]
            logger.warning(f"Span {i} is ambiguous. Found {num_matches} occurrences at {', '.join(match_positions)} for quote: '{quote[:50]}...'")
        
        # Anchor to the first occurrence
        best_match = matches[0]
        span.start = best_match.start()
        span.end = best_match.end()
        span.excerpt = segment_text[span.start:span.end] # Sync for legacy compatibility

        # Minimum length check (logic moved here to apply after anchoring)
        span_len = span.end - span.start
        # Broaden markers to include backticks and other common technical syntax
        # Case-insensitive check for markers like ERROR/error
        hard_markers = ["```", "traceback", "exception", "error", "stack trace", "`", "<!--", "-->"]
        if span_len < 10:
            has_marker = any(marker.lower() in span.quote.lower() for marker in hard_markers)
            if not has_marker:
                errors.append(f"Span {i} too short ({span_len} chars) and contains no hard markers. Quote: '{span.quote}'")

    is_valid = len(errors) == 0
    return is_valid, errors, additional_flags

def require_proof_for_promotion(result: RouterResult) -> bool:
    """Determines if the result requires proof spans for the recommended promotion."""
    # Stricter check: INDEX_READY always requires proof. 
    # TECHNICAL label also requires proof to be considered valid evidence.
    if result.recommended_state == RecommendedState.INDEX_READY or result.label == RouterLabel.TECHNICAL:
        return True
    return False

def decide_final_state(result: RouterResult, segment_text: str) -> Tuple[RecommendedState, List[RiskFlag]]:
    """Deterministic policy to decide the final state of a segment.
    
    Policy (conservative):
    1. NOISE label -> NOISE state
    2. FINANCIAL label -> REVIEW state (conservative)
    3. TECHNICAL/AMBIGUOUS -> Validate proof, check signals and confidence
    """
    risk_flags = list(result.risk_flags)
    
    # 1. NOISE early exit
    if result.label == RouterLabel.NOISE:
        return RecommendedState.NOISE, risk_flags

    # 2. FINANCIAL early exit (conservative)
    if result.label == RouterLabel.FINANCIAL:
        return RecommendedState.REVIEW, risk_flags

    # 3. Proof Validation
    is_valid, validation_errors, additional_flags = validate_proof_spans(segment_text, result.proof_spans)
    risk_flags.extend(additional_flags)
    
    if not is_valid:
        risk_flags.append(RiskFlag.PROOF_INVALID)
        return RecommendedState.REVIEW, risk_flags

    # 4. Promotion Requirements
    if require_proof_for_promotion(result) and not result.proof_spans:
        risk_flags.append(RiskFlag.PROOF_INVALID) # Missing proof for required promotion
        return RecommendedState.REVIEW, risk_flags

    # 5. TECHNICAL/AMBIGUOUS Promotion Logic
    if result.label in {RouterLabel.TECHNICAL, RouterLabel.AMBIGUOUS}:
        # Blocking risks
        if RiskFlag.CONTRADICTORY_SIGNALS in risk_flags or RiskFlag.NO_TECHNICAL_MARKERS in risk_flags:
            return RecommendedState.REVIEW, risk_flags
            
        # Promotion criteria
        has_hard_signals = len(result.signals) > 0
        is_high_confidence = result.confidence >= 0.90
        
        if is_high_confidence and has_hard_signals and result.recommended_state == RecommendedState.INDEX_READY:
            return RecommendedState.INDEX_READY, risk_flags

    # Default fallback
    return RecommendedState.REVIEW, risk_flags
