import hashlib
import json
from typing import Tuple, List, Set, Optional
from sqlmodel import Session, select, desc
from datetime import datetime, UTC

from ..models.models import DocSegment, SegmentDecisionLog, ProcessingState, ClassificationLabel
from .router_contract import RouterResult, RouterLabel, RecommendedState, RiskFlag, HardSignal
from .router_validation import validate_proof_spans
from .decision_log_payload import (
    DecisionLogPayload,
    Fingerprinting,
    RouterSuggested,
    ValidationResult,
    PolicyOutcome,
    PromotionRequirements,
    TournamentResults
)

BLOCKING_FLAGS = {
    RiskFlag.PROOF_INVALID,
    RiskFlag.CONTRADICTORY_SIGNALS,
    RiskFlag.NO_TECHNICAL_MARKERS,
    RiskFlag.MARKETING_LANGUAGE,
    RiskFlag.AMBIGUOUS_CONTEXT
}

def generate_decision_fingerprint(result: RouterResult, segment_text: str) -> str:
    """Generates a SHA256 hash representing the decision context.
    
    fingerprint = sha256(model_id|prompt_version|policy_version|segment_text_hash)
    """
    text_hash = hashlib.sha256(segment_text.encode('utf-8')).hexdigest()
    
    # Use defaults for non-LLM routers
    model_id = result.model_id or "stub"
    prompt_version = result.prompt_version or "stub"
    policy_version = result.policy_version or "router_policy_v1"
    
    components = f"{model_id}|{prompt_version}|{policy_version}|{text_hash}"
    return hashlib.sha256(components.encode('utf-8')).hexdigest()

def validate_router_result(segment_content: str, result: RouterResult) -> Tuple[bool, bool, List[str], List[RiskFlag]]:
    """Strict zero-trust validation of router proof spans."""
    # 1. Quote-First anchoring
    proof_valid, validation_errors, additional_flags = validate_proof_spans(segment_content, result.proof_spans)
    
    # 2. Taint Logic (PROOF_INVALID)
    is_tainted = not proof_valid
    added_risk_flags = list(additional_flags)
    
    if is_tainted:
        added_risk_flags.append(RiskFlag.PROOF_INVALID)
    
    # 3. TECHNICAL promotion requirements
    if result.label == RouterLabel.TECHNICAL and not result.proof_spans:
        is_tainted = True
        validation_errors.append("TECHNICAL label provided without proof spans")
        if RiskFlag.PROOF_INVALID not in added_risk_flags:
            added_risk_flags.append(RiskFlag.PROOF_INVALID)

    return proof_valid, is_tainted, validation_errors, added_risk_flags

def determine_final_state(result: RouterResult, is_tainted: bool, current_risks: Set[RiskFlag]) -> RecommendedState:
    """Deterministic policy engine to decide segment's final state."""
    
    # Check for any blocking flags
    has_blocking_flag = any(flag in current_risks for flag in BLOCKING_FLAGS)
    
    # 1. INDEX_READY Rule
    # label == TECHNICAL AND signals has at least one HardSignal.
    # proof_spans must be non-empty AND valid.
    # ZERO Blocking Flags present.
    if (not is_tainted and 
        result.label == RouterLabel.TECHNICAL and 
        len(result.signals) > 0 and 
        result.proof_spans and
        not has_blocking_flag):
        return RecommendedState.INDEX_READY

    # 2. NOISE Rule
    # label == NOISE and no HardSignal is present.
    if result.label == RouterLabel.NOISE and not result.signals:
        return RecommendedState.NOISE

    # 3. REVIEW Rule (Fallback)
    # If label == FINANCIAL
    # If "Technical" but contains any Blocking Flag (e.g., marketing_language)
    # If label == AMBIGUOUS
    # If tainted (handled via PROOF_INVALID in current_risks)
    return RecommendedState.REVIEW

def evaluate_router_decision(segment_content: str, result: RouterResult) -> Tuple[RecommendedState, DecisionLogPayload]:
    """Strictly evaluates a router decision without database side effects.
    
    Returns the computed final state and the full typed payload for auditing.
    """
    # 1. Fingerprinting
    fingerprint = generate_decision_fingerprint(result, segment_content)
    segment_text_hash = hashlib.sha256(segment_content.encode('utf-8')).hexdigest()

    # 2. Validation
    proof_valid, is_tainted, validation_errors, added_risk_flags = validate_router_result(segment_content, result)
    
    # 3. Policy Evaluation
    current_risks = set(result.risk_flags) | set(added_risk_flags)
    final_recommended_state = determine_final_state(result, is_tainted, current_risks)

    # 4. Build Typed Payload
    payload = DecisionLogPayload(
        fingerprinting=Fingerprinting(
            router_fingerprint=fingerprint,
            segment_text_hash=segment_text_hash,
            model_id=result.model_id or "stub",
            prompt_version=result.prompt_version or "stub",
            policy_version=result.policy_version or "router_policy_v1"
        ),
        router_suggested=RouterSuggested(
            label=result.label,
            recommended_state=result.recommended_state or RecommendedState.REVIEW,
            confidence=result.confidence,
            signals=result.signals,
            risk_flags=result.risk_flags,
            proof_spans=result.proof_spans,
            reasoning=result.reasoning
        ),
        validation=ValidationResult(
            proof_valid=proof_valid,
            tainted=is_tainted,
            validation_errors=validation_errors,
            added_risk_flags=added_risk_flags
        ),
        policy=PolicyOutcome(
            final_state=final_recommended_state,
            blocking_flags_present=[f for f in current_risks if f in BLOCKING_FLAGS],
            promotion_requirements=PromotionRequirements()
        )
    )
    return final_recommended_state, payload

def apply_router_decision(
    session: Session, 
    segment: DocSegment, 
    result: RouterResult,
    shadow_mode: bool = False,
    tournament_stub_result: Optional[RouterResult] = None
) -> Tuple[DocSegment, SegmentDecisionLog, bool]:
    """Validates and persists a router decision with structural provenance and idempotent auditing.
    
    If shadow_mode is True, segment state is NOT updated.
    If tournament_stub_result is provided, a tournament comparison is performed and logged.
    """
    
    # 1. Evaluate primary decision (usually LLM if in tournament)
    final_recommended_state, payload = evaluate_router_decision(segment.content, result)
    payload.shadow_mode = shadow_mode

    # 2. Tournament Logic (if applicable)
    if shadow_mode and tournament_stub_result:
        stub_final_state, _ = evaluate_router_decision(segment.content, tournament_stub_result)
        payload.tournament = TournamentResults(
            stub_final_state=stub_final_state,
            llm_final_state=final_recommended_state,
            disagreement=(stub_final_state != final_recommended_state)
        )

    # 3. Idempotent Logging Rule
    latest_log_stmt = select(SegmentDecisionLog).where(
        SegmentDecisionLog.segment_id == segment.id
    ).order_by(desc(SegmentDecisionLog.timestamp)).limit(1)
    latest_log = session.exec(latest_log_stmt).first()
    
    if latest_log and latest_log.reason.get("fingerprinting", {}).get("router_fingerprint") == payload.fingerprinting.router_fingerprint:
        # Check if shadow_mode matches to avoid skipping a real transition for a shadow log
        if latest_log.reason.get("shadow_mode") == shadow_mode:
            return segment, latest_log, False

    # 4. State Mapping
    state_map = {
        RecommendedState.INDEX_READY: ProcessingState.INDEX_READY,
        RecommendedState.REVIEW: ProcessingState.REVIEW,
        RecommendedState.NOISE: ProcessingState.NOISE,
        RecommendedState.QUARANTINE: ProcessingState.QUARANTINE
    }
    new_state = state_map[final_recommended_state]
    
    label_map = {
        RouterLabel.TECHNICAL: ClassificationLabel.TECHNICAL,
        RouterLabel.FINANCIAL: ClassificationLabel.FINANCIAL,
        RouterLabel.AMBIGUOUS: ClassificationLabel.AMBIGUOUS,
        RouterLabel.NOISE: ClassificationLabel.NOISE
    }
    new_label = label_map[result.label]

    # 5. Persistence
    old_state = segment.processing_state
    
    if not shadow_mode:
        segment.update_state(new_state)
        segment.classification_label = new_label
        session.add(segment)
    
    decision_log = SegmentDecisionLog(
        segment_id=segment.id,
        old_state=old_state,
        new_state=new_state if not shadow_mode else old_state, # In shadow mode, we log what WOULD happen
        actor=result.model_id or "stub",
        reason=payload.model_dump(),
        timestamp=datetime.now(UTC)
    )
    
    session.add(decision_log)
    
    return segment, decision_log, True
