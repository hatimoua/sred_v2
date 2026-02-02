from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from .router_contract import RouterLabel, RecommendedState, HardSignal, RiskFlag, ProofSpan

class Fingerprinting(BaseModel):
    router_fingerprint: str
    segment_text_hash: str
    model_id: str
    prompt_version: str
    policy_version: str

class RouterSuggested(BaseModel):
    label: RouterLabel
    recommended_state: RecommendedState
    confidence: float
    signals: List[HardSignal]
    risk_flags: List[RiskFlag]
    proof_spans: List[ProofSpan]
    reasoning: Optional[str] = None

class ValidationResult(BaseModel):
    proof_valid: bool
    tainted: bool
    validation_errors: List[str] = Field(default_factory=list)
    added_risk_flags: List[RiskFlag] = Field(default_factory=list)

class PromotionRequirements(BaseModel):
    requires_proof: bool = True
    requires_hard_signal: bool = True
    min_confidence: float = 0.9

class PolicyOutcome(BaseModel):
    final_state: RecommendedState
    blocking_flags_present: List[RiskFlag]
    promotion_requirements: PromotionRequirements = Field(default_factory=PromotionRequirements)

class TournamentResults(BaseModel):
    stub_final_state: RecommendedState
    llm_final_state: RecommendedState
    disagreement: bool

class DecisionLogPayload(BaseModel):
    log_schema_version: Literal["decision_log_v1"] = "decision_log_v1"
    shadow_mode: bool = False
    fingerprinting: Fingerprinting
    router_suggested: RouterSuggested
    validation: ValidationResult
    policy: PolicyOutcome
    tournament: Optional[TournamentResults] = None
