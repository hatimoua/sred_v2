from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class RouterLabel(str, Enum):
    TECHNICAL = "TECHNICAL"
    FINANCIAL = "FINANCIAL"
    AMBIGUOUS = "AMBIGUOUS"
    NOISE = "NOISE"

class RecommendedState(str, Enum):
    INDEX_READY = "INDEX_READY"
    REVIEW = "REVIEW"
    NOISE = "NOISE"
    QUARANTINE = "QUARANTINE"

class HardSignal(str, Enum):
    CODE_BLOCK = "code_block"
    STACK_TRACE = "stack_trace"
    EXCEPTION_MESSAGE = "exception_message"
    ARCHITECTURE_COMPONENT = "architecture_component"
    PERFORMANCE_METRIC = "performance_metric"
    TEST_RESULT = "test_result"
    TECHNICAL_CONSTRAINT = "technical_constraint"
    UNKNOWN_ROOT_CAUSE = "unknown_root_cause"

class RiskFlag(str, Enum):
    MARKETING_LANGUAGE = "marketing_language"
    SALES_PRICING_LANGUAGE = "sales_pricing_language"
    LEGAL_TERMS = "legal_terms"
    HR_PERFORMANCE = "hr_performance"
    NO_TECHNICAL_MARKERS = "no_technical_markers"
    AMBIGUOUS_CONTEXT = "ambiguous_context"
    CONTRADICTORY_SIGNALS = "contradictory_signals"
    PROOF_INVALID = "proof_invalid"
    PROOF_AMBIGUOUS = "proof_ambiguous"

class ProofSpan(BaseModel):
    quote: str = Field(..., description="Verbatim substring from the source text")
    start: Optional[int] = Field(default=None, ge=0)
    end: Optional[int] = Field(default=None)
    excerpt: Optional[str] = Field(default=None)
    kind: Optional[str] = Field(default=None)

    @field_validator('end')
    @classmethod
    def validate_offsets(cls, v: Optional[int], info):
        start = info.data.get('start')
        # Check consistency: both must be present or both must be None
        if (v is None and start is not None) or (v is not None and start is None):
            raise ValueError('Both start and end must be present, or both must be None')
        # Bounds check if both are present
        if v is not None and start is not None and v <= start:
            raise ValueError('end must be greater than start')
        return v

class RouterResult(BaseModel):
    label: RouterLabel
    confidence: float = Field(ge=0.0, le=1.0)
    signals: List[HardSignal] = Field(default_factory=list)
    risk_flags: List[RiskFlag] = Field(default_factory=list)
    proof_spans: List[ProofSpan] = Field(default_factory=list)
    recommended_state: RecommendedState = RecommendedState.REVIEW
    reasoning: Optional[str] = Field(None, max_length=800)
    model_id: Optional[str] = None
    prompt_version: Optional[str] = None
    policy_version: str = "router_policy_v1"

    @field_validator('signals', 'risk_flags', mode='after')
    @classmethod
    def deduplicate_preserve_order(cls, v: List) -> List:
        seen = set()
        return [x for x in v if not (x in seen or seen.add(x))]
