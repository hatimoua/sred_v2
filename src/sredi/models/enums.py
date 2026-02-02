from typing import Optional
from enum import Enum
from sqlmodel import AutoString

class ProcessingState(str, Enum):
    """Enumeration of possible states for a document segment in the pipeline.

    Attributes:
        QUARANTINE: Initial state for newly created segments.
        INDEX_READY: Segment has been validated as technical evidence and is ready for indexing.
        NOISE: Segment has been identified as non-technical noise.
        REVIEW: Segment requires manual human review.
    """
    QUARANTINE = "QUARANTINE"
    INDEX_READY = "INDEX_READY"
    NOISE = "NOISE"
    REVIEW = "REVIEW"

    @classmethod
    def validate_transition(cls, from_state: Optional["ProcessingState"], to_state: "ProcessingState"):
        """Enforces the state machine policy for segment transitions.

        Args:
            from_state: The current state of the segment. None if it's a new segment.
            to_state: The proposed new state for the segment.

        Raises:
            ValueError: If the transition from from_state to to_state is not allowed
                by the state machine policy.
        """
        # Ingestion (initial state)
        if from_state is None:
            return

        # Ensure we are working with Enum members if strings are passed
        if isinstance(from_state, str):
            from_state = cls(from_state)
        if isinstance(to_state, str):
            to_state = cls(to_state)

        if from_state == to_state:
            return

        allowed = {
            cls.QUARANTINE: [cls.INDEX_READY, cls.NOISE, cls.REVIEW],
            cls.REVIEW: [cls.INDEX_READY, cls.NOISE],
            # Dev/Human overrides
            cls.NOISE: [cls.REVIEW],
            cls.INDEX_READY: [cls.REVIEW],
        }

        if from_state not in allowed or to_state not in allowed[from_state]:
            raise ValueError(f"Invalid transition: {from_state} -> {to_state}")

class ClassificationLabel(str, Enum):
    """Labels assigned to segments by the router.

    Attributes:
        TECHNICAL: Content contains technical evidence for SR&ED.
        FINANCIAL: Content contains financial information.
        AMBIGUOUS: Content classification is uncertain.
        NOISE: Content identified as non-technical noise.
    """
    TECHNICAL = "TECHNICAL"
    FINANCIAL = "FINANCIAL"
    AMBIGUOUS = "AMBIGUOUS"
    NOISE = "NOISE"

class LinkType(str, Enum):
    """Types of links between projects and segments.

    Attributes:
        STRONG_ANCHOR: Direct match of project anchor in segment content.
        WEAK_INFERENCE: Inferred link based on context or ML.
    """
    STRONG_ANCHOR = "STRONG_ANCHOR"
    WEAK_INFERENCE = "WEAK_INFERENCE"
