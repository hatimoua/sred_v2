"""Router module using LangGraph for segment classification workflow.

Graph Structure:
    START → recall → enrich → [router_type?]
                                  ├─ "stub" → classify_stub ───────────────┬→ persist → END
                                  └─ "llm"  → classify_llm → [shadow?] ────┤
                                                                └─ stub ───┘
"""
import re
import logging
from typing import List, Optional, TypedDict, Literal
from sqlmodel import Session, select
import asyncio

from langgraph.graph import StateGraph, START, END

from ..models import DocSegment, ProcessingState, EntityAnchor
from ..db import get_session

logger = logging.getLogger(__name__)
from .router_contract import (
    RouterResult,
    RouterLabel,
    RecommendedState,
    HardSignal,
    ProofSpan,
)
from .validation import apply_router_decision
from .llm_client import LLMClient
from .router_llm import llm_route_segment
from .enrichment import EnrichmentService
from .vector_store import VectorStoreService

# Stub keyword patterns for technical evidence and their signals
TECH_PATTERNS = [
    {"pattern": r"architecture", "signal": HardSignal.ARCHITECTURE_COMPONENT, "strength": "weak"},
    {"pattern": r"infrastructure", "signal": HardSignal.ARCHITECTURE_COMPONENT, "strength": "weak"},
    {"pattern": r"implementation", "signal": HardSignal.TECHNICAL_CONSTRAINT, "strength": "weak"},
    {"pattern": r"experimental", "signal": HardSignal.TECHNICAL_CONSTRAINT, "strength": "weak"},
    {"pattern": r"error", "signal": HardSignal.EXCEPTION_MESSAGE, "strength": "strong"},
    {"pattern": r"exception", "signal": HardSignal.EXCEPTION_MESSAGE, "strength": "strong"},
    {"pattern": r"traceback", "signal": HardSignal.STACK_TRACE, "strength": "strong"},
    {"pattern": r"latency", "signal": HardSignal.PERFORMANCE_METRIC, "strength": "weak"},
    {"pattern": r"\b\d+(\.\d+)?\s?ms\b", "signal": HardSignal.PERFORMANCE_METRIC, "strength": "weak"},
    {"pattern": r"database", "signal": HardSignal.TECHNICAL_CONSTRAINT, "strength": "weak"},
    {"pattern": r"schema", "signal": HardSignal.TECHNICAL_CONSTRAINT, "strength": "weak"},
    {"pattern": r"\b(pytest|unit test|integration test|regression|benchmark|load test)\b", "signal": HardSignal.TEST_RESULT, "strength": "weak"},
]

class RouterState(TypedDict):
    """State managed by the LangGraph router workflow.

    Attributes:
        segment: The DocSegment being classified.
        result: Primary router classification result.
        tournament_stub_result: Stub result for shadow mode comparison.
        shadow_mode: Whether to run tournament (stub comparison).
        router_type: Which router to use ("stub" or "llm").
        db_session: Active database session.
        semantic_context: Recalled similar segments from vector store.
        related_context: Enriched anchor context strings.
    """
    segment: DocSegment
    result: Optional[RouterResult]
    tournament_stub_result: Optional[RouterResult]
    shadow_mode: bool
    router_type: Literal["stub", "llm"]
    db_session: Session
    semantic_context: Optional[str]
    related_context: Optional[str]

class RouterStub:
    """A deterministic stub implementation of the routing logic emitting structured RouterResults."""
    
    def classify(self, text: str) -> RouterResult:
        """Classifies a text segment into a structured RouterResult.

        Args:
            text: The text content to classify (must be the exact persisted content).

        Returns:
            RouterResult: Deterministic structured result with proof spans and signals.
        """
        signals = []
        proof_spans = []
        matched_patterns = []
        
        # Determine signals and proof spans based on patterns
        for item in TECH_PATTERNS:
            pattern = item["pattern"]
            signal = item["signal"]
            strength = item["strength"]
            
            # Use CASE-INSENSITIVE regex
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Check if this matched pattern is a "short" one that needs a hard marker
                # to pass zero-trust validation (if < 10 chars)
                start, end = match.span()
                excerpt = text[start:end]
                
                hard_markers = ["```", "traceback", "exception", "error", "stack trace", "`", "<!--", "-->"]
                is_short = (end - start) < 10
                has_marker = any(marker.lower() in excerpt.lower() for marker in hard_markers)
                
                # If it's short AND has no marker, it will fail validation.
                # We only count it as a signal if it will pass validation OR if it's strong.
                if is_short and not has_marker:
                    # Skip this weak short signal to avoid validation failure
                    continue

                matched_patterns.append({"signal": signal, "strength": strength})
                
                proof_spans.append(ProofSpan(
                    quote=excerpt,
                    start=start,
                    end=end,
                    excerpt=excerpt,
                    kind=signal.value
                ))
                signals.append(signal)
        
        # Promotion Gate Logic
        any_strong_match = any(p["strength"] == "strong" for p in matched_patterns)
        distinct_signals = {p["signal"] for p in matched_patterns}
        
        is_promoted = any_strong_match or len(distinct_signals) >= 2
        
        if is_promoted:
            label = RouterLabel.TECHNICAL
            confidence = 0.90
            recommended_state = RecommendedState.INDEX_READY
            reasoning = f"Promoted: {'Strong evidence' if any_strong_match else 'Compound evidence'} ({len(distinct_signals)} distinct types)."
        else:
            label = RouterLabel.AMBIGUOUS
            confidence = 0.50
            recommended_state = RecommendedState.REVIEW
            reasoning = f"Fallback: Insufficient evidence ({len(distinct_signals)} distinct types)."

        return RouterResult(
            label=label,
            confidence=confidence,
            signals=signals,
            proof_spans=proof_spans,
            recommended_state=recommended_state,
            reasoning=reasoning,
            model_id="RouterStub",
            prompt_version="stub_v2",
            policy_version="router_policy_v1"
        )

# =============================================================================
# GRAPH NODES
# =============================================================================

def node_recall_context(state: RouterState) -> RouterState:
    """Graph node to recall semantically similar past segments."""
    segment = state["segment"]
    semantic_context = None
    
    try:
        # Use default path for PoC. In prod this comes from config.
        vector_service = VectorStoreService()
        # Query for more than needed to filter out exact self-match
        results = vector_service.search_similar(segment.content, n_results=5)
        
        if results:
            # Simple dedup to avoid showing the exact same segment if it was just indexed
            matches = [r for r in results if r != segment.content][:3]
            
            if matches:
                formatted_matches = [f'- "{m}"' for m in matches]
                semantic_context = "### RECALLED MEMORY (SEMANTIC MATCHES)\n" + "\n".join(formatted_matches)
                
    except Exception as e:
        logger.error(f"Error recalling semantic context: {e}")
        
    return {**state, "semantic_context": semantic_context}


def node_enrich_context(state: RouterState) -> RouterState:
    """Enriches segment with resolved anchor context.

    Fetches EntityAnchors for the segment and resolves them via EnrichmentService
    to provide additional context for classification.

    Args:
        state: Current router state.

    Returns:
        RouterState: Updated state with related_context populated.
    """
    segment = state["segment"]
    session = state["db_session"]
    related_context = None

    anchors = session.exec(
        select(EntityAnchor).where(EntityAnchor.segment_id == segment.id)
    ).all()

    if anchors:
        enricher = EnrichmentService()
        context_lines = []
        for a in anchors:
            try:
                description = enricher.resolve_anchor(a.anchor_type, a.anchor_value)
                if description:
                    line = f"- [{a.anchor_type.value.upper()}] {a.anchor_value}: \"{description}\""
                else:
                    line = f"- [{a.anchor_type.value.upper()}] {a.anchor_value}"
            except Exception as e:
                logger.error(f"Error enriching anchor {a.anchor_value}: {e}")
                line = f"- [{a.anchor_type.value.upper()}] {a.anchor_value}"
            context_lines.append(line)

        related_context = "\n".join(context_lines)

    return {**state, "related_context": related_context}


def node_classify_stub(state: RouterState) -> RouterState:
    """Classifies segment using deterministic stub router.

    Args:
        state: Current router state.

    Returns:
        RouterState: Updated state with result from RouterStub.
    """
    segment = state["segment"]
    result = RouterStub().classify(segment.content)
    return {**state, "result": result}


async def node_classify_llm(state: RouterState) -> RouterState:
    """Classifies segment using LLM-based router.

    Args:
        state: Current router state with enriched context.

    Returns:
        RouterState: Updated state with result from LLM router.
    """
    segment = state["segment"]
    semantic_context = state.get("semantic_context")
    related_context = state.get("related_context")

    client = LLMClient()
    result = await llm_route_segment(
        client,
        segment_text=segment.content,
        context_before=segment.context_before,
        context_after=segment.context_after,
        related_context=related_context,
        semantic_context=semantic_context,
        parent_header=None,
    )
    return {**state, "result": result}


def node_shadow_stub(state: RouterState) -> RouterState:
    """Runs stub router in shadow mode for tournament comparison.

    Only called when shadow_mode=True and router_type="llm".

    Args:
        state: Current router state with LLM result.

    Returns:
        RouterState: Updated state with tournament_stub_result.
    """
    segment = state["segment"]
    stub_result = RouterStub().classify(segment.content)
    return {**state, "tournament_stub_result": stub_result}


def node_check_shadow(state: RouterState) -> RouterState:
    """Pass-through node before shadow mode conditional edge.

    This node exists purely to satisfy LangGraph's requirement that
    conditional edges must originate from a node.

    Args:
        state: Current router state.

    Returns:
        RouterState: Unchanged state.
    """
    return state


def node_flag_for_review(state: RouterState) -> RouterState:
    """Flags segment for human review due to low confidence.

    Sets the segment's recommended_state to REVIEW and logs the reason.

    Args:
        state: Current router state with low-confidence result.

    Returns:
        RouterState: State with result.recommended_state set to REVIEW.
    """
    result = state["result"]
    segment = state["segment"]

    if result:
        # Override recommended state to force human review
        result.recommended_state = RecommendedState.REVIEW
        result.reasoning = (
            f"HITL: Confidence {result.confidence:.0%} below {CONFIDENCE_THRESHOLD:.0%} threshold. "
            f"Original: {result.reasoning}"
        )
        logger.info(f"[HITL] Segment {segment.id} flagged for review (confidence={result.confidence:.2f})")
    else:
        logger.warning(f"[HITL] Segment {segment.id} flagged for review (no result)")

    return state


def node_persist_decision(state: RouterState) -> RouterState:
    """Persists the router decision to the database.

    Applies the classification result to the segment via apply_router_decision,
    which updates segment state and creates a SegmentDecisionLog entry.

    Args:
        state: Final router state with classification result.

    Returns:
        RouterState: Unchanged state (terminal node).
    """
    session = state["db_session"]
    segment = state["segment"]
    result = state["result"]
    shadow_mode = state["shadow_mode"]
    tournament_stub_result = state.get("tournament_stub_result")

    if result:
        apply_router_decision(
            session,
            segment,
            result,
            shadow_mode=shadow_mode,
            tournament_stub_result=tournament_stub_result,
        )

    return state


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================

def select_router(state: RouterState) -> Literal["classify_stub", "classify_llm"]:
    """Conditional edge: selects router based on router_type."""
    return "classify_stub" if state["router_type"] == "stub" else "classify_llm"


def should_run_shadow(state: RouterState) -> Literal["shadow_stub", "persist"]:
    """Conditional edge: runs shadow stub if shadow_mode is enabled."""
    return "shadow_stub" if state["shadow_mode"] else "persist"


CONFIDENCE_THRESHOLD = 0.90


def route_submission(state: RouterState) -> Literal["flag_for_review", "check_shadow"]:
    """Traffic cop: routes low-confidence LLM results to human review.

    Args:
        state: Current router state with LLM classification result.

    Returns:
        'flag_for_review' if confidence < 90% or result missing.
        'check_shadow' to proceed with shadow mode check.
    """
    result = state["result"]

    # Missing result edge case
    if not result:
        return "flag_for_review"

    # Low confidence triggers HITL
    if result.confidence < CONFIDENCE_THRESHOLD:
        return "flag_for_review"

    return "check_shadow"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_router_graph():
    """Builds and compiles the LangGraph workflow for segment routing.

    Graph Structure:
        START → recall → enrich → [router_type?]
                                      ├─ "stub" → classify_stub → persist → END
                                      └─ "llm"  → classify_llm → [route_submission?]
                                                                      ├─ "flag_for_review" → flag_for_review → persist → END
                                                                      └─ "check_shadow" → [shadow?]
                                                                                              ├─ shadow_stub → persist → END
                                                                                              └─ persist → END

    Returns:
        CompiledGraph: Ready-to-invoke LangGraph workflow.
    """
    workflow = StateGraph(RouterState)

    # Add nodes
    workflow.add_node("recall", node_recall_context)
    workflow.add_node("enrich", node_enrich_context)
    workflow.add_node("classify_stub", node_classify_stub)
    workflow.add_node("classify_llm", node_classify_llm)
    workflow.add_node("flag_for_review", node_flag_for_review)
    workflow.add_node("check_shadow", node_check_shadow)
    workflow.add_node("shadow_stub", node_shadow_stub)
    workflow.add_node("persist", node_persist_decision)

    # Entry: START → recall → enrich
    workflow.add_edge(START, "recall")
    workflow.add_edge("recall", "enrich")

    # Conditional: enrich → [stub or llm]
    workflow.add_conditional_edges(
        "enrich",
        select_router,
        {"classify_stub": "classify_stub", "classify_llm": "classify_llm"},
    )

    # Stub path: classify_stub → persist → END
    workflow.add_edge("classify_stub", "persist")

    # LLM path: classify_llm → [route_submission?] → HITL or shadow check
    workflow.add_conditional_edges(
        "classify_llm",
        route_submission,
        {"flag_for_review": "flag_for_review", "check_shadow": "check_shadow"},
    )

    # HITL path: flag_for_review → persist
    workflow.add_edge("flag_for_review", "persist")

    # Shadow check (only reached if confidence >= 90%)
    workflow.add_conditional_edges(
        "check_shadow",
        should_run_shadow,
        {"shadow_stub": "shadow_stub", "persist": "persist"},
    )
    workflow.add_edge("shadow_stub", "persist")

    # Terminal
    workflow.add_edge("persist", END)

    return workflow.compile()

# =============================================================================
# PUBLIC API
# =============================================================================

async def route_segments_async(
    limit: int = 100,
    router_type: Literal["stub", "llm"] = "stub",
    shadow_mode: bool = False,
    concurrency: int = 5,
    session: Optional[Session] = None,
) -> int:
    """Routes QUARANTINE segments through the classification workflow.

    Args:
        limit: Maximum number of segments to process.
        router_type: Router to use ("stub" for deterministic, "llm" for GPT).
        shadow_mode: If True and router_type="llm", also runs stub for comparison.
        concurrency: Max concurrent segment processing tasks.
        session: Database session. If None, creates and closes one.

    Returns:
        int: Number of segments processed.
    """
    if session is None:
        session_gen = get_session()
        session = next(session_gen)
        should_close = True
    else:
        should_close = False

    processed_count = 0
    graph = build_router_graph()

    try:
        statement = select(DocSegment).where(
            DocSegment.processing_state == ProcessingState.QUARANTINE
        ).limit(limit)

        segments = session.exec(statement).all()
        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(seg: DocSegment):
            async with semaphore:
                initial_state: RouterState = {
                    "segment": seg,
                    "result": None,
                    "tournament_stub_result": None,
                    "shadow_mode": shadow_mode,
                    "router_type": router_type,
                    "db_session": session,
                    "semantic_context": None,
                    "related_context": None,
                }
                await graph.ainvoke(initial_state)

        tasks = [process_one(seg) for seg in segments]
        await asyncio.gather(*tasks)
        processed_count = len(segments)

        session.commit()

    except Exception as e:
        logger.error(f"Error in routing: {e}")
        session.rollback()
        raise
    finally:
        if should_close:
            session.close()

    return processed_count


def route_segments(
    limit: int = 100,
    router_type: Literal["stub", "llm"] = "stub",
    shadow_mode: bool = False,
    concurrency: int = 5,
    session: Optional[Session] = None,
) -> int:
    """Synchronous entrypoint for routing segments.

    Wraps route_segments_async in asyncio.run().

    Args:
        limit: Maximum number of segments to process.
        router_type: Router to use ("stub" or "llm").
        shadow_mode: If True, runs tournament comparison.
        concurrency: Max concurrent tasks.
        session: Database session.

    Returns:
        int: Number of segments processed.
    """
    return asyncio.run(
        route_segments_async(
            limit=limit,
            router_type=router_type,
            shadow_mode=shadow_mode,
            concurrency=concurrency,
            session=session,
        )
    )
