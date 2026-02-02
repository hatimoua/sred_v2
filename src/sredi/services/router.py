import re
from typing import List, Optional, Tuple, Sequence, TypedDict, Annotated
from sqlmodel import Session, select
from datetime import datetime, UTC
import asyncio

from langgraph.graph import StateGraph, START, END

from ..models import (
    DocSegment, 
    SegmentDecisionLog, 
    ProcessingState, 
    ClassificationLabel,
    EntityAnchor
)
from ..db import get_session
from .router_contract import (
    RouterResult, 
    RouterLabel, 
    RecommendedState, 
    HardSignal, 
    ProofSpan, 
    RiskFlag
)
from .validation import apply_router_decision
from .llm_client import LLMClient
from .router_llm import llm_route_segment
from .enrichment import EnrichmentService

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
    """State managed by the LangGraph router workflow."""
    segment: DocSegment
    result: Optional[RouterResult]
    tournament_stub_result: Optional[RouterResult]
    shadow_mode: bool
    router_type: str
    db_session: Session

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

async def node_classify_segment(state: RouterState) -> RouterState:
    """Graph node to classify a segment using either stub or LLM router."""
    segment = state["segment"]
    router_type = state["router_type"]
    shadow_mode = state["shadow_mode"]
    session = state["db_session"]
    
    # 1. Fetch and Enrich Structural Context (Hard Anchors)
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
                    context_line = f"- [{a.anchor_type.value.upper()}] {a.anchor_value}: \"{description}\""
                else:
                    context_line = f"- [{a.anchor_type.value.upper()}] {a.anchor_value}"
            except Exception as e:
                # Fail Safe: Non-blocking decoration
                print(f"Error enriching anchor {a.anchor_value}: {e}")
                context_line = f"- [{a.anchor_type.value.upper()}] {a.anchor_value}"
            context_lines.append(context_line)
            
        related_context = "\n".join(context_lines)
    
    result = None
    tournament_stub_result = None
    
    if router_type == "stub":
        result = RouterStub().classify(segment.content)
    elif router_type == "llm":
        client = LLMClient()
        result = await llm_route_segment(
            client,
            segment_text=segment.content,
            context_before=segment.context_before,
            context_after=segment.context_after,
            related_context=related_context,
            parent_header=None # TODO: Fetch parent header if available
        )
        
        if shadow_mode:
            tournament_stub_result = RouterStub().classify(segment.content)
            
    return {
        **state,
        "result": result,
        "tournament_stub_result": tournament_stub_result
    }

def node_persist_decision(state: RouterState) -> RouterState:
    """Graph node to persist the router decision to the database."""
    session = state["db_session"]
    segment = state["segment"]
    result = state["result"]
    shadow_mode = state["shadow_mode"]
    tournament_stub_result = state["tournament_stub_result"]
    
    if result:
        apply_router_decision(
            session, 
            segment, 
            result, 
            shadow_mode=shadow_mode, 
            tournament_stub_result=tournament_stub_result
        )
        
    return state

def build_router_graph():
    """Builds and compiles the LangGraph workflow for routing."""
    workflow = StateGraph(RouterState)
    
    workflow.add_node("classify", node_classify_segment)
    workflow.add_node("persist", node_persist_decision)
    
    workflow.add_edge(START, "classify")
    workflow.add_edge("classify", "persist")
    workflow.add_edge("persist", END)
    
    return workflow.compile()

async def route_segments_async(
    limit: int = 100, 
    router_type: str = "stub", 
    shadow_mode: bool = False, 
    concurrency: int = 5,
    session: Optional[Session] = None
) -> int:
    """Async version of route_segments to be called from an existing loop."""
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
                initial_state = RouterState(
                    segment=seg,
                    result=None,
                    tournament_stub_result=None,
                    shadow_mode=shadow_mode,
                    router_type=router_type,
                    db_session=session
                )
                await graph.ainvoke(initial_state)

        tasks = [process_one(seg) for seg in segments]
        await asyncio.gather(*tasks)
        processed_count = len(segments)
        
        session.commit()
        
    except Exception as e:
        print(f"Error in routing: {e}")
        session.rollback()
        raise
    finally:
        if should_close:
            session.close()

    return processed_count

def route_segments(
    limit: int = 100,
    router_type: str = "stub",
    shadow_mode: bool = False,
    concurrency: int = 5,
    session: Optional[Session] = None
) -> int:
    """Synchronous entrypoint for routing segments."""
    return asyncio.run(route_segments_async(
        limit=limit,
        router_type=router_type,
        shadow_mode=shadow_mode,
        concurrency=concurrency,
        session=session
    ))
