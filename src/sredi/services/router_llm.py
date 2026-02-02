import logging
import json
from typing import Dict, Any, Optional
from .llm_client import LLMClient, LLMClientError
from .router_contract import RouterResult, RouterLabel, RecommendedState, RiskFlag, HardSignal

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a senior technical auditor for SRED.ai.
Analyze the provided document segment and classify it according to strict SRED (Scientific Research and Experimental Development) criteria.

### GOAL
Determine if the segment contains technical evidence of R&D activity (e.g., technical challenges, architectural decisions, testing results, technical constraints).

### OUTPUT FORMAT
You MUST output a valid JSON object only. No prose.
The JSON must follow this structure:
{
    "label": "TECHNICAL" | "FINANCIAL" | "AMBIGUOUS" | "NOISE",
    "confidence": float (0.0 to 1.0),
    "signals": ["code_block", "stack_trace", "exception_message", "architecture_component", "performance_metric", "test_result", "technical_constraint", "unknown_root_cause"],
    "risk_flags": ["marketing_language", "sales_pricing_language", "legal_terms", "hr_performance", "no_technical_markers", "ambiguous_context", "contradictory_signals"],
    "proof_spans": [
        {
            "quote": "verbatim substring from input",
            "kind": "optional category"
        }
    ],
    "recommended_state": "INDEX_READY" | "REVIEW" | "NOISE",
    "reasoning": "string (max 800 chars)"
}

### CRITICAL RULES FOR PROOF SPANS (ZERO-TRUST)
1. You MUST copy text EXACTLY from the segment.
2. Do NOT paraphrase. Do NOT summarize.
3. Preserve ALL Markdown symbols (*, **, backticks, brackets, comments).
4. Preserve ALL whitespace, newlines, and punctuation EXACTLY.
5. The quote MUST be contiguous and taken from ONE place in the segment.
6. No ellipses (...), no stitched fragments, no partial word cuts.
7. Avoid multi-line quotes if possible; pick a single sentence or continuous phrase within one line.
8. If the label is TECHNICAL and recommended_state is INDEX_READY, proof_spans MUST NOT be empty.

### ABSTENTION RULE (NO_PROOF)
If you cannot find a contiguous verbatim substring that supports your classification:
- Set quote: ""
- Set reasoning: "NO_PROOF: cannot provide verbatim quote"
- Set recommended_state: "REVIEW" (even if label is TECHNICAL)

This is a strict requirement. Accuracy is more important than providing proof.
"""

async def llm_route_segment(
    client: LLMClient,
    *,
    segment_text: str,
    context_before: Optional[str] = None,
    context_after: Optional[str] = None,
    parent_header: Optional[str] = None,
    policy_version: str = "router_policy_v1",
    prompt_version: str = "router_prompt_v1",
) -> RouterResult:
    """Routes a segment using LLM with context and fail-closed parsing."""
    
    # Construct input for LLM with context
    input_text = ""
    if parent_header:
        input_text += f"PARENT HEADER: {parent_header}\n\n"
    if context_before:
        input_text += f"CONTEXT BEFORE:\n---\n{context_before}\n---\n\n"
    
    input_text += f"SEGMENT TO CLASSIFY:\n===\n{segment_text}\n===\n\n"
    
    if context_after:
        input_text += f"CONTEXT AFTER:\n---\n{context_after}\n---\n"

    try:
        raw_result = await client.classify_segment(
            segment_text=input_text,
            metadata={"system_prompt": ROUTER_SYSTEM_PROMPT}
        )
        
        # Inject versions and model info
        raw_result["policy_version"] = policy_version
        raw_result["prompt_version"] = prompt_version
        raw_result["model_id"] = f"{client.provider}:{client.model}"
        
        # Parse into RouterResult
        return RouterResult(**raw_result)

    except (LLMClientError, Exception) as e:
        logger.error(f"LLM Routing failed or returned invalid JSON: {e}")
        
        # Fail-closed: Recommend REVIEW with proof_invalid flag
        return RouterResult(
            label=RouterLabel.AMBIGUOUS,
            confidence=0.0,
            risk_flags=[RiskFlag.PROOF_INVALID],
            recommended_state=RecommendedState.REVIEW,
            reasoning=f"LLM Error or Invalid Output: {str(e)[:200]}",
            model_id=f"{client.provider}:{client.model}",
            prompt_version=prompt_version,
            policy_version=policy_version
        )
