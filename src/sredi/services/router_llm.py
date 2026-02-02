import logging
import json
from typing import Dict, Any, Optional
from .llm_client import LLMClient, LLMClientError
from .router_contract import RouterResult, RouterLabel, RecommendedState, RiskFlag, HardSignal, ProofSpan

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are ROUTER_LLM for a Zero-Trust SR&ED document triage system.

You will be given:
1) SEGMENT TO CLASSIFY: The target text you must analyze and quote from.
2) CONTEXT BEFORE/AFTER: Surrounding text to help you understand the segment's meaning.

Your job is to:
1) Choose a routing label for the SEGMENT TO CLASSIFY.
2) Provide verifiable evidence as VERBATIM QUOTES copied EXACTLY from the SEGMENT TO CLASSIFY only.

CRITICAL ZERO-TRUST RULES (DO NOT VIOLATE)
- You MUST NOT copy quotes from the CONTEXT sections. Quotes MUST come from the "SEGMENT TO CLASSIFY" block.
- You MUST NOT paraphrase, summarize, or rewrite any evidence.
- Evidence MUST be a single contiguous substring copied EXACTLY from the provided segment text.
- Preserve ALL characters exactly: whitespace, punctuation, Markdown symbols (*, **, `, [, ], #, -, >), quotes, and capitalization.
- Do NOT use ellipses (...) inside quotes.
- Do NOT stitch together multiple separated parts.

OUTPUT FORMAT (STRICT JSON ONLY)
Return ONLY valid JSON with this exact schema:

{
  "final_state": "INDEX_READY" | "REVIEW" | "NOISE",
  "proofs": [
    {
      "quote": "<verbatim substring copied exactly from the SEGMENT TO CLASSIFY>",
      "reason": "<1 short sentence explaining why this quote supports the chosen final_state>"
    }
  ]
}

EVIDENCE REQUIREMENTS
- Provide 1 to 3 proofs.
- Each quote should be 80 to 240 characters when possible.
- If the SEGMENT TO CLASSIFY is shorter than 80 characters, copy the relevant portion exactly.

WHEN TO ABSTAIN (IMPORTANT)
If you cannot find a contiguous verbatim substring in the SEGMENT TO CLASSIFY that supports your classification:
- Set final_state to "REVIEW"
- Return an EMPTY proofs list: "proofs": []
- Do NOT invent a quote. Do NOT quote from context.

ROUTING GUIDANCE (HOW TO PICK final_state)
INDEX_READY: Clear technical evidence of R&D (uncertainty, systematic investigation, experiments).
REVIEW: Ambiguous, incomplete, or mixed content.
NOISE: Administrative, boilerplate, templates, checklists, marketing fluff.

IMPORTANT: If the evidence for a technical label is only found in the CONTEXT sections and not in the SEGMENT TO CLASSIFY itself, you MUST choose REVIEW and return proofs: [].
"""

async def llm_route_segment(
    client: LLMClient,
    *,
    segment_text: str,
    context_before: Optional[str] = None,
    context_after: Optional[str] = None,
    related_context: Optional[str] = None,
    semantic_context: Optional[str] = None,
    parent_header: Optional[str] = None,
    policy_version: str = "router_policy_v1",
    prompt_version: str = "router_prompt_v4",
) -> RouterResult:
    """Routes a segment using LLM with context and fail-closed parsing."""
    
    # Construct input for LLM with context
    input_text = ""
    if related_context:
        input_text += f"### STRUCTURAL CONTEXT (METADATA ONLY)\nThe following entities are linked to this segment. Use them to understand the technical intent (e.g. recognizing a bug fix vs maintenance).\n\n{related_context}\n\nCRITICAL RULES:\n1. You MUST NOT quote from the \"STRUCTURAL CONTEXT\" section.\n2. Your \"proof_spans\" must ONLY come from the \"SEGMENT TO CLASSIFY\" block below.\n3. If the context implies R&D but the segment text has zero evidence, you must still classify as AMBIGUOUS or NOISE.\n\n"
    
    if semantic_context:
        input_text += f"### RECALLED MEMORY (SEMANTIC MATCHES)\nThe following are PAST segments that are semantically similar to the current one. Use them as reference for how similar content might be classified or phrased.\n\n{semantic_context}\n\nCRITICAL RULE: These are PAST segments. Do NOT confuse them with the CURRENT segment.\n\n"

    if parent_header:
        input_text += f"PARENT HEADER: {parent_header}\n\n"
    if context_before:
        input_text += f"CONTEXT BEFORE:\n---\n{context_before}\n---\n\n"
    
    input_text += f"SEGMENT TO CLASSIFY:\n===\n{segment_text}\n===\n\n"
    input_text += "Copy quotes from the segment text exactly. Do not paraphrase."
    
    if context_after:
        input_text += f"\n\nCONTEXT AFTER:\n---\n{context_after}\n---\n"

    try:
        raw_llm_output = await client.classify_segment(
            segment_text=input_text,
            metadata={"system_prompt": ROUTER_SYSTEM_PROMPT}
        )
        
        # Map new schema to RouterResult
        # final_state -> recommended_state
        # proofs -> proof_spans
        # reasoning -> combined reasons
        
        final_state_str = raw_llm_output.get("final_state", "REVIEW")
        proofs = raw_llm_output.get("proofs", [])
        
        # Map final_state to internal Enum
        state_map = {
            "INDEX_READY": RecommendedState.INDEX_READY,
            "REVIEW": RecommendedState.REVIEW,
            "NOISE": RecommendedState.NOISE
        }
        recommended_state = state_map.get(final_state_str, RecommendedState.REVIEW)
        
        # Map label based on state (LLM doesn't output label directly anymore)
        label_map = {
            RecommendedState.INDEX_READY: RouterLabel.TECHNICAL,
            RecommendedState.NOISE: RouterLabel.NOISE,
            RecommendedState.REVIEW: RouterLabel.AMBIGUOUS
        }
        label = label_map.get(recommended_state, RouterLabel.AMBIGUOUS)
        
        proof_spans = []
        all_reasons = []
        for p in proofs:
            proof_spans.append(ProofSpan(
                quote=p.get("quote", ""),
                kind="evidence"
            ))
            if p.get("reason"):
                all_reasons.append(p.get("reason"))
        
        reasoning = " | ".join(all_reasons) if all_reasons else "No specific reasoning provided."
        if not proofs:
            reasoning = "NO_PROOF: cannot provide verbatim quote"
        
        return RouterResult(
            label=label,
            confidence=0.95 if recommended_state != RecommendedState.REVIEW else 0.5,
            signals=[], # LLM doesn't output signals in new schema, will be filled by post-processing or updated later
            risk_flags=[],
            proof_spans=proof_spans,
            recommended_state=recommended_state,
            reasoning=reasoning[:800],
            model_id=f"{client.provider}:{client.model}",
            prompt_version=prompt_version,
            policy_version=policy_version
        )

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
