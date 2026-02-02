import pytest
import json
import httpx
from unittest.mock import AsyncMock, patch
from sredi.services.llm_client import LLMClient, LLMClientError
from sredi.services.router_llm import llm_route_segment, ROUTER_SYSTEM_PROMPT
from sredi.services.router_contract import RouterLabel, RecommendedState, RiskFlag, HardSignal

@pytest.mark.asyncio
async def test_llm_client_openai_success():
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "label": "TECHNICAL",
                        "confidence": 0.95,
                        "signals": ["code_block"],
                        "risk_flags": [],
                        "proof_spans": [{"start": 0, "end": 10, "excerpt": "some code", "kind": "code"}],
                        "recommended_state": "INDEX_READY",
                        "reasoning": "Clear technical evidence"
                    })
                }
            }
        ]
    }
    
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(status_code=200, json=lambda: mock_response)
        
        client = LLMClient(provider="openai", api_key="fake_key")
        result = await client.classify_segment(segment_text="some code", metadata={})
        
        assert result["label"] == "TECHNICAL"
        assert result["confidence"] == 0.95

@pytest.mark.asyncio
async def test_llm_client_retry_logic():
    mock_final_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"success": True})
                }
            }
        ]
    }
    with patch("httpx.AsyncClient.post") as mock_post:
        # First call 429, second call 200
        mock_post.side_effect = [
            AsyncMock(status_code=429, text="Rate limit", raise_for_status=lambda: None),
            AsyncMock(status_code=200, json=lambda: mock_final_response)
        ]
        
        client = LLMClient(provider="openai", api_key="fake_key", max_retries=1)
        # Patch sleep to avoid waiting
        with patch("asyncio.sleep", return_value=None):
            result = await client.classify_segment(segment_text="test", metadata={})
            assert result["success"] is True
            assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_router_llm_parsing_success():
    mock_raw = {
        "label": "TECHNICAL",
        "confidence": 0.9,
        "signals": ["code_block"],
        "risk_flags": [],
        "proof_spans": [{"start": 0, "end": 5, "excerpt": "print", "kind": "code"}],
        "recommended_state": "INDEX_READY",
        "reasoning": "Found print statement"
    }
    
    client = AsyncMock(spec=LLMClient)
    client.provider = "openai"
    client.model = "gpt-4o"
    client.classify_segment.return_value = mock_raw
    
    result = await llm_route_segment(client, segment_text="print('hi')")
    
    assert result.label == RouterLabel.TECHNICAL
    assert result.recommended_state == RecommendedState.INDEX_READY
    assert result.model_id == "openai:gpt-4o"

@pytest.mark.asyncio
async def test_router_llm_fail_closed_invalid_json():
    client = AsyncMock(spec=LLMClient)
    client.provider = "openai"
    client.model = "gpt-4o"
    client.classify_segment.side_effect = Exception("Malformed JSON from LLM")
    
    result = await llm_route_segment(client, segment_text="garbage")
    
    assert result.label == RouterLabel.AMBIGUOUS
    assert result.recommended_state == RecommendedState.REVIEW
    assert RiskFlag.PROOF_INVALID in result.risk_flags
    assert result.reasoning is not None and "Malformed JSON" in result.reasoning

@pytest.mark.asyncio
async def test_router_llm_fail_closed_missing_fields():
    # Missing 'label'
    mock_raw = {
        "confidence": 0.9,
        "reasoning": "incomplete"
    }
    
    client = AsyncMock(spec=LLMClient)
    client.provider = "openai"
    client.model = "gpt-4o"
    client.classify_segment.return_value = mock_raw
    
    result = await llm_route_segment(client, segment_text="some text")
    
    assert result.label == RouterLabel.AMBIGUOUS
    assert result.recommended_state == RecommendedState.REVIEW
    assert RiskFlag.PROOF_INVALID in result.risk_flags
