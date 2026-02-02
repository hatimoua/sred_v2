import pytest
import uuid
from unittest.mock import patch, AsyncMock
from sqlmodel import Session
from sredi.services import router
from sredi.models.models import DocSegment, EntityAnchor, Workspace, Document
from sredi.models.enums import ProcessingState, AnchorType

@pytest.mark.asyncio
async def test_graph_injects_context(session: Session):
    # 1. Setup - Create Workspace and Document (to avoid FK errors)
    ws = Workspace(name="context_test_ws")
    session.add(ws)
    session.flush()

    doc = Document(
        workspace_id=ws.id,
        content_hash="context_test_hash",
        filename="test.md",
        file_path="/test.md",
        file_size_bytes=100
    )
    session.add(doc)
    session.flush()

    # 2. Setup - Create DocSegment
    seg = DocSegment(
        id=uuid.uuid4(),
        document_id=doc.id,
        content="This is a segment referencing JIRA-123 and UNKNOWN-999.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.flush()

    # 3. Setup - Create EntityAnchors (one known, one unknown)
    anchor1 = EntityAnchor(
        segment_id=seg.id,
        anchor_type=AnchorType.TICKET,
        anchor_value="JIRA-123",
        confidence=1.0
    )
    anchor2 = EntityAnchor(
        segment_id=seg.id,
        anchor_type=AnchorType.TICKET,
        anchor_value="UNKNOWN-999",
        confidence=1.0
    )
    session.add(anchor1)
    session.add(anchor2)
    session.commit()

    # 4. Action - Mock llm_route_segment and run the graph
    with patch("sredi.services.router.llm_route_segment", new_callable=AsyncMock) as mock_llm:
        # Mock a return value for the LLM
        from sredi.services.router_contract import RouterResult, RouterLabel, RecommendedState
        mock_llm.return_value = RouterResult(
            label=RouterLabel.TECHNICAL,
            confidence=0.9,
            signals=[],
            proof_spans=[],
            recommended_state=RecommendedState.INDEX_READY,
            reasoning="Mocked reason",
            model_id="mock",
            prompt_version="mock",
            policy_version="mock"
        )

        # Build graph and run
        graph = router.build_router_graph()
        initial_state = router.RouterState(
            segment=seg,
            result=None,
            tournament_stub_result=None,
            shadow_mode=False,
            router_type="llm",
            db_session=session
        )
        await graph.ainvoke(initial_state)

        # 5. Assertion - Verify enriched context delivery
        assert mock_llm.called
        args, kwargs = mock_llm.call_args
        related_context = kwargs.get("related_context")
        
        assert related_context is not None
        # Verify Enriched Anchor (Known ID)
        assert "JIRA-123" in related_context
        assert "Critical Sharding Failure" in related_context
        
        # Verify Fallback Anchor (Unknown ID)
        assert "UNKNOWN-999" in related_context
        # It should appear without a description
        assert "UNKNOWN-999\"" not in related_context 
