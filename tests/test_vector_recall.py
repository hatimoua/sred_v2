import pytest
import uuid
import tempfile
import shutil
from unittest.mock import patch, AsyncMock
from sqlmodel import Session
from sredi.services import router
from sredi.models.models import DocSegment, Workspace, Document
from sredi.models.enums import ProcessingState
from sredi.services.vector_store import VectorStoreService

@pytest.fixture
def clean_vector_store_path():
    # Setup: Create temp dir
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Teardown: Delete temp dir
    shutil.rmtree(temp_dir)

@pytest.mark.asyncio
async def test_router_recalls_semantic_context(session: Session, clean_vector_store_path):
    # 1. Setup - Initialize Vector Store with temp path
    vector_service = VectorStoreService(path=clean_vector_store_path)
    
    # 2. Setup - Create Workspace and Document
    ws = Workspace(name="semantic_test_ws")
    session.add(ws)
    session.flush()

    doc = Document(
        workspace_id=ws.id,
        content_hash="semantic_test_hash",
        filename="db_logs.md",
        file_path="/db_logs.md",
        file_size_bytes=100
    )
    session.add(doc)
    session.flush()

    # 3. Ingest Historical Segments (The "Memory")
    # Segment A: Explicit "Database is slow"
    seg_a = DocSegment(
        id=uuid.uuid4(),
        document_id=doc.id,
        content="The database is running very slow during peak load.",
        processing_state=ProcessingState.INDEX_READY
    )
    # Segment B: "Latency in SQL layer"
    seg_b = DocSegment(
        id=uuid.uuid4(),
        document_id=doc.id,
        content="We observed high latency in the SQL layer queries.",
        processing_state=ProcessingState.INDEX_READY
    )
    
    session.add(seg_a)
    session.add(seg_b)
    
    # Add to Vector Store
    vector_service.add_segment(seg_a)
    vector_service.add_segment(seg_b)
    
    # 4. Setup - The "Current" Segment (Query)
    # Segment C: "Query performance dropped" (Semantically similar to A and B)
    seg_c = DocSegment(
        id=uuid.uuid4(),
        document_id=doc.id,
        content="Investigating why query performance dropped significantly.",
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg_c)
    session.commit()

    # 5. Action - Mock LLM and VectorService inside Router
    # We need to mock VectorStoreService in router.py to use our temp-path instance
    with patch("sredi.services.router.VectorStoreService") as MockVectorService, \
         patch("sredi.services.router.llm_route_segment", new_callable=AsyncMock) as mock_llm:
        
        # Configure the mock to return our pre-initialized service instance
        MockVectorService.return_value = vector_service
        
        # Mock LLM return value
        from sredi.services.router_contract import RouterResult, RouterLabel, RecommendedState
        mock_llm.return_value = RouterResult(
            label=RouterLabel.TECHNICAL,
            confidence=0.85,
            signals=[],
            proof_spans=[],
            recommended_state=RecommendedState.INDEX_READY,
            reasoning="Mocked reasoning",
            model_id="mock",
            prompt_version="mock",
            policy_version="mock"
        )

        # Build and run the graph
        graph = router.build_router_graph()
        initial_state = router.RouterState(
            segment=seg_c,
            result=None,
            tournament_stub_result=None,
            shadow_mode=False,
            router_type="llm",
            db_session=session,
            semantic_context=None
        )
        await graph.ainvoke(initial_state)

        # 6. Assertion - Verify Semantic Context Injection
        assert mock_llm.called
        args, kwargs = mock_llm.call_args
        semantic_context = kwargs.get("semantic_context")
        
        assert semantic_context is not None
        # Should contain recalled memories
        assert "RECALLED MEMORY" in semantic_context
        # Should contain content from Segment A or B (depending on Chroma's retrieval score)
        # Both are very similar, so at least one should appear.
        match_a = "database is running very slow" in semantic_context
        match_b = "high latency in the SQL layer" in semantic_context
        
        assert match_a or match_b, f"Expected semantic recall of DB/SQL issues. Got: {semantic_context}"
