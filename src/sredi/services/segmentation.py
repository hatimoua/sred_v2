import re
from pathlib import Path
from typing import List, Optional
from sqlmodel import Session, select
import uuid
from ..models import Document, DocSegment, ProcessingState, EntityAnchor, AnchorType
from ..db import get_session

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".py"}

def segment_documents(workspace_name: str = "default", session: Optional[Session] = None) -> int:
    """Finds documents without segments and splits them into atomic units with structural provenance.

    Args:
        workspace_name: Name of the workspace to process. Defaults to "default".
        session: Optional database session. If None, a new one is created and closed.

    Returns:
        The total number of new segments created across all processed documents.
    """
    if session is None:
        session_gen = get_session()
        session = next(session_gen)
        should_close = True
    else:
        should_close = False
    
    total_segments = 0
    
    try:
        from ..services.ingestion import get_or_create_workspace
        workspace = get_or_create_workspace(session, workspace_name)

        statement = select(Document).where(Document.workspace_id == workspace.id)
        results = session.exec(statement).all()
        
        for doc in results:
            if len(doc.segments) > 0:
                continue

            file_path = Path(doc.file_path)
            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue
            
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                
                # Split boundaries: Markdown headers OR double newlines
                # Use re.MULTILINE to catch headers at start of any line
                pattern = re.compile(r'(?m)(^#+ .*$|\n\n+)')
                
                last_pos = 0
                sequence_index = 0
                current_parent_id = None
                
                # Find all matches
                for match in pattern.finditer(content):
                    start, end = match.span()
                    
                    # Capture preceding chunk if it's not empty
                    chunk_text = content[last_pos:start]
                    if chunk_text.strip():
                        current_parent_id = _create_segment(
                            session, doc.id, chunk_text, last_pos, start, 
                            content, sequence_index, current_parent_id, is_header=False
                        )
                        sequence_index += 1
                        total_segments += 1
                    
                    # Capture the boundary itself if it's a header
                    boundary_text = match.group(0)
                    if boundary_text.startswith('#'):
                        current_parent_id = _create_segment(
                            session, doc.id, boundary_text, start, end, 
                            content, sequence_index, None, is_header=True
                        )
                        sequence_index += 1
                        total_segments += 1
                    
                    last_pos = end
                
                # Capture final chunk
                final_chunk = content[last_pos:]
                if final_chunk.strip():
                    _create_segment(
                        session, doc.id, final_chunk, last_pos, len(content), 
                        content, sequence_index, current_parent_id, is_header=False
                    )
                    total_segments += 1
                
                session.commit()
                
            except Exception as e:
                print(f"Error segmenting {doc.filename}: {e}")
                session.rollback()
                continue

    finally:
        if should_close:
            session.close()

    return total_segments

def _create_segment(
    session: Session, 
    doc_id: uuid.UUID, 
    text: str, 
    start: int, 
    end: int, 
    full_content: str,
    sequence_index: int,
    parent_id: Optional[uuid.UUID],
    is_header: bool
) -> uuid.UUID:
    """Helper to create a DocSegment with exact offsets and context shadows.

    Args:
        session: Active database session.
        doc_id: ID of the source document.
        text: Raw text content for this segment.
        start: Character start offset in the source file.
        end: Character end offset in the source file.
        full_content: Entire content of the source file for context extraction.
        sequence_index: Monotonic position index in the document.
        parent_id: Optional ID of the parent header segment.
        is_header: Flag indicating if this segment is a structural header.

    Returns:
        uuid.UUID: The generated ID of the new segment.
    """
    context_before = full_content[max(0, start - 300):start]
    context_after = full_content[end:end + 300]
    
    seg = DocSegment(
        document_id=doc_id,
        content=text.strip(),
        start_offset=start,
        end_offset=end,
        sequence_index=sequence_index,
        parent_id=parent_id,
        context_before=context_before,
        context_after=context_after,
        processing_state=ProcessingState.QUARANTINE
    )
    session.add(seg)
    session.flush() # Ensure ID is generated for parent tracking

    # Step 1.2 & 1.3: Extract and persist anchors
    anchors = extract_anchors(seg.content)
    for anchor_data in anchors:
        anchor = EntityAnchor(
            segment_id=seg.id,
            anchor_type=anchor_data["type"],
            anchor_value=anchor_data["value"],
            confidence=anchor_data["confidence"]
        )
        session.add(anchor)
        
    # Step 5.3: Semantic Ingestion Hook
    try:
        from .vector_store import VectorStoreService
        # Use default path for PoC. In prod this comes from config.
        vector_service = VectorStoreService()
        vector_service.add_segment(seg)
    except Exception as e:
        # Fail safe: Do not stop ingestion if vector store fails
        print(f"Vector store ingestion failed for segment {seg.id}: {e}")

    return seg.id

def extract_anchors(text: str) -> List[dict]:
    r"""Extracts and normalizes hard anchors (tickets, PRs, files) from text.
    
    Patterns:
    - Jira: [A-Z]{2,10}-\d+
    - GitHub PR/Issue: (#\d+) with prefix normalization
    - File Refs: paths ending in code extensions
    """
    anchors = []
    
    # 1. Jira / Ticket IDs (Normalized to uppercase)
    jira_pattern = re.compile(r'\b([A-Z]{2,10}-\d{1,10})\b')
    for match in jira_pattern.finditer(text):
        anchors.append({
            "type": AnchorType.TICKET,
            "value": match.group(1).upper(),
            "confidence": 1.0
        })
        
    # 2. GitHub PRs/Issues (Normalized to #123)
    # Pattern matches optional prefix and then #digits
    gh_pattern = re.compile(r'(?i)(?:close|fix|ref|see)?\s*(#\d+)\b')
    for match in gh_pattern.finditer(text):
        anchors.append({
            "type": AnchorType.PR,
            "value": match.group(1), # group(1) is only the #digits part
            "confidence": 1.0
        })
        
    # 3. File References (Normalized to lowercase)
    file_pattern = re.compile(r'\b([\w\-/]+\.(?:py|md|ts|go|json|yaml|rst))\b')
    for match in file_pattern.finditer(text):
        anchors.append({
            "type": AnchorType.FILE_REF,
            "value": match.group(1).lower(),
            "confidence": 1.0
        })

    # 4. Error Codes (Basic patterns for stack trace indicators)
    error_pattern = re.compile(r'\b(Traceback|Exception|RuntimeError|ValueError|TypeError|ERROR|CRITICAL)\b', re.IGNORECASE)
    for match in error_pattern.finditer(text):
         anchors.append({
            "type": AnchorType.ERROR_CODE,
            "value": match.group(1),
            "confidence": 1.0
        })
        
    return anchors

def reconstruct_document(segments: List[DocSegment]) -> str:
    """Reconstructs a visual representation of the document from its segments.

    This utility is used to verify the structural integrity of the segmentation
    by joining the contents of all segments in their original sequence.

    Args:
        segments: List of DocSegments belonging to a single document.

    Returns:
        str: A concatenated string of all segment contents.
    """
    # Since segments are ordered by sequence_index via relationship, 
    # but the input might be a filtered list, we sort manually to be sure.
    sorted_segments = sorted(segments, key=lambda s: s.sequence_index)
    
    result = []
    current_pos = 0
    
    # This reconstruction assumes segments cover the whole file or we fill gaps with whitespace.
    # To be perfectly deterministic, we'd store the "gap" text too, 
    # but for audit we check if content exists at correct offsets.
    for seg in sorted_segments:
        # If there's a gap between segments (like \n\n that were consumed by regex),
        # we could fill it or just append.
        # Strict reconstruction:
        result.append(seg.content)
        
    return "\n\n".join(result) # Rough approximation for visual check
