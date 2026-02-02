import re
from pathlib import Path
from typing import List, Optional
from sqlmodel import Session, select
import uuid

from ..models import Document, DocSegment, ProcessingState
from ..db import get_session

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst"}

def segment_documents(workspace_name: str = "default", session: Session = None) -> int:
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
    return seg.id

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
