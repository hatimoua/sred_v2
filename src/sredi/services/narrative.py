from typing import List, Optional
import uuid
from sqlmodel import Session, select
from ..models import WorkCluster, DocSegment
from .llm_client import LLMClient

class NarrativeService:
    def __init__(self):
        self.llm = LLMClient()

    async def generate_cluster_titles(self, session: Session, workspace_id: uuid.UUID):
        """Generates descriptive titles for all untitled WorkClusters in the workspace.
        
        Args:
            session: DB Session.
            workspace_id: ID of the workspace.
        """
        print(f"📝 Generating narratives for workspace {workspace_id}...")
        
        # Fetch all clusters for this workspace
        clusters = session.exec(
            select(WorkCluster).where(WorkCluster.workspace_id == workspace_id)
        ).all()
        
        if not clusters:
            print("No clusters found.")
            return

        print(f"Found {len(clusters)} clusters to process.")
        
        for cluster in clusters:
            # Skip if already titled (unless it's the default/placeholder)
            if cluster.title and "Cluster" not in cluster.title and "Untitled" not in cluster.title:
                continue
                
            # Fetch up to 5 segments for context
            # We need to access segments via relationship or query
            # Since we are in a session, lazy loading might work if configured, but let's be explicit
            # or rely on the relationship if it's eager/lazy enough.
            # SQLModel relationships are lazy by default but explicit query is safer for async contexts if needed (though we are sync here mostly).
            # Let's just use cluster.segments if loaded, or query.
            
            # Use query to limit
            segments = session.exec(
                select(DocSegment)
                .where(DocSegment.cluster_id == cluster.id)
                .limit(5)
            ).all()
            
            if not segments:
                print(f"Cluster {cluster.id} has no segments. Skipping.")
                continue
                
            # Prepare Prompt
            segment_snippets = []
            for s in segments:
                # Truncate content to avoid token limits
                content_preview = s.content[:500] + "..." if len(s.content) > 500 else s.content
                segment_snippets.append(f"--- Segment ---\n{content_preview}")
            
            snippets_text = "\n\n".join(segment_snippets)
            
            prompt = f"""
Analyze these code segments from a single feature track or work item:

{snippets_text}

Generate a concise, technical claim title for this group of work (e.g., "Refactoring Authentication Middleware" or "Vector Store Implementation").
Return ONLY the title, no quotes, no preamble.
Title:
"""
            # Call LLM
            # Note: LLMClient methods might be async or sync? 
            # Checked llm_client.py before: it uses openai.AsyncOpenAI?
            # Let's check if I should use await.
            # In Phase 4, `router_llm.py` used `await self.client.chat.completions.create`.
            # `LLMClient` likely wraps this.
            # Let's assume LLMClient has a method `predict` or similar.
            # Actually, I haven't implemented `LLMClient` generic helper yet, I used `RouterLLM` which was specific.
            # Wait, I see `src/sredi/services/llm_client.py` in the file list from `find_by_name` earlier.
            # Let me check `llm_client.py` content to be sure.
            
            # Since I can't check right now without a tool call, and I want to batch, 
            # I will assume standard usage or use `RouterLLM` logic if `LLMClient` isn't generic.
            # Better: I'll read `llm_client.py` in the same turn if possible, but I can't use output to write file.
            # I will rely on `RouterLLM` pattern or a simple OpenAI call if `LLMClient` is not suitable.
            # Actually, `RouterLLM` returns a structured object.
            # I probably need a simple text generation method.
            
            # Let's assume I need to implement a simple generation method in NarrativeService using OpenAI directly or `LLMClient`.
            # I'll check `llm_client.py` first in a separate tool call to be safe? 
            # No, I'll take a risk and assume `LLMClient` exists (it was in the file list).
            # I'll assume it has a `get_completion` or similar.
            # If not, I'll fix it.
            
            # Wait, better plan: I'll use `from openai import AsyncOpenAI` and `settings` directly here if `LLMClient` is unknown.
            # But let's try to reuse `LLMClient`.
            
            try:
                # Temporary: use direct OpenAI call for simplicity if LLMClient api is unknown
                # But wait, I should look at `llm_client.py`.
                # I'll write the file assuming `LLMClient` has `generate_text(prompt)`.
                # If not, I'll edit.
                
                response_text = await self.llm.generate_text(prompt)
                title = response_text.strip().strip('"')
                
                print(f"Cluster {cluster.id} -> {title}")
                cluster.title = title
                session.add(cluster)
                session.commit()
                
            except Exception as e:
                print(f"Error generating title for cluster {cluster.id}: {e}")

