from typing import Optional
import uuid
from sqlmodel import Session, select
from ..models import WorkCluster, DocSegment
from .llm_client import LLMClient

SRED_SYSTEM_PROMPT = """You are an expert SR&ED Tax Consultant.
Map the following technical evidence into the 3 mandatory criteria for an R&D claim.

CRITERIA DEFINITIONS:
1. Technological Uncertainty: What standard approaches failed? What complex constraints conflicted? (Do NOT describe business risks).
2. Technological Advancement: What new knowledge did you generate? (Do NOT describe the product features).
3. Systematic Investigation: What experiments, testing, or analysis were performed?

OUTPUT FORMAT (JSON):
{
  "uncertainty": "...",
  "advancement": "...",
  "investigation": "..."
}
"""


class ComplianceService:
    def __init__(self):
        self.llm = LLMClient()

    async def generate_compliance_narrative(self, session: Session, workspace_id: uuid.UUID):
        """Generates SR&ED compliance narratives for all WorkClusters in the workspace.
        
        Args:
            session: DB Session.
            workspace_id: ID of the workspace.
        """
        print(f"⚖️ Generating compliance narratives for workspace {workspace_id}...")
        
        clusters = session.exec(
            select(WorkCluster).where(WorkCluster.workspace_id == workspace_id)
        ).all()
        
        if not clusters:
            print("No clusters found.")
            return

        print(f"Found {len(clusters)} clusters to process.")
        
        for cluster in clusters:
            # Skip if already has compliance data
            if cluster.tech_uncertainty and cluster.tech_advancement and cluster.systematic_investigation:
                print(f"Cluster {cluster.id} already has compliance data. Skipping.")
                continue
                
            # Fetch up to 10 segments for context
            segments = session.exec(
                select(DocSegment)
                .where(DocSegment.cluster_id == cluster.id)
                .limit(10)
            ).all()
            
            if not segments:
                print(f"Cluster {cluster.id} has no segments. Skipping.")
                continue
                
            # Prepare evidence snippets
            evidence_snippets = []
            for s in segments:
                content_preview = s.content[:800] + "..." if len(s.content) > 800 else s.content
                evidence_snippets.append(f"--- Evidence Segment ---\n{content_preview}")
            
            evidence_text = "\n\n".join(evidence_snippets)
            
            user_prompt = f"""
PROJECT: {cluster.title}

TECHNICAL EVIDENCE:
{evidence_text}

Based on the above evidence, generate the SR&ED compliance mapping.
"""
            
            try:
                # Use classify_segment which returns JSON
                response = await self.llm.classify_segment(
                    segment_text=user_prompt,
                    metadata={"system_prompt": SRED_SYSTEM_PROMPT}
                )
                
                cluster.tech_uncertainty = response.get("uncertainty", "")
                cluster.tech_advancement = response.get("advancement", "")
                cluster.systematic_investigation = response.get("investigation", "")
                
                print(f"✅ Cluster '{cluster.title}' -> Compliance mapped")
                session.add(cluster)
                session.commit()
                
            except Exception as e:
                print(f"❌ Error generating compliance for cluster {cluster.id}: {e}")
