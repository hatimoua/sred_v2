import os
import asyncio
from pathlib import Path
from sqlmodel import Session, select
from sredi.db import engine
from sredi.services.compliance import ComplianceService
from sredi.models import WorkCluster, DocSegment, Workspace

# Configuration
REPORT_DIR = "./audit_reports"
REPORT_FILE = os.path.join(REPORT_DIR, "FINAL_CLAIM.md")


async def generate_compliance_narratives():
    """Generates SR&ED compliance narratives for all clusters."""
    print("⚖️ Starting Compliance Narrative Generation...")
    
    with Session(engine) as session:
        ws = session.exec(select(Workspace).where(Workspace.name == "default")).first()
        if not ws:
            print("Error: Default workspace not found.")
            return
            
        compliance_service = ComplianceService()
        await compliance_service.generate_compliance_narrative(session, ws.id)


def generate_markdown_report():
    """Generates the final Markdown claim document."""
    print("\n📄 Generating FINAL_CLAIM.md...")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    with Session(engine) as session:
        ws = session.exec(select(Workspace).where(Workspace.name == "default")).first()
        if not ws:
            print("Error: Default workspace not found.")
            return
            
        clusters = session.exec(
            select(WorkCluster).where(WorkCluster.workspace_id == ws.id)
        ).all()
        
        if not clusters:
            print("No clusters found.")
            return
        
        # Sort by number of segments (largest first)
        clusters.sort(key=lambda c: len(c.segments), reverse=True)
        
        markdown_lines = [
            "# SR&ED Technical Narrative Report",
            "",
            f"*Generated for workspace: {ws.name}*",
            "",
            "---",
            ""
        ]
        
        for cluster in clusters:
            # Fetch segments for evidence list
            segments = session.exec(
                select(DocSegment)
                .where(DocSegment.cluster_id == cluster.id)
                .limit(10)
            ).all()
            
            markdown_lines.append(f"# Project: {cluster.title}")
            markdown_lines.append("")
            markdown_lines.append(f"*{len(cluster.segments)} evidence segments*")
            markdown_lines.append("")
            
            # Uncertainty
            markdown_lines.append("## 1. Technological Uncertainty")
            markdown_lines.append("")
            uncertainty = cluster.tech_uncertainty or "*Not yet generated*"
            markdown_lines.append(uncertainty)
            markdown_lines.append("")
            
            # Advancement
            markdown_lines.append("## 2. Technological Advancement")
            markdown_lines.append("")
            advancement = cluster.tech_advancement or "*Not yet generated*"
            markdown_lines.append(advancement)
            markdown_lines.append("")
            
            # Investigation
            markdown_lines.append("## 3. Systematic Investigation")
            markdown_lines.append("")
            investigation = cluster.systematic_investigation or "*Not yet generated*"
            markdown_lines.append(investigation)
            markdown_lines.append("")
            
            # Evidence Used
            markdown_lines.append("### Evidence Used")
            markdown_lines.append("")
            for seg in segments:
                # Truncate for readability
                preview = seg.content[:150].replace("\n", " ").strip()
                if len(seg.content) > 150:
                    preview += "..."
                markdown_lines.append(f"- {preview}")
            markdown_lines.append("")
            markdown_lines.append("---")
            markdown_lines.append("")
        
        # Write file
        report_content = "\n".join(markdown_lines)
        with open(REPORT_FILE, "w") as f:
            f.write(report_content)
        
        print(f"✅ Report saved to: {REPORT_FILE}")
        print(f"   Clusters documented: {len(clusters)}")


async def main():
    await generate_compliance_narratives()
    generate_markdown_report()


if __name__ == "__main__":
    asyncio.run(main())
