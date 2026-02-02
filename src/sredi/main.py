import typer
import subprocess
from pathlib import Path
from sqlmodel import select, delete, text, col, Session
from .db import engine, get_session
# Import models so SQLModel knows about them (though we use Alembic for migrations mostly)
from .models import * 
from .services import ingestion, segmentation, router as router_service, grouping, llm_client, router_llm
import asyncio
import json

app = typer.Typer(help="SRED.ai MVP - Local Evidence Brain")

@app.command()
def setup():
    """Runs Alembic migrations and ensures the database schema is up to date.

    This command applies all pending migrations to the database specified
    in the DATABASE_URL environment variable.
    """
    typer.echo("Running Alembic migrations...")
    try:
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        typer.echo("Migrations applied successfully.")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running migrations: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def ingest(directory: Path, workspace: str = "default"):
    """Ingests a directory of files into the evidence brain.

    Args:
        directory: The local directory path containing files to ingest.
        workspace: The name of the workspace to ingest into. Defaults to "default".
    """
    if not directory.exists():
        typer.echo(f"Directory not found: {directory}", err=True)
        raise typer.Exit(code=1)
    
    typer.echo(f"Ingesting {directory} into workspace '{workspace}'...")
    scanned, new_docs, skipped = ingestion.ingest_directory(directory, workspace)
    typer.echo(f"Ingestion complete.")
    typer.echo(f"  Scanned: {scanned}")
    typer.echo(f"  New:     {new_docs}")
    typer.echo(f"  Skipped: {skipped} (deduped)")

@app.command()
def segment(workspace: str = "default"):
    """Segments pending documents into atomic units (DocSegments).

    Args:
        workspace: The name of the workspace to process. Defaults to "default".
    """
    typer.echo(f"Segmenting documents in workspace '{workspace}'...")
    num_segments = segmentation.segment_documents(workspace)
    typer.echo(f"Segmentation complete. Created {num_segments} new segments.")

@app.command()
def route(
    limit: int = 100,
    router: str = typer.Option("stub", "--router", help="Router type: stub or llm"),
    shadow: bool = typer.Option(False, "--shadow/--no-shadow", help="Enable shadow mode (no state mutation)"),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrency for LLM routing")
):
    """Routes quarantined segments using specified router policy.

    Args:
        limit: Maximum number of segments to process in this run. Defaults to 100.
        router: Router policy to use (stub or llm).
        shadow: If True, logs decisions but does not update segment states.
        concurrency: Max concurrent LLM requests.
    """
    if router == "llm" and not shadow:
        typer.echo("Error: LLM router in 'Apply Mode' (--no-shadow) is not yet implemented.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Routing up to {limit} segments using {router} router (shadow={shadow})...")
    count = router_service.route_segments(
        limit=limit,
        router_type=router,
        shadow_mode=shadow,
        concurrency=concurrency
    )
    typer.echo(f"Routing complete. Processed {count} segments.")

# Sub-app for anchors
anchors_app = typer.Typer(help="Manage project anchors (DEPRECATED)")
app.add_typer(anchors_app, name="anchors")

@anchors_app.command("load")
def load_anchors(file: Path, workspace: str = "default"):
    """[DEPRECATED] Loads project anchors from a CSV file.
    
    This command is deprecated and will be removed in a future version.
    Grouping logic is moving towards leveraging Shadow-Mode Tournament results.
    """
    typer.echo("Warning: 'anchors load' is deprecated. Grouping will soon rely on tournament results.", err=True)
    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(code=1)
    
    count = grouping.load_anchors_from_csv(file, workspace)
    typer.echo(f"Loaded {count} new anchors.")

@app.command()
def group(
    workspace: str = "default",
    tournament: bool = typer.Option(True, "--tournament/--no-tournament", help="Leverage Shadow-Mode Tournament results for grouping")
):
    """Groups linked evidence to projects.

    By default, leverages the latest Shadow-Mode Tournament results to identify 
    high-confidence technical evidence and project associations.

    Args:
        workspace: The name of the workspace to process. Defaults to "default".
        tournament: Whether to use tournament results.
    """
    typer.echo(f"Grouping segments in workspace '{workspace}' (tournament={tournament})...")
    count = grouping.group_segments(workspace, use_tournament=tournament)
    typer.echo(f"Grouping complete. Created {count} links.")

@app.command()
def status(workspace: str = "default"):
    """Shows detailed system status for a workspace: counts of docs, segments by state, projects, and links.

    Args:
        workspace: The name of the workspace to show status for. Defaults to "default".
    """
    session_gen = get_session()
    session = next(session_gen)
    try:
        from .services.ingestion import get_or_create_workspace
        ws = get_or_create_workspace(session, workspace)

        # 1. Documents Stats
        docs = session.exec(select(Document).where(Document.workspace_id == ws.id)).all()
        last_ingest_run = session.exec(
            select(PipelineRun)
            .where(PipelineRun.workspace_id == ws.id, PipelineRun.command == "ingest")
            .order_by(col(PipelineRun.timestamp).desc())
        ).first()
        
        new_last_ingest = 0
        skipped_last_ingest = 0
        if last_ingest_run:
            new_last_ingest = last_ingest_run.parameters.get("new", 0)
            skipped_last_ingest = last_ingest_run.parameters.get("skipped", 0)

        # 2. Segments Stats
        segs_stmt = select(DocSegment).join(Document).where(Document.workspace_id == ws.id)
        segs = session.exec(segs_stmt).all()
        states = {state: 0 for state in ProcessingState}
        for s in segs:
            states[s.processing_state] += 1

        # 3. Decisions Stats
        decisions_stmt = select(SegmentDecisionLog).join(DocSegment).join(Document).where(Document.workspace_id == ws.id)
        decisions = session.exec(decisions_stmt).all()
        actors = {"RouterStub": 0, "Human": 0, "GPT": 0}
        for d in decisions:
            actors[d.actor] = actors.get(d.actor, 0) + 1

        # 4. Projects & Links Stats
        projs = session.exec(select(Project).where(Project.workspace_id == ws.id)).all()
        links_stmt = select(ProjectSegmentLink).join(Project).where(Project.workspace_id == ws.id)
        links = session.exec(links_stmt).all()
        
        index_ready_segs = [s for s in segs if s.processing_state == ProcessingState.INDEX_READY]
        linked_index_ready_ids = {link.segment_id for link in links}
        num_linked_index_ready = len([s for s in index_ready_segs if s.id in linked_index_ready_ids])
        num_unlinked_index_ready = len(index_ready_segs) - num_linked_index_ready

        # Readable Output
        typer.echo(f"\n- Workspace: {workspace}")
        typer.echo(f"- Docs: {len(docs)} (new last ingest: {new_last_ingest}, deduped: {skipped_last_ingest})")
        typer.echo(f"- Segments: {len(segs):,}")
        for state in ProcessingState:
            typer.echo(f"  - {state.value}: {states[state]}")
        
        actor_str = ", ".join([f"{k}: {v}" for k, v in actors.items()])
        typer.echo(f"- Decisions: {len(decisions)} ({actor_str})")
        typer.echo(f"- Projects: {len(projs)}")
        typer.echo(f"- Links: {len(links)} (INDEX_READY linked: {num_linked_index_ready}, INDEX_READY unlinked: {num_unlinked_index_ready})")
        
        # Optional Top Projects (if helpful)
        project_link_counts = {}
        for link in links:
            proj_name = link.project.name
            project_link_counts[proj_name] = project_link_counts.get(proj_name, 0) + 1
        top_projects = sorted(project_link_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_projects:
            typer.echo("\nTop Projects:")
            for name, count in top_projects:
                typer.echo(f"  - {name}: {count} segments")
    finally:
        session.close()

def reset_db(session: Session = None):
    """Wipes all data from the database.
    
    Args:
        session: Optional session. If None, a new one is created.
    """
    if session is None:
        session_gen = get_session()
        session = next(session_gen)
        should_close = True
    else:
        should_close = False
        
    try:
        # Delete in order of dependencies
        session.exec(delete(ProjectSegmentLink))
        session.exec(delete(EntityAnchor))
        session.exec(delete(SegmentDecisionLog))
        session.exec(delete(DocSegment))
        session.exec(delete(Document))
        session.exec(delete(Project))
        session.exec(delete(PipelineRun))
        session.exec(delete(Workspace))
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        if should_close:
            session.close()

@app.command()
def reset(hard: bool = typer.Option(False, "--hard", help="Wipe all data")):
    """Resets the workspace or wipes the entire database.

    Args:
        hard: If True, wipes all data from the database. Use with caution.
    """
    if hard:
        typer.confirm("Are you sure you want to WIPE the database? This cannot be undone.", abort=True)
        try:
            reset_db()
            typer.echo("Database wiped successfully.")
        except Exception as e:
            typer.echo(f"Error wiping database: {e}", err=True)
    else:
        typer.echo("Use --hard to wipe data. This command does nothing without it.")

@app.command()
def llm_route_sample(file: Path):
    """Dry-run LLM routing on a local text file.
    
    This command does NOT mutate the database. It is used to verify LLM classification
    and parsing logic on a sample piece of evidence.
    """
    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(code=1)
    
    content = file.read_text()
    typer.echo(f"Routing content from {file} via LLM...")
    
    async def run_routing():
        client = llm_client.LLMClient()
        result = await router_llm.llm_route_segment(client, segment_text=content)
        return result

    result = asyncio.run(run_routing())
    
    typer.echo("\n--- Router Result ---")
    typer.echo(json.dumps(result.model_dump(), indent=2))
    typer.echo("----------------------")

if __name__ == "__main__":
    app()
