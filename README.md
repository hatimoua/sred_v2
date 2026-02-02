# SRED.ai MVP — Local Evidence Brain

A strict, CLI-first pipeline for processing, validating, and grouping technical evidence for SR&ED/T661 claims.

## Philosophy
This system is built on three core principles:
1.  **Containment**: All new content starts in **QUARANTINE**. Nothing is treated as evidence until explicitly routed.
2.  **Provenance**: Every state change (Quarantine → Index Ready) is logged in an append-only `SegmentDecisionLog`.
3.  **Anchors**: Evidence is grouped deterministically against known "Anchors" (Jira Epics, Git repositories) rather than unsupervised clustering.

## Pipeline Architecture

The pipeline runs locally via the `sredi` CLI. Data flows through four distinct stages:

### 1. Ingestion (`sredi ingest <path>`)
*   **Goal**: Securely store raw files and prevent duplicates.
*   **Process**:
    1.  Recursively scans the target directory.
    2.  Computes a SHA256 hash of the raw file content.
    3.  Checks the database for existing hashes within the Workspace.
    4.  **Deduplication**: If the hash exists, the file is skipped. Use `status` to see skip counts.
    5.  Creates a `Document` record linked to a `PipelineRun` for auditability.

### 2. Segmentation (`sredi segment`)
*   **Goal**: Break documents into atomic, independently referencable units ("DocSegments").
*   **Process**:
    1.  Finds `Document`s that have not yet been segmented.
    2.  Reads content (currently supports `.txt`, `.md`, `.rst`).
    3.  Splits content by paragraph (double newline `\n\n`).
    4.  **Quarantine**: Every new segment is created with `processing_state="QUARANTINE"`.
    5.  Segments are stored with their source `Document` ID.

### 3. Routing (`sredi route`)
*   **Goal**: Identify technical evidence and filter out noise.
*   **Process**:
    1.  Fetches a batch of `QUARANTINE` segments.
    2.  **Router Logic** (Current Stub):
        *   Scans for technical keywords (e.g., "error", "latency", "schema", "architecture").
        *   If keywords are found → Recommends `INDEX_READY` (Confidence: 0.9).
        *   If no keywords → Recommends `NOISE` (Confidence: 0.5).
    3.  **Provenance**: A `SegmentDecisionLog` entry is written *before* the state is updated, recording:
        *   Old State (`QUARANTINE`)
        *   New State (`INDEX_READY` / `NOISE`)
        *   Actor (`RouterStub`)
        *   Reasoning (Keywords found)
    4.  Updates the `DocSegment` state.

### 4. Grouping (`sredi group`)
*   **Goal**: Link proven evidence to specific Projects.
*   **Process**:
    1.  **Load Anchors**: First, use `sredi anchors load <csv>` to define Projects (e.g., "Project Alpha", "JIRA-123").
    2.  Fetches `INDEX_READY` segments.
    3.  **Strong Linking**: Checks if the segment content *explicitly contains* a Project's `source_anchor` string.
    4.  If a match is found, creates a `ProjectSegmentLink` with `type="STRONG_ANCHOR"` and `confidence=1.0`.

## Usage Guide

### Prerequisites
- Python 3.10+
- Docker (for PostgreSQL + pgvector)

### Setup
1.  Start the database:
    ```bash
    docker compose up -d
    ```
2.  Initialize the schema:
    ```bash
    # Ensure .env has DATABASE_URL=postgresql://video_user:video_password@localhost:5433/sred_db
    python -m src.sredi.main setup
    ```

### Running the Pipeline
```bash
# 1. Ingest files
python -m src.sredi.main ingest ./my_docs

# 2. Segment
python -m src.sredi.main segment

# 3. Route (Classify)
python -m src.sredi.main route

# 4. Load Anchors
python -m src.sredi.main anchors load ./anchors.csv

# 5. Group
python -m src.sredi.main group

# 6. Check Status
python -m src.sredi.main status
```

### Resetting
To wipe the database and start over (Dev only):
```bash
python -m src.sredi.main reset --hard
```

## Data Schema
The system uses **SQLModel** (SQLAlchemy) with `pgvector` support.
*   `Document`: Physical file record (SHA256 hash).
*   `DocSegment`: The atomic unit of evidence (Status: QUARANTINE/index_ready/NOISE).
*   `SegmentDecisionLog`: Append-only audit trail of routing decisions.
*   `Project`: The "Anchor" (e.g., Jira Epic).
*   `ProjectSegmentLink`: The edge connecting evidence to a project.
