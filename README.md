# SRED.ai — Automated SR&ED Evidence Pipeline

An end-to-end system that transforms raw technical documentation into audit-ready SR&ED compliance reports. The pipeline ingests documents, classifies evidence using LLM routing, clusters related work into projects, and generates legal narratives for the three SR&ED criteria.

## Overview

```
Raw Docs → Ingestion → Segmentation → LLM Routing → Clustering → Narrative → Compliance Report
```

**Output:** A Markdown report (`FINAL_CLAIM.md`) containing:
- Project titles derived from clustered evidence
- Three SR&ED compliance sections per project:
  1. **Technological Uncertainty** — What standard approaches failed?
  2. **Technological Advancement** — What new knowledge was generated?
  3. **Systematic Investigation** — What experiments were performed?
- Evidence citations linking back to source files

---

## Core Principles

| Principle | Description |
|-----------|-------------|
| **Containment** | All content starts in `QUARANTINE`. Nothing becomes evidence until explicitly routed. |
| **Provenance** | Every state change is logged in an append-only `SegmentDecisionLog` with actor, reasoning, and timestamp. |
| **Dual-Router** | A `RouterStub` (rule-based) and `RouterLLM` (GPT-4o-mini) can run in parallel for validation. |
| **Vector Clustering** | Technical segments are grouped by semantic similarity using ChromaDB embeddings + AgglomerativeClustering. |

---

## Pipeline Stages

### 1. Ingestion
**Service:** `src/sredi/services/ingestion.py`

- Recursively scans directories for `.md`, `.txt`, `.rst`, `.yaml` files
- Computes SHA256 hash for deduplication
- Creates `Document` records linked to a `Workspace`

### 2. Segmentation
**Service:** `src/sredi/services/segmentation.py`

- Breaks documents into atomic `DocSegment` units
- Supports hierarchical segmentation (Markdown headers → paragraphs)
- Stores context windows (`context_before`, `context_after`) for LLM routing
- Embeds segments into ChromaDB via `VectorStoreService`

### 3. Routing (Classification)
**Service:** `src/sredi/services/router.py`, `router_llm.py`

- **RouterStub:** Rule-based classifier using keyword detection
- **RouterLLM:** GPT-4o-mini with structured JSON output
- **Classification Labels:** `TECHNICAL`, `AMBIGUOUS`, `NOISE`
- **Processing States:** `QUARANTINE` → `INDEX_READY` | `REVIEW` | `NOISE`
- **Shadow Mode:** Run LLM alongside stub without committing decisions (for validation)

### 4. Clustering
**Service:** `src/sredi/services/clustering.py`

- Fetches embeddings from ChromaDB for `TECHNICAL` segments
- Runs `AgglomerativeClustering` (cosine metric, distance_threshold=1.0)
- Creates `WorkCluster` records grouping related evidence

### 5. Narrative Generation
**Service:** `src/sredi/services/narrative.py`

- Generates descriptive titles for each cluster using LLM
- Summarizes the technical work represented by grouped segments

### 6. Compliance Mapping
**Service:** `src/sredi/services/compliance.py`

- Maps each cluster to the three SR&ED criteria via LLM
- Populates `tech_uncertainty`, `tech_advancement`, `systematic_investigation` fields
- Uses a specialized legal prompt to ensure compliance language

### 7. Report Generation
**Scripts:** `scripts/generate_final_claim.py`, `scripts/test_full_pipeline.py`

- Outputs `audit_reports/FINAL_CLAIM.md` or `TEST_DATA_CLAIM.md`
- Includes evidence citations with source file paths

---

## Project Structure

```
src/sredi/
├── main.py                 # CLI entry point
├── config.py               # Settings (DATABASE_URL, OPENAI_API_KEY, etc.)
├── db.py                   # SQLModel engine setup
├── models/
│   ├── models.py           # ORM: Workspace, Document, DocSegment, WorkCluster, etc.
│   └── enums.py            # ProcessingState, ClassificationLabel
└── services/
    ├── ingestion.py        # File ingestion
    ├── segmentation.py     # Document → Segments
    ├── router.py           # Routing orchestrator
    ├── router_llm.py       # LLM-based classifier
    ├── clustering.py       # Semantic clustering
    ├── narrative.py        # Title generation
    ├── compliance.py       # SR&ED criteria mapping
    ├── vector_store.py     # ChromaDB integration
    └── llm_client.py       # OpenAI/Gemini client

scripts/
├── test_full_pipeline.py   # Full E2E test on ./test_data
├── generate_final_claim.py # Generate compliance report
├── run_phase7_full.py      # Phase 7 demo script
└── run_tournament.py       # Stub vs LLM validation

audit_reports/              # Generated Markdown reports
alembic/                    # Database migrations
```

---

## Data Models

| Model | Description |
|-------|-------------|
| `Workspace` | Isolation boundary for documents and segments |
| `Document` | Physical file record (SHA256 hash, file path) |
| `DocSegment` | Atomic evidence unit with `processing_state` and `classification_label` |
| `WorkCluster` | Grouped segments representing a "Work Item" with compliance fields |
| `SegmentDecisionLog` | Append-only audit trail of routing decisions |
| `EntityAnchor` | Extracted entities (dates, identifiers) from segments |
| `Project` | External anchor (Jira Epic, Git repo) |
| `ProjectSegmentLink` | Edge connecting evidence to a project |

### Segment States

```
QUARANTINE ──┬──> INDEX_READY (proven technical)
             ├──> REVIEW (technical but unproven)
             └──> NOISE (filtered out)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (PostgreSQL + pgvector)
- OpenAI API key (for LLM routing)

### Setup

```bash
# 1. Start database
docker compose up -d

# 2. Install dependencies
pip install -e .

# 3. Set environment variables
export DATABASE_URL="postgresql://video_user:video_password@localhost:5433/sred_db"
export OPENAI_API_KEY="sk-..."

# 4. Initialize schema
python -m src.sredi.main setup
```

### Run Full Pipeline (Recommended)

```bash
# Process ./test_data and generate compliance report
PYTHONPATH=src python scripts/test_full_pipeline.py
```

**Output:** `audit_reports/TEST_DATA_CLAIM.md`

### Step-by-Step CLI

```bash
# 1. Ingest documents
python -m src.sredi.main ingest ./my_docs

# 2. Segment into units
python -m src.sredi.main segment

# 3. Route (classify) segments
python -m src.sredi.main route --router-type llm

# 4. Check status
python -m src.sredi.main status
```

### Reset (Dev Only)

```bash
python -m src.sredi.main reset --hard
```

---

## Configuration

Environment variables (or `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OPENAI_API_KEY` | OpenAI API key for LLM routing | Required for LLM |
| `ROUTER_TYPE` | `stub` or `llm` | `stub` |
| `SHADOW_MODE` | Run LLM without committing | `false` |
| `LLM_MODEL` | Model for routing | `gpt-4o-mini` |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_full_pipeline.py` | Full E2E test: clean → ingest → segment → route → cluster → narrative → compliance → report |
| `scripts/generate_final_claim.py` | Generate compliance report from existing clusters |
| `scripts/run_tournament.py` | Compare RouterStub vs RouterLLM decisions |
| `scripts/run_phase7_full.py` | Phase 7 demo (clustering + narrative) |

---

## Example Output

```markdown
# SR&ED Technical Narrative Report

# Project: CSINode Integration and Experimentation

*166 evidence segments*

## 1. Technological Uncertainty
The project faced technological uncertainty due to the need to create or extend 
the existing `StorageInfos` interface to enable both the scheduler and CAS to 
work with the previously created fake `CSINode` objects...

## 2. Technological Advancement
The project generated new knowledge by proposing enhancements to the 
`SetClusterState` function to capture and add `CSINode` information...

## 3. Systematic Investigation
A systematic investigation was conducted through a series of experiments 
(EXP-001, EXP-002, EXP-003) that tested various changes to the system...

### Evidence Used
- **[test_data/sig-autoscaling__5030-attach-limit-autoscaler__README.md]**: ...
- **[test_data/product_1.md]**: ...
```

---

## Development

### Database Migrations

```bash
# Generate migration after model changes
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

### Testing

```bash
# Run full pipeline test
PYTHONPATH=src python scripts/test_full_pipeline.py

# Verify output
cat audit_reports/TEST_DATA_CLAIM.md
```

---

## Architecture Decisions

1. **ChromaDB for embeddings** — Local vector store, no external dependencies
2. **SQLModel ORM** — Type-safe models with SQLAlchemy backend
3. **Async routing** — Concurrent LLM calls for throughput
4. **Append-only logs** — Full audit trail for compliance
5. **Hierarchical segmentation** — Preserves document structure for context
