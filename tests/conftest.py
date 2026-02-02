import pytest
from sqlmodel import SQLModel, create_engine, Session
from sredi.db import get_session
from sredi.models.models import Workspace

@pytest.fixture(name="session")
def session_fixture():
    # Use a check_same_thread=False for sqlite if needed, but here standard is fine.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    # Explicitly dispose the engine to close all connections and avoid ResourceWarnings
    engine.dispose()

@pytest.fixture(name="test_workspace")
def workspace_fixture(session: Session):
    workspace = Workspace(name="test_workspace")
    session.add(workspace)
    session.commit()
    session.refresh(workspace)
    return workspace
