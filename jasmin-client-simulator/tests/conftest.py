import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.database.base import Base
from app.database import models as _models  # noqa: F401 - ensure models registered
from app.customers.default_personas import DEFAULT_PERSONAS
from app.database.models import CustomerProfileModel

# Use StaticPool so all connections share the same in-memory database
test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, bind=test_engine
)


@pytest.fixture(scope="function", autouse=True)
def override_llm_provider(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()

    # Seed default personas
    for persona in DEFAULT_PERSONAS:
        data = persona.model_dump()
        data["travel_dates"] = persona.travel_dates.model_dump()
        db.add(CustomerProfileModel(**data))
    
    # Seed default user
    from app.database.models import UserModel
    from app.core.security import (
        DEFAULT_USER_EMAIL,
        DEFAULT_USER_PASSWORD,
        DEFAULT_USER_NAME,
        DEFAULT_USER_ROLE,
        hash_password
    )
    admin_user = UserModel(
        email=DEFAULT_USER_EMAIL,
        hashed_password=hash_password(DEFAULT_USER_PASSWORD),
        full_name=DEFAULT_USER_NAME,
        role=DEFAULT_USER_ROLE
    )
    db.add(admin_user)
    db.add(UserModel(
        email="rep@simulator.com",
        hashed_password=hash_password("RepPass123!"),
        full_name="Sales Rep",
        role="sales_rep"
    ))
    db.commit()
    from app.customers.accounts import backfill_client_accounts
    backfill_client_accounts(db)  # every persona has its client login, as init_db does

    yield db

    db.close()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session, monkeypatch):
    from app.main import app
    from app.database.session import get_db
    from app.conversation.voice_service import voice_service

    # Keep tests offline and away from the real simulator.db
    monkeypatch.setattr("app.main.init_db", lambda: None)
    monkeypatch.setattr(voice_service, "synthesize_speech_base64", lambda text, lang="fr": "")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()
