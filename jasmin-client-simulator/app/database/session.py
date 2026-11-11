import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.database.base import Base

logger = logging.getLogger("simulator")

# sqlite connection setting connect_args for multithread access if sqlite
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from app.database import models  # noqa: F401
    from app.customers.default_personas import DEFAULT_PERSONAS
    from app.database.models import CustomerProfileModel, UserModel
    from app.core.security import (
        DEFAULT_USER_EMAIL,
        DEFAULT_USER_PASSWORD,
        DEFAULT_USER_NAME,
        DEFAULT_USER_ROLE,
        hash_password
    )
    from sqlalchemy import inspect, text

    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)

    # Perform column migration for existing SQLite DBs if columns were added later
    with engine.connect() as conn:
        inspector = inspect(engine)
        if "customer_profiles" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("customer_profiles")]
            if "archetype" not in columns:
                logger.info("Migrating customer_profiles table: adding 'archetype' column...")
                conn.execute(text("ALTER TABLE customer_profiles ADD COLUMN archetype VARCHAR(50) DEFAULT 'satisfied' NOT NULL"))
                conn.commit()
            if "budget_currency" not in columns:
                # One-time move of budgets from USD to TND (the app now shows and reasons in dinars only)
                logger.info("Migrating customer_profiles: budgets USD -> TND...")
                conn.execute(text("ALTER TABLE customer_profiles ADD COLUMN budget_currency VARCHAR(3) DEFAULT 'USD' NOT NULL"))
                conn.execute(text("UPDATE customer_profiles SET budget_total = ROUND(budget_total * :rate) "
                                  "WHERE budget_currency = 'USD' AND scenario_id LIKE 'twin_%'"), {"rate": settings.TND_PER_USD})
                conn.execute(text("UPDATE customer_profiles SET budget_currency = 'TND'"))
                conn.commit()
            if "human_profile" not in columns:
                logger.info("Migrating customer_profiles table: adding 'human_profile' column...")
                conn.execute(text("ALTER TABLE customer_profiles ADD COLUMN human_profile JSON"))
                conn.commit()
        if "simulations" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("simulations")]
            if "language" not in columns:
                logger.info("Migrating simulations table: adding 'language' column...")
                conn.execute(text("ALTER TABLE simulations ADD COLUMN language VARCHAR(5) DEFAULT 'fr' NOT NULL"))
                conn.commit()
            for name, ddl in [("scenario", "VARCHAR(40) DEFAULT 'premiere_demande' NOT NULL"),
                              ("difficulty", "VARCHAR(20) DEFAULT 'normal' NOT NULL"), ("outcome", "VARCHAR(20)"),
                              ("outcome_reason", "TEXT"), ("decision_state", "JSON"), ("debrief", "JSON")]:
                if name not in columns:
                    logger.info(f"Migrating simulations table: adding '{name}' column...")
                    conn.execute(text(f"ALTER TABLE simulations ADD COLUMN {name} {ddl}"))
                    conn.commit()
        if "users" in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns("users")]
            if "profile_id" not in columns:
                logger.info("Migrating users table: adding 'profile_id' column...")
                conn.execute(text("ALTER TABLE users ADD COLUMN profile_id VARCHAR(36) REFERENCES customer_profiles(id)"))
                conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_profile_id ON users (profile_id)"))
                conn.commit()

    db = SessionLocal()
    try:
        # Seed default user if not present
        existing_user = db.query(UserModel).filter(UserModel.email == DEFAULT_USER_EMAIL).first()
        if not existing_user:
            logger.info(f"Seeding default user ({DEFAULT_USER_EMAIL})...")
            admin_user = UserModel(
                email=DEFAULT_USER_EMAIL,
                hashed_password=hash_password(DEFAULT_USER_PASSWORD),
                full_name=DEFAULT_USER_NAME,
                role=DEFAULT_USER_ROLE
            )
            db.add(admin_user)
            db.commit()
            logger.info("Default user successfully seeded.")

        existing_count = db.query(CustomerProfileModel).count()
        if existing_count == 0:
            logger.info("Seeding initial customer profiles into customer_profiles table...")
            for persona in DEFAULT_PERSONAS:
                data = persona.model_dump()
                data["travel_dates"] = persona.travel_dates.model_dump()
                db.add(CustomerProfileModel(**data))
            db.commit()
            logger.info(f"Successfully seeded {len(DEFAULT_PERSONAS)} customer profiles.")
        else:
            # Synchronize standard default profiles with latest attributes
            for persona in DEFAULT_PERSONAS:
                prof = db.query(CustomerProfileModel).filter(CustomerProfileModel.scenario_id == persona.scenario_id).first()
                data = persona.model_dump()
                data["travel_dates"] = persona.travel_dates.model_dump()
                if prof:
                    for k, v in data.items():
                        setattr(prof, k, v)
                else:
                    db.add(CustomerProfileModel(**data))
            db.commit()

        from app.customers.accounts import backfill_client_accounts
        created = backfill_client_accounts(db)
        if created:
            logger.info(f"Created {created} client logins for the personas.")

    except Exception as e:
        logger.error(f"Error seeding database: {str(e)}")
        db.rollback()
    finally:
        db.close()
