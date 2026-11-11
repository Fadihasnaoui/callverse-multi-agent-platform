import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database.base import Base


def generate_uuid():
    return str(uuid.uuid4())


def utcnow():
    return datetime.now(timezone.utc)


class CustomerProfileModel(Base):
    __tablename__ = "customer_profiles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    scenario_id = Column(String(64), nullable=False, default="default_scenario")
    archetype = Column(String(50), nullable=False, default="satisfied") # satisfied | unsatisfied | hesitant
    name = Column(String(100), nullable=False)
    age_range = Column(String(50), nullable=False)
    persona_description = Column(Text, nullable=False)

    # Destination & Travel Details
    destination_preferences = Column(JSON, nullable=False)
    travel_dates = Column(JSON, nullable=False)
    number_of_travelers = Column(Integer, nullable=False)

    # Budget & Financial Sensitivity
    budget_total = Column(Float, nullable=False)  # TND, whole trip
    budget_currency = Column(String(3), nullable=False, default="TND")  # "USD" only on rows awaiting the one-time conversion
    price_sensitivity = Column(String(50), nullable=False)

    # Accommodation & Travel Preferences
    preferred_accommodation = Column(Text, nullable=False)
    preferred_transportation = Column(Text, nullable=False)

    # Motivations & Fixed Constraints
    motivations = Column(JSON, nullable=False)
    constraints = Column(JSON, nullable=False)

    # Personality & Communication Style
    personality_traits = Column(JSON, nullable=False)
    communication_style = Column(Text, nullable=False)

    # Objections & Decision Criteria
    objections = Column(JSON, nullable=False)
    decision_criteria = Column(JSON, nullable=False)

    # Baseline Psychological Metrics
    initial_trust = Column(Float, nullable=False, default=5.0)
    initial_interest = Column(Float, nullable=False, default=5.0)
    initial_booking_intention = Column(Float, nullable=False, default=3.0)

    desired_outcome = Column(Text, nullable=False)
    human_profile = Column(JSON, nullable=True)  # app/customers/human.py
    created_at = Column(DateTime, default=utcnow)

    # Relationships
    simulations = relationship(
        "SimulationModel", back_populates="profile",
        cascade="all, delete-orphan", lazy="select"
    )


class SimulationModel(Base):
    __tablename__ = "simulations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    profile_id = Column(String(36), ForeignKey("customer_profiles.id"), nullable=False)
    scenario_title = Column(String(200), nullable=False, default="Standard Sales Inquiry")

    # State tracking
    current_stage = Column(String(50), nullable=False, default="greetings")
    status = Column(String(50), nullable=False, default="active")
    language = Column(String(5), nullable=False, default="fr")  # language the client speaks: fr | en
    scenario = Column(String(40), nullable=False, default="premiere_demande")
    difficulty = Column(String(20), nullable=False, default="normal")
    outcome = Column(String(20), nullable=True)        # app/conversation/decision.py OUTCOMES
    outcome_reason = Column(Text, nullable=True)
    decision_state = Column(JSON, nullable=True)       # needs met, patience, mood, events, promises, intent log
    debrief = Column(JSON, nullable=True)

    trust_level = Column(Float, nullable=False, default=5.0)
    interest_level = Column(Float, nullable=False, default=5.0)
    booking_intention = Column(Float, nullable=False, default=3.0)

    known_requirements = Column(JSON, nullable=False, default=dict)
    questions_answered = Column(JSON, nullable=False, default=list)
    unresolved_objections = Column(JSON, nullable=False, default=list)
    provided_info = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    profile = relationship("CustomerProfileModel", back_populates="simulations", lazy="select")
    messages = relationship(
        "ConversationMessageModel", back_populates="simulation",
        cascade="all, delete-orphan", order_by="ConversationMessageModel.timestamp",
        lazy="select"
    )


class ConversationMessageModel(Base):
    __tablename__ = "conversation_messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    simulation_id = Column(String(36), ForeignKey("simulations.id"), nullable=False)
    sender = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    stage_at_message = Column(String(50), nullable=False)
    state_snapshot = Column(JSON, nullable=False, default=dict)
    timestamp = Column(DateTime, default=utcnow)

    # Relationships
    simulation = relationship("SimulationModel", back_populates="messages", lazy="select")


class UserSessionModel(Base):
    """Login tokens, persisted so sessions survive a server restart."""
    __tablename__ = "user_sessions"

    # ponytail: no expiry; add an expires_at column and check it in current_user if sessions must time out
    token = Column(String(80), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=utcnow)


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(100), nullable=False, default="Conseiller Voyage")
    role = Column(String(50), nullable=False, default="sales_rep") # sales_rep | admin | client
    profile_id = Column(String(36), ForeignKey("customer_profiles.id"), nullable=True, unique=True)  # client: their persona
    created_at = Column(DateTime, default=utcnow)
