from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

class TravelDatesSchema(BaseModel):
    start_date: str = Field(..., example="2026-11-01")
    end_date: str = Field(..., example="2026-11-10")
    flexible: bool = Field(True, example=True)

class CustomerProfileBase(BaseModel):
    name: str = Field(..., example="Carlos Mendez")
    scenario_id: str = Field("default_scenario", example="price_sensitive_scenario")
    archetype: str = Field("satisfied", example="satisfied") # satisfied | unsatisfied | hesitant
    age_range: str = Field(..., example="30-40")
    persona_description: str = Field(..., example="Extremely budget-conscious traveler looking for best value.")
    
    destination_preferences: List[str] = Field(..., example=["Hammamet", "Djerba"])
    travel_dates: TravelDatesSchema
    number_of_travelers: int = Field(..., example=2)
    
    budget_total: float = Field(..., example=4500.0, description="Budget total du séjour, en dinars tunisiens (TND)")
    price_sensitivity: str = Field(..., example="high") # high, medium, low
    
    preferred_accommodation: str = Field(..., example="3-star resort or apartment")
    preferred_transportation: str = Field(..., example="Economy flight")
    
    motivations: List[str] = Field(..., example=["Relaxation", "Beach"])
    constraints: List[str] = Field(..., example=["Budget max 4500 TND tout compris", "Vol direct de préférence"])
    
    personality_traits: List[str] = Field(..., example=["cautious", "frugal", "detail-oriented"])
    communication_style: str = Field(..., example="Direct, asks repeatedly about final prices and hidden fees.")
    
    objections: List[str] = Field(..., example=["Too expensive", "Hidden service fees", "Non-refundable policy"])
    decision_criteria: List[str] = Field(..., example=["Lowest total cost", "Free cancellation"])
    
    initial_trust: float = Field(5.0, ge=0.0, le=10.0)
    initial_interest: float = Field(5.0, ge=0.0, le=10.0)
    initial_booking_intention: float = Field(3.0, ge=0.0, le=10.0)
    
    desired_outcome: str = Field(..., example="Secure a complete package under budget with clear pricing.")
    # "Portrait humain" (app/customers/human.py): story, voice, hidden facts, how they decide. Optional.
    human_profile: Optional[Dict[str, Any]] = None

class CustomerProfileCreate(CustomerProfileBase):
    pass

class CustomerProfileResponse(CustomerProfileBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True

class CustomerStateSnapshot(BaseModel):
    trust_level: float = Field(..., ge=0.0, le=10.0)
    interest_level: float = Field(..., ge=0.0, le=10.0)
    booking_intention: float = Field(..., ge=0.0, le=10.0)
    current_stage: str
    unresolved_objections: List[str]
    known_requirements: Dict[str, Any]
    questions_answered: List[str]
    provided_info: Dict[str, Any]
    language: str = "fr"
    decision_state: Dict[str, Any] = Field(default_factory=dict)

class SimulationCreate(BaseModel):
    profile_id: str
    scenario_title: Optional[str] = "Standard Sales Simulation"
    language: Literal["fr", "en"] = "fr"  # language the simulated client writes and speaks
    # Situation and difficulty belong to the persona; set them here only to override it (tests, other agents)
    scenario: Optional[Literal["premiere_demande", "rappel_devis", "derniere_minute", "modification",
                               "reclamation", "client_fidele", "groupe"]] = None
    difficulty: Optional[Literal["facile", "normal", "difficile"]] = None
    seed: Optional[int] = Field(None, description="Replay a simulation with the same mood and events")

class SimulationResponse(BaseModel):
    id: str
    profile_id: str
    scenario_title: str
    current_stage: str
    status: str
    language: str = "fr"
    trust_level: float
    interest_level: float
    booking_intention: float
    known_requirements: Dict[str, Any]
    questions_answered: List[str]
    unresolved_objections: List[str]
    provided_info: Dict[str, Any]
    initial_message: Optional[str] = None
    initial_audio: Optional[str] = None
    scenario: str = "premiere_demande"
    difficulty: str = "normal"
    outcome: Optional[str] = None
    outcome_reason: Optional[str] = None
    decision_state: Optional[Dict[str, Any]] = None
    debrief: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class RepMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, example="Bonjour ! Je peux vous proposer un séjour à Djerba dans votre budget.")

TONES = ("chaleureux", "neutre", "pressant", "evasif")
PRICE_FEELINGS = ("confort", "limite", "trop_cher", "inconnu")
OUTCOME_KEYS = ("reserve", "option", "rappel", "devis_email", "perdu")


class TurnSignals(BaseModel):
    """What happened in the turn, as judged by the client (app/conversation/decision.py turns it into scores)."""
    must_haves_met: List[str] = Field(default_factory=list)
    deal_breaker_hit: Optional[str] = None
    delighter: Optional[str] = None
    red_flags: List[str] = Field(default_factory=list)
    advisor_tone: str = "neutre"
    price_feeling: str = "inconnu"
    hidden_revealed: List[str] = Field(default_factory=list)
    advisor_promises: List[str] = Field(default_factory=list)
    client_facts: List[str] = Field(default_factory=list)   # what the client itself stated this turn (dates, budget, wishes)
    buying_signal: bool = False
    wants_to_leave: bool = False
    proposed_outcome: Optional[str] = None

    @field_validator("advisor_tone", "price_feeling", "proposed_outcome", mode="before")
    @classmethod
    def known_value(cls, v, info):
        # LLM output is untrusted: unknown labels fall back to neutral values instead of failing the turn
        allowed = {"advisor_tone": TONES, "price_feeling": PRICE_FEELINGS, "proposed_outcome": OUTCOME_KEYS}[info.field_name]
        if isinstance(v, str):
            v = v.strip().lower().replace("é", "e").replace(" ", "_")
        if v in allowed:
            return v
        return {"advisor_tone": "neutre", "price_feeling": "inconnu"}.get(info.field_name)

    @field_validator("must_haves_met", "red_flags", "hidden_revealed", "advisor_promises", "client_facts", mode="before")
    @classmethod
    def as_list(cls, v):
        return [str(x) for x in v if x] if isinstance(v, list) else ([str(v)] if v else [])

    @field_validator("deal_breaker_hit", "delighter", mode="before")
    @classmethod
    def as_text(cls, v):
        return str(v) if v not in (None, "", False, [], "null") else None


class CustomerTurnResponse(BaseModel):
    customer_response: str
    transcription: Optional[str] = None
    audio_response: Optional[str] = None
    current_stage: Literal["greetings", "needs_discovery", "recommendation", "objection_handling", "negotiation", "closing"]
    # Scores are computed by the decision engine from `signals`; the model does not need to send them
    trust_level: float = 0.0
    interest_level: float = 0.0
    booking_intention: float = 0.0
    signals: TurnSignals = Field(default_factory=TurnSignals)
    intent_reason: Optional[str] = None
    patience_left: Optional[int] = None
    event: Optional[str] = None           # surprise the client brought up this turn
    outcome: Optional[str] = None         # set when the conversation ends (decision.OUTCOMES)
    outcome_reason: Optional[str] = None
    debrief: Optional[Dict[str, Any]] = None
    detected_rep_quality: Literal["poor", "neutral", "good", "excellent"]
    addressed_objections: List[str]
    new_unresolved_objections: List[str]
    reasoning: str
    # Hotel the client prefers among those the advisor proposed (kept only if the advisor really named it)
    hotel_vise: Optional[str] = None
    hotel_vise_details: Optional[Dict[str, Any]] = None

    @field_validator("trust_level", "interest_level", "booking_intention")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        # LLM output is untrusted: keep scores on the 0-10 scale
        return min(10.0, max(0.0, v))

class ConversationMessageResponse(BaseModel):
    id: str
    simulation_id: str
    sender: str
    content: str
    stage_at_message: str
    state_snapshot: Dict[str, Any]
    timestamp: datetime

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    email: str = Field(..., example="admin@simulator.com")
    password: str = Field(..., example="Password123!")


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    profile_id: Optional[str] = None  # set for role "client": the persona this login belongs to
    created_at: datetime

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

