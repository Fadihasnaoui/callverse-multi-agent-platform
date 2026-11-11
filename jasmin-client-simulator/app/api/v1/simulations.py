import random
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.hotels.catalog import profile_region, trip_nights
from app.database.models import CustomerProfileModel, SimulationModel, ConversationMessageModel
from app.customers.schemas import (
    SimulationCreate,
    SimulationResponse,
    RepMessageRequest,
    CustomerTurnResponse,
    ConversationMessageResponse
)
from app.conversation.engine import ConversationEngine
from app.conversation.voice_service import voice_service
from app.core.exceptions import LLMProviderException
from app.conversation import decision, scenarios
from app.conversation.debrief import build_debrief
from app.customers.human import load_human
from app.customers.schemas import CustomerProfileResponse

router = APIRouter(prefix="/simulations", tags=["Simulations"])

@router.post("", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
def start_simulation(payload: SimulationCreate, db: Session = Depends(get_db)):
    profile = db.query(CustomerProfileModel).filter(CustomerProfileModel.id == payload.profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Customer profile not found.")
    
    # The client walks in with a tourist region, not a hotel: the advisor proposes hotels and prices
    engine = ConversationEngine(db)
    region = profile_region(profile)

    # Scenario, mood and surprise events are drawn once from a seed (stored: a simulation can be replayed)
    seed = payload.seed if payload.seed is not None else random.randrange(1_000_000)
    human = load_human(profile)
    scenario = payload.scenario or human.scenario       # the persona's own situation
    difficulty = payload.difficulty or human.difficulty
    setup = scenarios.setup(scenario, difficulty, seed, profile.archetype, trip_nights(profile.travel_dates))
    decision_state = decision.init_state(human, setup)

    simulation = SimulationModel(
        profile_id=profile.id,
        scenario_title=payload.scenario_title or f"{scenarios.SCENARIOS[scenario]['label']} avec {profile.name}",
        current_stage="greetings",
        status="active",
        language=payload.language,
        scenario=scenario,
        difficulty=difficulty,
        decision_state=decision_state,
        trust_level=max(0.0, min(10.0, profile.initial_trust + setup["trust_delta"])),
        interest_level=profile.initial_interest,
        booking_intention=profile.initial_booking_intention,
        unresolved_objections=profile.objections,
        known_requirements={"destination": profile.destination_preferences, "region": region,
                            "budget": profile.budget_total, "hotel_vise": None}
    )
    
    db.add(simulation)
    db.commit()
    db.refresh(simulation)

    # Generate personalized initial greeting text & TTS voice audio
    initial_text = engine.generate_opening(profile, None, simulation.language, decision_state)
    # Short-term memory: the client must never contradict its own first words
    simulation.decision_state = {**decision_state, "opening": initial_text, "client_facts": engine.trip_facts(profile, decision_state)}
    decision.remember_questions(simulation.decision_state, initial_text)
    initial_audio = voice_service.synthesize_speech_base64(initial_text, lang=simulation.language)

    # Save initial customer greeting to message history
    first_msg = ConversationMessageModel(
        simulation_id=simulation.id,
        sender="customer",
        content=initial_text,
        stage_at_message="greetings",
        state_snapshot={
            "trust_level": simulation.trust_level,
            "interest_level": simulation.interest_level,
            "booking_intention": simulation.booking_intention
        }
    )
    db.add(first_msg)
    db.commit()

    resp = SimulationResponse.model_validate(simulation)
    resp.initial_message = initial_text
    resp.initial_audio = initial_audio
    return resp

@router.get("/{simulation_id}", response_model=SimulationResponse)
def get_simulation(simulation_id: str, db: Session = Depends(get_db)):
    simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    return SimulationResponse.model_validate(simulation)

@router.post("/{simulation_id}/messages", response_model=CustomerTurnResponse)
def send_rep_message(
    simulation_id: str,
    payload: RepMessageRequest,
    db: Session = Depends(get_db)
):
    simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    
    if simulation.status != "active":
        raise HTTPException(status_code=400, detail="Cannot send message to an inactive or completed simulation session.")

    engine = ConversationEngine(db)
    turn_response = engine.process_rep_message(simulation_id, payload.message)
    
    # Synthesize customer voice audio response
    turn_response.audio_response = voice_service.synthesize_speech_base64(turn_response.customer_response, lang=simulation.language)
    return turn_response

@router.post("/{simulation_id}/voice_message", response_model=CustomerTurnResponse)
def send_rep_voice_message(
    simulation_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Plain def: Groq/gTTS calls are blocking, so FastAPI runs this in a threadpool
    simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation session not found.")

    if simulation.status != "active":
        raise HTTPException(status_code=400, detail="Cannot send message to an inactive or completed simulation session.")

    audio_bytes = file.file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    # 1. Transcribe voice input using Groq Whisper
    try:
        transcript = voice_service.transcribe_audio(audio_bytes, filename=file.filename or "audio.webm", language=simulation.language)
    except LLMProviderException as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    if not transcript:
        raise HTTPException(status_code=422, detail="Audio non reconnu ou silence.")

    # 2. Process conversation message
    engine = ConversationEngine(db)
    turn_response = engine.process_rep_message(simulation_id, transcript)
    
    # 3. Attach transcript & synthesize voice output for customer persona
    turn_response.transcription = transcript
    turn_response.audio_response = voice_service.synthesize_speech_base64(turn_response.customer_response, lang=simulation.language)
    
    return turn_response


@router.get("/{simulation_id}/history", response_model=List[ConversationMessageResponse])
def get_conversation_history(simulation_id: str, db: Session = Depends(get_db)):
    simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    
    messages = (
        db.query(ConversationMessageModel)
        .filter(ConversationMessageModel.simulation_id == simulation_id)
        .order_by(ConversationMessageModel.timestamp.asc())
        .all()
    )
    return [ConversationMessageResponse.model_validate(m) for m in messages]

@router.post("/{simulation_id}/end", response_model=SimulationResponse)
def end_simulation(simulation_id: str, db: Session = Depends(get_db)):
    simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    
    simulation.status = "completed"
    db.commit()
    db.refresh(simulation)
    build_debrief(db, simulation, CustomerProfileResponse.model_validate(simulation.profile), ConversationEngine(db).llm_provider)
    return SimulationResponse.model_validate(simulation)


@router.get("/{simulation_id}/debrief")
def get_debrief(simulation_id: str, db: Session = Depends(get_db)):
    simulation = db.query(SimulationModel).filter(SimulationModel.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation session not found.")
    if simulation.status != "completed":
        raise HTTPException(status_code=409, detail="Le bilan est disponible à la fin de la simulation.")
    return build_debrief(db, simulation, CustomerProfileResponse.model_validate(simulation.profile), ConversationEngine(db).llm_provider)
