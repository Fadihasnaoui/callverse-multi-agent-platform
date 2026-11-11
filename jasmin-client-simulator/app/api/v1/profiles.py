from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import CustomerProfileModel
from app.customers.schemas import CustomerProfileCreate, CustomerProfileResponse
from app.customers.accounts import create_client_account
from app.customers.default_personas import DEFAULT_PERSONAS
from app.customers.generator import generate_customer_persona
from app.hotels.catalog import is_tunisian_place

OUTSIDE_TUNISIA = "Jasmin propose uniquement des séjours en Tunisie : choisissez une ville ou un hôtel tunisien (ex : Hammamet, Djerba)."

router = APIRouter(prefix="/profiles", tags=["Customer Profiles"])

@router.post("", response_model=CustomerProfileResponse, status_code=status.HTTP_201_CREATED)
def create_profile(profile_in: CustomerProfileCreate, db: Session = Depends(get_db)):
    foreign = [d for d in profile_in.destination_preferences if not is_tunisian_place(d, db)]
    if foreign:
        raise HTTPException(status_code=422, detail=f"{OUTSIDE_TUNISIA} Refusé : {', '.join(foreign)}.")
    profile_data = profile_in.model_dump()
    profile_data["travel_dates"] = profile_in.travel_dates.model_dump()
    
    db_profile = CustomerProfileModel(**profile_data)
    db.add(db_profile)
    db.flush()
    create_client_account(db, db_profile)
    db.commit()
    db.refresh(db_profile)
    return CustomerProfileResponse.model_validate(db_profile)

@router.post("/generate", response_model=CustomerProfileResponse, status_code=status.HTTP_201_CREATED)
def generate_profile(
    archetype: str = Query("satisfied", description="Archetype key, see GET /api/v1/archetypes"),
    destination: Optional[str] = Query(None, description="Optional destination preference"),
    budget: Optional[float] = Query(None, description="Optional max budget, TND"),
    background: Optional[str] = Query(None, description="Optional custom background & persona story"),
    scenario: Optional[str] = Query(None, description="Situation: premiere_demande, rappel_devis, derniere_minute, modification, reclamation, client_fidele, groupe (default: the type's usual one)"),
    difficulty: Optional[str] = Query(None, description="facile | normal | difficile (default: the type's usual one)"),
    db: Session = Depends(get_db)
):
    from app.conversation.scenarios import DIFFICULTIES, SCENARIOS
    if (scenario and scenario not in SCENARIOS) or (difficulty and difficulty not in DIFFICULTIES):
        raise HTTPException(status_code=422, detail="Scénario ou difficulté inconnus.")
    if destination and not is_tunisian_place(destination, db):
        raise HTTPException(status_code=422, detail=OUTSIDE_TUNISIA)
    persona_in = generate_customer_persona(archetype=archetype, destination=destination, budget=budget, background=background)
    # The LLM may still invent foreign places: keep only Tunisian ones
    persona_in.destination_preferences = [d for d in persona_in.destination_preferences if is_tunisian_place(d, db)] or [
        destination or "Tunisie"]
    persona_in.human_profile = {**(persona_in.human_profile or {}), "scenario": scenario or "", "difficulty": difficulty or ""}
    profile_data = persona_in.model_dump()
    profile_data["travel_dates"] = persona_in.travel_dates.model_dump()
    
    db_profile = CustomerProfileModel(**profile_data)
    db.add(db_profile)
    db.flush()
    create_client_account(db, db_profile)
    db.commit()
    db.refresh(db_profile)
    return CustomerProfileResponse.model_validate(db_profile)
@router.post("/reset-defaults", response_model=List[CustomerProfileResponse])
def reset_default_profiles(db: Session = Depends(get_db)):
    """Reset or update standard default customer personas to latest configurations."""
    updated = []
    for persona in DEFAULT_PERSONAS:
        prof = db.query(CustomerProfileModel).filter(CustomerProfileModel.scenario_id == persona.scenario_id).first()
        data = persona.model_dump()
        data["travel_dates"] = persona.travel_dates.model_dump()
        if prof:
            for k, v in data.items():
                setattr(prof, k, v)
            db.commit()
            db.refresh(prof)
            updated.append(prof)
        else:
            new_prof = CustomerProfileModel(**data)
            db.add(new_prof)
            db.commit()
            db.refresh(new_prof)
            updated.append(new_prof)
    return [CustomerProfileResponse.model_validate(p) for p in updated]

@router.get("", response_model=List[CustomerProfileResponse])
def list_profiles(db: Session = Depends(get_db)):
    profiles = db.query(CustomerProfileModel).all()
    if not profiles:
        # Auto-seed default personas if table is empty
        for persona in DEFAULT_PERSONAS:
            data = persona.model_dump()
            data["travel_dates"] = persona.travel_dates.model_dump()
            db.add(CustomerProfileModel(**data))
        db.commit()
        profiles = db.query(CustomerProfileModel).all()
    return [CustomerProfileResponse.model_validate(p) for p in profiles]

@router.get("/{profile_id}", response_model=CustomerProfileResponse)
def get_profile(profile_id: str, db: Session = Depends(get_db)):
    profile = db.query(CustomerProfileModel).filter(CustomerProfileModel.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Customer profile not found.")
    return CustomerProfileResponse.model_validate(profile)

