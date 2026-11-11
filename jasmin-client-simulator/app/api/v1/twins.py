from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import CustomerProfileModel, UserModel
from app.customers.accounts import create_client_account
from app.customers.archetypes import get_archetype
from app.customers.twin import TwinQuestionnaire, build_twin_profile

router = APIRouter(prefix="/twins", tags=["Jumeau numérique"])


@router.post("", status_code=status.HTTP_201_CREATED)
def create_twin(answers: TwinQuestionnaire, db: Session = Depends(get_db)):
    # Public endpoint (real clients fill the questionnaire): input is bounded by the schema.
    # ponytail: no rate limiting; add one (or a captcha) before exposing this page on the internet
    if db.query(UserModel).filter(UserModel.email == answers.email).first():
        raise HTTPException(status_code=409, detail="Un compte existe déjà avec cet email : connectez-vous à votre espace client.")
    profile_in = build_twin_profile(answers)
    data = profile_in.model_dump()
    data["travel_dates"] = profile_in.travel_dates.model_dump()
    profile = CustomerProfileModel(**data)
    db.add(profile)
    db.flush()
    create_client_account(db, profile, email=answers.email, password=answers.password)
    db.commit()
    archetype = get_archetype(profile.archetype)
    return {
        "profile_id": profile.id,
        "email": answers.email,
        "first_name": profile.name,
        "archetype": profile.archetype,
        "archetype_label": archetype["label"],
        "archetype_color": archetype["color"],
        "archetype_description": archetype["description"],
    }
