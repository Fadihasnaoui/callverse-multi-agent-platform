from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload
from app.database.session import get_db
from app.database.models import CustomerProfileModel, SimulationModel, UserModel
from app.api.v1.auth import current_user
from app.api.v1.admin import CONVERSION_THRESHOLD, _avg, is_won
from app.customers.archetypes import get_archetype
from app.conversation.decision import OUTCOMES

router = APIRouter(prefix="/me", tags=["Espace client"])


def require_client(user: UserModel = Depends(current_user)) -> UserModel:
    if user.role != "client" or not user.profile_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Espace réservé aux clients.")
    return user


def _summary(s: SimulationModel) -> dict:
    hotel = (s.known_requirements or {}).get("hotel_vise") or {}
    return {
        "id": s.id,
        "title": s.scenario_title,
        "created_at": s.created_at,
        "status": s.status,
        "language": s.language,
        "stage": s.current_stage,
        "trust_level": s.trust_level,
        "interest_level": s.interest_level,
        "booking_intention": s.booking_intention,
        "won": is_won(s),
        "outcome": s.outcome,
        "outcome_label": OUTCOMES.get(s.outcome),
        "outcome_reason": s.outcome_reason,
        "hotel": hotel.get("name"),
        "messages": len(s.messages),
    }


@router.get("/dashboard")
def my_dashboard(user: UserModel = Depends(require_client), db: Session = Depends(get_db)):
    profile = db.get(CustomerProfileModel, user.profile_id)
    sims = (db.query(SimulationModel).options(selectinload(SimulationModel.messages))
            .filter(SimulationModel.profile_id == profile.id).order_by(SimulationModel.created_at.desc()).all())
    archetype = get_archetype(profile.archetype)
    return {
        "profile": {
            "name": profile.name,
            "archetype": profile.archetype,
            "archetype_label": archetype["label"],
            "archetype_color": archetype["color"],
            "archetype_description": archetype["description"],
            "destinations": profile.destination_preferences,
            "travel_dates": profile.travel_dates,
            "travelers": profile.number_of_travelers,
            "budget_total": profile.budget_total,
        },
        "stats": {
            "simulations": len(sims),
            "completed": sum(1 for s in sims if s.status == "completed"),
            "won": sum(1 for s in sims if is_won(s)),
            "avg_trust": _avg([s.trust_level for s in sims]),
            "avg_interest": _avg([s.interest_level for s in sims]),
            "avg_intent": _avg([s.booking_intention for s in sims]),
        },
        "conversion_threshold": CONVERSION_THRESHOLD,
        "simulations": [_summary(s) for s in sims],
    }


@router.get("/simulations/{simulation_id}")
def my_simulation(simulation_id: str, user: UserModel = Depends(require_client), db: Session = Depends(get_db)):
    sim = db.get(SimulationModel, simulation_id)
    if not sim or sim.profile_id != user.profile_id:  # 404, not 403: don't reveal other clients' simulations
        raise HTTPException(status_code=404, detail="Simulation introuvable.")
    return {
        **_summary(sim),
        "transcript": [
            {"sender": m.sender, "content": m.content, "timestamp": m.timestamp, "stage": m.stage_at_message,
             "trust_level": (m.state_snapshot or {}).get("trust_level"),
             "booking_intention": (m.state_snapshot or {}).get("booking_intention")}
            for m in sim.messages
        ],
    }
