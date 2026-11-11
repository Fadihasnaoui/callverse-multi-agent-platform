from fastapi import APIRouter
from app.customers.archetypes import ARCHETYPES

router = APIRouter(prefix="/archetypes", tags=["Archetypes"])


@router.get("")
def list_archetypes():
    # Public, presentation-only fields; the simulator and admin UIs build their lists from this
    return [
        {"key": k, "label": a["label"], "color": a["color"], "difficulty": a["difficulty"], "description": a["description"]}
        for k, a in ARCHETYPES.items()
    ]
