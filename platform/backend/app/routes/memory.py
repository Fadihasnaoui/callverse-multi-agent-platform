from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import memory as memory_service

router = APIRouter(prefix="/businesses/{business_id}/memory", tags=["memory"])


class NewMemory(BaseModel):
    content: str
    scope: str = "global"  # global / content / scheduling / voice / leads / learning
    kind: str = "note"
    pinned: bool = True  # notes typed by the owner are always sent to the agent


@router.get("")
def list_memory(business_id: str, scope: str | None = None):
    return memory_service.list_memories(business_id, scope)


@router.post("")
def add_memory(business_id: str, payload: NewMemory):
    memory_id = memory_service.remember(
        business_id, payload.scope, payload.kind, payload.content, source="owner", pinned=payload.pinned
    )
    if not memory_id:
        raise HTTPException(
            status_code=400,
            detail=f"content is required, scope must be one of {memory_service.SCOPES}, kind one of {memory_service.KINDS}",
        )
    return {"id": memory_id}


@router.post("/seed-ozone")
def seed_ozone_memory(business_id: str):
    """Idempotent, owner-visible starter knowledge for the Ozone validation campaign."""
    entries = [
        ("global", "preference", "Ozone Voyages serves Tunisian families, couples and groups. Write primarily in natural French with a warm Tunisian tone."),
        ("global", "preference", "Never invent prices, dates, hotel availability or package inclusions. These facts require human confirmation before publishing."),
        ("content", "preference", "Prefer concrete scenes, useful travel details and direct questions. Avoid generic tourism clichés and exaggerated promises."),
        ("scheduling", "schedule", "Historical Facebook cadence averaged 7.9 days between posts with an 18.1-day maximum gap. Test a consistent three-post weekly rhythm."),
        ("leads", "feedback", "Treat requests for price, availability, dates, departure city or booking as strong lead intent; compliments alone are not leads."),
        ("learning", "insight", "Baseline hypothesis only: compare Reels against static posts for at least 10 published posts before declaring a winning format."),
    ]
    ids = [memory_service.remember(business_id, scope, kind, content, source="ozone_baseline", pinned=True)
           for scope, kind, content in entries]
    return {"seeded": sum(bool(memory_id) for memory_id in ids), "total": len(entries)}


@router.delete("/{memory_id}")
def delete_memory(business_id: str, memory_id: str):
    if not memory_service.delete_memory(business_id, memory_id):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": memory_id}
