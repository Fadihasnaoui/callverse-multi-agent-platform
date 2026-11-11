from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import glossary as glossary_service

router = APIRouter(prefix="/businesses/{business_id}/glossary", tags=["glossary"])


class NewPair(BaseModel):
    heard: str
    meant: str


@router.get("")
def list_glossary(business_id: str):
    return glossary_service.list_pairs(business_id)


@router.post("")
def add_pair(business_id: str, payload: NewPair):
    if not glossary_service.add_pair(business_id, payload.heard, payload.meant):
        raise HTTPException(status_code=400, detail="heard and meant are required and must differ")
    return {"ok": True}


@router.delete("/{pair_id}")
def delete_pair(business_id: str, pair_id: str):
    if not glossary_service.delete_pair(business_id, pair_id):
        raise HTTPException(status_code=404, detail="pair not found")
    return {"deleted": pair_id}
