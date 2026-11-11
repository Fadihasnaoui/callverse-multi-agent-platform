"""Notification endpoints — in-app notification feed."""
from fastapi import APIRouter

from app.services import notifications as notif_service

router = APIRouter(prefix="/businesses/{business_id}", tags=["notifications"])


@router.get("/notifications")
def list_notifications(business_id: str, unread_only: bool = False, limit: int = 50):
    return notif_service.list_notifications(business_id, limit=limit, unread_only=unread_only)


@router.get("/notifications/count")
def unread_count(business_id: str):
    return {"unread": notif_service.unread_count(business_id)}


@router.post("/notifications/{notification_id}/read")
def mark_read(business_id: str, notification_id: str):
    notif_service.mark_read(business_id, notification_id)
    return {"ok": True}


@router.post("/notifications/read-all")
def mark_all_read(business_id: str):
    count = notif_service.mark_all_read(business_id)
    return {"marked": count}
