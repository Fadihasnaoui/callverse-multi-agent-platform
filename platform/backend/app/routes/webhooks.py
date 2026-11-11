import hashlib
import hmac
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from app import config
from app.services import leads as leads_service
from app.services import jobs

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.get("/meta")
def verify(request: Request):
    q = request.query_params
    if q.get("hub.mode") == "subscribe" and config.META_VERIFY_TOKEN and q.get("hub.verify_token") == config.META_VERIFY_TOKEN:
        return Response(content=q.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(status_code=403, detail="verification failed")


@router.post("/meta")
async def receive(request: Request, background: BackgroundTasks):
    raw = await request.body()
    if config.META_APP_SECRET:
        expected = "sha256=" + hmac.new(config.META_APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, request.headers.get("X-Hub-Signature-256", "")):
            raise HTTPException(status_code=403, detail="bad signature")
    elif not config.ALLOW_UNSIGNED_WEBHOOKS:
        # Fail closed: without the app secret anyone could forge leads.
        raise HTTPException(status_code=503, detail="webhook not configured: set META_APP_SECRET")
    try:
        event = json.loads(raw or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="body is not valid JSON")
    # Meta expects a fast 200; classification (an LLM call) runs after replying.
    if not isinstance(event, dict):
        raise HTTPException(400, 'webhook body must be an object')
    try:
        jobs.enqueue('webhook', event, key=hashlib.sha256(raw).hexdigest())
    except Exception:
        raise HTTPException(503, 'event could not be durably saved; retry delivery')
    return {"status": "ok"}
