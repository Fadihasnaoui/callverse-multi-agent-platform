import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app import config, db
from app.services import assets as assets_service
from app.services import avatar as avatar_service
from app.services import content as content_service
from app.services import memory
from app.services import music as music_service
from app.services import image_creative
from app.services import publisher as publisher_service
from app.services import scheduling as scheduling_service
from app.services import video as video_service
from app.services import strategist as strategist_service
from app.services import weekly as weekly_service
from app.services import content_state
from app.services import jobs
from fastapi.responses import JSONResponse

TUNIS = ZoneInfo("Africa/Tunis")

router = APIRouter(prefix="/businesses/{business_id}/content", tags=["content"])


class GenerateContent(BaseModel):
    platform: str  # facebook / instagram
    topic: str
    content_type: str = "static_post"  # static_post / reel / avatar_clip


class ResolveDraft(BaseModel):
    decision: str  # approve / reject / edit
    edited_caption: str | None = None
    scheduled_for: str | None = None  # ISO timestamp
    reviewer_notes: str | None = None
    expected_version: int | None = None


class PlanWeek(BaseModel):
    main_idea: str
    week_start: str | None = None  # any date in the target week (YYYY-MM-DD); default = this week
    posts: int = 5
    platforms: list[str] = ["facebook", "instagram"]
    content_types: list[str] = ["static_post", "reel"]
    generate_images: bool = False


class ScheduleSlot(BaseModel):
    local_datetime: str | None = None  # Tunisia local time, e.g. 2026-10-14T19:00; null clears the slot


class TranscriptDecision(BaseModel):
    transcript: str
    approve: bool = False


class AttachPhotos(BaseModel):
    hotel_name: str
    destination: str
    max_photos: int = 5


@router.post("")
def generate(business_id: str, payload: GenerateContent, request: Request):
    if request.headers.get('prefer') == 'respond-async':
        return JSONResponse(jobs.enqueue('generate',payload.model_dump(),business_id,request.headers.get('idempotency-key')),status_code=202)
    try:
        return content_service.generate_content(
            business_id, payload.platform, payload.topic, payload.content_type
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.post("/plan-week")
def plan_week(business_id: str, payload: PlanWeek, request: Request):
    if request.headers.get('prefer') == 'respond-async':
        return JSONResponse(jobs.enqueue('plan-week',payload.model_dump(),business_id,request.headers.get('idempotency-key')),status_code=202)
    try:
        return weekly_service.plan_week(
            business_id, payload.main_idea, payload.week_start, payload.posts,
            payload.platforms, payload.content_types, payload.generate_images,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


def _public_photo(url: str) -> str:
    """Photo URLs sourced from Google Places embed the server's API key; never send that to a browser."""
    return re.sub(r"([?&])key=[^&]*&?", r"\1", url).rstrip("?&")


@router.get("")
def list_content(business_id: str, status: str | None = None):
    query = (
        "SELECT id, type, platform, caption, status, scheduled_for, created_at, hashtags, "
        "asset_refs, rendered_video_path, publish_error, meta_post_id, avatar_spoken_text, "
        "avatar_language, avatar_transcript_status, avatar_job_id, features, content_version, approved_version, proposed_schedule FROM content_items "
        "WHERE business_id = %s"
    )
    params = [business_id]
    if status:
        query += " AND status = %s"
        params.append(status)
    with db.get_conn() as conn:
        rows = conn.execute(query + " ORDER BY created_at DESC", params).fetchall()
    return [
        {
            "id": str(r[0]),
            "type": r[1],
            "platform": r[2],
            "caption": r[3],
            "status": r[4],
            "scheduled_for": r[5].isoformat() if r[5] else None,
            "created_at": r[6].isoformat(),
            "hashtags": r[7] or [],
            "photo_urls": [_public_photo(u) for u in (r[8] or [])],
            "video_url": f"/renders/{config.render_filename(r[9])}" if r[9] else None,
            "publish_error": r[10],
            "meta_post_id": r[11],
            "avatar_text": r[12],
            "avatar_language": r[13],
            "avatar_transcript_status": r[14],
            "avatar_job_id": r[15],
            "visual_brief": (r[16] or {}).get("visual_brief"),
            "music_url": f"/renders/{config.render_filename((r[16] or {}).get('music_path'))}" if (r[16] or {}).get("music_path") else None,
            "content_version": r[17], "approved_version": r[18],
            "proposed_schedule": r[19].isoformat() if r[19] else None,
        }
        for r in rows
    ]


@router.post("/{content_id}/resolve")
def resolve(business_id: str, content_id: str, payload: ResolveDraft):
    if payload.decision not in {"approve", "reject", "edit"}:
        raise HTTPException(status_code=400, detail="decision must be approve, reject, or edit")

    with db.get_conn() as conn:
        existing = conn.execute(
            "SELECT caption, platform, type, status, scheduled_for, rendered_video_path, content_version "
            "FROM content_items WHERE id = %s AND business_id = %s FOR UPDATE",
            (content_id, business_id),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="content item not found")
        current_status = existing[3]
        if current_status in ("publishing", "published", "publish_unknown"):
            raise HTTPException(status_code=409, detail=f"item is already {current_status}")

        new_status = "approved" if payload.decision in {"approve", "edit"} else "rejected"
        if payload.expected_version is not None and payload.expected_version != existing[6]:
            raise HTTPException(status_code=409, detail="content changed since review - reload and review the new version")
        new_schedule = None
        if new_status == "approved":
            caption = payload.edited_caption if payload.edited_caption is not None else existing[0]
            if not (caption or "").strip():
                raise HTTPException(status_code=400, detail="cannot approve an empty caption")
            if existing[2] in ("reel", "avatar_clip") and not existing[5]:
                raise HTTPException(status_code=400, detail="this item has no video yet - generate it first")
            if payload.scheduled_for is not None:
                try:
                    new_schedule = datetime.fromisoformat(payload.scheduled_for.replace("Z", "+00:00"))
                except ValueError:
                    raise HTTPException(status_code=400, detail="scheduled_for must be an ISO timestamp")
                if new_schedule.tzinfo is None:
                    new_schedule = new_schedule.replace(tzinfo=TUNIS)
            effective = new_schedule or existing[4]
            if effective and effective < datetime.now(TUNIS) - timedelta(minutes=5):
                raise HTTPException(status_code=400, detail="that publish time has already passed - pick a new time")
            try:
                content_state.validate_artifact(conn, content_id, caption)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error))

        changed = (payload.edited_caption is not None and payload.edited_caption != existing[0]) or (new_schedule is not None and new_schedule != existing[4])
        fields = ["status = %s", "reviewer_notes = %s", "approved_at = " + ("now()" if new_status == 'approved' else "NULL"), "approved_version = %s"]
        params = [new_status, payload.reviewer_notes]
        params.append(existing[6] + int(changed) if new_status == 'approved' else None)
        if payload.edited_caption is not None:
            fields.append("caption = %s")
            params.append(payload.edited_caption)
        if new_schedule is not None:
            fields.append("scheduled_for = %s")
            params.append(new_schedule)
        params.extend([content_id, business_id, existing[6]])

        updated = conn.execute(
            f"UPDATE content_items SET {', '.join(fields)} WHERE id = %s AND business_id = %s "
            "AND status NOT IN ('publishing', 'published', 'publish_unknown') AND content_version=%s RETURNING id",
            params,
        ).fetchone()
        conn.commit()
        if not updated:  # the publisher claimed it between our read and write
            raise HTTPException(status_code=409, detail="item changed while you were editing - reload")

    if payload.decision == "reject":
        memory.learn_from_rejection(business_id, existing[1], existing[2], existing[0], payload.reviewer_notes)
    elif payload.edited_caption is not None:
        memory.learn_from_edit(business_id, existing[0], payload.edited_caption)

    return {"id": content_id, "status": new_status}


@router.post("/{content_id}/schedule")
def schedule_item(business_id: str, content_id: str, payload: ScheduleSlot):
    """Calendar drag-and-drop. Placing an item on a slot is the owner's explicit
    'post this then' decision, so it approves it; clearing the slot returns it
    to pending. Published/publishing items are never touched."""
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT status, type, platform, rendered_video_path, caption FROM content_items "
            "WHERE id = %s AND business_id = %s FOR UPDATE",
            (content_id, business_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="content item not found")
        status, ctype, platform, video_path, caption = row
        if status not in ("pending", "approved"):
            raise HTTPException(status_code=400, detail=f"item is {status} and cannot be scheduled")

        if payload.local_datetime is None:
            cleared = conn.execute(
                "UPDATE content_items SET status = 'pending', scheduled_for = NULL "
                "WHERE id = %s AND status IN ('pending', 'approved') RETURNING id",
                (content_id,),
            ).fetchone()
            conn.commit()
            if not cleared:
                raise HTTPException(status_code=409, detail="item changed - reload")
            return {"id": content_id, "status": "pending", "scheduled_for": None}

        try:
            when = datetime.fromisoformat(payload.local_datetime).replace(tzinfo=TUNIS)
        except ValueError:
            raise HTTPException(status_code=400, detail="local_datetime must look like 2026-10-14T19:00")
        if when <= datetime.now(TUNIS):
            raise HTTPException(status_code=400, detail="that time is in the past")
        if not (caption or "").strip():
            raise HTTPException(status_code=400, detail="this item has no caption yet")
        if ctype in ("reel", "avatar_clip") and not video_path:
            raise HTTPException(status_code=400, detail="this item has no video yet - generate it first")
        try:
            content_state.validate_artifact(conn, content_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        placed = conn.execute(
            "UPDATE content_items SET status = 'approved', scheduled_for = %s, approved_at = now(), "
            "publish_error = NULL, approved_version=content_version + CASE WHEN scheduled_for IS DISTINCT FROM %s THEN 1 ELSE 0 END "
            "WHERE id = %s AND status IN ('pending', 'approved') RETURNING id",
            (when, when, content_id),
        ).fetchone()
        conn.commit()
        if not placed:
            raise HTTPException(status_code=409, detail="item changed - reload")

    memory.remember(
        business_id, "scheduling", "schedule",
        f"Owner scheduled a {platform} {ctype} for {when:%A} {when:%H:%M} (Tunisia time)",
        source="calendar",
    )
    return {"id": content_id, "status": "approved", "scheduled_for": when.isoformat()}


@router.post("/{content_id}/attach-photos")
def attach_photos(business_id: str, content_id: str, payload: AttachPhotos):
    with db.get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM content_items WHERE id = %s AND business_id = %s", (content_id, business_id)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="content item not found")

    version = content_state.media_version(content_id)

    try:
        sourced = assets_service.source_photos(payload.hotel_name, payload.destination, max_photos=payload.max_photos)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not sourced["photo_urls"]:
        raise HTTPException(
            status_code=400,
            detail=f"no photos available for '{payload.hotel_name}' in {payload.destination} - "
            "register some first via POST /assets/manual, or set up Google Places.",
        )

    content_state.save_media(content_id, version, {"asset_refs": sourced["photo_urls"], "rendered_video_path": None})

    return {"content_id": content_id, "photo_urls": sourced["photo_urls"], "source": sourced["source"]}


@router.post("/{content_id}/render")
def render(business_id: str, content_id: str, request: Request):
    if request.headers.get('prefer') == 'respond-async':
        _check_owned(business_id,content_id)
        return JSONResponse(jobs.enqueue('render',{'content_id':content_id},business_id,request.headers.get('idempotency-key')),status_code=202)
    with db.get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM content_items WHERE id = %s AND business_id = %s", (content_id, business_id)
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="content item not found")

    try:
        return video_service.render_reel(content_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/{content_id}/music")
def generate_music(business_id: str, content_id: str, request: Request):
    _check_owned(business_id, content_id)
    if request.headers.get('prefer') == 'respond-async':
        return JSONResponse(jobs.enqueue('music',{'content_id':content_id},business_id,request.headers.get('idempotency-key')),status_code=202)
    try:
        return music_service.generate_reel_music(content_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/{content_id}/generate-image")
def generate_image(business_id: str, content_id: str, request: Request):
    _check_owned(business_id, content_id)
    if request.headers.get('prefer') == 'respond-async':
        return JSONResponse(jobs.enqueue('generate-image',{'content_id':content_id},business_id,request.headers.get('idempotency-key')),status_code=202)
    try:
        return image_creative.generate_for_content(content_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


def _check_owned(business_id: str, content_id: str):
    with db.get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM content_items WHERE id = %s AND business_id = %s", (content_id, business_id)
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="content item not found")


@router.post("/{content_id}/avatar/dictation")
async def avatar_dictation(business_id: str, content_id: str, request: Request):
    # Raw audio body (browser MediaRecorder blob) - avoids a multipart dependency.
    await run_in_threadpool(_check_owned, business_id, content_id)
    audio = await request.body()
    try:
        # blocking network + LLM calls: keep them off the event loop
        return await run_in_threadpool(
            avatar_service.transcribe_dictation, content_id, audio, request.headers.get("content-type", "")
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/{content_id}/avatar/transcript")
def avatar_transcript(business_id: str, content_id: str, payload: TranscriptDecision):
    _check_owned(business_id, content_id)
    try:
        return avatar_service.resolve_transcript(content_id, payload.transcript, payload.approve)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.post("/{content_id}/avatar")
def start_avatar(business_id: str, content_id: str, request: Request, language: str = "fr"):
    _check_owned(business_id, content_id)
    if request.headers.get('prefer') == 'respond-async':
        return JSONResponse(jobs.enqueue('avatar',{'content_id':content_id,'language':language},business_id,request.headers.get('idempotency-key')),status_code=202)
    try:
        return avatar_service.start_avatar_clip(content_id, language)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/{content_id}/avatar/refresh")
def refresh_avatar(business_id: str, content_id: str):
    _check_owned(business_id, content_id)
    try:
        return avatar_service.refresh_avatar_clip(content_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/{content_id}/publish-now")
def publish_now(business_id: str, content_id: str):
    # Skips waiting for scheduled_for, NOT the approval gate: publisher only
    # touches items already in 'approved' status.
    _check_owned(business_id, content_id)
    try:
        return publisher_service.publish_item(content_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.post("/{content_id}/retry")
def retry_failed(business_id: str, content_id: str):
    # A failed publish goes back to 'approved' (already human-approved
    # content) so the scheduler picks it up again after the cause is fixed.
    _check_owned(business_id, content_id)
    with db.get_conn() as conn:
        updated = conn.execute(
            "UPDATE content_items SET status = 'approved', publish_error = NULL "
            "WHERE id = %s AND status = 'failed' RETURNING id",
            (content_id,),
        ).fetchone()
        conn.commit()
    if not updated:
        raise HTTPException(status_code=400, detail="only failed items can be retried")
    return {"id": content_id, "status": "approved"}


class ReconcilePublish(BaseModel):
    remote_post_id: str | None = None
    confirmed_absent: bool = False


@router.post('/{content_id}/reconcile')
def reconcile_publish(business_id: str, content_id: str, body: ReconcilePublish):
    _check_owned(business_id, content_id)
    try:
        return publisher_service.reconcile(content_id, body.remote_post_id, body.confirmed_absent)
    except ValueError as error:
        raise HTTPException(409, str(error))


@router.delete("/{content_id}")
def delete_content(business_id: str, content_id: str):
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM content_items WHERE id = %s AND business_id = %s",
            (content_id, business_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="content item not found")
        if row[0] in ("publishing", "published", "publish_unknown"):
            raise HTTPException(status_code=409, detail=f"cannot delete a {row[0]} item")
        removed = conn.execute("DELETE FROM content_items WHERE id = %s AND business_id=%s AND status NOT IN ('publishing','published','publish_unknown') RETURNING id", (content_id, business_id)).fetchone()
        if not removed:
            raise HTTPException(status_code=409, detail="content changed - reload")
        conn.commit()
    return {"deleted": content_id}


class PatchContent(BaseModel):
    caption: str | None = None
    hashtags: list[str] | None = None


@router.patch("/{content_id}")
def patch_content(business_id: str, content_id: str, body: PatchContent):
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM content_items WHERE id = %s AND business_id = %s",
            (content_id, business_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="content item not found")
        if row[0] in ("publishing", "published", "publish_unknown"):
            raise HTTPException(status_code=409, detail=f"cannot edit a {row[0]} item")
        sets, vals = [], []
        if body.caption is not None:
            sets.append("caption = %s")
            vals.append(body.caption)
        if body.hashtags is not None:
            sets.append("hashtags = %s")
            vals.append(list(dict.fromkeys(tag.strip().lstrip('#') for tag in body.hashtags if tag.strip().lstrip('#'))))
        if not sets:
            raise HTTPException(status_code=400, detail="nothing to update")
        vals.extend([content_id, business_id])
        updated = conn.execute(f"UPDATE content_items SET {', '.join(sets)} WHERE id = %s AND business_id=%s AND status NOT IN ('publishing','published','publish_unknown') RETURNING id", vals).fetchone()
        if not updated:
            raise HTTPException(status_code=409, detail="content changed - reload")
        conn.commit()
    return {"updated": content_id}


@router.post("/{content_id}/suggest-schedule")
def suggest_schedule(business_id: str, content_id: str):
    # Proposes a time, writes it to scheduled_for - still just a proposal:
    # the existing /resolve endpoint is what actually approves (and can
    # override the time) before anything publishes. Agent-chosen timing
    # does not skip the human approval gate.
    try:
        return scheduling_service.suggest_schedule(business_id, content_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get('/jobs/{job_id}')
def get_job(business_id: str, job_id: str):
    try: return jobs.get(business_id,job_id)
    except ValueError as error: raise HTTPException(404,str(error))


@router.get('/jobs')
def list_jobs(business_id: str):
    with db.get_conn() as conn:
        rows = conn.execute('SELECT id,kind,status,error,progress FROM app_jobs WHERE business_id=%s ORDER BY created_at DESC LIMIT 20',(business_id,)).fetchall()
    return [dict(zip(('id','kind','status','error','progress'),row)) for row in rows]


class RestartJob(BaseModel):
    reviewed_outcome: bool = False


@router.post('/jobs/{job_id}/restart')
def restart_job(business_id: str, job_id: str, payload: RestartJob):
    if not payload.reviewed_outcome: raise HTTPException(400,'Review existing artifacts and possible vendor charges before restarting')
    with db.get_conn() as conn:
        row = conn.execute("UPDATE app_jobs SET status='queued',error=NULL,available_at=now(),finished_at=NULL WHERE id=%s AND business_id=%s AND status IN ('failed','unknown') RETURNING id",(job_id,business_id)).fetchone()
        if not row: raise HTTPException(409,'Job is not awaiting review')
        conn.execute("DELETE FROM job_steps WHERE job_id=%s AND status='running'",(job_id,))
        conn.commit()
    return JSONResponse({'job_id':str(row[0]),'status':'queued'},status_code=202)


@router.get("/strategist/brief")
def strategic_brief(business_id: str):
    """The senior strategist's current assessment — what's working, what's not,
    and what the next move should be. This is the brain of the agent."""
    try:
        brief = strategist_service.build_strategic_brief(business_id, context="content")
        recommendation = strategist_service.recommend_next_content(business_id)
        return {"brief": brief, "recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategist/recommend")
def strategic_recommendation(business_id: str, platform: str = None):
    """What should the next post be? The senior strategist's data-driven recommendation."""
    try:
        return strategist_service.recommend_next_content(business_id, platform=platform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
