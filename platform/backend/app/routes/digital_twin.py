"""Shared orchestration API for the Ozone Digital Twin.

This module deliberately does not import or modify any agent implementation.
It calls each public agent API, records the neutral transcript, and exposes a
single session model to the unified dashboard.
"""
from __future__ import annotations

import json
from collections import Counter
import random
import re
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from psycopg.rows import dict_row

from app import config, db
from app.services import twin_clients, twin_report

router = APIRouter(prefix="/digital-twin", tags=["digital-twin"])

# Lab runs are deliberately isolated from customer data.  Individual sessions
# remain persisted in PostgreSQL; this small live registry drives the control
# room while a batch is running.
_batches: dict[str, dict[str, Any]] = {}
# The Quality Analyst runs one local-LLM analysis at a time and refuses any
# concurrent one (ANALYSIS_ALREADY_IN_PROGRESS_NO_RETRY), which left most
# batch reviews rule-only.  Reviews therefore queue here, one after another.
_quality_lock = asyncio.Lock()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


async def _request(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: int = 60) -> Any:
    try:
        response = await run_in_threadpool(
            requests.request, method, url, json=payload, timeout=timeout, allow_redirects=False
        )
    except requests.Timeout:
        raise HTTPException(504, "The connected agent timed out. Its state was not advanced.")
    except requests.RequestException:
        raise HTTPException(502, "The connected agent is unavailable. Check that its service is running.")
    if not response.ok:
        raise HTTPException(502, "The connected agent rejected the request. Its state was not advanced.")
    try:
        return response.json()
    except ValueError:
        raise HTTPException(502, "The connected agent returned an invalid response.")


def _service_root(url: str) -> str:
    return url.split("/v1/", 1)[0]


def _record_event(conn: Any, session_id: str, event_type: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO digital_twin_events(twin_session_id,event_type,payload) VALUES (%s,%s,%s::jsonb)",
        (session_id, event_type, _json(payload)),
    )


def _get_session(session_id: str) -> dict[str, Any]:
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute("SELECT * FROM digital_twin_sessions WHERE id=%s", (session_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Simulation session not found.")
    return row


def _messages(session_id: str) -> list[dict[str, Any]]:
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT id,sequence,role,content,source_message_id,payload,created_at "
            "FROM digital_twin_messages WHERE twin_session_id=%s ORDER BY sequence",
            (session_id,),
        ).fetchall()


def _view(session_id: str) -> dict[str, Any]:
    session = _get_session(session_id)
    session["messages"] = _messages(session_id)
    return session


def _route_for(advisor: dict[str, Any]) -> dict[str, Any]:
    reason = str(advisor.get("reason", ""))
    reply = str(advisor.get("reply", "")).lower()
    if any(term in f"{reason} {reply}" for term in ("reservation", "booking", "paiement", "payment")):
        return {"queue": "reservations", "priority": "normal", "reason": "booking_or_payment_interest"}
    if any(term in f"{reason} {reply}" for term in ("conseiller", "adviser", "reclamation", "complaint")):
        return {"queue": "human_advisor", "priority": "high", "reason": "human_follow_up_requested"}
    return {"queue": "digital_twin_monitor", "priority": "low", "reason": "normal_simulation_turn"}


async def _voice(method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any] | None:
    """Call the LiveKit transport; audio problems never break the saved call."""
    if not config.DIGITAL_TWIN_VOICE_URL:
        return None
    try:
        return await _request(method, config.DIGITAL_TWIN_VOICE_URL + path, payload=payload, timeout=timeout)
    except HTTPException:
        return None


async def _speak(session: dict[str, Any], sequence: int, role: str, text: str) -> None:
    """Voice one already-saved turn into this call's room, then attach the audio reference to it.

    The text is the exact persisted message, so the room, the transcript and
    the Quality input are the same dialogue.  The request waits for playout,
    which paces a voiced call at real speaking speed.
    """
    voice = session.get("voice") or {}
    if voice.get("status") != "open":
        return
    spoken = await _voice("POST", f"/calls/{session['call_id']}/turns", {
        "sequence": sequence, "speaker": role, "text": text, "language": session["language"], "wait": True,
    }, timeout=260)
    reference = {"room": session.get("room_name"), "status": (spoken or {}).get("status", "unavailable"),
                 "audio_seconds": (spoken or {}).get("audio_seconds"), "spoken_at": (spoken or {}).get("spoken_at")}
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE digital_twin_messages SET payload = payload || jsonb_build_object('voice', %s::jsonb) "
            "WHERE twin_session_id=%s AND sequence=%s",
            (_json(reference), session["id"], sequence),
        )
        if reference["status"] != "spoken":
            _record_event(conn, str(session["id"]), "voice.turn_failed", {"sequence": sequence, **reference})
        conn.commit()


# "Late" alone is not a complaint: "late check-out" is a booking question.
_COMPLAINT = re.compile(r"r[ée]clamation|plainte|complaint|didn'?t go well|s'est mal pass|bruyant|noisy|en retard"
                        r"|\b(?:an?|one|two) hours? late\b|\b(?:was|arrived) late\b|apolog|excuses|geste commercial|\bgesture\b")
_CHANGE = re.compile(r"\bannul|\bmodifi|\bcancel|\bchange\b|\bdéplacer|\bdecaler|\bdécaler|\bshift\b|\bmove the arrival")
# Word boundaries matter: "already booked" is a change request, not a ready booking.
_RESERVATION = re.compile(
    r"je r[ée]serve\b|on r[ée]serve\b|on confirme la r[ée]servation|\br[ée]server\b|je vais prendre|je prends\b"
    r"|bloquer (?:d[èe]s maintenant )?(?:la|une) chambre|let'?s confirm the booking|\bbook it\b|\bi'?ll take\b"
    r"|i'?d like to book|i want to book|\bbook (?:the|this|that)\b|secure the reservation")
_HUMAN = re.compile(r"\bconseill(er|ère)\b|\badvis[oe]r\b|\bhumain\b|\bhuman\b|\bune personne\b|\ba person\b"
                    r"|\bmanager\b|\bresponsable\b|\bsuperviseur\b|\bsupervisor\b|\bescalat\w*"
                    r"|\b(?:talk|speak) to (?:someone|somebody|whoever|a real|an agent|a manager)"
                    r"|\bparler (?:à|a) (?:quelqu'un|un responsable|une vraie personne)")


def _ai_stalled_turns(messages: list[dict[str, Any]]) -> int:
    """How many consecutive advisor turns ended with the same decision reason.

    An observable signal: Amira asking the same question again means the AI is
    not resolving the contact, which Workforce should be allowed to weigh.
    """
    reasons = [str((m.get("payload") or {}).get("reason") or "") for m in messages if m["role"] == "advisor"]
    count = 0
    for reason in reversed(reasons):
        if not reason or reason != reasons[-1]:
            break
        count += 1
    return count


def _workforce_labels(session: dict[str, Any]) -> dict[str, str]:
    """Display-only labels so the Workforce dashboard can show which call a decision belongs to."""
    client = session.get("client") or {}
    return {"call_id": str(session.get("call_id") or "")[:40], "batch_id": str(session.get("batch_id") or "")[:60],
            "client_label": str(client.get("name") or client.get("persona") or session.get("profile_id") or "")[:120]}


def _offers_shown(messages: list[dict[str, Any]]) -> bool:
    """Whether Amira has already presented verified offers in this call."""
    return any(m["role"] == "advisor" and str((m.get("payload") or {}).get("reason") or "").startswith(
        ("destination_offers", "recommendation_comparison", "quote_", "booking_")) for m in messages)


def _workforce_request(turn: str, language: str, sequence: int, channel: str = "CHAT", stalled: int = 0,
                       offers_shown: bool = True) -> dict[str, Any]:
    """Map an observed customer turn to the shared Workforce contract.

    This intentionally uses no hidden customer/simulation state and only the
    explicit message intent. A ready reservation always goes to a human.
    """
    text = turn.lower()
    complaint = bool(_COMPLAINT.search(text))
    change = bool(_CHANGE.search(text))
    # "Ready to book" needs something to book: before any offer, "je veux
    # réserver un hôtel" is a first request, not a reservation for a human.
    reservation = bool(_RESERVATION.search(text)) and not change and offers_shown
    human = bool(_HUMAN.search(text))
    request_type = "COMPLAINT_INCIDENT" if complaint else "CHANGE_CANCEL" if change else "BOOKING" if reservation else "FAQ"
    ai_score = 0.45 if complaint else 0.75
    if stalled >= 3:
        # Three identical advisor decisions in a row is a loop: below
        # Workforce's failing-AI threshold (0.3), so the contact goes to a human.
        ai_score = 0.25
    elif stalled == 2:
        ai_score = max(0.2, ai_score - 0.15)
    return {
        "turn_id": str(sequence), "request_type": request_type,
        "urgency": "HIGH" if complaint else "MEDIUM", "sensitivity": "HIGH" if complaint else "MEDIUM",
        "channel": channel, "language": {"fr": "FR", "en": "EN", "ar": "AR_MSA", "aeb-Latn": "AR_DIALECT", "aeb-Arab": "AR_DIALECT"}.get(language, "FR"),
        "request_value": 0.8 if reservation else 0.5, "human_requested": human,
        "callback_allowed": True, "ai_resolution_score": round(ai_score, 2),
        "reservation_ready": reservation,
    }


async def _deliver_workforce_event(event: dict[str, Any]) -> dict[str, Any]:
    if not config.WORKFORCE_ROUTING_WEBHOOK_URL:
        return {**event, "delivery": "stored_locally"}
    try:
        await _request("POST", config.WORKFORCE_ROUTING_WEBHOOK_URL, payload=event, timeout=12)
        return {**event, "delivery": "webhook_delivered"}
    except HTTPException:
        # Routing must not erase an otherwise valid customer/advisor turn.
        return {**event, "delivery": "webhook_failed"}


@router.get("/health")
async def health() -> dict[str, Any]:
    targets = {
        "advisor": _service_root(config.DIGITAL_TWIN_ADVISOR_URL) + "/health",
        "client_simulator": config.DIGITAL_TWIN_SIMULATOR_URL + "/health",
        "quality_analyst": _service_root(config.DIGITAL_TWIN_QUALITY_URL) + "/health",
        "workforce": config.WORKFORCE_API_URL + "/health",
    }
    if config.DIGITAL_TWIN_VOICE_URL:
        targets["voice_rooms"] = config.DIGITAL_TWIN_VOICE_URL + "/health"
    states: dict[str, str] = {}
    for name, url in targets.items():
        try:
            response = await run_in_threadpool(requests.get, url, timeout=4)
            states[name] = "ready" if response.ok else "unavailable"
        except requests.RequestException:
            states[name] = "unavailable"
    return {"services": states}


@router.get("/personas")
async def personas() -> Any:
    return await _request("GET", config.DIGITAL_TWIN_SIMULATOR_URL + "/api/v1/profiles")


@router.get("/sessions")
async def sessions(limit: int = 50, offset: int = 0, batch_id: str | None = None) -> list[dict[str, Any]]:
    """Conversation history from the database, newest first, with a short Quality summary."""
    limit, offset = min(max(limit, 1), 200), max(offset, 0)
    where, params = ("WHERE s.batch_id = %s", [batch_id]) if batch_id else ("", [])
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT s.id, s.call_id, s.batch_id, s.language, s.scenario, s.status, s.client, s.voice, "
            "s.created_at, s.completed_at, s.quality_review, "
            "(SELECT count(*) FROM digital_twin_messages m WHERE m.twin_session_id = s.id) AS messages "
            f"FROM digital_twin_sessions s {where} ORDER BY s.created_at DESC LIMIT %s OFFSET %s",
            (*params, limit, offset),
        ).fetchall()
    for row in rows:
        row["quality"] = _quality_summary(row.pop("quality_review"))
    return rows


@router.get("/quality/reports")
async def quality_reports(batch_id: str | None = None, limit: int = 60) -> list[dict[str, Any]]:
    """Full Quality reviews of finished Digital Twin calls, for the Quality Analyst page."""
    where = "WHERE s.quality_review <> '{}'::jsonb" + (" AND s.batch_id = %s" if batch_id else "")
    params = [batch_id] if batch_id else []
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            "SELECT s.id, s.call_id, s.batch_id, s.language, s.status, s.client, s.created_at, s.completed_at, "
            "s.quality_review, s.ai_quality_report, (SELECT count(*) FROM digital_twin_messages m WHERE m.twin_session_id = s.id) AS messages "
            f"FROM digital_twin_sessions s {where} ORDER BY s.completed_at DESC NULLS LAST LIMIT %s",
            (*params, min(max(limit, 1), 200)),
        ).fetchall()
        # AI reviews written with each run's report (API model where the local check was skipped).
        batches = sorted({row["batch_id"] for row in rows if row.get("batch_id")})
        stored = cur.execute("SELECT call_reviews FROM digital_twin_reports WHERE batch_id = ANY(%s)", (batches,)).fetchall() if batches else []
    ai = {str(r.get("session_id")): r for report in stored for r in report["call_reviews"] or []}
    for row in rows:
        row["quality"] = _quality_summary(row["quality_review"])
        row["ai_review"] = ai.get(str(row["id"]))
    return rows


async def _advance_with_retry(session_id: str) -> dict[str, Any]:
    """One retry for a transient agent error while many calls share Amira.

    Retrying is safe: Workforce decisions are keyed by conversation/turn and
    Amira's message_id is idempotent, and nothing is saved before her reply.
    """
    try:
        return await advance_session(session_id)
    except HTTPException as exc:
        if exc.status_code not in {502, 504}:
            raise
        await asyncio.sleep(3)
        return await advance_session(session_id)


async def _run_batch(batch_id: str, settings: dict[str, Any]) -> None:
    batch = _batches[batch_id]
    semaphore = asyncio.Semaphore(settings["concurrency"])
    used_personas: set[str] = set()  # scripted personas already drawn in this run

    async def one(index: int) -> None:
        item = batch["runs"][index]
        await semaphore.acquire()
        holding = True
        item["status"] = "running"
        try:
            # Each call draws its own client from the seed, so a run is varied
            # yet replayable: the same seed gives the same mix of clients.
            rng = random.Random(settings["seed"] * 1000 + index)
            client = twin_clients.pick(rng, settings["client_mix"], settings["profiles"], settings["language"], used_personas)
            if settings.get("scenario"):
                client["scenario"] = settings["scenario"]
            if settings.get("difficulty"):
                client["difficulty"] = settings["difficulty"]
            client["fallback_profile_id"] = rng.choice(settings["profiles"])
            if client["source"] == "jasmin_profile":
                client["name"] = settings["profile_names"].get(client["profile_id"])
            item["client"] = {k: client.get(k) for k in ("source", "name", "persona", "archetype", "scenario", "difficulty", "language")}
            started = await start_session({
                "client": client, "seed": settings["seed"] + index,
                "voice": settings["voice"], "batch_id": batch_id,
            })
            item["session_id"] = str(started["id"])
            current = started
            for _ in range(settings["max_turns"]):
                if current["status"] != "active":
                    break
                current = await _advance_with_retry(str(current["id"]))
                item["turns"] += 1
                # Every client source can close the call itself (booking, decline, goodbye).
                if not await _client_active(current):
                    break
                # Jasmin sometimes returns no reply while her simulation stays
                # open; the client has nothing more to say, so the call ends.
                if not current.get("messages") or current["messages"][-1]["role"] != "client":
                    break
            # The conversation is over: free the parallel slot before the
            # one-at-a-time Quality review, so waiting reviews do not stop
            # the next queued calls from starting.
            semaphore.release(); holding = False
            completed = await complete_session(str(current["id"]))
            item["status"] = completed["status"]
            item["quality_score"] = _quality_summary(completed.get("quality_review")).get("score")
        except Exception as exc:
            item["status"] = "failed"; item["error"] = str(getattr(exc, "detail", exc))
            if item["session_id"]:
                # A failed call still closes its room, releases Workforce
                # capacity and is reviewed on whatever transcript exists.
                try:
                    await _fail_session(item["session_id"], item["error"])
                except Exception:
                    pass
        finally:
            if holding:
                semaphore.release()
            batch["completed"] += 1

    await asyncio.gather(*(one(index) for index in range(settings["count"])))
    batch["status"] = "completed"; batch["completed_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/batches")
async def start_batch(body: dict[str, Any]) -> dict[str, Any]:
    count = min(max(int(body.get("count", 12)), 1), 30)
    concurrency = min(max(int(body.get("concurrency", 8)), 1), 12)
    profiles = await personas()
    profile_ids = [str(item.get("id")) for item in profiles if item.get("id")]
    profile_names = {str(item.get("id")): item.get("name") or item.get("display_name") for item in profiles if item.get("id")}
    if not profile_ids:
        raise HTTPException(502, "The client simulator has no available personas.")
    language = str(body.get("language") or "fr").lower()
    if language not in {"fr", "en", "mixed"}:
        raise HTTPException(422, "language must be fr, en or mixed.")
    batch_id = str(uuid.uuid4())
    settings = {"count": count, "concurrency": concurrency, "profiles": profile_ids, "profile_names": profile_names,
                "language": language, "scenario": body.get("scenario"),
                "difficulty": body.get("difficulty"), "seed": int(body.get("seed", 1000)),
                "max_turns": min(max(int(body.get("max_turns", 10)), 2), 16),
                "voice": bool(body.get("voice", True)) and bool(config.DIGITAL_TWIN_VOICE_URL),
                # random: Jasmin profiles + newly generated Jasmin clients + Voice Lab scripted clients.
                "client_mix": "jasmin" if body.get("client_mix") == "jasmin" else "random"}
    _batches[batch_id] = {"id": batch_id, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                          "completed_at": None, "settings": {k: v for k, v in settings.items() if k not in {"profiles", "profile_names"}},
                          "completed": 0,
                          "runs": [{"index": i + 1, "status": "queued", "session_id": None, "error": None, "turns": 0}
                                   for i in range(count)]}
    asyncio.create_task(_run_batch(batch_id, settings))
    return await batch(batch_id)


def _quality_summary(review: Any) -> dict[str, Any]:
    if not isinstance(review, dict) or not review:
        return {}
    combined = review.get("combined_scorecard") or {}
    card = review.get("scorecard") or {}
    findings = card.get("findings")
    return {
        "score": combined.get("composite_score_100", combined.get("official_score_100", card.get("score_100"))),
        "status": combined.get("status") or card.get("status"), "coverage": card.get("coverage"),
        "turns_checked": card.get("turns_checked"), "findings": len(findings) if isinstance(findings, list) else 0,
        "codes": [f.get("code") for f in findings if isinstance(f, dict)] if isinstance(findings, list) else [],
        "llm_checked": (review.get("result") or {}).get("status") == "free_quality_checked",
        "human_review_required": bool(combined.get("human_review_required") or card.get("human_review_required")),
    }


def _batch_calls(batch_id: str) -> list[dict[str, Any]]:
    """The call-centre board, built only from persisted Twin call records."""
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """
            SELECT s.id, s.call_id, s.room_name, s.profile_id, s.language, s.status, s.voice, s.client,
                   s.workforce_route, s.quality_review, s.created_at, s.completed_at,
                   (SELECT count(*) FROM digital_twin_messages m WHERE m.twin_session_id = s.id) AS messages,
                   (SELECT json_build_object('role', m.role, 'content', left(m.content, 220))
                      FROM digital_twin_messages m WHERE m.twin_session_id = s.id
                     ORDER BY m.sequence DESC LIMIT 1) AS last_message,
                   (SELECT coalesce(json_agg(m.payload->>'reason' ORDER BY m.sequence), '[]'::json)
                      FROM digital_twin_messages m WHERE m.twin_session_id = s.id AND m.role = 'advisor') AS advisor_reasons,
                   (SELECT coalesce(json_agg(e.payload->>'queue' ORDER BY e.created_at), '[]'::json)
                      FROM digital_twin_events e
                     WHERE e.twin_session_id = s.id AND e.event_type = 'workforce.route_proposed'
                       AND e.payload ? 'session_id') AS route_path,
                   (SELECT count(*) FROM digital_twin_events e
                     WHERE e.twin_session_id = s.id AND e.event_type = 'voice.turn_failed') AS voice_failures
              FROM digital_twin_sessions s WHERE s.batch_id = %s ORDER BY s.created_at
            """,
            (batch_id,),
        ).fetchall()
    calls = []
    for row in rows:
        route = row.pop("workforce_route") or {}
        row["route"] = {key: route.get(key) for key in ("decision", "queue", "reason", "policy", "humans_available", "task_id", "advisor_reason")}
        row["quality"] = _quality_summary(row.pop("quality_review"))
        # Longest run of identical consecutive advisor decisions: a repetition
        # loop the transcript shows even when a rubric scores the call highly.
        streak = longest = 0
        previous = None
        for reason in row.pop("advisor_reasons") or []:
            streak = streak + 1 if reason and reason == previous else 1
            longest, previous = max(longest, streak), reason
        row["advisor_repeat_streak"] = longest
        calls.append(row)
    return calls


def _by_client_source(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Calls and mean Quality score per client source, to compare how Amira copes with each."""
    groups: dict[str, list[Any]] = {}
    for call in calls:
        groups.setdefault((call.get("client") or {}).get("source") or "jasmin_profile", []).append(call["quality"].get("score"))
    return {source: {"calls": len(scores),
                     "mean_score": round(sum(s for s in scores if isinstance(s, (int, float))) / max(1, sum(isinstance(s, (int, float)) for s in scores)), 1)
                     if any(isinstance(s, (int, float)) for s in scores) else None}
            for source, scores in groups.items()}


def _evaluation(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate what the run shows about each agent's output."""
    queues: dict[str, int] = {}
    for call in calls:
        for queue in call["route_path"] or []:
            queues[queue or "unknown"] = queues.get(queue or "unknown", 0) + 1
    scores = [c["quality"]["score"] for c in calls if isinstance(c["quality"].get("score"), (int, float))]
    done = [c for c in calls if c["status"] != "active"]
    return {
        "calls": len(calls), "finished": len(done), "active": len(calls) - len(done),
        "messages": sum(int(c["messages"] or 0) for c in calls),
        "advisor": {"calls_with_repeat_loop": sum(1 for c in calls if c["advisor_repeat_streak"] >= 3),
                    "longest_repeat": max((c["advisor_repeat_streak"] for c in calls), default=0)},
        "routing": {"decisions": sum(queues.values()), "by_queue": queues,
                    "waiting_for_human": queues.get("waiting_human", 0),
                    "calls_escalated": sum(1 for c in calls if any(q in {"reservations", "customer_advisors", "waiting_human"} for q in c["route_path"] or []))},
        "quality": {"reviewed": len(scores), "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
                    "min_score": min(scores) if scores else None,
                    "below_80": sum(1 for s in scores if s < 80),
                    "human_review_required": sum(1 for c in calls if c["quality"].get("human_review_required")),
                    "missing_review": sum(1 for c in done if not c["quality"]),
                    "llm_checked": sum(1 for c in calls if c["quality"].get("llm_checked")),
                    # Most frequent findings across the run: what to fix first.
                    "top_findings": dict(Counter(code for c in calls for code in c["quality"].get("codes", [])).most_common(6))},
        "clients": _by_client_source(calls),
        "client_details": {"languages": dict(Counter(c["language"] for c in calls)),
                           "scenarios": dict(Counter((c.get("client") or {}).get("scenario") or "—" for c in calls))},
        "voice": {"rooms": sum(1 for c in calls if (c["voice"] or {}).get("room")),
                  "turns_spoken": sum(int((c["voice"] or {}).get("turns_spoken") or 0) for c in calls),
                  "turn_failures": sum(int(c["voice_failures"] or 0) for c in calls),
                  "audio_minutes": round(sum(float((c["voice"] or {}).get("audio_seconds") or 0) for c in calls) / 60, 1)},
        "outcomes": {"completed": sum(1 for c in calls if c["status"] == "completed"),
                     "failed": sum(1 for c in calls if c["status"] == "failed")},
    }


@router.get("/batches")
async def batches(limit: int = 10) -> list[dict[str, Any]]:
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT batch_id AS id, count(*) AS calls, min(created_at) AS created_at, "
            "count(*) FILTER (WHERE status='active') AS active FROM digital_twin_sessions "
            "WHERE batch_id IS NOT NULL GROUP BY batch_id ORDER BY min(created_at) DESC LIMIT %s",
            (min(max(limit, 1), 50),),
        ).fetchall()


@router.get("/batches/{batch_id}")
async def batch(batch_id: str) -> dict[str, Any]:
    calls = _batch_calls(batch_id)
    live = _batches.get(batch_id)
    if not live and not calls:
        raise HTTPException(404, "Simulation batch not found.")
    # After an API restart the registry is gone; the persisted calls remain.
    result = dict(live) if live else {"id": batch_id, "status": "interrupted" if any(c["status"] == "active" for c in calls) else "completed",
                                      "settings": {"count": len(calls)}, "completed": sum(1 for c in calls if c["status"] != "active"), "runs": []}
    state = None
    try:
        state = await _request("GET", f"{config.WORKFORCE_API_URL}/external/state", timeout=8)
    except HTTPException:
        pass
    result["calls"] = calls
    result["evaluation"] = _evaluation(calls)
    result["workforce"] = None if not state else {key: state.get(key) for key in ("human_capacity", "humans_available", "active_human_assignments")}
    return result


@router.get("/sessions/{session_id}")
async def session(session_id: str) -> dict[str, Any]:
    return _view(session_id)


async def _open_jasmin(profile_id: str, language: str, body: dict[str, Any]) -> tuple[str, str]:
    # Jasmin publishes only French and English clients; rejecting others here
    # avoids a session which starts but fails on the simulator's validation.
    if language not in {"fr", "en"}:
        raise HTTPException(422, "Jasmin clients speak fr or en.")
    simulation_body = {key: body[key] for key in ("scenario", "difficulty", "seed") if body.get(key) is not None}
    simulation_body.update(profile_id=profile_id, language=language)
    simulation = await _request("POST", config.DIGITAL_TWIN_SIMULATOR_URL + "/api/v1/simulations", payload=simulation_body)
    simulator_id = str(simulation.get("id", "")).strip()
    initial = str(simulation.get("initial_message", "")).strip()
    if not simulator_id or not initial:
        raise HTTPException(502, "The client simulator did not return an initial customer message.")
    return simulator_id, initial


async def _open_client(session_id: str, body: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    """Start this call's simulated client: (simulator id, opening line, language, client record)."""
    client = dict(body.get("client") or {"source": "jasmin_profile", "profile_id": body.get("profile_id")})
    source = client.get("source")
    language = str(client.get("language") or body.get("language") or "fr")
    if source == "scripted":
        try:
            initial, persona = twin_clients.start_scripted(session_id, str(client.get("persona")))
        except (KeyError, OSError) as exc:
            raise HTTPException(422, f"Unknown scripted persona: {exc}")
        return "scripted-" + session_id, initial, persona["language"], {**client, "language": persona["language"]}
    if source == "jasmin_generated":
        query = urlencode({key: client[key] for key in ("archetype", "destination", "budget", "scenario", "difficulty") if client.get(key)})
        try:
            profile = await _request("POST", f"{config.DIGITAL_TWIN_SIMULATOR_URL}/api/v1/profiles/generate?{query}", payload={}, timeout=90)
            client.update(profile_id=str(profile["id"]), name=profile.get("name"))
        except (HTTPException, KeyError):
            # Generation depends on Jasmin's LLM; a stored persona keeps the call going.
            client.update(source="jasmin_profile", profile_id=client.get("fallback_profile_id"), generation="failed")
    profile_id = str(client.get("profile_id") or "").strip()
    if not profile_id:
        raise HTTPException(422, "profile_id is required.")
    simulator_id, initial = await _open_jasmin(profile_id, language.lower(), {**body, **{k: client.get(k) for k in ("scenario", "difficulty")}})
    return simulator_id, initial, language.lower(), {**client, "language": language.lower()}


async def _client_reply(session: dict[str, Any], advisor: dict[str, Any]) -> tuple[str, bool]:
    """The simulated client's next line after Amira's reply, and whether it closed the call."""
    if (session.get("client") or {}).get("source") == "scripted":
        try:
            return twin_clients.scripted_next(str(session["id"]), advisor)
        except KeyError:
            raise HTTPException(409, "The scripted client is no longer running (the API restarted).")
    simulated = await _request(
        "POST", f"{config.DIGITAL_TWIN_SIMULATOR_URL}/api/v1/simulations/{session['simulator_session_id']}/messages",
        payload={"message": str(advisor.get("reply", ""))},
    )
    return str(simulated.get("customer_response") or "").strip(), False


async def _client_active(session: dict[str, Any]) -> bool:
    if (session.get("client") or {}).get("source") == "scripted":
        return twin_clients.scripted_active(str(session["id"]))
    sim = await _request("GET", f"{config.DIGITAL_TWIN_SIMULATOR_URL}/api/v1/simulations/{session['simulator_session_id']}")
    return sim.get("status") == "active"


async def _end_client(session: dict[str, Any]) -> None:
    if (session.get("client") or {}).get("source") == "scripted":
        twin_clients.end_scripted(str(session["id"]))
        return
    await _request("POST", f"{config.DIGITAL_TWIN_SIMULATOR_URL}/api/v1/simulations/{session['simulator_session_id']}/end", payload={})


@router.get("/client-sources")
async def client_sources() -> dict[str, Any]:
    return {"scripted_personas": [{"id": p["id"], "name": p.get("name"), "language": p["language"]} for p in twin_clients.scripted_personas()],
            "archetypes": twin_clients.ARCHETYPES, "scenarios": twin_clients.SCENARIOS, "difficulties": twin_clients.DIFFICULTIES}


@router.post("/sessions")
async def start_session(body: dict[str, Any]) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    simulator_id, initial, language, client = await _open_client(session_id, body)
    profile_id = str(client.get("profile_id") or client.get("persona") or "")
    advisor_session_id = "twin-" + session_id
    customer_id = "simulated-" + simulator_id
    # A dedicated tenant guarantees that synthetic Jasmine turns cannot appear
    # in the live customer-advisor workspace, while still using its exact API.
    twin_tenant_id = "ozone-digital-twin"
    # The call ID is the single key shared by the transcript, the LiveKit room,
    # the Workforce conversation and the Quality review of this call.
    call_id = session_id.replace("-", "")[:12]
    voice: dict[str, Any] = {}
    if body.get("voice"):
        room = await _voice("POST", "/calls", {"call_id": call_id, "twin_session_id": session_id,
                                               "language": language, "label": client.get("name") or profile_id})
        voice = ({"status": "open", "room": room.get("room"), "listener_url": room.get("listener_url")}
                 if room else {"status": "unavailable"})
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO digital_twin_sessions(id,tenant_id,customer_id,advisor_session_id,simulator_session_id,language,scenario,"
            "call_id,room_name,batch_id,profile_id,voice,client) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)",
            (session_id, twin_tenant_id, customer_id, advisor_session_id, simulator_id, language,
             client.get("scenario") or body.get("scenario"),
             call_id, voice.get("room"), body.get("batch_id"), profile_id, _json(voice), _json(client)),
        )
        conn.execute(
            "INSERT INTO digital_twin_messages(twin_session_id,sequence,role,content,payload) VALUES (%s,1,'client',%s,%s::jsonb)",
            (session_id, initial, _json({"source": client.get("source"), "profile_id": profile_id})),
        )
        _record_event(conn, session_id, "simulation.started", {"profile_id": profile_id, "simulator_session_id": simulator_id,
                                                               "call_id": call_id, "voice": voice, "client": client})
        conn.commit()
    session = _get_session(session_id)
    await _speak(session, 1, "client", initial)
    return _view(session_id)


@router.post("/sessions/{session_id}/advance")
async def advance_session(session_id: str) -> dict[str, Any]:
    current = _get_session(session_id)
    if current["status"] != "active":
        raise HTTPException(409, "This simulation is already closed.")
    transcript = _messages(session_id)
    if not transcript or transcript[-1]["role"] != "client":
        raise HTTPException(409, "The simulation is not waiting for an advisor response.")
    client_turn = transcript[-1]
    channel = "PHONE" if (current.get("voice") or {}).get("room") else "CHAT"
    workforce_payload = _workforce_request(client_turn["content"], current["language"], int(client_turn["sequence"]),
                                           channel, _ai_stalled_turns(transcript), _offers_shown(transcript))
    workforce_payload["conversation_id"] = session_id
    workforce_payload.update(_workforce_labels(current))
    workforce = await _request("POST", config.WORKFORCE_EXTERNAL_ROUTING_URL, payload=workforce_payload, timeout=20)
    # Workforce makes an operational assignment, not a replacement reply.
    # Even when a reservation or complaint is queued for a human, Amira must
    # continue the customer conversation and explain the next confirmed step.
    with db.get_conn() as conn:
        _record_event(
            conn,
            session_id,
            "workforce.routed_external_contact" if workforce.get("decision") != "AI" else "workforce.route_proposed",
            workforce,
        )
        conn.commit()
    # Exactly the request the Customer Advisor page sends: the client's own
    # words, with Amira keeping the conversation state herself.  A comparison
    # run showed the page path keeps destination, dates and party size without
    # any replayed context, so the simulation tests the same Amira as real use.
    advisor_payload = {
        "tenant_id": current["tenant_id"], "agent_id": "customer-advisor",
        "customer_id": current["customer_id"], "session_id": current["advisor_session_id"],
        "message_id": f"{session_id}-{client_turn['sequence']}", "channel": "digital_twin",
        "language": current["language"],
        "message": client_turn["content"],
    }
    advisor = await _request("POST", config.DIGITAL_TWIN_ADVISOR_URL, payload=advisor_payload)
    reply = str(advisor.get("reply", "")).strip()
    if not reply or advisor.get("status") not in {None, "completed"}:
        raise HTTPException(502, "Amira did not return a valid response. The simulation remains at the client turn.")
    route = {"session_id": session_id, **workforce, "advisor_reason": advisor.get("reason")}
    advisor_sequence = int(client_turn["sequence"]) + 1
    review_record = advisor.get("quality_review") if isinstance(advisor.get("quality_review"), dict) else None
    if review_record:
        # The analyst must cite the customer's actual utterance, never the
        # internal resilience context supplied to the advisor API.
        review_record = {**review_record, "customer_message": {"role": "customer", "text": client_turn["content"]}}
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO digital_twin_messages(twin_session_id,sequence,role,content,payload) VALUES (%s,%s,'advisor',%s,%s::jsonb)",
            (session_id, advisor_sequence, reply, _json({
                "source": "customer_advisor", "reason": advisor.get("reason"),
                "trip": advisor.get("trip") if isinstance(advisor.get("trip"), dict) else {},
                # This is Amira's compact, schema-validated operational proof:
                # request, response, offers, tools and handoff information.
                "review_record": review_record,
            })),
        )
        conn.execute("UPDATE digital_twin_sessions SET workforce_route=%s::jsonb WHERE id=%s", (_json(route), session_id))
        _record_event(conn, session_id, "workforce.route_proposed", route)
        conn.commit()
    await _speak(current, advisor_sequence, "advisor", reply)
    next_message, closed = await _client_reply(current, advisor)
    if next_message:
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO digital_twin_messages(twin_session_id,sequence,role,content,payload) VALUES (%s,%s,'client',%s,%s::jsonb)",
                (session_id, advisor_sequence + 1, next_message,
                 _json({"source": (current.get("client") or {}).get("source") or "jasmin_profile", "closing": closed})),
            )
            conn.commit()
        await _speak(current, advisor_sequence + 1, "client", next_message)
    return _view(session_id)


async def _run_quality_review(session_id: str, provider: Any = None) -> Any:
    """Send the immutable saved dialogue to the analyst and persist its evidence."""
    current = _get_session(session_id)
    messages = _messages(session_id)
    if len([item for item in messages if item["role"] in {"client", "advisor"}]) < 2:
        raise HTTPException(409, "Run at least one advisor turn before requesting a quality review.")
    review_payload = {
        "messages": [
            {"role": "agent" if item["role"] == "advisor" else "client", "text": item["content"]}
            for item in messages if item["role"] in {"client", "advisor"}
        ],
        # The analyst names Tunisian Arabic "aeb" whatever the script.
        "language": "aeb" if str(current["language"]).startswith("aeb") else current["language"],
        "conversation_complete": current["status"] == "completed",
        "provider": provider,
        # The local deterministic rubric is always permitted.  No cloud model
        # is requested unless an operator selects one explicitly.
        "cloud_consent": False,
        "free_tier_confirmed": False,
    }
    review_records = [
        item.get("payload", {}).get("review_record")
        for item in messages
        if item["role"] == "advisor" and isinstance(item.get("payload"), dict)
        and isinstance(item["payload"].get("review_record"), dict)
    ]
    # Records saved before the shared-schema fix hold the customer turn as a
    # bare string; the contract requires {"role": "customer", "text": ...}.
    review_records = [
        {**record, "customer_message": {"role": "customer", "text": record["customer_message"]}}
        if isinstance(record.get("customer_message"), str) else record
        for record in review_records
    ]
    if review_records:
        review_payload["review_records"] = review_records
    async with _quality_lock:
        try:
            # The analyst's own local-LLM limit is 180 s on CPU.
            review = await _request("POST", config.DIGITAL_TWIN_QUALITY_URL + "/analyze", payload=review_payload, timeout=200)
        except HTTPException as exc:
            if exc.status_code not in {502, 504}:
                raise
            await asyncio.sleep(5)
            review = await _request("POST", config.DIGITAL_TWIN_QUALITY_URL + "/analyze", payload=review_payload, timeout=200)
    with db.get_conn() as conn:
        conn.execute("UPDATE digital_twin_sessions SET quality_review=%s::jsonb WHERE id=%s", (_json(review), session_id))
        _record_event(conn, session_id, "quality.review_completed", {"provider": provider or "deterministic"})
        conn.commit()
    return review


async def _close_simulated_workforce_tasks(session_id: str) -> None:
    """Release human capacity after an isolated simulation finishes.

    The assignment remains in Workforce history with a completed status, but
    synthetic test work must never consume a real adviser slot forever.
    """
    try:
        state = await _request(
            "GET", f"{config.WORKFORCE_API_URL}/external/state?conversation_id={session_id}", timeout=12
        )
        closed: list[str] = []
        for task in state.get("tasks", []):
            if task.get("status") != "assigned" or not task.get("task_id"):
                continue
            await _request(
                "POST", f"{config.WORKFORCE_API_URL}/external/outcome",
                payload={"conversation_id": session_id, "task_id": task["task_id"], "status": "completed"}, timeout=12,
            )
            closed.append(str(task["task_id"]))
        if closed:
            with db.get_conn() as conn:
                _record_event(conn, session_id, "workforce.simulated_tasks_completed", {"task_ids": closed})
                conn.commit()
    except HTTPException:
        # Completion of a transcript must not fail because a monitoring system
        # is briefly unavailable. The visible session remains completed.
        return


async def _close_voice(session: dict[str, Any]) -> None:
    voice = session.get("voice") or {}
    if voice.get("status") != "open":
        return
    room = await _voice("POST", f"/calls/{session['call_id']}/close", {}) or {}
    closed = {**voice, "status": "closed", "turns_spoken": room.get("turns_spoken"),
              "turns_failed": room.get("turns_failed"), "audio_seconds": room.get("audio_seconds")}
    with db.get_conn() as conn:
        conn.execute("UPDATE digital_twin_sessions SET voice=%s::jsonb WHERE id=%s", (_json(closed), session["id"]))
        _record_event(conn, str(session["id"]), "voice.closed", closed)
        conn.commit()


async def _fail_session(session_id: str, error: str) -> None:
    """Close a call that could not continue without discarding its evidence."""
    current = _get_session(session_id)
    if current["status"] != "active":
        return
    try:
        await _end_client(current)
    except HTTPException:
        pass
    with db.get_conn() as conn:
        conn.execute("UPDATE digital_twin_sessions SET status='failed',completed_at=now() WHERE id=%s", (session_id,))
        _record_event(conn, session_id, "simulation.failed", {"error": error})
        conn.commit()
    await _close_voice(current)
    await _close_simulated_workforce_tasks(session_id)
    try:
        await _run_quality_review(session_id)
    except HTTPException:
        pass
    await _report_call_outcome(session_id)


async def recover_interrupted_calls() -> None:
    """Close run calls left active by an app restart.

    A run is driven by an in-memory task, so after a restart nothing will
    advance its calls again: their rooms would stay open and silent. Each one is
    closed like a failed call (room closed, Workforce released, transcript reviewed).
    """
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id FROM digital_twin_sessions WHERE status='active' AND batch_id IS NOT NULL").fetchall()
    for (session_id,) in rows:
        try:
            await _fail_session(str(session_id), "interrupted: the Digital Twin app restarted during the run")
        except Exception:
            pass


async def _report_call_outcome(session_id: str) -> None:
    """Tell Workforce how the call ended, with its Quality result, so its KPIs are real.

    Sent once, after the review.  Workforce journals it; an outage there must
    never fail a finished call, so it is recorded as an event and skipped.
    """
    current = _get_session(session_id)
    summary = _quality_summary(current.get("quality_review"))
    result = (current.get("quality_review") or {}).get("result") or {}
    payload = {
        "conversation_id": session_id,
        "status": "failed" if current["status"] == "failed" else "completed",
        "turns": sum(1 for m in _messages(session_id) if m["role"] == "client"),
        "quality_score": summary.get("score"),
        "quality_llm_checked": result.get("status") == "free_quality_checked",
        # The real end time, so waiting-time KPIs stay right even for late reports.
        "ended_at": current["completed_at"].isoformat() if current.get("completed_at") else None,
        **_workforce_labels(current),
    }
    try:
        await _request("POST", f"{config.WORKFORCE_API_URL}/external/call", payload=payload, timeout=12)
        event = "workforce.call_outcome_reported"
    except HTTPException:
        event = "workforce.call_outcome_failed"
    with db.get_conn() as conn:
        _record_event(conn, session_id, event, payload)
        conn.commit()


async def _route_closing_turn(current: dict[str, Any]) -> None:
    """Route the client's last utterance, which no advisor turn follows.

    Jasmin often closes with the decision itself ("confirm the booking"); that
    is exactly the contact Workforce must hand to a human, so it cannot be
    dropped just because the simulated conversation ends there.
    """
    transcript = _messages(str(current["id"]))
    if not transcript or transcript[-1]["role"] != "client":
        return
    last = transcript[-1]
    channel = "PHONE" if (current.get("voice") or {}).get("room") else "CHAT"
    payload = _workforce_request(last["content"], current["language"], int(last["sequence"]), channel, _ai_stalled_turns(transcript), _offers_shown(transcript))
    payload["conversation_id"] = str(current["id"])
    payload.update(_workforce_labels(current))
    try:
        workforce = await _request("POST", config.WORKFORCE_EXTERNAL_ROUTING_URL, payload=payload, timeout=20)
    except HTTPException:
        return
    route = {"session_id": str(current["id"]), **workforce, "advisor_reason": "closing_client_turn"}
    with db.get_conn() as conn:
        conn.execute("UPDATE digital_twin_sessions SET workforce_route=%s::jsonb WHERE id=%s", (_json(route), current["id"]))
        _record_event(conn, str(current["id"]), "workforce.route_proposed", route)
        conn.commit()


@router.post("/sessions/{session_id}/complete")
async def complete_session(session_id: str) -> dict[str, Any]:
    current = _get_session(session_id)
    if current["status"] == "active":
        await _route_closing_turn(current)
        await _end_client(current)
        with db.get_conn() as conn:
            conn.execute("UPDATE digital_twin_sessions SET status='completed',completed_at=now() WHERE id=%s", (session_id,))
            _record_event(conn, session_id, "simulation.completed", {})
            conn.commit()
        await _close_voice(current)
        await _close_simulated_workforce_tasks(session_id)
        # A completed Digital Twin always enters Quality automatically.  A
        # Quality outage must not undo the completed customer transcript.
        try:
            await _run_quality_review(session_id)
        except HTTPException as exc:
            with db.get_conn() as conn:
                _record_event(conn, session_id, "quality.review_failed", {"status": exc.status_code, "detail": str(exc.detail)})
                conn.commit()
        await _report_call_outcome(session_id)
    return _view(session_id)


@router.post("/sessions/{session_id}/analyze")
async def analyze_session(session_id: str, body: dict[str, Any]) -> Any:
    return await _run_quality_review(session_id, body.get("provider"))


# ---------------------------------------------------------------------------
# Written run report (stored; generated on request through Amira's LLM gateway)

def _stored_report(batch_id: str) -> dict[str, Any] | None:
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute("SELECT batch_id, sections, provider, model, warnings, call_reviews, created_at FROM digital_twin_reports WHERE batch_id=%s",
                           (batch_id,)).fetchone()


async def _report_ask(messages: list[dict[str, str]]) -> dict[str, Any]:
    """One chat call through Amira's LLM gateway with the report model; {} if no provider answered."""
    try:
        return await _request("POST", config.DIGITAL_TWIN_REPORT_LLM_URL, payload={
            "messages": messages, "provider_order": config.DIGITAL_TWIN_REPORT_PROVIDERS, "timeout_seconds": 60,
            "openrouter_model": config.DIGITAL_TWIN_REPORT_MODEL,
        }, timeout=90)
    except HTTPException:
        return {}


@router.post("/sessions/{session_id}/ai-quality-report")
async def ai_quality_report(session_id: str) -> dict[str, Any]:
    """Structured AI quality report of one finished call, written by the API model and stored."""
    session = _get_session(session_id)
    if session.get("status") == "active":
        raise HTTPException(409, "Wait for the call to end before writing its quality report.")
    transcript = _messages(session_id)
    if not transcript:
        raise HTTPException(409, "This call has no conversation to review.")
    try:
        report = await twin_report.call_quality_report(session, _quality_summary(session.get("quality_review")), transcript, _report_ask)
    except ValueError as exc:
        raise HTTPException(502, f"The AI quality report could not be written: {exc}.") from exc
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    with db.get_conn() as conn:
        conn.execute("UPDATE digital_twin_sessions SET ai_quality_report=%s::jsonb WHERE id=%s", (_json(report), session_id))
        conn.commit()
    return report


@router.get("/batches/{batch_id}/report")
async def run_report(batch_id: str) -> dict[str, Any]:
    report = _stored_report(batch_id)
    if not report:
        raise HTTPException(404, "No written report for this run yet.")
    return report


@router.post("/batches/{batch_id}/report")
async def generate_run_report(batch_id: str) -> dict[str, Any]:
    """Write the run report section by section with an LLM, check its numbers, store it."""
    data = await batch(batch_id)
    if data.get("status") == "running":
        raise HTTPException(409, "Wait for the run to finish before writing its report.")
    reviews = await quality_reports(batch_id=batch_id, limit=200)

    ask = _report_ask

    # AI reading of every call: the analyst's local-model check where it ran, an API model otherwise.
    gate = asyncio.Semaphore(3)

    async def review_one(review: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            return await twin_report.review_call(review, _messages(str(review["id"])), ask)

    call_reviews = list(await asyncio.gather(*(review_one(r) for r in reviews)))
    facts = twin_report.build_facts(data, reviews, call_reviews)

    sections = []
    for key, title, focus in twin_report.SECTIONS:
        sections.append(await twin_report.write_section(key, title, focus, facts[key], ask))
    if not any(s["text"] for s in sections):
        raise HTTPException(502, "No language model answered; the report was not written.")
    first = next((s for s in sections if s.get("provider")), {})
    warnings = [{"section": s["title"], "warning": s["warning"]} for s in sections if s.get("warning")]
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO digital_twin_reports(batch_id, sections, facts, provider, model, warnings, call_reviews, created_at) "
            "VALUES (%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,now()) ON CONFLICT (batch_id) DO UPDATE SET "
            "sections=EXCLUDED.sections, facts=EXCLUDED.facts, provider=EXCLUDED.provider, model=EXCLUDED.model, "
            "warnings=EXCLUDED.warnings, call_reviews=EXCLUDED.call_reviews, created_at=now()",
            (batch_id, _json(sections), _json(facts), first.get("provider"), first.get("model"), _json(warnings), _json(call_reviews)),
        )
        conn.commit()
    return _stored_report(batch_id)
