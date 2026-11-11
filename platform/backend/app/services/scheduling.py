"""Phase 5 (scheduling intelligence) - the agent proposes WHEN to post, not
just what to post. For now, before any business has enough real performance
history for Phase 8's learning engine to have anything to learn from, this
runs on two real, defensible inputs instead of guessing: (1) general digital
marketing scheduling principles plus genuinely known facts about the
Tunisian market specifically (not invented, not business-specific data -
general market knowledge, same category as "weekends are different in
different countries", not a claim about Ozone itself), and (2) the
business's own real posting-cadence history from its Phase 0.5 audit, so it
never repeats the kind of 78-day silent gap that audit actually found.

The moment Phase 8 exists and a business has real engagement/conversion
history of its own, that becomes the dominant signal here and this general
knowledge becomes the fallback for businesses still too new to have any
data yet - not a permanent state."""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app import config, db
from app.ai import prompts, run as ai_run
from app.ai.schemas import ScheduleOut
from app.services import memory, strategist

TUNIS = ZoneInfo("Africa/Tunis")


def _parse_local(value: str) -> datetime:
    """The model proposes Tunisia local time; a bare timestamp is read as such."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=TUNIS)


def _schedule_guard(output) -> list[str]:
    when = _parse_local(output.proposed_datetime)
    now = datetime.now(TUNIS)
    if when < now + timedelta(hours=1):
        return [f"proposed_datetime must be at least 1 hour after now ({now:%Y-%m-%dT%H:%M} Tunisia time)"]
    if when > now + timedelta(days=21):
        return ["proposed_datetime must be within the next 21 days"]
    return []


def _latest_audit_cadence(conn, business_id: str) -> dict | None:
    row = conn.execute(
        "SELECT findings->'cadence' FROM social_audits WHERE business_id = %s "
        "ORDER BY created_at DESC LIMIT 1",
        (business_id,),
    ).fetchone()
    return row[0] if row else None


def _latest_learning_insight(conn, business_id: str) -> dict | None:
    row = conn.execute(
        "SELECT insights FROM learning_insights WHERE business_id = %s "
        "AND source <> 'simulation_baseline' ORDER BY created_at DESC LIMIT 1",
        (business_id,),
    ).fetchone()
    return row[0] if row else None


def _recently_scheduled(conn, business_id: str) -> list[str]:
    """Other content already scheduled/approved but not yet published -
    without this, the agent could cluster several posts on the same day
    with nothing else to go on."""
    rows = conn.execute(
        "SELECT scheduled_for FROM content_items WHERE business_id = %s "
        "AND status = 'approved' AND scheduled_for IS NOT NULL AND published_at IS NULL "
        "ORDER BY scheduled_for",
        (business_id,),
    ).fetchall()
    return [r[0].isoformat() for r in rows]


def suggest_schedule(business_id: str, content_id: str) -> dict:
    with db.get_conn() as conn:
        cadence = _latest_audit_cadence(conn, business_id)
        learning = _latest_learning_insight(conn, business_id)
        upcoming = _recently_scheduled(conn, business_id)

        exists = conn.execute(
            "SELECT 1 FROM content_items WHERE id = %s AND business_id = %s", (content_id, business_id)
        ).fetchone()
        if not exists:
            raise ValueError("content item not found")

    # Strategic context for smarter scheduling
    try:
        strategic_brief = strategist.build_strategic_brief(business_id, context="schedule")
    except Exception:
        strategic_brief = ""

    now = datetime.now(TUNIS).strftime("%Y-%m-%dT%H:%M (%A)")
    result = ai_run(
        "schedule", prompts.SCHEDULE.system,
        f"Current date/time in Tunisia: {now}. proposed_datetime must be Tunisia local time.\n"
        f"This business's real posting cadence history (from their audit, or null if none yet): {cadence}\n"
        f"What has actually worked for this business before (or null if none yet): {learning}\n"
        f"Other content already scheduled, do not cluster on the same day: {upcoming}\n"
        f"{'--- STRATEGIC CONTEXT ---' + chr(10) + strategic_brief + chr(10) if strategic_brief else ''}"
        f"{memory.recall_block(business_id, 'scheduling')}\n"
        "Propose the single best date/time for this one new piece of content, at least a few "
        "hours from now.",
        ScheduleOut, version=prompts.SCHEDULE.version, guard=_schedule_guard,
    )

    proposed = _parse_local(result.proposed_datetime)
    rationale = result.rationale

    with db.get_conn() as conn:
        conn.execute(
            "UPDATE content_items SET proposed_schedule = %s, proposed_schedule_rationale=%s WHERE id = %s AND business_id=%s",
            (proposed, rationale, content_id, business_id),
        )
        conn.commit()

    return {"content_id": content_id, "proposed_datetime": proposed.isoformat(), "rationale": rationale}
