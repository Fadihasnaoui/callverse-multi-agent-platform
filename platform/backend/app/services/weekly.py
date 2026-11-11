"""Weekly planning - the owner gives one main idea for the week and the agent
lays out a whole week of posts: which day/time, which platform, which format,
and a distinct angle on the idea for each. Every slot is then written by the
normal content engine (same truthfulness rules, same ref-coded CTA, same
learning-insights loop) and lands as a *pending* item with its slot already in
scheduled_for. Nothing is approved here - the owner reviews the week and
approves; the approval gate is untouched."""
import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app import config, db
from app.services import content as content_service
from app.services import image_creative
from app.ai import prompts, run as ai_run
from app.ai.schemas import PlanOut
from app.services import memory, strategist
from app.services import jobs

log = logging.getLogger("weekly")
TUNIS = ZoneInfo("Africa/Tunis")
ALLOWED_TYPES = ("static_post", "reel", "avatar_clip")
ALLOWED_PLATFORMS = ("facebook", "instagram")
MAX_POSTS = 14

def _plan_guard(first_day: int, platforms: list[str], types: list[str], posts: int, min_reels: int = 0):
    """Hard rules the model must satisfy; a violation is sent back for one repair."""
    def guard(plan) -> list[str]:
        problems = []
        if len(plan.slots) != posts:
            problems.append(f"you returned {len(plan.slots)} slots; exactly {posts} are required")
        for slot in plan.slots:
            if slot.day_offset < first_day:
                problems.append(f"day_offset {slot.day_offset} is in the past; use {first_day} to 6")
            if slot.platform not in platforms:
                problems.append(f"platform '{slot.platform}' is not allowed; use one of {platforms}")
            if slot.content_type not in types:
                problems.append(f"content_type '{slot.content_type}' is not allowed; use one of {types}")
        if posts >= 3 and len({slot.time for slot in plan.slots}) < 2:
            problems.append("every post has the same time; vary the posting times (for example 12:30, 18:00, 20:30)")
        if min_reels > 0 and "reel" in types:
            future_reels = sum(1 for s in plan.slots if s.content_type == "reel" and s.day_offset >= first_day)
            if future_reels < min_reels:
                problems.append(
                    f"only {future_reels} reel(s) on day_offset {first_day}-6 but at least {min_reels} are required "
                    f"(industry data shows video outperforms static; convert some static_post slots to reel, "
                    f"placing reels on day_offset {first_day} or later)"
                )
        return problems[:6]
    return guard


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _clean_slots(raw: list, week_start: date, platforms: list[str], types: list[str], count: int) -> list[dict]:
    slots = []
    for s in raw:
        try:
            offset = int(s["day_offset"])
            hour, minute = (int(x) for x in str(s["time"]).split(":")[:2])
            topic = str(s["topic"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 <= offset <= 6 and 0 <= hour <= 23 and 0 <= minute <= 59 and topic):
            continue
        platform = s.get("platform") if s.get("platform") in platforms else platforms[0]
        ctype = s.get("content_type") if s.get("content_type") in types else types[0]
        when = datetime.combine(week_start + timedelta(days=offset), datetime.min.time(), TUNIS).replace(
            hour=hour, minute=minute
        )
        slots.append({"when": when, "platform": platform, "content_type": ctype, "topic": topic})
    slots.sort(key=lambda x: x["when"])
    return slots[:count]


def plan_week(
    business_id: str,
    main_idea: str,
    week_start: str | None = None,
    posts: int = 5,
    platforms: list[str] | None = None,
    content_types: list[str] | None = None,
    generate_images: bool = False,
) -> dict:
    main_idea = (main_idea or "").strip()
    if not main_idea:
        raise ValueError("main_idea is required")
    platforms = [p for p in (platforms or ["facebook", "instagram"]) if p in ALLOWED_PLATFORMS]
    types = [t for t in (content_types or ["static_post", "reel"]) if t in ALLOWED_TYPES]
    if not platforms or not types:
        raise ValueError(f"platforms must be from {ALLOWED_PLATFORMS}, content_types from {ALLOWED_TYPES}")
    if not 1 <= posts <= MAX_POSTS:
        raise ValueError(f"posts must be between 1 and {MAX_POSTS}")

    today = datetime.now(TUNIS).date()
    if week_start:
        start = _monday(date.fromisoformat(week_start))
    else:
        # From Thursday on, "this week" has too little time left for a complete campaign.
        start = _monday(today) + timedelta(days=7) if today.weekday() >= 3 else _monday(today)
    if start + timedelta(days=6) < today:
        raise ValueError("that week is already over")
    first_day = max(0, (today - start).days)  # days of this week that are already gone

    with db.get_conn() as conn:
        business = conn.execute("SELECT brand_voice FROM businesses WHERE id = %s", (business_id,)).fetchone()
    if not business:
        raise ValueError("business not found")

    try:
        strategic_brief = strategist.build_strategic_brief(business_id, context="weekly")
    except Exception:
        strategic_brief = ""

    try:
        content_directives = strategist.build_content_directives(business_id)
    except Exception:
        content_directives = ""

    # Compute minimum reel count from benchmarks if reels are allowed
    min_reels = 0
    if "reel" in types and posts >= 3:
        try:
            bm = strategist._load_industry_benchmarks()
            if bm:
                # If Ozone's reel ratio is below 30%, enforce at least 40% reels
                with db.get_conn() as conn:
                    counts = conn.execute(
                        "SELECT COUNT(*) FILTER (WHERE type = 'reel'), COUNT(*) "
                        "FROM content_items WHERE business_id = %s AND status = 'published'",
                        (business_id,),
                    ).fetchone()
                reel_pct = counts[0] / max(counts[1], 1) * 100
                if reel_pct < 30:
                    min_reels = max(1, round(posts * 0.4))
        except Exception:
            pass

    def make_plan():
        return ai_run(
            "weekly", prompts.WEEKLY.system,
            f"Brand voice: {business[0] or 'friendly, direct, Tunisian'}\n"
            f"Week: Monday {start.isoformat()} to Sunday {(start + timedelta(days=6)).isoformat()} "
            f"(today in Tunisia is {today.isoformat()} and it is now {datetime.now(TUNIS).strftime('%H:%M')}; "
            f"use only day_offset {first_day} to 6; for day_offset {first_day} (today) "
            f"use times AFTER {(datetime.now(TUNIS) + timedelta(hours=2)).strftime('%H:%M')} only)\n"
            f"Number of posts: exactly {posts}\n"
            f"Allowed platforms: {platforms}\nAllowed formats: {types}\n\n"
            f"{'--- STRATEGIC BRIEF (use this to plan strategically, not randomly) ---' + chr(10) + strategic_brief + chr(10) + chr(10) if strategic_brief else ''}"
            f"{content_directives + chr(10) + chr(10) if content_directives else ''}"
            f"{memory.recall_block(business_id, 'weekly', main_idea)}\n"
            f"Owner's main idea for the week:\n{main_idea}",
            PlanOut, timeout=90, model=config.CONTENT_MODEL, version=prompts.WEEKLY.version,
            guard=_plan_guard(first_day, platforms, types, posts, min_reels=min_reels),
        )
    result = PlanOut.model_validate(jobs.step('plan', lambda: make_plan().model_dump()))
    now = datetime.now(TUNIS)
    raw_slots = [slot.model_dump() for slot in result.slots]
    all_clean = _clean_slots(raw_slots, start, platforms, types, posts)
    slots = [s for s in all_clean if s["when"] > now]
    if not slots:
        log.warning("planner produced no usable slots: %s", raw_slots)
        raise ValueError("the planner returned no slots that are still in the future - choose a later week or try again")

    memory.remember(business_id, "content", "plan", f"Week of {start.isoformat()}: owner's main idea was \"{main_idea[:300]}\"", source="weekly_plan")
    created, failed, media_failed = [], [], []
    for index, slot in enumerate(slots):
        try:
            item = jobs.step(f'draft-{index}',lambda: content_service.generate_content(
                business_id, slot["platform"], slot["topic"], slot["content_type"]
            ))
            with db.get_conn() as conn:
                conn.execute(
                    "UPDATE content_items SET scheduled_for = %s, features = features || %s::jsonb WHERE id = %s",
                    (slot["when"], json.dumps({"week_of": start.isoformat(), "week_idea": main_idea}), item["id"]),
                )
                conn.commit()
            image_url = None
            image_status = "not_requested"
            if generate_images:
                try:
                    image = image_creative.generate_for_content(item["id"])
                    image_url = image["image_url"]
                    image_status = "generated"
                except Exception as image_error:
                    image_status = "failed"
                    media_failed.append({"id": item["id"], "topic": slot["topic"], "error": str(image_error)})
            created.append({"id": item["id"], "platform": slot["platform"], "type": slot["content_type"],
                            "scheduled_for": slot["when"].isoformat(), "topic": slot["topic"],
                            "image_url": image_url, "image_status": image_status})
        except Exception as error:  # one bad slot must not lose the rest of the week
            failed.append({"topic": slot["topic"], "error": str(error)})
    try:
        from app.services.notifications import notify
        notify(business_id, "week_planned",
               f"Week of {start.isoformat()} planned: {len(created)} posts",
               f"Idea: {main_idea[:200]}",
               {"week_start": start.isoformat(), "posts": len(created), "failed": len(failed)})
    except Exception:
        pass

    return {"week_start": start.isoformat(), "created": created, "failed": failed,
            "media_failed": media_failed, "status": "pending"}
