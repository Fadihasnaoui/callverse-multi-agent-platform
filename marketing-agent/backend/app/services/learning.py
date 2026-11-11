"""Phase 8 - the learning engine. Groups a business's published content by
its features (format, platform, weekday, hour, media) and measures each
group against real downstream outcomes - leads and conversions/revenue, not
likes. Deliberately interpretable (plain grouped rates, with sample sizes
shown) rather than a model: one business's post volume is far too small
for anything fancier to be honest. The result is written to
learning_insights, which the Phase 1 content engine reads before every
generation - that is the loop closing."""
import logging
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from app import config, db
from app.ai import prompts, run as ai_run
from app.ai.schemas import SummaryOut
from app.services import memory

log = logging.getLogger("learning")
TUNIS = ZoneInfo("Africa/Tunis")
MIN_POSTS_FOR_CONFIDENCE = 10

def _daypart(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _load_items(conn, business_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.id, c.type, c.platform, c.asset_refs, c.rendered_video_path,
               COALESCE(c.published_at, c.scheduled_for) AS posted_at, c.analytics_features->>'topic',
               (SELECT count(*) FROM leads l WHERE l.source_content_item_id = c.id AND l.status = 'new'),
               (SELECT count(*) FROM conversions v JOIN leads l ON l.id = v.lead_id
                  WHERE l.source_content_item_id = c.id),
               (SELECT COALESCE(sum(v.value_amount), 0) FROM conversions v JOIN leads l ON l.id = v.lead_id
                  WHERE l.source_content_item_id = c.id AND v.value_currency='TND'),
               COALESCE((c.analytics_features->>'like_count')::int,
                        (c.analytics_features->>'reactions_count')::int, 0),
               COALESCE((c.analytics_features->>'comments_count')::int, 0), c.measured_engagement,c.metrics_as_of,c.measured_reach
        FROM content_analytics c WHERE c.business_id = %s AND c.status = 'published'
        """,
        (business_id,),
    ).fetchall()
    items = []
    for r in rows:
        posted = r[5].astimezone(TUNIS) if r[5] else None
        likes = r[10]
        comments = r[11]
        items.append({
            "id": str(r[0]),
            "format": r[1],
            "platform": r[2],
            "media": "video" if r[4] else ("photo" if r[3] else "text"),
            "weekday": posted.strftime("%A") if posted else None,
            "daypart": _daypart(posted.hour) if posted else None,
            "topic": r[6],
            "leads": r[7],
            "conversions": r[8],
            "revenue": float(r[9]),
            "likes": likes,
            "comments": comments,
            "engagement": r[12] if r[12] is not None else likes + comments,
            "metrics_available": r[13] is not None,
            "reach": r[14],
        })
    return items


def _group(items: list[dict], dimension: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item[dimension]:
            buckets[item[dimension]].append(item)
    groups = []
    for value, members in buckets.items():
        n = len(members)
        total_eng = sum(m["engagement"] for m in members)
        comparable=[m for m in members if m.get('reach') is not None and m['reach']>0]
        reach=sum(m['reach'] for m in comparable)
        groups.append({
            "value": value,
            "posts": n,
            "leads": sum(m["leads"] for m in members),
            "conversions": sum(m["conversions"] for m in members),
            "revenue": round(sum(m["revenue"] for m in members), 2),
            "leads_per_post": round(sum(m["leads"] for m in members) / n, 2),
            "conversions_per_post": round(sum(m["conversions"] for m in members) / n, 2),
            "total_likes": sum(m["likes"] for m in members),
            "total_comments": sum(m["comments"] for m in members),
            "avg_engagement": round(total_eng / n, 1),
            "measured_posts":len(comparable),
            "leads_per_1000_post_reach":round(sum(m['leads'] for m in comparable)/reach*1000,2) if reach else None,
            "conversions_per_1000_post_reach":round(sum(m['conversions'] for m in comparable)/reach*1000,2) if reach else None,
            "evidence":"descriptive association; exposure and conversion delay must be comparable",
        })
    return sorted(groups, key=lambda g: (g["conversions_per_post"], g['leads_per_post'], g["avg_engagement"]), reverse=True)


def run_learning(business_id: str) -> dict:
    with db.get_conn() as conn:
        items = _load_items(conn, business_id)
    if not items:
        return {"status": "insufficient_data", "detail": "no published content yet"}

    by_dimension = {d: _group(items, d) for d in ("format", "platform", "media", "weekday", "daypart")}
    has_outcomes = any(i["leads"] > 0 or i["conversions"] > 0 for i in items)
    if has_outcomes:
        top_content = sorted(items, key=lambda i: (i["conversions"], i["leads"], i["engagement"]), reverse=True)[:5]
    else:
        top_content = sorted(items, key=lambda i: i["engagement"], reverse=True)[:5]
    total_eng = sum(i["engagement"] for i in items)
    insights = {
        "sample_size": len(items),
        "total_engagement": total_eng,
        "avg_engagement_per_post": round(total_eng / len(items), 1) if items else 0,
        "learning_from": "leads+conversions+engagement" if has_outcomes else "engagement (no leads/conversions yet)",
        "confidence": "descriptive - not evidence of improvement" if len(items) >= MIN_POSTS_FOR_CONFIDENCE and has_outcomes else "low - insufficient comparable outcomes",
        "measured_posts": sum(bool(i.get('metrics_available')) for i in items),
        "attributed_posts": sum(i['leads']>0 or i['conversions']>0 for i in items),
        "objective": "conversions per post, then qualified messages per post; descriptive associations only",
        "revenue_currency": "TND",
        "by_feature": by_dimension,
        "top_content": [{k: i[k] for k in ("topic", "format", "media", "weekday", "daypart", "leads", "conversions", "likes", "comments", "engagement")} for i in top_content],
    }

    # Enrich with comment sentiment data if available
    try:
        from app.services.sentiment import load_sentiment_context
        sent = load_sentiment_context(business_id)
        if sent:
            insights["comment_sentiment"] = sent
    except Exception:
        log.exception("loading sentiment context failed")

    # Compare against industry benchmarks
    try:
        from app.services.strategist import _load_industry_benchmarks
        bm = _load_industry_benchmarks()
        if bm:
            overall = bm.get("overall", {}).get("engagement_rate", {})
            travel = bm.get("travel_content", {}).get("engagement_rate", {})
            our_avg = insights["avg_engagement_per_post"]
            insights["industry_benchmark"] = {'comparable':False,'reason':'Raw interactions per post cannot be compared with percentage engagement rates.'}
    except Exception:
        log.exception("loading benchmarks failed")

    summary_text = None
    try:
        topics = " ".join(str(i.get("topic") or "") for i in insights["top_content"])
        summary = ai_run(
            "learning", prompts.LEARNING.system,
            str(insights) + memory.recall_block(business_id, "learning", topics),
            SummaryOut, model=config.CONTENT_MODEL, version=prompts.LEARNING.version,
        )
        insights["recommendations"] = summary.recommendations
        summary_text = summary.summary
    except Exception:  # noqa: BLE001 - the numbers are still worth storing without prose
        log.exception("learning summary failed")

    import json
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO learning_insights (business_id, source, insights, summary_text) VALUES (%s,'periodic_job',%s,%s)",
            (business_id, json.dumps(insights), summary_text),
        )
        conn.commit()
    if summary_text:
        memory.remember(business_id, "learning", "insight", f"{datetime.now():%Y-%m-%d}: {summary_text}", source="learning_job")
    try:
        from app.services.notifications import notify
        notify(business_id, "learning_updated",
               f"Learning updated ({len(items)} posts analyzed)",
               (summary_text or "")[:300],
               {"sample_size": len(items), "has_outcomes": has_outcomes})
    except Exception:
        pass
    return {"status": "ok", "summary": summary_text, "insights": insights}


@db.single_task('marketing-learning')
def run_learning_all() -> int:
    """Scheduler entry point — runs every 3 days.
    First refreshes comment sentiment, then runs the learning engine."""
    with db.get_conn() as conn:
        ids = [str(r[0]) for r in conn.execute(
            "SELECT DISTINCT business_id FROM content_analytics WHERE status = 'published'").fetchall()]
    for business_id in ids:
        # Step 1: refresh comment sentiment (feeds into learning)
        try:
            from app.services.sentiment import run_sentiment
            run_sentiment(business_id)
        except Exception:  # noqa: BLE001
            log.exception("sentiment refresh failed for %s", business_id)
        # Step 2: run learning with fresh sentiment data
        try:
            run_learning(business_id)
        except Exception:  # noqa: BLE001
            log.exception("learning failed for %s", business_id)
    return len(ids)
