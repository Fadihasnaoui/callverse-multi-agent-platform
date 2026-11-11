"""Phases 6-9 read/write endpoints: leads (6), conversions (7), learning (8),
owner report (9)."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import db
from app.services import memory
from app.services import learning as learning_service
from app.services import metrics as metrics_service
from app.services import sentiment as sentiment_service
from app.services import simulation
from app.services import strategist
from app.services import jobs

router = APIRouter(prefix="/businesses/{business_id}", tags=["outcomes"])


class LeadFeedback(BaseModel):
    is_lead: bool


class ReportConversion(BaseModel):
    lead_id: str | None = None
    value_amount: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    value_currency: str = Field(default="TND", pattern=r'^[A-Z]{3}$')
    notes: str | None = None
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=200)


@router.get("/leads")
def list_leads(business_id: str, status: str | None = None):
    query = (
        "SELECT l.id, l.source, l.platform, l.source_content_item_id, l.author_name, l.message_text, "
        "l.intent_confidence, l.status, l.created_at, left(c.caption, 90), "
        "EXISTS (SELECT 1 FROM conversions v WHERE v.lead_id = l.id) "
        "FROM leads l LEFT JOIN content_items c ON c.id = l.source_content_item_id WHERE l.business_id = %s"
    )
    params = [business_id]
    if status:
        query += " AND l.status = %s"
        params.append(status)
    with db.get_conn() as conn:
        rows = conn.execute(query + " ORDER BY l.created_at DESC LIMIT 200", params).fetchall()
    return [
        {"id": str(r[0]), "source": r[1], "platform": r[2],
         "source_content_item_id": str(r[3]) if r[3] else None, "author_name": r[4],
         "message_text": r[5], "intent_confidence": r[6], "status": r[7], "created_at": r[8].isoformat(),
         "source_caption": r[9], "converted": r[10]}
        for r in rows
    ]


@router.post("/leads/{lead_id}/feedback")
def lead_feedback(business_id: str, lead_id: str, payload: LeadFeedback):
    """Owner corrects the agent's lead scoring. The status flips immediately and
    the judgement goes into the 'leads' memory, which steers future scoring."""
    with db.get_conn() as conn:
        row = conn.execute(
            "UPDATE leads SET status = %s WHERE id = %s AND business_id = %s "
            "RETURNING message_text, intent_confidence",
            ("new" if payload.is_lead else "ignored", lead_id, business_id),
        ).fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="lead not found")
    memory.learn_lead_judgement(business_id, row[0], payload.is_lead, row[1])
    return {"id": lead_id, "status": "new" if payload.is_lead else "ignored"}


@router.post("/conversions")
def report_conversion(business_id: str, payload: ReportConversion):
    fingerprint = hashlib.sha256(json.dumps(payload.model_dump(exclude={'idempotency_key'}, mode='json'),sort_keys=True).encode()).hexdigest()
    key = payload.idempotency_key or fingerprint
    with db.get_conn() as conn:
        conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (business_id+key,))
        prior = conn.execute('SELECT id,request_fingerprint FROM conversions WHERE business_id=%s AND idempotency_key=%s', (business_id,key)).fetchone()
        if prior:
            if prior[1] != fingerprint: raise HTTPException(409,'idempotency key was already used with different sale details')
            return {'id':str(prior[0]),'replayed':True}
        if payload.lead_id:
            lead = conn.execute(
                "SELECT 1 FROM leads WHERE id = %s AND business_id = %s", (payload.lead_id, business_id)
            ).fetchone()
            if not lead:
                raise HTTPException(status_code=404, detail="lead not found for this business")
        lead_text = None
        if payload.lead_id:
            lead_text = conn.execute("SELECT message_text FROM leads WHERE id = %s", (payload.lead_id,)).fetchone()[0]
        row = conn.execute(
            "INSERT INTO conversions (business_id, lead_id, value_amount, value_currency, notes,idempotency_key,request_fingerprint) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (business_id, payload.lead_id, payload.value_amount, payload.value_currency, payload.notes,key,fingerprint),
        ).fetchone()
        jobs.enqueue('conversion_memory', {'conversion_id':str(row[0]),'lead_text':lead_text,'notes':payload.notes},business_id,key=str(row[0]),conn=conn)
        conn.commit()
    try:
        from app.services.notifications import notify
        notify(business_id, "lead_converted",
               f"Sale recorded: {payload.value_amount or 0} {payload.value_currency}",
               payload.notes or "",
               {"lead_id": payload.lead_id, "value": payload.value_amount})
    except Exception:
        pass
    return {"id": str(row[0])}


@router.post("/learning/run")
def run_learning(business_id: str):
    # Refresh sentiment first, then run learning with fresh data
    try:
        sentiment_service.run_sentiment(business_id)
    except Exception:
        pass  # sentiment failure shouldn't block learning
    return learning_service.run_learning(business_id)


@router.post("/sentiment/run")
def run_sentiment(business_id: str):
    return sentiment_service.run_sentiment(business_id)


@router.get("/sentiment")
def get_sentiment(business_id: str):
    ctx = sentiment_service.load_sentiment_context(business_id)
    if not ctx:
        return {"status": "no_data", "detail": "no comment analysis available yet"}
    return {"status": "ok", **ctx}


@router.get("/content-directives")
def get_content_directives(business_id: str):
    directives = strategist.build_content_directives(business_id)
    return {"directives": directives}


@router.get("/self-benchmark")
def get_self_benchmark(business_id: str, days: int = 30):
    result = strategist.self_benchmark(business_id, period_days=days)
    if not result:
        return {"status": "insufficient_data", "detail": "not enough published posts in both periods"}
    return {"status": "ok", **result}


@router.get("/content-recommendations")
def get_content_recommendations(business_id: str):
    """Return AI-powered topic recommendations based on sentiment analysis,
    benchmarks, and learning data — shown to the user before they generate."""
    recommendations = []

    # 1. Sentiment-driven topics
    ctx = sentiment_service.load_sentiment_context(business_id)
    if ctx and ctx.get("top_topics"):
        topics = list(ctx["top_topics"].items())[:5]
        question_pct = 0
        total = ctx.get("total_comments", 0)
        if total > 0:
            with db.get_conn() as conn:
                q_count = conn.execute(
                    """SELECT SUM((ca.sentiment->>'question')::int)
                    FROM comment_analysis ca JOIN content_items c ON c.id = ca.content_item_id
                    WHERE c.business_id = %s""", (business_id,),
                ).fetchone()
                question_pct = round((q_count[0] or 0) / total * 100) if q_count[0] else 0

        for topic_name, count in topics:
            recommendations.append({
                "type": "sentiment",
                "topic": topic_name,
                "reason": f"Mentioned {count} times in comments — your audience is asking about this.",
                "weight": count,
            })
        if ctx.get("avg_buying_intent", 0) > 40:
            recommendations.append({
                "type": "intent",
                "topic": "pricing / offers",
                "reason": f"Buying intent is {ctx['avg_buying_intent']}/100 with {ctx.get('high_intent_comments', 0)} high-intent comments. Create content with clear pricing.",
                "weight": ctx["avg_buying_intent"],
            })
        if question_pct > 30:
            recommendations.append({
                "type": "engagement",
                "topic": "FAQ / answers",
                "reason": f"{question_pct}% of comments are questions. Address them in content to boost engagement.",
                "weight": question_pct,
            })

    # 2. Best-performing categories from learning
    try:
        with db.get_conn() as conn:
            insight_row = conn.execute(
                "SELECT insights FROM learning_insights WHERE business_id = %s "
                "AND source <> 'simulation_baseline' ORDER BY created_at DESC LIMIT 1",
                (business_id,),
            ).fetchone()
        if insight_row and insight_row[0]:
            top_content = insight_row[0].get("top_content", [])[:3]
            for item in top_content:
                if item.get("topic"):
                    recommendations.append({
                        "type": "learning",
                        "topic": item["topic"][:80],
                        "reason": f"Similar content got {item.get('engagement', 0)} engagement ({item.get('format', '?')} format).",
                        "weight": item.get("engagement", 0),
                    })
    except Exception:
        pass

    # 3. Content directives summary
    directives = strategist.build_content_directives(business_id)

    # 4. Self-benchmark insight
    benchmark_insight = None
    try:
        sb = strategist.self_benchmark(business_id)
        if sb and sb.get("insights"):
            benchmark_insight = sb["insights"]
    except Exception:
        pass

    return {
        "recommendations": sorted(recommendations, key=lambda r: r.get("weight", 0), reverse=True),
        "directives_summary": directives[:500] if directives else None,
        "benchmark_insight": benchmark_insight,
        "sentiment_summary": {
            "overall_sentiment": ctx.get("overall_sentiment", 0),
            "buying_intent": ctx.get("avg_buying_intent", 0),
            "top_topics": list((ctx or {}).get("top_topics", {}).keys())[:5],
        } if ctx else None,
    }


@router.get("/learning/simulation")
def learning_simulation(business_id: str):
    return simulation.campaign_sandbox(business_id)


@router.post("/learning/seed-baseline")
def seed_learning_baseline(business_id: str):
    baseline = simulation.campaign_sandbox(business_id)
    import json
    with db.get_conn() as conn:
        conn.execute("DELETE FROM learning_insights WHERE business_id = %s AND source = 'simulation_baseline'", (business_id,))
        row = conn.execute(
            "INSERT INTO learning_insights (business_id, source, insights, summary_text) "
            "VALUES (%s, 'simulation_baseline', %s, %s) RETURNING id, created_at",
            (business_id, json.dumps(baseline), baseline["learning"]["summary"]),
        ).fetchone()
        conn.commit()
    return {"id": str(row[0]), "created_at": row[1].isoformat(), "label": baseline["label"],
            "excluded_from_live_learning": True}


@router.post("/metrics/sync")
def sync_metrics(business_id: str,request: Request):
    if request.headers.get('prefer')=='respond-async':
        return JSONResponse(jobs.enqueue('metrics-sync',{},business_id,request.headers.get('idempotency-key')),status_code=202)
    return metrics_service.sync_business(business_id)


@router.get("/report")
def report(business_id: str, days: int = 30):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with db.get_conn() as conn:
        posts = conn.execute(
            "SELECT count(*) FILTER (WHERE status='published'), count(*) FILTER (WHERE status='pending'), "
            "count(*) FILTER (WHERE status='failed') FROM content_items WHERE business_id = %s AND created_at >= %s",
            (business_id, since),
        ).fetchone()
        leads = conn.execute(
            "SELECT count(*) FILTER (WHERE status='new'), count(*) FILTER (WHERE source_content_item_id IS NOT NULL AND status='new') "
            "FROM leads WHERE business_id = %s AND created_at >= %s",
            (business_id, since),
        ).fetchone()
        sales = conn.execute(
            "SELECT count(*), COALESCE(sum(value_amount) FILTER(WHERE value_currency='TND'),0) FROM conversions WHERE business_id = %s AND reported_at >= %s",
            (business_id, since),
        ).fetchone()
        top = conn.execute(
            """
            SELECT c.id, c.type, c.platform, left(c.caption, 90),
                   count(DISTINCT l.id) FILTER (WHERE l.status='new'), count(DISTINCT v.id),
                   COALESCE(sum(v.value_amount) FILTER(WHERE v.value_currency='TND'),0)
            FROM content_items c LEFT JOIN leads l ON l.source_content_item_id = c.id
            LEFT JOIN conversions v ON v.lead_id = l.id
            WHERE c.business_id = %s AND c.status='published' GROUP BY c.id
            ORDER BY 6 DESC, 5 DESC LIMIT 5
            """,
            (business_id,),
        ).fetchall()
        insight = conn.execute(
            "SELECT summary_text, insights->'recommendations', created_at FROM learning_insights "
            "WHERE business_id = %s AND source <> 'simulation_baseline' ORDER BY created_at DESC LIMIT 1",
            (business_id,),
        ).fetchone()
        audit = conn.execute(
            "SELECT posts_analyzed, findings, created_at FROM social_audits "
            "WHERE business_id = %s ORDER BY created_at DESC LIMIT 1",
            (business_id,),
        ).fetchone()
        real_learning_posts = conn.execute(
            "SELECT count(*) FROM content_items WHERE business_id = %s AND status = 'published'",
            (business_id,),
        ).fetchone()[0]
        performance = conn.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (pm.content_item_id) pm.reach, pm.impressions, pm.engagement
                FROM performance_metrics pm JOIN content_items ci ON ci.id = pm.content_item_id
                WHERE ci.business_id = %s AND COALESCE(ci.published_at,ci.created_at) >= %s
                ORDER BY pm.content_item_id, pm.fetched_at DESC, pm.id DESC
            )
            SELECT COALESCE(sum(reach),0), COALESCE(sum(impressions),0), COALESCE(sum(engagement),0) FROM latest
            """, (business_id, since),
        ).fetchone()
    baseline = simulation.campaign_sandbox(business_id)
    cadence = (audit[1] or {}).get("cadence", {}) if audit else {}
    audit_findings = audit[1] or {} if audit else {}

    try:
        strategic_brief = strategist.build_strategic_brief(business_id, context="report")
    except Exception:
        strategic_brief = ""
    try:
        strat_rec = strategist.recommend_next_content(business_id)
    except Exception:
        strat_rec = None

    audit_recommendations = []
    if cadence:
        audit_recommendations = [
            "Move from roughly one post per week to three consistent posts per week for the next four weeks.",
            f"Keep every publishing gap below 7 days; the historical maximum was {cadence.get('longest_gap_days', 'unknown')} days.",
            "Run a controlled mix of Reels and static posts, then compare reach, leads and sales after at least 10 published posts.",
            "Connect every comment or DM to its source post so the agent learns from business outcomes, not vanity metrics alone.",
        ]
    summary = insight[0] if insight else None
    recommendations = insight[1] if insight else None
    if not summary or "None" in summary:
        summary = (
            "Historical cadence is loaded as the starting baseline. Meta returned no comparable engagement metrics, "
            "so format performance remains an experiment rather than a proven result."
        )
        recommendations = audit_recommendations
    return {
        "period_days": days,
        "content": {"published": posts[0], "awaiting_approval": posts[1], "failed": posts[2]},
        "leads": {"total": leads[0], "traced_to_a_post": leads[1]},
        "sales": {"count": sales[0], "revenue": float(sales[1])},
        "performance": {"reach": int(performance[0]), "impressions": int(performance[1]),
                        "engagement": int(performance[2])},
        "top_content": [
            {"id": str(r[0]), "type": r[1], "platform": r[2], "caption_start": r[3],
             "leads": r[4], "sales": r[5], "revenue": float(r[6])} for r in top
        ],
        "latest_insight": {"summary": summary, "recommendations": recommendations or audit_recommendations,
                           "as_of": insight[2].isoformat()} if insight else None,
        "audit": {
            "posts_analyzed": audit[0], "cadence": cadence,
            "engagement_available": bool((audit[1] or {}).get("engagement_data_available")),
            "as_of": audit[2].isoformat(), "recommendations": audit_recommendations,
        } if audit else None,
        "learning": {
            "stage": "descriptive" if real_learning_posts >= 10 else "baseline",
            "real_published_posts": real_learning_posts,
            "confidence_target": 10,
            "progress_percent": min(real_learning_posts * 10, 100),
            "data_quality": "Published count is not statistical confidence; verify comparable metrics and attributed outcomes.",
        },
        "baseline": baseline,
        "strategist": strat_rec,
        "strategic_brief": strategic_brief,
        "audit_detail": {
            "health_score": audit_findings.get("health_score", {}),
            "strategy_mix": audit_findings.get("strategy_mix", {}),
            "format_mix": audit_findings.get("format_mix", {}),
            "category_performance": audit_findings.get("category_performance", {}),
            "best_and_worst": audit_findings.get("best_and_worst", {}),
            "time_distribution": audit_findings.get("time_distribution", {}),
            "platform_breakdown": audit_findings.get("platform_breakdown", {}),
        } if audit else None,
        "sentiment": sentiment_service.load_sentiment_context(business_id),
        "industry_benchmarks": strategist._load_industry_benchmarks(),
        "self_benchmark": strategist.self_benchmark(business_id),
    }
