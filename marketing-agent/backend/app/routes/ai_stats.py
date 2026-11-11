from fastapi import APIRouter

from app import config, db

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/stats")
def stats(days: int = 7):
    """Reliability of every AI task over the last N days, straight from the call log:
    how often it succeeded, how often it needed a repair, how slow and how many tokens."""
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT task, count(*), count(*) FILTER (WHERE ok), count(*) FILTER (WHERE ok AND attempts > 1),
                   round(avg(attempts)::numeric, 2), round(avg(latency_ms)), sum(prompt_tokens), sum(completion_tokens),
                   max(prompt_version)
            FROM ai_calls WHERE created_at > now() - make_interval(days => %s)
              AND (%s = '' OR created_at >= %s::timestamptz)
            GROUP BY task ORDER BY task
            """,
            (days, config.AI_STATS_SINCE, config.AI_STATS_SINCE or "1970-01-01T00:00:00Z"),
        ).fetchall()
        recent_errors = conn.execute(
            "SELECT task, created_at, left(error, 200) FROM ai_calls WHERE ok = false "
            "AND created_at > now() - make_interval(days => %s) "
            "AND (%s = '' OR created_at >= %s::timestamptz) ORDER BY id DESC LIMIT 10",
            (days, config.AI_STATS_SINCE, config.AI_STATS_SINCE or "1970-01-01T00:00:00Z"),
        ).fetchall()
    return {
        "days": days,
        "tracking_since": config.AI_STATS_SINCE or None,
        "tasks": [
            {"task": r[0], "calls": r[1], "succeeded": r[2], "success_rate": round(r[2] / r[1], 3),
             "needed_repair": r[3], "avg_attempts": float(r[4]), "avg_latency_ms": int(r[5] or 0),
             "prompt_tokens": r[6], "completion_tokens": r[7], "prompt_version": r[8]}
            for r in rows
        ],
        "recent_failures": [{"task": r[0], "at": r[1].isoformat(), "error": r[2]} for r in recent_errors],
    }


@router.get("/costs")
def cost_report(days: int = 30):
    """Exact OpenRouter charges recorded from each API response."""
    days = max(1, min(days, 365))
    with db.get_conn() as conn:
        summary = conn.execute(
            """
            SELECT COALESCE(sum(cost_usd), 0),
                   COALESCE(sum(cost_usd) FILTER (WHERE created_at >= current_date), 0),
                   count(*) FILTER (WHERE cost_usd IS NOT NULL),
                   COALESCE(avg(cost_usd) FILTER (WHERE ok AND cost_usd IS NOT NULL), 0)
            FROM ai_calls WHERE created_at > now() - make_interval(days => %s)
              AND (%s = '' OR created_at >= %s::timestamptz)
            """,
            (days, config.AI_STATS_SINCE, config.AI_STATS_SINCE or "1970-01-01T00:00:00Z"),
        ).fetchone()
        month_total = conn.execute(
            "SELECT COALESCE(sum(cost_usd), 0) FROM ai_calls "
            "WHERE created_at >= date_trunc('month', now()) "
            "AND (%s = '' OR created_at >= %s::timestamptz)",
            (config.AI_STATS_SINCE, config.AI_STATS_SINCE or "1970-01-01T00:00:00Z"),
        ).fetchone()[0]
        breakdown = conn.execute(
            """
            SELECT task, model, count(*), count(*) FILTER (WHERE ok), COALESCE(sum(cost_usd), 0),
                   COALESCE(avg(cost_usd) FILTER (WHERE cost_usd IS NOT NULL), 0)
            FROM ai_calls WHERE created_at > now() - make_interval(days => %s)
              AND (%s = '' OR created_at >= %s::timestamptz)
              AND cost_usd IS NOT NULL
            GROUP BY task, model ORDER BY sum(cost_usd) DESC NULLS LAST, count(*) DESC
            """,
            (days, config.AI_STATS_SINCE, config.AI_STATS_SINCE or "1970-01-01T00:00:00Z"),
        ).fetchall()
        recent = conn.execute(
            """
            SELECT created_at, task, model, ok, cost_usd, prompt_tokens, completion_tokens, latency_ms
            FROM ai_calls WHERE created_at > now() - make_interval(days => %s)
              AND (%s = '' OR created_at >= %s::timestamptz)
              AND cost_usd IS NOT NULL
            ORDER BY id DESC LIMIT 50
            """,
            (days, config.AI_STATS_SINCE, config.AI_STATS_SINCE or "1970-01-01T00:00:00Z"),
        ).fetchall()
        all_time = conn.execute(
            "SELECT COALESCE(sum(cost_usd), 0) FROM ai_calls WHERE cost_usd IS NOT NULL"
        ).fetchone()[0]
    budget = config.MONTHLY_AI_BUDGET_USD
    month_spend = float(month_total or 0)
    from app.services import costs
    committed = costs.current_month_spend()
    with db.get_conn() as conn:
        reserved = conn.execute("SELECT provider,status,COALESCE(sum(amount),0),count(*) FROM spend_reservations WHERE status IN ('reserved','estimated') AND created_at>=date_trunc('month',now()) GROUP BY provider,status").fetchall()
    total_spent = float(all_time or 0)
    total_credits = config.OPENROUTER_TOTAL_CREDITS
    return {
        "days": days,
        "currency": "USD",
        "period_spend": float(summary[0] or 0),
        "today_spend": float(summary[1] or 0),
        "priced_calls": summary[2],
        "average_cost": float(summary[3] or 0),
        "month_spend": month_spend,
        "budget_committed": committed,
        "reservations": [{'provider':r[0],'status':r[1],'amount_usd':float(r[2]),'operations':r[3]} for r in reserved],
        "all_time_spend": total_spent,
        "monthly_budget": budget,
        "remaining_budget": round(budget - committed, 6) if budget > 0 else None,
        "budget_percent": min(round(committed / budget * 100, 1), 100) if budget > 0 else None,
        "total_credits": total_credits,
        "credits_remaining": round(total_credits - total_spent, 6) if total_credits > 0 else None,
        "breakdown": [
            {"task": row[0], "model": row[1], "calls": row[2], "succeeded": row[3],
             "cost_usd": float(row[4] or 0), "average_cost_usd": float(row[5] or 0)}
            for row in breakdown
        ],
        "recent": [
            {"at": row[0].isoformat(), "task": row[1], "model": row[2], "ok": row[3],
             "cost_usd": float(row[4]) if row[4] is not None else None,
             "prompt_tokens": row[5], "completion_tokens": row[6], "latency_ms": row[7]}
            for row in recent
        ],
    }
