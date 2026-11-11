import json

from fastapi import APIRouter, HTTPException

from app import db, security
from app.services import audit as audit_service

router = APIRouter(prefix="/businesses/{business_id}/audit", tags=["audit"])


def _get_credential(conn, business_id: str, platform: str) -> tuple[str, dict] | None:
    row = conn.execute(
        "SELECT access_token, meta FROM business_credentials WHERE business_id = %s AND platform = %s",
        (business_id, platform),
    ).fetchone()
    if not row:
        return None
    return security.decrypt_token(row[0]), row[1] or {}


@router.post("")
def run_audit(business_id: str, post_limit: int = 100):
    """Run a full digital marketing audit across all connected platforms."""
    with db.get_conn() as conn:
        if not conn.execute("SELECT 1 FROM businesses WHERE id = %s", (business_id,)).fetchone():
            raise HTTPException(status_code=404, detail="business not found")
        fb_cred = _get_credential(conn, business_id, "facebook")
        ig_cred = _get_credential(conn, business_id, "instagram")

    if not fb_cred and not ig_cred:
        raise HTTPException(
            status_code=400,
            detail="No platform credentials connected. Connect Facebook and/or Instagram first.",
        )

    fb_token, fb_meta = fb_cred if fb_cred else (None, {})
    ig_token, ig_meta = ig_cred if ig_cred else (None, {})

    page_id = fb_meta.get("page_id") if fb_meta else None
    ig_user_id = ig_meta.get("ig_user_id") if ig_meta else None

    if fb_cred and not page_id:
        raise HTTPException(status_code=400, detail="Facebook credential is missing meta.page_id")

    result = audit_service.run_audit(
        page_id=page_id,
        fb_access_token=fb_token if page_id else None,
        ig_user_id=ig_user_id,
        ig_access_token=ig_token if ig_user_id else None,
        post_limit=post_limit,
    )

    if result.get("error") and result.get("posts_analyzed", 0) == 0:
        raise HTTPException(status_code=502, detail=result["error"])

    with db.get_conn() as conn:
        audit_row = conn.execute(
            "INSERT INTO social_audits (business_id, platform, posts_analyzed, findings, report_text) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id, created_at",
            (
                business_id,
                ",".join(result.get("platforms_audited", ["facebook"])),
                result["posts_analyzed"],
                json.dumps(result["findings"]),
                _format_report_text(result),
            ),
        ).fetchone()
        conn.execute(
            "INSERT INTO learning_insights (business_id, source, insights, summary_text) "
            "VALUES (%s, 'historical_audit', %s, %s)",
            (
                business_id,
                json.dumps(result["learning_seed"]),
                _format_learning_summary(result),
            ),
        )
        conn.commit()

    return {
        "audit_id": str(audit_row[0]),
        "created_at": audit_row[1].isoformat(),
        "posts_analyzed": result["posts_analyzed"],
        "platforms_audited": result.get("platforms_audited", []),
        "strengths": result.get("strengths", []),
        "weaknesses": result.get("weaknesses", []),
        "opportunities": result.get("opportunities", []),
        "kpi_summary": result.get("kpi_summary", {}),
        "platform_comparison": result.get("platform_comparison", ""),
        "closing_note": result.get("closing_note", ""),
        "raw_data": result["findings"],
    }


@router.get("/latest")
def get_latest_audit(business_id: str):
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT id, posts_analyzed, findings, report_text, created_at FROM social_audits "
            "WHERE business_id = %s ORDER BY created_at DESC LIMIT 1",
            (business_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="no audit run yet for this business")
    findings = row[2] or {}
    return {
        "audit_id": str(row[0]),
        "posts_analyzed": row[1],
        "findings": findings,
        "report_text": row[3],
        "created_at": row[4].isoformat(),
        "cadence": findings.get("cadence", {}),
        "format_mix": findings.get("format_mix", {}),
        "category_performance": findings.get("category_performance", {}),
        "strategy_mix": findings.get("strategy_mix", {}),
        "health_score": findings.get("health_score", {}),
        "time_distribution": findings.get("time_distribution", {}),
        "best_and_worst": findings.get("best_and_worst", {}),
        "platform_breakdown": findings.get("platform_breakdown", {}),
        "platforms_audited": findings.get("platforms_audited", []),
    }


def _format_report_text(result: dict) -> str:
    parts = []
    if result.get("strengths"):
        parts.append("STRENGTHS:\n" + "\n".join(f"  + {s}" for s in result["strengths"]))
    if result.get("weaknesses"):
        parts.append("WEAKNESSES:\n" + "\n".join(f"  - {w}" for w in result["weaknesses"]))
    if result.get("opportunities"):
        parts.append("OPPORTUNITIES:\n" + "\n".join(f"  > {o}" for o in result["opportunities"]))
    if result.get("platform_comparison"):
        parts.append(f"PLATFORM COMPARISON:\n  {result['platform_comparison']}")
    if result.get("closing_note"):
        parts.append(f"NEXT STEP:\n  {result['closing_note']}")
    return "\n\n".join(parts)


def _format_learning_summary(result: dict) -> str:
    seed = result.get("learning_seed", {})
    best = seed.get("best_performing_format")
    if best:
        return f"Historical audit: best-performing format is {best}. Platforms: {', '.join(result.get('platforms_audited', []))}."
    return f"Historical cadence baseline loaded. Platforms: {', '.join(result.get('platforms_audited', []))}."
