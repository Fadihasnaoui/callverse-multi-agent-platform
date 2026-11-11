import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import db, security
from app.ai.rules import StyleRules
from app.services import graph_api, strategist

log = logging.getLogger(__name__)

router = APIRouter(prefix="/businesses", tags=["businesses"])


class CreateBusiness(BaseModel):
    name: str
    industry: str = ""
    brand_voice: str = ""


class UpdateBusiness(BaseModel):
    name: str | None = None
    industry: str | None = None
    brand_voice: str | None = None
    status: str | None = None  # onboarding / active / paused


class ConnectCredential(BaseModel):
    platform: str  # facebook / instagram / google_places
    access_token: str
    meta: dict = {}


@router.get("")
def list_businesses():
    """All customers, newest first, with which platforms are connected (never the tokens)."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT b.id, b.name, b.industry, b.brand_voice, b.status, b.created_at, "
            "COALESCE((SELECT array_agg(platform) FROM business_credentials c WHERE c.business_id = b.id), '{}') "
            "FROM businesses b ORDER BY b.created_at DESC"
        ).fetchall()
    return [{**_business_row_to_dict(r), "connected": list(r[6])} for r in rows]


@router.put("/{business_id}")
def update_business(business_id: str, payload: UpdateBusiness):
    if payload.status is not None and payload.status not in ("onboarding", "active", "paused"):
        raise HTTPException(status_code=400, detail="status must be onboarding, active or paused")
    with db.get_conn() as conn:
        row = conn.execute(
            "UPDATE businesses SET name = COALESCE(%s, name), industry = COALESCE(%s, industry), "
            "brand_voice = COALESCE(%s, brand_voice), status = COALESCE(%s, status) WHERE id = %s "
            "RETURNING id, name, industry, brand_voice, status, created_at",
            (payload.name, payload.industry, payload.brand_voice, payload.status, business_id),
        ).fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="business not found")
    return _business_row_to_dict(row)


def _get_strategist_recommendation(business_id: str) -> dict | None:
    try:
        return strategist.recommend_next_content(business_id)
    except Exception:
        return None


@router.get("/{business_id}/overview")
def overview(business_id: str):
    """One call for the dashboard header: what needs attention, and what is coming up."""
    with db.get_conn() as conn:
        if not conn.execute("SELECT 1 FROM businesses WHERE id = %s", (business_id,)).fetchone():
            raise HTTPException(status_code=404, detail="business not found")
        counts = dict(conn.execute(
            "SELECT status, count(*) FROM content_analytics WHERE business_id = %s GROUP BY status", (business_id,)
        ).fetchall())
        upcoming = conn.execute(
            "SELECT id, type, platform, left(caption, 120), scheduled_for FROM content_analytics WHERE business_id = %s "
            "AND status = 'approved' AND scheduled_for > now() ORDER BY scheduled_for LIMIT 5", (business_id,)
        ).fetchall()
        needs_video = conn.execute(
            "SELECT count(*) FROM content_analytics WHERE business_id = %s AND status = 'pending' "
            "AND type IN ('reel', 'avatar_clip') AND rendered_video_path IS NULL", (business_id,)
        ).fetchone()[0]
        leads_new = conn.execute(
            "SELECT count(*) FROM leads WHERE business_id = %s AND status = 'new' AND created_at > now() - interval '30 days'",
            (business_id,),
        ).fetchone()[0]
        sales = conn.execute(
            "SELECT count(*), COALESCE(sum(value_amount) FILTER(WHERE value_currency='TND'), 0) FROM conversions WHERE business_id = %s "
            "AND reported_at > now() - interval '30 days'", (business_id,)
        ).fetchone()
        currency_rows = conn.execute("SELECT value_currency,COALESCE(sum(value_amount),0) FROM conversions WHERE business_id=%s AND reported_at>now()-interval '30 days' GROUP BY value_currency", (business_id,)).fetchall()
        scheduled_total = conn.execute("SELECT count(*) FROM content_items WHERE business_id=%s AND status='approved' AND scheduled_for>now()", (business_id,)).fetchone()[0]
        connected = [r[0] for r in conn.execute(
            "SELECT platform FROM business_credentials WHERE business_id = %s", (business_id,)).fetchall()]
        performance = conn.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (pm.content_item_id) pm.content_item_id, pm.reach, pm.impressions,
                       pm.engagement, pm.saves, pm.video_views, pm.fetched_at
                FROM performance_metrics pm
                JOIN content_items ci ON ci.id = pm.content_item_id
                WHERE ci.business_id = %s AND COALESCE(ci.published_at,ci.created_at)>now()-interval '30 days'
                ORDER BY pm.content_item_id, pm.fetched_at DESC, pm.id DESC
            )
            SELECT COALESCE(sum(reach),0), COALESCE(sum(impressions),0), COALESCE(sum(engagement),0),
                   COALESCE(sum(saves),0), COALESCE(sum(video_views),0), max(fetched_at), count(*),
                   bool_or(reach IS NOT NULL),bool_or(impressions IS NOT NULL),bool_or(engagement IS NOT NULL)
            FROM latest
            """,
            (business_id,),
        ).fetchone()
        mix_rows = conn.execute(
            "SELECT type, count(*) FROM content_analytics WHERE business_id = %s GROUP BY type", (business_id,),
        ).fetchall()
        # Per-platform post counts
        platform_counts = conn.execute(
            "SELECT platform, source, count(*) FROM content_analytics WHERE business_id = %s "
            "AND status = 'published' GROUP BY platform, source", (business_id,),
        ).fetchall()
        # Page-level metrics (latest per platform/metric)
        page_metric_rows = conn.execute(
            """
            SELECT DISTINCT ON (platform, metric_name) platform, metric_name, metric_value, fetched_at
            FROM page_metrics WHERE business_id = %s
            ORDER BY platform, metric_name, fetched_at DESC
            """, (business_id,),
        ).fetchall()
        weekly_rows = conn.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (pm.content_item_id) pm.content_item_id, pm.engagement
                FROM performance_metrics pm ORDER BY pm.content_item_id, pm.fetched_at DESC, pm.id DESC
            )
            SELECT date_trunc('week', COALESCE(ci.published_at, ci.created_at)), ci.platform,
                   COALESCE(sum(latest.engagement),0)
            FROM content_analytics ci LEFT JOIN latest ON latest.content_item_id = ci.id
            WHERE ci.business_id = %s AND ci.status = 'published'
              AND COALESCE(ci.published_at, ci.created_at) > now() - interval '28 days'
            GROUP BY 1, 2 ORDER BY 1
            """,
            (business_id,),
        ).fetchall()
        insight = conn.execute(
            "SELECT summary_text, insights->'recommendations', created_at FROM learning_insights "
            "WHERE business_id = %s AND source NOT LIKE 'simulation%%' ORDER BY created_at DESC LIMIT 1",
            (business_id,),
        ).fetchone()
        audit = conn.execute(
            "SELECT posts_analyzed, created_at FROM social_audits WHERE business_id = %s "
            "ORDER BY created_at DESC LIMIT 1", (business_id,),
        ).fetchone()
        # Content category distribution
        category_rows = conn.execute(
            "SELECT content_category, count(*) FROM content_analytics "
            "WHERE business_id = %s AND content_category IS NOT NULL GROUP BY content_category",
            (business_id,),
        ).fetchall()
        # Strategy layer distribution
        strategy_rows = conn.execute(
            "SELECT strategy_layer, count(*) FROM content_analytics "
            "WHERE business_id = %s AND strategy_layer IS NOT NULL GROUP BY strategy_layer",
            (business_id,),
        ).fetchall()
        # Per-platform performance (engagement from features JSON for IG)
        platform_perf_rows = conn.execute(
            """
            SELECT ci.platform,
                   count(*) AS posts,
                   COALESCE(sum((ci.analytics_features->>'like_count')::int), 0) AS total_likes,
                   COALESCE(sum((ci.analytics_features->>'comments_count')::int), 0) AS total_comments,
                   COALESCE(sum((ci.analytics_features->>'shares')::int), 0) +
                   COALESCE(sum((ci.analytics_features->>'reactions_count')::int), 0) AS total_reactions_shares
            FROM content_analytics ci
            WHERE ci.business_id = %s AND ci.status = 'published'
            GROUP BY ci.platform
            """,
            (business_id,),
        ).fetchall()
        # Category performance (avg engagement per category from IG inline metrics)
        category_perf_rows = conn.execute(
            """
            SELECT content_category,
                   count(*) AS posts,
                   COALESCE(avg((analytics_features->>'like_count')::int), 0) AS avg_likes,
                   COALESCE(avg((analytics_features->>'comments_count')::int), 0) AS avg_comments
            FROM content_analytics
            WHERE business_id = %s AND content_category IS NOT NULL
              AND platform = 'instagram' AND status = 'published'
            GROUP BY content_category
            ORDER BY COALESCE(avg((analytics_features->>'like_count')::int), 0) + COALESCE(avg((analytics_features->>'comments_count')::int), 0) DESC
            """,
            (business_id,),
        ).fetchall()
        # Posting cadence: posts per week by platform (last 90 days)
        cadence_rows = conn.execute(
            """
            SELECT platform,
                   count(*) AS total,
                   count(*) / GREATEST(EXTRACT(EPOCH FROM (max(COALESCE(published_at, created_at))
                       - min(COALESCE(published_at, created_at)))) / 604800, 1) AS per_week
            FROM content_analytics
            WHERE business_id = %s AND status = 'published'
              AND COALESCE(published_at, created_at) > now() - interval '90 days'
            GROUP BY platform
            """,
            (business_id,),
        ).fetchall()
    reach, impressions, engagement = int(performance[0]), int(performance[1]), int(performance[2])

    # Build per-platform post breakdown
    platform_posts = {}
    for plat, source, cnt in platform_counts:
        if plat not in platform_posts:
            platform_posts[plat] = {"agent": 0, "page_import": 0, "total": 0}
        platform_posts[plat][source] = cnt
        platform_posts[plat]["total"] += cnt

    # Build page-level KPI dict
    page_kpis: dict = {}
    for plat, metric_name, metric_value, fetched_at in page_metric_rows:
        if plat not in page_kpis:
            page_kpis[plat] = {"last_synced": fetched_at.isoformat() if fetched_at else None}
        page_kpis[plat][metric_name] = float(metric_value) if metric_value is not None else None

    # Parse audit findings for strengths/weaknesses if available
    audit_analysis = None
    if audit:
        try:
            findings = audit[1] or {}
            audit_analysis = {
                "posts_analyzed": audit[0],
                "as_of": audit[1].isoformat() if hasattr(audit[1], "isoformat") else None,
            }
        except Exception:
            pass

    # Build per-platform engagement from stored data
    platform_performance = {}
    for plat, posts_count, likes, comments, reactions_shares in platform_perf_rows:
        total_eng = likes + comments + reactions_shares
        followers = float(page_kpis.get(plat, {}).get("followers_count") or 0)
        platform_performance[plat] = {
            "posts": posts_count,
            "total_likes": likes,
            "total_comments": comments,
            "total_reactions_shares": reactions_shares,
            "total_engagement": total_eng,
            "avg_engagement_per_post": round(total_eng / max(posts_count, 1), 1),
            "engagement_rate": round(total_eng / followers * 100, 3) if followers else None,
        }

    # Category performance for learning
    category_performance = {}
    for cat, posts_count, avg_likes, avg_comments in category_perf_rows:
        category_performance[cat or "uncategorized"] = {
            "posts": posts_count,
            "avg_likes": round(float(avg_likes), 1),
            "avg_comments": round(float(avg_comments), 1),
            "avg_engagement": round(float(avg_likes) + float(avg_comments), 1),
        }

    # Posting cadence
    posting_cadence = {}
    for plat, total, per_week in cadence_rows:
        posting_cadence[plat] = {
            "total_90d": total,
            "per_week": round(float(per_week), 1),
        }

    return {
        "content": {k: counts.get(k, 0) for k in ("pending", "approved", "published", "failed", "rejected")},
        "needs_media": needs_video,
        "upcoming": [{"id": str(r[0]), "type": r[1], "platform": r[2], "caption": r[3],
                      "scheduled_for": r[4].isoformat()} for r in upcoming],
        "leads_new_30d": leads_new,
        "sales_30d": {"count": sales[0], "revenue": float(sales[1]),"revenue_currency":"TND","revenue_by_currency":{r[0]:float(r[1]) for r in currency_rows}},
        "scheduled_total": scheduled_total,
        "connected": connected,
        "performance_30d": {
            "availability": {'reach':bool(performance[7]),'impressions':bool(performance[8]),'engagement':bool(performance[9])},
            "definition":"Lifetime outcomes of posts published in the last 30 days. Reach is summed post reach, not deduplicated campaign audience.",
            "reach": reach,
            "impressions": impressions,
            "engagement": engagement,
            "engagement_rate": round(engagement / reach * 100, 1) if reach else None,
            "saves": int(performance[3]), "video_views": int(performance[4]),
            "last_synced_at": performance[5].isoformat() if performance[5] else None,
            "posts_with_metrics": performance[6],
        },
        "content_mix": {row[0]: row[1] for row in mix_rows},
        "platform_posts": platform_posts,
        "page_kpis": page_kpis,
        "platform_performance": platform_performance,
        "category_distribution": {row[0] or "uncategorized": row[1] for row in category_rows},
        "category_performance": category_performance,
        "strategy_mix": {row[0] or "distribution": row[1] for row in strategy_rows},
        "posting_cadence": posting_cadence,
        "weekly_performance": [
            {"week": row[0].date().isoformat(), "platform": row[1], "engagement": int(row[2])}
            for row in weekly_rows
        ],
        "latest_insight": {
            "summary": insight[0], "recommendations": insight[1] or [], "as_of": insight[2].isoformat()
        } if insight else None,
        "latest_audit": {"posts_analyzed": audit[0], "as_of": audit[1].isoformat()} if audit else None,
        "strategist": _get_strategist_recommendation(business_id),
    }


@router.post("")
def create_business(payload: CreateBusiness):
    with db.get_conn() as conn:
        row = conn.execute(
            "INSERT INTO businesses (name, industry, brand_voice) VALUES (%s, %s, %s) "
            "RETURNING id, name, industry, brand_voice, status, created_at",
            (payload.name, payload.industry, payload.brand_voice),
        ).fetchone()
        conn.commit()
    return _business_row_to_dict(row)


@router.get("/{business_id}")
def get_business(business_id: str):
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT id, name, industry, brand_voice, status, created_at FROM businesses WHERE id = %s",
            (business_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="business not found")
    return _business_row_to_dict(row)


@router.post("/{business_id}/credentials")
def connect_credential(business_id: str, payload: ConnectCredential):
    with db.get_conn() as conn:
        exists = conn.execute("SELECT 1 FROM businesses WHERE id = %s", (business_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="business not found")

    token_to_store = payload.access_token
    token_type = "user_provided"
    token_expires_at = None
    token_scopes: list[str] = []
    exchange_info = {}

    # Auto-exchange Facebook tokens to permanent page tokens
    if payload.platform == "facebook" and payload.meta.get("page_id"):
        try:
            exchange_info = graph_api.exchange_to_permanent_page_token(
                payload.access_token, payload.meta["page_id"]
            )
            token_to_store = exchange_info["access_token"]
            token_type = exchange_info.get("token_type", "unknown")
            token_scopes = exchange_info.get("scopes", [])
            if exchange_info.get("exchanged"):
                log.info("Token exchanged to %s for page %s", token_type, payload.meta["page_id"])
        except graph_api.GraphAPIError as e:
            log.warning("Token exchange failed (storing as-is): %s", e)

    encrypted = security.encrypt_token(token_to_store)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO business_credentials (business_id, platform, access_token, meta, "
            "token_type, token_expires_at, token_scopes, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (business_id, platform) DO UPDATE SET "
            "access_token = EXCLUDED.access_token, meta = EXCLUDED.meta, "
            "token_type = EXCLUDED.token_type, token_expires_at = EXCLUDED.token_expires_at, "
            "token_scopes = EXCLUDED.token_scopes, updated_at = now()",
            (business_id, payload.platform, encrypted, json.dumps(payload.meta),
             token_type, token_expires_at, token_scopes or None),
        )
        conn.commit()
    return {
        "status": "connected",
        "platform": payload.platform,
        "token_type": token_type,
        "exchanged": exchange_info.get("exchanged", False),
        "scopes": token_scopes,
    }


@router.get("/{business_id}/credentials/health")
def credential_health(business_id: str):
    """Check token validity for all connected platforms."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT platform, access_token, meta, token_type, token_expires_at, token_scopes, updated_at "
            "FROM business_credentials WHERE business_id = %s",
            (business_id,),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="no credentials found")

    results = {}
    for platform, enc_token, meta, token_type, expires_at, scopes, updated_at in rows:
        token = security.decrypt_token(enc_token)
        health: dict = {"platform": platform, "token_type": token_type, "updated_at": updated_at.isoformat() if updated_at else None}
        if platform == "facebook":
            debug = graph_api.debug_token(token)
            health["valid"] = debug.get("valid")
            health["expires_at"] = debug.get("expires_at")
            health["scopes"] = debug.get("scopes", [])
            health["never_expires"] = debug.get("expires_at") in (0, None) and debug.get("valid")
            if debug.get("error"):
                health["error"] = debug["error"]
        elif platform == "instagram":
            ig_user_id = (meta or {}).get("ig_user_id")
            if ig_user_id:
                try:
                    profile = graph_api.fetch_ig_profile(ig_user_id, token)
                    health["valid"] = bool(profile.get("id"))
                    health["username"] = profile.get("username")
                    health["followers"] = profile.get("followers_count")
                except Exception:
                    health["valid"] = False
            else:
                health["valid"] = None
                health["error"] = "ig_user_id not set in meta"
        results[platform] = health
    return results


@router.post("/{business_id}/credentials/discover")
def discover_platforms(business_id: str, payload: ConnectCredential):
    """Give a user token — get back all Facebook Pages and linked Instagram accounts.
    Use this to find page_id and ig_user_id before connecting."""
    pages = graph_api.discover_pages(payload.access_token)
    if not pages:
        raise HTTPException(status_code=400, detail="This token has no managed Facebook Pages. Make sure you're a Page admin.")
    return {"pages": pages}


@router.get("/{business_id}/credentials/diagnose")
def diagnose_credentials(business_id: str):
    """Test what the stored tokens can actually do — read page, list pages, fetch posts."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT platform, access_token, meta FROM business_credentials WHERE business_id = %s",
            (business_id,),
        ).fetchall()
    if not rows:
        return {"error": "no credentials"}

    results = {}
    for platform, enc_token, meta in rows:
        token = security.decrypt_token(enc_token)
        diag: dict = {"platform": platform}
        try:
            me = graph_api._get("me", token, {"fields": "name,id"})
            diag["token_identity"] = me
        except Exception as e:
            diag["token_identity"] = f"error: {e}"

        if platform == "facebook":
            page_id = (meta or {}).get("page_id", "")
            # Can we read the page directly?
            try:
                page = graph_api._get(page_id, token, {"fields": "name,id,access_token"})
                diag["page_direct_access"] = {"name": page.get("name"), "id": page.get("id"), "has_page_token": "access_token" in page}
            except Exception as e:
                diag["page_direct_access"] = f"error: {e}"
            # Can we list pages?
            try:
                pages = graph_api._get("me/accounts", token, {"fields": "id,name"})
                diag["me_accounts"] = pages.get("data", [])
            except Exception as e:
                diag["me_accounts"] = f"error: {e}"
            # Can we read page posts?
            try:
                posts = graph_api._get(f"{page_id}/posts", token, {"fields": "id,message", "limit": "1"})
                diag["can_read_posts"] = len(posts.get("data", [])) > 0
                if posts.get("data"):
                    diag["sample_post"] = posts["data"][0].get("message", "")[:80]
            except Exception as e:
                diag["can_read_posts"] = f"error: {e}"

        elif platform == "instagram":
            ig_user_id = (meta or {}).get("ig_user_id", "")
            try:
                profile = graph_api.fetch_ig_profile(ig_user_id, token)
                diag["ig_profile"] = {"username": profile.get("username"), "followers": profile.get("followers_count")}
            except Exception as e:
                diag["ig_profile"] = f"error: {e}"

        results[platform] = diag
    return results


@router.post("/{business_id}/credentials/connect-all")
def connect_all_platforms(business_id: str, payload: ConnectCredential):
    """One-shot setup: give a user token and a page_id, and this connects both
    Facebook (with permanent token exchange) AND Instagram (auto-discovered).
    No need to find the ig_user_id yourself."""
    with db.get_conn() as conn:
        if not conn.execute("SELECT 1 FROM businesses WHERE id = %s", (business_id,)).fetchone():
            raise HTTPException(status_code=404, detail="business not found")

    page_id = payload.meta.get("page_id")
    if not page_id:
        raise HTTPException(status_code=400, detail="meta.page_id is required")

    results = {}

    # 1. Connect Facebook with token exchange
    fb_payload = ConnectCredential(platform="facebook", access_token=payload.access_token, meta={"page_id": page_id})
    results["facebook"] = connect_credential(business_id, fb_payload)

    # 2. Auto-discover and connect Instagram using the permanent page token
    # The FB page token works for IG API calls since the IG business account is linked
    fb_token = results["facebook"].get("stored_token_type")
    ig_token_to_use = payload.access_token
    if results["facebook"].get("exchanged"):
        # Use the permanent page token for IG too — it never expires
        with db.get_conn() as conn:
            fb_cred = conn.execute(
                "SELECT access_token FROM business_credentials WHERE business_id = %s AND platform = 'facebook'",
                (business_id,),
            ).fetchone()
            if fb_cred:
                ig_token_to_use = security.decrypt_token(fb_cred[0])

    ig_account = graph_api.discover_instagram_account(page_id, ig_token_to_use)
    if ig_account:
        ig_payload = ConnectCredential(
            platform="instagram",
            access_token=ig_token_to_use,
            meta={"ig_user_id": ig_account["ig_user_id"], "username": ig_account.get("username")},
        )
        results["instagram"] = connect_credential(business_id, ig_payload)
        results["instagram"]["discovered"] = ig_account
    else:
        results["instagram"] = {
            "status": "not_found",
            "detail": "No Instagram Business account linked to this Facebook Page. "
                      "Link it in Facebook Page Settings > Linked Accounts > Instagram, "
                      "and make sure the Instagram account is set to Professional (Business or Creator).",
        }

    return results


def _business_row_to_dict(row) -> dict:
    return {
        "id": str(row[0]),
        "name": row[1],
        "industry": row[2],
        "brand_voice": row[3],
        "status": row[4],
        "created_at": row[5].isoformat() if row[5] else None,
    }


@router.get("/{business_id}/rules")
def get_rules(business_id: str):
    with db.get_conn() as conn:
        row = conn.execute("SELECT style_rules FROM businesses WHERE id = %s", (business_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="business not found")
    return StyleRules.model_validate(row[0] or {}).model_dump()


@router.put("/{business_id}/rules")
def put_rules(business_id: str, rules: StyleRules):
    with db.get_conn() as conn:
        row = conn.execute(
            "UPDATE businesses SET style_rules = %s::jsonb WHERE id = %s RETURNING id",
            (json.dumps(rules.model_dump()), business_id),
        ).fetchone()
        conn.commit()
    if not row:
        raise HTTPException(status_code=404, detail="business not found")
    return rules.model_dump()
