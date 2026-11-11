"""Synchronize real metrics from Facebook and Instagram pages.

This is the core data engine for the live dashboard. It:
1. Imports ALL posts from a page (not just agent-published ones) into content_items
2. Detects content category and strategy layer for each post
3. Fetches per-post insights for every known post
4. Fetches page-level metrics (followers, reach, impressions)
5. Works for both Facebook and Instagram
"""
import logging
import re
from datetime import datetime, timezone

from app import db, security
from app.services import graph_api

log = logging.getLogger(__name__)

# --- Content categorization for the learning algorithm ---

_CATEGORY_PATTERNS = {
    "promotion": re.compile(
        r"(promo|offre|réduction|discount|flash|solde|prix|%\s*off|remise|vente|تخفيض|عرض)", re.I
    ),
    "destination": re.compile(
        r"(istanbul|antalya|tunisie|bodrum|sharm|marrakech|dubai|maldives|zanzibar|"
        r"bali|paris|rome|barcelone|hurghada|destination|voyage|سفر|رحلة)", re.I
    ),
    "hotel": re.compile(
        r"(hotel|hôtel|resort|spa|palace|all\s*inclusive|séjour|إقامة|فندق)", re.I
    ),
    "engagement": re.compile(
        r"(comment|avis|partagez|dites.nous|which|choisi|vote|sondage|question|\?$)", re.I
    ),
    "event": re.compile(
        r"(soirée|event|fête|gala|concert|célébr|anniversaire|حفل|مهرجان)", re.I
    ),
    "testimonial": re.compile(
        r"(avis|témoignage|client|review|merci|شكر|تجربة)", re.I
    ),
    "brand": re.compile(
        r"(ozone|notre\s*équipe|notre\s*agence|bienvenue|about\s*us|qui\s*sommes)", re.I
    ),
}

_STRATEGY_SIGNALS = {
    "pillar": re.compile(
        r"(guide|conseil|astuce|tips|how\s*to|comment\s+faire|savoir|découvr|explore|"
        r"top\s*\d+|les\s+\d+\s+meilleur|raison)", re.I
    ),
    "engagement": re.compile(
        r"(comment|partagez|tag|mentionnez|dites.nous|\?\s*$|sondage|vote|which|choisi)", re.I
    ),
}


def _detect_category(caption: str) -> str:
    if not caption:
        return "uncategorized"
    for category, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(caption):
            return category
    return "general"


def _detect_strategy_layer(caption: str, media_type: str) -> str:
    if not caption:
        return "distribution"
    for layer, pattern in _STRATEGY_SIGNALS.items():
        if pattern.search(caption):
            return layer
    return "distribution"


def _get_credential(conn, business_id: str, platform: str) -> tuple[str, dict] | None:
    row = conn.execute(
        "SELECT access_token, meta FROM business_credentials "
        "WHERE business_id = %s AND platform = %s",
        (business_id, platform),
    ).fetchone()
    if not row:
        return None
    return security.decrypt_token(row[0]), row[1] or {}


def _import_fb_posts(conn, business_id: str, page_id: str, token: str) -> int:
    """Import Facebook page posts into content_items as source='page_import'.
    Skips posts that already exist (by meta_post_id). Returns count of new imports."""
    posts = graph_api.fetch_page_posts(page_id, token, limit=500)
    imported = 0
    for post in posts:
        post_id = post.get("id")
        if not post_id:
            continue
        exists = conn.execute(
            "SELECT id FROM content_items WHERE business_id = %s AND platform='facebook' AND (meta_post_id = %s OR meta_post_id=%s)",
            (business_id, post_id,post_id.split('_',1)[-1]),
        ).fetchone()
        if exists:
            import json
            counts = {'reactions_count': (post.get('reactions',{}).get('summary') or {}).get('total_count'),
                      'comments_count': (post.get('comments',{}).get('summary') or {}).get('total_count'),
                      'shares':(post.get('shares') or {}).get('count')}
            conn.execute("UPDATE content_items SET features=features || %s::jsonb WHERE id=%s", (json.dumps({k:v for k,v in counts.items() if v is not None}),str(exists[0])))
            conn.commit()
            continue

        attachments = (post.get("attachments") or {}).get("data") or []
        media_type = attachments[0].get("media_type", "").lower() if attachments else "text"
        if "video" in media_type:
            content_type = "reel"
        elif "photo" in media_type:
            content_type = "static_post"
        else:
            content_type = "static_post"

        created = post.get("created_time", "")
        published_at = None
        if created:
            try:
                published_at = datetime.strptime(created, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                pass

        shares = (post.get("shares") or {}).get("count", 0)
        reactions_count = (post.get("reactions", {}).get("summary") or {}).get("total_count", 0)
        comments_count = (post.get("comments", {}).get("summary") or {}).get("total_count", 0)

        caption_text = (post.get("message") or "")[:2000]
        category = _detect_category(caption_text)
        strategy = _detect_strategy_layer(caption_text, media_type)

        import json
        conn.execute(
            "INSERT INTO content_items (business_id, type, platform, caption, status, "
            "meta_post_id, published_at, source, content_category, strategy_layer, features) "
            "VALUES (%s, %s, 'facebook', %s, 'published', %s, %s, 'page_import', %s, %s, %s)",
            (
                business_id, content_type, caption_text,
                post_id, published_at, category, strategy,
                json.dumps({
                    "shares": shares,
                    "reactions_count": reactions_count,
                    "comments_count": comments_count,
                    "permalink": post.get("permalink_url"),
                    "imported": True,
                }),
            ),
        )
        imported += 1
    if imported:
        conn.commit()
    return imported


def _import_ig_posts(conn, business_id: str, ig_user_id: str, token: str) -> int:
    """Import Instagram media into content_items as source='page_import'."""
    media_list = graph_api.fetch_ig_media(ig_user_id, token, limit=500)
    imported = 0
    for media in media_list:
        media_id = media.get("id")
        if not media_id:
            continue
        exists = conn.execute(
            "SELECT 1 FROM content_items WHERE business_id = %s AND meta_post_id = %s",
            (business_id, media_id),
        ).fetchone()
        if exists:
            import json
            counts = {k:media.get(k) for k in ('like_count','comments_count') if media.get(k) is not None}
            conn.execute("UPDATE content_items SET features=features || %s::jsonb WHERE business_id=%s AND meta_post_id=%s", (json.dumps(counts),business_id,media_id))
            conn.commit()
            continue

        media_type = media.get("media_type", "IMAGE")
        if media_type in ("VIDEO", "REELS"):
            content_type = "reel"
        else:
            content_type = "static_post"

        published_at = None
        ts = media.get("timestamp", "")
        if ts:
            try:
                published_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass

        caption_text = (media.get("caption") or "")[:2000]
        category = _detect_category(caption_text)
        strategy = _detect_strategy_layer(caption_text, media_type)

        import json
        conn.execute(
            "INSERT INTO content_items (business_id, type, platform, caption, status, "
            "meta_post_id, published_at, source, content_category, strategy_layer, features) "
            "VALUES (%s, %s, 'instagram', %s, 'published', %s, %s, 'page_import', %s, %s, %s)",
            (
                business_id, content_type, caption_text,
                media_id, published_at, category, strategy,
                json.dumps({
                    "like_count": media.get("like_count", 0),
                    "comments_count": media.get("comments_count", 0),
                    "media_type": media_type,
                    "permalink": media.get("permalink"),
                    "imported": True,
                }),
            ),
        )
        imported += 1
    if imported:
        conn.commit()
    return imported


def _sync_fb_post_insights(conn, business_id: str, token: str) -> dict:
    """Fetch insights for all Facebook posts (agent AND imported)."""
    posts = conn.execute(
        "SELECT id, meta_post_id FROM content_items "
        "WHERE business_id = %s AND platform = 'facebook' "
        "AND status = 'published' AND meta_post_id IS NOT NULL",
        (business_id,),
    ).fetchall()
    synced = 0
    with_data = 0
    for content_id, post_id in posts:
        try:
            insights = graph_api.fetch_post_insights(post_id, token)
        except Exception as e:
            log.warning("Failed to fetch insights for %s: %s", post_id, e)
            continue
        if not insights:
            continue
        engagement = sum(insights.get(key) or 0 for key in ('post_clicks','post_reactions_by_type_total')) if any(key in insights for key in ('post_clicks','post_reactions_by_type_total')) else None
        if insights:
            with_data += 1
        conn.execute(
            "INSERT INTO performance_metrics (content_item_id, reach, impressions, engagement, saves, video_views,reactions) "
            "VALUES (%s, %s, %s, %s, %s, %s,%s)",
            (
                str(content_id),
                insights.get("reach",insights.get("post_impressions_unique")),
                insights.get("impressions",insights.get("post_impressions")),
                engagement,
                insights.get("saves"),
                insights.get("post_video_views"),
                insights.get('post_reactions_by_type_total'),
            ),
        )
        synced += 1
    if synced:
        conn.commit()
    return {"synced": synced, "with_data": with_data}


def _sync_ig_post_insights(conn, business_id: str, token: str) -> dict:
    """Fetch insights for all Instagram posts (agent AND imported)."""
    posts = conn.execute(
        "SELECT id, meta_post_id, features->>'media_type' FROM content_items "
        "WHERE business_id = %s AND platform = 'instagram' "
        "AND status = 'published' AND meta_post_id IS NOT NULL",
        (business_id,),
    ).fetchall()
    synced = 0
    with_data = 0
    for content_id, media_id, media_type in posts:
        try:
            insights = graph_api.fetch_ig_media_insights(media_id, token, media_type or "IMAGE")
        except Exception as e:
            log.warning("Failed to fetch IG insights for %s: %s", media_id, e)
            continue
        if not insights:
            continue
        if insights:
            with_data += 1
        reach = insights.get("reach")
        impressions = insights.get("impressions")
        engagement = sum(insights.get(key) or 0 for key in ('likes','comments','shares')) if any(key in insights for key in ('likes','comments','shares')) else None
        saves = insights.get("saved")
        video_views = insights.get("plays")
        conn.execute(
            "INSERT INTO performance_metrics (content_item_id, reach, impressions, engagement, saves, video_views, likes, comments, shares) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (str(content_id), reach, impressions, engagement, saves, video_views,
             insights.get("likes"), insights.get("comments"), insights.get("shares")),
        )
        synced += 1
    if synced:
        conn.commit()
    return {"synced": synced, "with_data": with_data}


def _sync_page_level_metrics(conn, business_id: str, platform: str, metrics: dict):
    """Store page-level metric snapshots (numeric values only)."""
    for name, value in metrics.items():
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            continue
        conn.execute(
            "INSERT INTO page_metrics (business_id, platform, metric_name, metric_value) "
            "VALUES (%s, %s, %s, %s)",
            (business_id, platform, name, value),
        )
    conn.commit()


def backfill_categories(business_id: str) -> int:
    """Backfill content_category and strategy_layer on existing imported posts."""
    updated = 0
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, caption, type FROM content_items "
            "WHERE business_id = %s AND content_category IS NULL AND caption IS NOT NULL",
            (business_id,),
        ).fetchall()
        for row_id, caption, content_type in rows:
            category = _detect_category(caption)
            strategy = _detect_strategy_layer(caption, content_type or "")
            conn.execute(
                "UPDATE content_items SET content_category = %s, strategy_layer = %s WHERE id = %s",
                (category, strategy, str(row_id)),
            )
            updated += 1
        if updated:
            conn.commit()
    return updated


def sync_business(business_id: str) -> dict:
    with db.get_conn() as lease:
        locked = lease.execute('SELECT pg_try_advisory_lock(hashtext(%s))', ('metrics:'+business_id,)).fetchone()[0]
        if not locked:
            return {'status':'busy','detail':'A metrics sync is already in progress'}
        try:
            return _sync_business(business_id)
        finally:
            lease.execute('SELECT pg_advisory_unlock(hashtext(%s))', ('metrics:'+business_id,))


def _sync_business(business_id: str) -> dict:
    """Full sync for one business: import posts, fetch insights, page metrics.
    Called by the 6-hour scheduler and the manual /metrics/sync endpoint."""
    categorized = backfill_categories(business_id)

    result = {
        "facebook": {"status": "not_connected", "imported": 0, "posts_synced": 0, "with_insights": 0},
        "instagram": {"status": "not_connected", "imported": 0, "posts_synced": 0, "with_insights": 0},
        "page_metrics": {},
        "categories_backfilled": categorized,
    }

    with db.get_conn() as conn:
        # --- Facebook ---
        fb_cred = _get_credential(conn, business_id, "facebook")
        if fb_cred:
            token, meta = fb_cred
            page_id = meta.get("page_id")
            if page_id:
                try:
                    imported = _import_fb_posts(conn, business_id, page_id, token)
                    fb_insights = _sync_fb_post_insights(conn, business_id, token)
                    result["facebook"] = {
                        "status": "ok",
                        "imported": imported,
                        "posts_synced": fb_insights["synced"],
                        "with_insights": fb_insights["with_data"],
                    }
                except graph_api.GraphAPIError as e:
                    # A Page post edge may be unavailable for a token/app
                    # combination even though basic Page fields are allowed.
                    # Keep the post result separate so followers and other
                    # page-level metrics can still reach the dashboard.
                    log.warning("Facebook post sync failed for %s: %s", business_id, e)
                    result["facebook"] = {
                        "status": f"posts_unavailable: {e}",
                        "imported": 0,
                        "posts_synced": 0,
                        "with_insights": 0,
                    }
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                # Page-level metrics must be attempted independently of post
                # import; this makes a partial Meta permission set visible,
                # rather than rendering the entire Live dashboard empty.
                try:
                    page_metrics = graph_api.fetch_page_insights(page_id, token)
                    if page_metrics:
                        _sync_page_level_metrics(conn, business_id, "facebook", page_metrics)
                        result["page_metrics"]["facebook"] = page_metrics
                except Exception as e:
                    log.warning("FB page metrics failed: %s", e)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

        # --- Instagram ---
        ig_cred = _get_credential(conn, business_id, "instagram")
        if ig_cred:
            token, meta = ig_cred
            ig_user_id = meta.get("ig_user_id")
            # Fall back to FB page token if IG token is expired
            # (the permanent page token works for IG API calls too)
            if fb_cred:
                fb_token_fallback, _ = fb_cred
            else:
                fb_token_fallback = None
            if ig_user_id:
                try:
                    imported = _import_ig_posts(conn, business_id, ig_user_id, token)
                    ig_insights = _sync_ig_post_insights(conn, business_id, token)
                    result["instagram"] = {
                        "status": "ok",
                        "imported": imported,
                        "posts_synced": ig_insights["synced"],
                        "with_insights": ig_insights["with_data"],
                    }
                    # Account-level metrics
                    try:
                        ig_profile = graph_api.fetch_ig_profile(ig_user_id, token)
                        ig_account = graph_api.fetch_ig_account_insights(ig_user_id, token)
                        ig_metrics = {
                            "followers_count": ig_profile.get("followers_count"),
                            "follows_count": ig_profile.get("follows_count"),
                            "media_count": ig_profile.get("media_count"),
                            **ig_account,
                        }
                        _sync_page_level_metrics(conn, business_id, "instagram", ig_metrics)
                        result["page_metrics"]["instagram"] = ig_metrics
                    except Exception as e:
                        log.warning("IG account metrics failed: %s", e)
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                except graph_api.GraphAPIError as e:
                    if fb_token_fallback and "expired" in str(e).lower():
                        log.info("IG token expired, retrying with FB page token")
                        try:
                            token = fb_token_fallback
                            imported = _import_ig_posts(conn, business_id, ig_user_id, token)
                            ig_insights = _sync_ig_post_insights(conn, business_id, token)
                            result["instagram"] = {
                                "status": "ok (via fb token)",
                                "imported": imported,
                                "posts_synced": ig_insights["synced"],
                                "with_insights": ig_insights["with_data"],
                            }
                        except Exception as e2:
                            log.error("Instagram sync failed with FB token too: %s", e2)
                            result["instagram"]["status"] = f"error: {e2}"
                    else:
                        log.error("Instagram sync failed for %s: %s", business_id, e)
                        result["instagram"]["status"] = f"error: {e}"

    return result


def sync_all() -> int:
    """Sync all businesses that have any connected platform."""
    with db.get_conn() as conn:
        business_ids = [str(row[0]) for row in conn.execute(
            "SELECT DISTINCT business_id FROM business_credentials"
        ).fetchall()]
    for business_id in business_ids:
        try:
            sync_business(business_id)
        except Exception as e:
            log.error("sync_all failed for %s: %s", business_id, e)
    return len(business_ids)
