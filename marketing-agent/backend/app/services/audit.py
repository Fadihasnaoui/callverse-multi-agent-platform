"""Full digital marketing audit — the core intelligence of the dashboard.

Pulls a business's REAL post history from both Facebook and Instagram,
computes structured findings (cadence, format mix, engagement, best/worst),
fetches page-level KPIs, then asks the AI to produce a SWOT-style analysis
that pinpoints specific weak points and actionable opportunities.
"""
import logging
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean

from app.ai import prompts, run as ai_run
from app.ai.schemas import AuditOut
from app.services import graph_api

log = logging.getLogger(__name__)


def _parse_created_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _media_type(post: dict) -> str:
    attachments = (post.get("attachments") or {}).get("data") or []
    if not attachments:
        # Instagram uses media_type directly
        ig_type = post.get("media_type", "").upper()
        if ig_type in ("VIDEO", "REELS"):
            return "video"
        if ig_type == "CAROUSEL_ALBUM":
            return "carousel"
        return "text" if not ig_type else "photo"
    return attachments[0].get("media_type", "unknown").lower()


def _posting_cadence(posts: list[dict]) -> dict:
    timestamps = []
    for p in posts:
        ts = p.get("created_time") or p.get("timestamp")
        if ts:
            timestamps.append(_parse_created_time(ts))
    timestamps.sort()
    if len(timestamps) < 2:
        return {"posts_count": len(timestamps), "avg_gap_days": None, "longest_gap_days": None}
    gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 86400 for i in range(len(timestamps) - 1)]
    # Posts per week
    span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    posts_per_week = round(len(timestamps) / max(span_days / 7, 1), 1)
    return {
        "posts_count": len(timestamps),
        "avg_gap_days": round(mean(gaps), 1),
        "longest_gap_days": round(max(gaps), 1),
        "shortest_gap_days": round(min(gaps), 1),
        "posts_per_week": posts_per_week,
        "first_post": timestamps[0].isoformat(),
        "last_post": timestamps[-1].isoformat(),
        "days_since_last_post": round((datetime.now(timezone.utc) - timestamps[-1]).total_seconds() / 86400, 1),
    }


def _engagement_score(insights: dict) -> int | None:
    if not insights:
        return None
    # FB insights / inline
    clicks = insights.get("post_clicks") or 0
    reactions = insights.get("post_reactions_by_type_total") or insights.get("reactions_inline") or 0
    video_views = insights.get("post_video_views") or 0
    shares = insights.get("shares") or 0
    comments_inline = insights.get("comments_inline") or 0
    # IG inline metrics
    likes = insights.get("like_count") or 0
    comments = insights.get("comments_count") or 0
    total = clicks + reactions + video_views + shares + comments_inline + likes + comments
    return total if total > 0 else None


def _format_mix(posts_with_insights: list[dict]) -> dict:
    by_type = defaultdict(list)
    for post in posts_with_insights:
        score = _engagement_score(post.get("insights", {}))
        media = _media_type(post)
        by_type[media].append(score or 0)
    return {
        media_type: {
            "count": len(values),
            "avg_engagement": round(mean(values), 1) if values else 0,
            "pct_of_total": round(len(values) / max(len(posts_with_insights), 1) * 100, 1),
        }
        for media_type, values in by_type.items()
    }


def _best_and_worst(posts_with_insights: list[dict], top_n: int = 3) -> dict:
    ranked = [p for p in posts_with_insights if _engagement_score(p.get("insights", {})) is not None]
    ranked.sort(key=lambda p: _engagement_score(p.get("insights", {})), reverse=True)

    def summarize(post: dict) -> dict:
        return {
            "message": (post.get("message") or post.get("caption") or "")[:200],
            "media_type": _media_type(post),
            "engagement_score": _engagement_score(post.get("insights", {})),
            "created_time": post.get("created_time") or post.get("timestamp"),
            "platform": post.get("_platform", "unknown"),
        }

    return {
        "best": [summarize(p) for p in ranked[:top_n]],
        "worst": [summarize(p) for p in ranked[-top_n:]] if len(ranked) > top_n else [],
    }


def _time_distribution(posts: list[dict]) -> dict:
    """When does this business post? Helps spot missed time slots."""
    hours = defaultdict(int)
    days = defaultdict(int)
    for p in posts:
        ts = p.get("created_time") or p.get("timestamp")
        if not ts:
            continue
        dt = _parse_created_time(ts)
        hours[dt.hour] += 1
        days[dt.strftime("%A")] += 1
    return {
        "by_hour": dict(sorted(hours.items())),
        "by_day": dict(sorted(days.items(), key=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(x[0]) if x[0] in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] else 7)),
        "peak_hour": max(hours, key=hours.get) if hours else None,
        "peak_day": max(days, key=days.get) if days else None,
    }


def _tone_sample(posts: list[dict], sample_size: int = 10) -> list[str]:
    captions = [p.get("message") or p.get("caption") or "" for p in posts if (p.get("message") or p.get("caption"))]
    return captions[:sample_size]


def _category_performance(posts_with_insights: list[dict]) -> dict:
    """Which content categories drive the most engagement."""
    from app.services.metrics import _detect_category
    by_cat = defaultdict(list)
    for post in posts_with_insights:
        caption = post.get("message") or post.get("caption") or ""
        cat = _detect_category(caption)
        score = _engagement_score(post.get("insights", {}))
        by_cat[cat].append(score or 0)
    return {
        cat: {
            "count": len(scores),
            "avg_engagement": round(mean(scores), 1) if scores else 0,
            "pct_of_total": round(len(scores) / max(len(posts_with_insights), 1) * 100, 1),
        }
        for cat, scores in sorted(by_cat.items(), key=lambda kv: -mean(kv[1]) if kv[1] else 0)
    }


def _strategy_layer_mix(posts: list[dict]) -> dict:
    """Classify posts into pillar/distribution/engagement and measure the split."""
    from app.services.metrics import _detect_strategy_layer
    layers = defaultdict(int)
    for post in posts:
        caption = post.get("message") or post.get("caption") or ""
        media_type = _media_type(post)
        layer = _detect_strategy_layer(caption, media_type)
        layers[layer] += 1
    total = sum(layers.values()) or 1
    return {
        layer: {
            "count": count,
            "pct": round(count / total * 100, 1),
        }
        for layer, count in layers.items()
    }


def _content_health_score(cadence: dict, format_mix: dict, strategy_mix: dict) -> dict:
    """Score 0-100 across key health dimensions for the learning algorithm."""
    scores = {}

    # Consistency score (0-100): based on posts_per_week and longest gap
    ppw = cadence.get("posts_per_week", 0)
    longest_gap = cadence.get("longest_gap_days", 999)
    consistency = min(100, int((ppw / 3) * 60 + max(0, 40 - longest_gap)))
    scores["consistency"] = max(0, consistency)

    # Format diversity (0-100): how many formats used, penalize >70% single format
    n_formats = len(format_mix)
    max_pct = max((f["pct_of_total"] for f in format_mix.values()), default=0)
    scores["format_diversity"] = min(100, int(n_formats * 20 + max(0, 40 - (max_pct - 30))))

    # Strategy balance (0-100): how close to 20/50/30 ideal
    pillar_pct = strategy_mix.get("pillar", {}).get("pct", 0)
    dist_pct = strategy_mix.get("distribution", {}).get("pct", 0)
    engage_pct = strategy_mix.get("engagement", {}).get("pct", 0)
    deviation = abs(pillar_pct - 20) + abs(dist_pct - 50) + abs(engage_pct - 30)
    scores["strategy_balance"] = max(0, int(100 - deviation))

    scores["overall"] = int(mean(scores.values()))
    return scores


def _fetch_fb_data(page_id: str, access_token: str, post_limit: int) -> dict:
    """Fetch Facebook posts + insights + page-level metrics."""
    posts = graph_api.fetch_page_posts(page_id, access_token, limit=post_limit)
    posts_with_insights = []
    for post in posts:
        shares = (post.get("shares") or {}).get("count", 0)
        reactions_summary = (post.get("reactions", {}).get("summary") or {})
        comments_summary = (post.get("comments", {}).get("summary") or {})
        insights = {
            "reactions_inline": reactions_summary.get("total_count", 0),
            "comments_inline": comments_summary.get("total_count", 0),
            "shares": shares,
        }
        # Only call per-post insights API if we got inline engagement
        # (avoids hundreds of empty API calls when App Review is pending)
        if insights["reactions_inline"] or insights["comments_inline"] or shares:
            try:
                api_insights = graph_api.fetch_post_insights(post["id"], access_token)
                insights.update(api_insights)
            except Exception:
                pass
        posts_with_insights.append({**post, "insights": insights, "_platform": "facebook"})

    page_metrics = {}
    try:
        page_metrics = graph_api.fetch_page_insights(page_id, access_token)
    except Exception as e:
        log.warning("FB page insights unavailable: %s", e)

    return {"posts": posts_with_insights, "page_metrics": page_metrics}


def _fetch_ig_data(ig_user_id: str, access_token: str, post_limit: int) -> dict:
    """Fetch Instagram media + insights + profile metrics."""
    media_list = graph_api.fetch_ig_media(ig_user_id, access_token, limit=post_limit)
    posts_with_insights = []
    for media in media_list:
        inline_engagement = (media.get("like_count") or 0) + (media.get("comments_count") or 0)
        insights = {
            "like_count": media.get("like_count", 0),
            "comments_count": media.get("comments_count", 0),
        }
        # Only call per-media insights API if inline metrics exist
        if inline_engagement > 0:
            try:
                api_insights = graph_api.fetch_ig_media_insights(
                    media["id"], access_token, media.get("media_type", "IMAGE")
                )
                insights.update(api_insights)
            except Exception:
                pass
        media["insights"] = insights
        media["_platform"] = "instagram"
        posts_with_insights.append(media)

    profile = graph_api.fetch_ig_profile(ig_user_id, access_token)
    account_insights = graph_api.fetch_ig_account_insights(ig_user_id, access_token)

    return {
        "posts": posts_with_insights,
        "profile": profile,
        "account_insights": account_insights,
    }


def run_audit(
    page_id: str | None = None,
    fb_access_token: str | None = None,
    ig_user_id: str | None = None,
    ig_access_token: str | None = None,
    post_limit: int = 100,
) -> dict:
    """Run a full digital marketing audit across available platforms.
    Returns structured findings ready for the dashboard and learning engine."""

    all_posts = []
    fb_data = {}
    ig_data = {}

    # Facebook
    if page_id and fb_access_token:
        try:
            fb_data = _fetch_fb_data(page_id, fb_access_token, post_limit)
            all_posts.extend(fb_data["posts"])
        except graph_api.GraphAPIError as e:
            log.error("Facebook audit failed: %s", e)
            fb_data = {"error": str(e)}

    # Instagram
    if ig_user_id and ig_access_token:
        try:
            ig_data = _fetch_ig_data(ig_user_id, ig_access_token, post_limit)
            all_posts.extend(ig_data["posts"])
        except graph_api.GraphAPIError as e:
            log.error("Instagram audit failed: %s", e)
            ig_data = {"error": str(e)}

    if not all_posts:
        return {
            "posts_analyzed": 0,
            "error": "No posts found on any connected platform",
            "platforms": {"facebook": bool(fb_data), "instagram": bool(ig_data)},
        }

    # Cross-platform analysis
    cadence = _posting_cadence(all_posts)
    format_mix = _format_mix(all_posts)
    best_worst = _best_and_worst(all_posts)
    time_dist = _time_distribution(all_posts)
    tone_sample = _tone_sample(all_posts)
    cat_perf = _category_performance(all_posts)
    strategy_mix = _strategy_layer_mix(all_posts)

    # Per-platform breakdown
    fb_posts = [p for p in all_posts if p.get("_platform") == "facebook"]
    ig_posts = [p for p in all_posts if p.get("_platform") == "instagram"]

    platform_breakdown = {}
    if fb_posts:
        platform_breakdown["facebook"] = {
            "posts": len(fb_posts),
            "cadence": _posting_cadence(fb_posts),
            "format_mix": _format_mix(fb_posts),
            "page_metrics": fb_data.get("page_metrics", {}),
        }
    if ig_posts:
        platform_breakdown["instagram"] = {
            "posts": len(ig_posts),
            "cadence": _posting_cadence(ig_posts),
            "format_mix": _format_mix(ig_posts),
            "profile": ig_data.get("profile", {}),
            "account_insights": ig_data.get("account_insights", {}),
        }

    engagement_data_available = bool(format_mix)
    health_score = _content_health_score(cadence, format_mix, strategy_mix)

    structured_findings = {
        "cadence": cadence,
        "format_mix": format_mix,
        "best_and_worst": best_worst,
        "time_distribution": time_dist,
        "category_performance": cat_perf,
        "strategy_mix": strategy_mix,
        "health_score": health_score,
        "tone_sample_size": len(tone_sample),
        "engagement_data_available": engagement_data_available,
        "platform_breakdown": platform_breakdown,
        "platforms_audited": list(platform_breakdown.keys()),
        "total_posts": len(all_posts),
    }

    # AI analysis
    ai_input = (
        f"Real data for this business (both platforms combined):\n{structured_findings}\n\n"
        f"Sample of recent captions (for tone only, do not quote numbers from here):\n{tone_sample}\n\n"
    )
    if not engagement_data_available:
        ai_input += (
            "engagement_data_available is false: Meta did not return usable engagement numbers. "
            "Say so plainly as one finding and base the rest on cadence and content data."
        )
    if len(platform_breakdown) == 1:
        missing = "Instagram" if "facebook" in platform_breakdown else "Facebook"
        ai_input += f"\nOnly one platform is connected. {missing} is not connected — flag this as a weakness."

    report = ai_run(
        "audit", prompts.AUDIT.system, ai_input,
        AuditOut, version=prompts.AUDIT.version,
    )

    # Learning seed
    best_format = (
        max(format_mix.items(), key=lambda kv: kv[1]["avg_engagement"])[0]
        if format_mix
        else None
    )
    # Best category by engagement
    best_category = (
        max(cat_perf.items(), key=lambda kv: kv[1]["avg_engagement"])[0]
        if cat_perf
        else None
    )

    learning_seed = {
        "best_performing_format": best_format,
        "best_performing_category": best_category,
        "format_mix": format_mix,
        "category_performance": cat_perf,
        "strategy_mix": strategy_mix,
        "health_score": health_score,
        "posting_cadence": cadence,
        "time_distribution": time_dist,
        "platform_breakdown": {k: {"posts": v["posts"]} for k, v in platform_breakdown.items()},
        "confidence": "cadence_only" if not engagement_data_available else "engagement_available",
        "recommendations": [str(o) for o in (report.opportunities or [])[:3]] or [
            "Publish three times per week for four weeks and keep gaps below seven days.",
            "Test Reels and static posts in parallel before choosing a preferred format.",
            "Track each post through to leads and sales before making performance claims.",
        ],
    }

    return {
        "posts_analyzed": len(all_posts),
        "platforms_audited": list(platform_breakdown.keys()),
        "findings": structured_findings,
        "strengths": [str(s) for s in report.strengths],
        "weaknesses": [str(w) for w in report.weaknesses],
        "opportunities": [str(o) for o in report.opportunities],
        "kpi_summary": report.kpi_summary.model_dump() if report.kpi_summary else {},
        "platform_comparison": report.platform_comparison,
        "closing_note": report.closing_note,
        "health_score": health_score,
        "learning_seed": learning_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
