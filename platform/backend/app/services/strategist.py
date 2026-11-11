"""The senior digital marketing strategist brain.

This module assembles the strategic context that a senior marketer would
consider before making ANY content decision. It reads performance history,
audience signals, platform-specific patterns, and competitive positioning
to produce a strategic brief that feeds into content generation, weekly
planning, and scheduling.

The key insight: a senior marketer doesn't just write posts — they think
about WHY this post, WHY now, WHY this format, and HOW it connects to
the business goal. This module captures that reasoning."""
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from app import db

log = logging.getLogger("strategist")
TUNIS = ZoneInfo("Africa/Tunis")

_BENCHMARKS_CACHE: dict | None = None


def _load_industry_benchmarks() -> dict | None:
    global _BENCHMARKS_CACHE
    if _BENCHMARKS_CACHE is not None:
        return _BENCHMARKS_CACHE
    p = Path(__file__).resolve().parent.parent.parent / "data" / "industry_benchmarks.json"
    if not p.exists():
        return None
    candidate = json.loads(p.read_text(encoding="utf-8"))
    # An unmatched TikTok sample is not a Facebook/Instagram performance baseline.
    if candidate.get('platform') not in ('facebook','instagram') or candidate.get('metric_definition') != 'interactions_per_reach_percent':
        return None
    _BENCHMARKS_CACHE = candidate
    return _BENCHMARKS_CACHE


def build_content_directives(business_id: str) -> str:
    """Convert raw benchmarks + Ozone's own performance into concrete creative rules
    that the content generation AI can execute on directly.

    Returns a block of text with numbered rules, not raw numbers."""
    bm = _load_industry_benchmarks()
    if not bm:
        return ""

    directives = []

    # 1. Duration directive from benchmark data
    by_dur = bm.get("by_duration", {})
    if by_dur:
        valid_buckets = [(k, v) for k, v in by_dur.items() if k != "unknown" and v.get("sample_size", 0) > 10]
        if not valid_buckets:
            valid_buckets = [(k, v) for k, v in by_dur.items() if k != "unknown"]
        best_bucket = max(valid_buckets, key=lambda x: x[1].get("engagement_rate", {}).get("median", 0))
        worst_bucket = min(
            valid_buckets,
            key=lambda x: x[1].get("engagement_rate", {}).get("median", 0),
        )
        directives.append(
            f"VIDEO LENGTH: {best_bucket[0]} videos get the highest engagement "
            f"({best_bucket[1]['engagement_rate']['median']}% median). "
            f"Keep reels in this range. Avoid {worst_bucket[0]} "
            f"({worst_bucket[1]['engagement_rate']['median']}% — nearly half the engagement)."
        )

    # 2. Engagement type directive
    overall = bm.get("overall", {})
    save_rate = overall.get("save_rate", {}).get("median", 0)
    share_rate = overall.get("share_rate", {}).get("median", 0)
    comment_rate = overall.get("comment_rate", {}).get("median", 0)
    if save_rate > share_rate:
        directives.append(
            "SAVE-WORTHY CONTENT WINS: saves outpace shares in the industry. "
            "Create content people bookmark — practical guides, destination lists, "
            "packing tips, budget breakdowns, itineraries. Add 'Save this for your next trip' cues."
        )
    else:
        directives.append(
            "SHAREABLE CONTENT WINS: shares outpace saves in the industry. "
            "Create content people send to friends — emotional reels, surprising facts, "
            "jaw-dropping visuals. Make posts that trigger 'I need to show this to someone'."
        )

    # 3. Compare Ozone's performance to benchmarks and give specific improvement directives
    travel_eng = bm.get("travel_content", {}).get("engagement_rate", {})
    overall_eng = overall.get("engagement_rate", {})
    try:
        with db.get_conn() as conn:
            ozone_stats = conn.execute(
                """SELECT
                    COALESCE(AVG(COALESCE((analytics_features->>'like_count')::int,
                                          (analytics_features->>'reactions_count')::int, 0) +
                                 COALESCE((analytics_features->>'comments_count')::int, 0)), 0),
                    COUNT(*) FILTER (WHERE type = 'reel'),
                    COUNT(*) FILTER (WHERE type = 'static_post'),
                    COUNT(*)
                FROM content_analytics WHERE business_id = %s AND status = 'published'""",
                (business_id,),
            ).fetchone()
        our_avg = float(ozone_stats[0])
        reel_count = ozone_stats[1]
        static_count = ozone_stats[2]
        total_count = ozone_stats[3]

        if total_count > 0:
            reel_pct = reel_count / total_count * 100
            if reel_pct < 30 and reel_count < static_count:
                directives.append(
                    f"FORMAT MIX (MANDATORY): Only {reel_pct:.0f}% of Ozone's content is Reels "
                    f"({reel_count} vs {static_count} static). Industry data shows video "
                    f"consistently outperforms. When planning a week of 5 posts, at least 2 MUST be reels. "
                    f"For 3 posts, at least 1 must be a reel."
                )
    except Exception:
        log.exception("ozone stats for directives failed")

    # 4. Travel-specific insight
    if travel_eng and travel_eng.get("sample_size", 0) > 10:
        directives.append(
            f"TRAVEL NICHE: Travel content gets {travel_eng.get('median', 0)}% median engagement "
            f"vs {overall_eng.get('median', 0)}% overall. Travel audiences engage slightly less "
            f"but SAVE more (planning behavior). Lean into 'save for later' content: itineraries, "
            f"price comparisons, seasonal guides, packing lists."
        )

    # 5. Comment engagement
    if comment_rate:
        directives.append(
            f"CONVERSATION STARTERS: Industry median comment rate is only {comment_rate}%. "
            "Posts that ask a specific question ('Have you tried X?', 'Which would you pick: A or B?') "
            "break through. End every post with a genuine question, not a generic 'What do you think?'"
        )

    if not directives:
        return ""

    return (
        "--- CONTENT DIRECTIVES (apply these to every piece of content) ---\n"
        + "\n".join(f"{i+1}. {d}" for i, d in enumerate(directives))
    )

# --- Platform playbooks ---
# What a senior marketer knows about each platform's algorithm and audience

PLATFORM_PLAYBOOKS = {
    "facebook": {
        "algorithm_signals": [
            "test content that invites meaningful comments and shares; no verified algorithm weighting is assumed",
            "conversation chains (replies to comments) are the #1 organic boost",
            "compare native video with link posts in matched experiments",
            "compare photo collections with single images where the publishing adapter supports them",
            "posting consistency matters more than volume — 3-5x/week is sweet spot",
            "shares are the strongest signal — content worth sharing = content that reaches new audiences",
        ],
        "content_rules": {
            "caption_length": "medium (150-300 chars performs best for engagement)",
            "hook_window": "first 2 lines must stop the scroll — question or bold claim",
            "hashtags": "5-8 max, FB doesn't reward hashtag stuffing",
            "cta_style": "implicit (question at end) outperforms explicit (click here)",
            "video_length": "60-90 seconds for reels, 3-5 min for native video",
        },
        "audience_behavior_tunisia": {
            "peak_hours": "18:00-22:00 weekdays, 10:00-14:00 weekends",
            "high_engagement_days": "Tuesday, Thursday, Sunday",
            "language": "French + Darija mix performs best, pure MSA underperforms",
            "seasonal": "June-August domestic tourism peak, Ramadan shifts to post-iftar",
        },
    },
    "instagram": {
        "algorithm_signals": [
            "saves are the #1 signal — saveable content (tips, guides, itineraries) dominates",
            "shares via DM are weighted heavily — 'send this to someone who needs this'",
            "test reels against static posts for discovery rather than assuming a fixed reach multiplier",
            "carousel completion rate matters — if people swipe all slides, huge boost",
            "story interactions (polls, questions) feed into feed ranking",
            "posting 4-7 reels/week is current sweet spot for growth",
        ],
        "content_rules": {
            "caption_length": "long captions (500-1000 chars) work IF they tell a story",
            "hook_window": "first 125 chars before 'more' — make it count",
            "hashtags": "3-5 targeted > 30 generic, mix niche + mid-volume",
            "cta_style": "save-bait ('save for your next trip') and share-bait ('tag someone')",
            "video_length": "15-30 seconds for reels (hook in first 3 seconds)",
        },
        "audience_behavior_tunisia": {
            "peak_hours": "19:00-23:00 weekdays, 11:00-15:00 weekends",
            "high_engagement_days": "Wednesday, Friday, Sunday",
            "language": "French-forward with Darija flavor, visual-first",
            "seasonal": "same as FB but younger audience = more responsive to trends",
        },
    },
}

# --- Content funnel framework ---
# A senior marketer thinks in funnels, not isolated posts

FUNNEL_STAGES = {
    "awareness": {
        "goal": "reach new audiences who don't know the brand yet",
        "formats": ["reels", "shareable carousels", "destination showcases"],
        "metrics": "reach, shares, profile visits",
        "ratio": 0.30,
    },
    "consideration": {
        "goal": "educate and build trust with people who've seen the brand",
        "formats": ["detailed carousels", "testimonials", "behind-the-scenes", "guides"],
        "metrics": "saves, comments, website clicks",
        "ratio": 0.40,
    },
    "conversion": {
        "goal": "drive bookings and inquiries from warm audience",
        "formats": ["limited offers", "social proof", "direct CTA posts"],
        "metrics": "DMs, link clicks, leads, bookings",
        "ratio": 0.20,
    },
    "retention": {
        "goal": "keep past customers engaged and generate referrals",
        "formats": ["UGC reposts", "loyalty offers", "community highlights"],
        "metrics": "repeat engagement, referral DMs, reviews",
        "ratio": 0.10,
    },
}


def _load_performance_context(conn, business_id: str) -> dict:
    """Load the latest performance data a strategist would review."""
    # Latest audit findings
    audit_row = conn.execute(
        "SELECT findings, created_at FROM social_audits WHERE business_id = %s "
        "ORDER BY created_at DESC LIMIT 1",
        (business_id,),
    ).fetchone()
    audit = audit_row[0] if audit_row else {}
    audit_age = (datetime.now(TUNIS) - audit_row[1].astimezone(TUNIS)).days if audit_row else None

    # Latest learning insights
    learning_row = conn.execute(
        "SELECT insights, summary_text FROM learning_insights WHERE business_id = %s "
        "AND source <> 'simulation_baseline' ORDER BY created_at DESC LIMIT 1",
        (business_id,),
    ).fetchone()
    learning = learning_row[0] if learning_row else {}

    # Recent post performance by category and platform
    recent_perf = conn.execute(
        """SELECT content_category, strategy_layer, platform, type,
                  COUNT(*) as posts,
                  AVG(COALESCE((analytics_features->>'like_count')::int,
                              (analytics_features->>'reactions_count')::int, 0)
                    + COALESCE((analytics_features->>'comments_count')::int, 0)
                    + COALESCE((analytics_features->>'shares')::int, 0)) as avg_engagement
           FROM content_analytics
           WHERE business_id = %s AND status = 'published'
             AND published_at > now() - interval '90 days'
           GROUP BY content_category, strategy_layer, platform, type
           ORDER BY avg_engagement DESC""",
        (business_id,),
    ).fetchall()

    category_perf = defaultdict(lambda: {"posts": 0, "avg_engagement": 0})
    platform_perf = defaultdict(lambda: {"posts": 0, "avg_engagement": 0})
    format_perf = defaultdict(lambda: {"posts": 0, "avg_engagement": 0})
    strategy_perf = defaultdict(lambda: {"posts": 0, "avg_engagement": 0})

    for row in recent_perf:
        cat, strategy, platform, fmt, posts, avg_eng = row
        avg_eng = float(avg_eng or 0)
        if cat:
            category_perf[cat]["posts"] += posts
            category_perf[cat]["avg_engagement"] = max(category_perf[cat]["avg_engagement"], avg_eng)
        if platform:
            platform_perf[platform]["posts"] += posts
            platform_perf[platform]["avg_engagement"] = max(platform_perf[platform]["avg_engagement"], avg_eng)
        if fmt:
            format_perf[fmt]["posts"] += posts
            format_perf[fmt]["avg_engagement"] = max(format_perf[fmt]["avg_engagement"], avg_eng)
        if strategy:
            strategy_perf[strategy]["posts"] += posts

    # Recent posting gaps
    gap_row = conn.execute(
        """SELECT platform,
                  MAX(published_at) as last_post,
                  COUNT(*) FILTER (WHERE published_at > now() - interval '7 days') as posts_last_week,
                  COUNT(*) FILTER (WHERE published_at > now() - interval '30 days') as posts_last_month
           FROM content_analytics
           WHERE business_id = %s AND status = 'published' AND published_at IS NOT NULL
           GROUP BY platform""",
        (business_id,),
    ).fetchall()

    posting_status = {}
    for row in gap_row:
        platform, last_post, last_week, last_month = row
        days_since = (datetime.now(TUNIS) - last_post.astimezone(TUNIS)).days if last_post else None
        posting_status[platform] = {
            "days_since_last_post": days_since,
            "posts_last_week": last_week,
            "posts_last_month": last_month,
            "urgency": "critical" if days_since and days_since > 14
                       else "high" if days_since and days_since > 7
                       else "normal",
        }

    # Upcoming scheduled content
    upcoming = conn.execute(
        "SELECT platform, type, scheduled_for, content_category, strategy_layer "
        "FROM content_analytics WHERE business_id = %s AND status IN ('approved', 'pending') "
        "AND scheduled_for IS NOT NULL AND scheduled_for > now() "
        "ORDER BY scheduled_for LIMIT 10",
        (business_id,),
    ).fetchall()

    pipeline = [
        {"platform": r[0], "type": r[1], "scheduled": r[2].isoformat() if r[2] else None,
         "category": r[3], "strategy_layer": r[4]}
        for r in upcoming
    ]

    return {
        "audit": {
            "health_score": audit.get("health_score", {}),
            "strategy_mix": audit.get("strategy_mix", {}),
            "category_performance": audit.get("category_performance", {}),
            "cadence": audit.get("cadence", {}),
            "age_days": audit_age,
        },
        "learning": learning,
        "category_performance": dict(category_perf),
        "platform_performance": dict(platform_perf),
        "format_performance": dict(format_perf),
        "strategy_balance": dict(strategy_perf),
        "posting_status": posting_status,
        "pipeline": pipeline,
    }


def _load_page_kpis(conn, business_id: str) -> dict:
    """Load the latest page-level KPIs."""
    rows = conn.execute(
        """SELECT platform, metric_name, metric_value
           FROM page_metrics WHERE business_id = %s
           AND fetched_at = (SELECT MAX(fetched_at) FROM page_metrics WHERE business_id = %s)""",
        (business_id, business_id),
    ).fetchall()
    kpis = defaultdict(dict)
    for platform, name, value in rows:
        kpis[platform][name] = float(value) if value else 0
    return dict(kpis)


def build_strategic_brief(business_id: str, context: str = "content") -> str:
    """Build the strategic context brief that feeds into content generation.

    This is what a senior marketer would think through before writing anything:
    - What's working and what's not (data)
    - What gap in the content mix needs filling (strategy)
    - What the platform algorithm rewards right now (platform intelligence)
    - What's already in the pipeline (coordination)
    - What the business goal is this period (direction)

    Args:
        business_id: The business to strategize for
        context: One of 'content', 'weekly', 'schedule', 'audit' — changes emphasis
    """
    with db.get_conn() as conn:
        perf = _load_performance_context(conn, business_id)
        kpis = _load_page_kpis(conn, business_id)

        # Business info
        biz = conn.execute(
            "SELECT name, industry, brand_voice FROM businesses WHERE id = %s",
            (business_id,),
        ).fetchone()

    brief_parts = []

    # 1. SITUATION ASSESSMENT
    health = perf["audit"].get("health_score", {})
    overall_health = health.get("overall", "unknown")
    brief_parts.append(
        f"=== STRATEGIC SITUATION ===\n"
        f"Business: {biz[0] if biz else 'Unknown'} ({biz[1] if biz else 'travel'})\n"
        f"Content health score: {overall_health}/100 "
        f"(consistency: {health.get('consistency', '?')}, "
        f"format diversity: {health.get('format_diversity', '?')}, "
        f"strategy balance: {health.get('strategy_balance', '?')})"
    )

    # 2. WHAT'S WORKING (from real data)
    winners = []
    cat_perf = perf.get("category_performance", {})
    if cat_perf:
        top_cats = sorted(cat_perf.items(), key=lambda x: x[1].get("avg_engagement", 0), reverse=True)[:3]
        cat_parts = ["{} ({:.1f} avg eng)".format(c, d.get("avg_engagement", 0)) for c, d in top_cats]
        winners.append("Top content categories: " + ", ".join(cat_parts))

    fmt_perf = perf.get("format_performance", {})
    if fmt_perf:
        top_fmts = sorted(fmt_perf.items(), key=lambda x: x[1].get("avg_engagement", 0), reverse=True)[:2]
        fmt_parts = ["{} ({:.1f} avg eng)".format(f, d.get("avg_engagement", 0)) for f, d in top_fmts]
        winners.append("Best formats: " + ", ".join(fmt_parts))

    learning = perf.get("learning", {})
    if learning.get("recommendations"):
        recs = learning["recommendations"][:2]
        winners.append("Learning engine says: " + "; ".join(recs))

    if winners:
        brief_parts.append("=== WHAT'S WORKING ===\n" + "\n".join(f"- {w}" for w in winners))

    # 3. STRATEGY GAP ANALYSIS
    strategy = perf.get("strategy_balance", {})
    total_strat = sum(d.get("posts", 0) for d in strategy.values()) or 1
    current_mix = {
        k: round(d.get("posts", 0) / total_strat * 100)
        for k, d in strategy.items()
    }
    target_mix = {"pillar": 20, "distribution": 50, "engagement": 30}
    gaps = []
    for layer, target in target_mix.items():
        current = current_mix.get(layer, 0)
        diff = target - current
        if abs(diff) > 10:
            direction = "NEEDS MORE" if diff > 0 else "has excess"
            gaps.append(f"{layer}: {current}% (target {target}%) — {direction}")

    if gaps:
        brief_parts.append(
            "=== STRATEGY GAPS ===\n"
            "Current vs. target content layer mix:\n" +
            "\n".join(f"- {g}" for g in gaps) +
            "\nPrioritize creating content for the layers marked NEEDS MORE."
        )

    # 4. POSTING URGENCY
    posting = perf.get("posting_status", {})
    urgency_notes = []
    for platform, status in posting.items():
        days = status.get("days_since_last_post")
        if status["urgency"] == "critical":
            urgency_notes.append(f"CRITICAL: {platform} has been silent for {days} days — post immediately")
        elif status["urgency"] == "high":
            urgency_notes.append(f"WARNING: {platform} last posted {days} days ago — prioritize this platform")
        else:
            urgency_notes.append(f"{platform}: {status['posts_last_week']} posts last week, {days} days since last post")

    if urgency_notes:
        brief_parts.append("=== POSTING STATUS ===\n" + "\n".join(f"- {n}" for n in urgency_notes))

    # 5. PIPELINE AWARENESS
    pipeline = perf.get("pipeline", [])
    if pipeline:
        pipe_summary = ", ".join(
            f"{p['platform']}/{p['type']} on {p['scheduled'][:10]}" for p in pipeline[:5]
        )
        brief_parts.append(f"=== ALREADY IN PIPELINE ===\n{pipe_summary}\nAvoid duplicating these topics/angles.")

    # 6. PLATFORM INTELLIGENCE (context-specific)
    if context in ("content", "weekly"):
        platform_notes = []
        for platform in ("facebook", "instagram"):
            pb = PLATFORM_PLAYBOOKS.get(platform, {})
            if platform in posting:
                signals = pb.get("algorithm_signals", [])[:3]
                rules = pb.get("content_rules", {})
                platform_notes.append(
                    f"\n{platform.upper()} algorithm priorities: {'; '.join(signals)}\n"
                    f"Content rules: hook in {rules.get('hook_window', 'first 2 lines')}, "
                    f"caption {rules.get('caption_length', 'medium')}, "
                    f"hashtags {rules.get('hashtags', '5-8')}"
                )
        if platform_notes:
            brief_parts.append("=== PLATFORM INTELLIGENCE ===" + "".join(platform_notes))

    # 7. FUNNEL GUIDANCE
    if context == "weekly":
        brief_parts.append(
            "=== CONTENT FUNNEL FRAMEWORK ===\n"
            "A senior strategist plans the week as a funnel, not random posts:\n"
            "- 30% Awareness: reach new people (reels, shareable content, destination showcases)\n"
            "- 40% Consideration: build trust (guides, testimonials, detailed carousels)\n"
            "- 20% Conversion: drive action (offers, social proof, urgency)\n"
            "- 10% Retention: reward loyalty (UGC, community, exclusive content)\n"
            "Each post should serve ONE funnel stage. The week should cover all four."
        )

    # 8. AUDIENCE SIZE CONTEXT
    if kpis:
        audience_parts = []
        for platform, metrics in kpis.items():
            followers = metrics.get("followers_count", metrics.get("fan_count", 0))
            if followers:
                audience_parts.append(f"{platform}: {int(followers):,} followers")
        if audience_parts:
            brief_parts.append(f"=== AUDIENCE ===\n{', '.join(audience_parts)}")

    # 9. COMMENT SENTIMENT INTELLIGENCE
    try:
        from app.services.sentiment import load_sentiment_context
        sent = load_sentiment_context(business_id)
        if sent and sent.get("total_comments", 0) > 0:
            sent_parts = [
                f"Overall sentiment: {sent['overall_sentiment']}% "
                f"(positive minus negative, across {sent['total_comments']} comments)",
                f"Average buying intent: {sent['avg_buying_intent']}/100 "
                f"({sent['high_intent_comments']} high-intent comments detected)",
            ]
            if sent.get("top_topics"):
                topics = ", ".join(f"{t} ({n})" for t, n in list(sent["top_topics"].items())[:5])
                sent_parts.append(f"What people talk about: {topics}")
            if sent.get("by_category"):
                cat_lines = []
                for cat, d in list(sent["by_category"].items())[:4]:
                    cat_lines.append(
                        f"{cat}: sentiment {d['avg_sentiment']}%, "
                        f"buying intent {d['avg_intent']}/100 ({d['comments']} comments)"
                    )
                sent_parts.append("Sentiment by content category:\n  " + "\n  ".join(cat_lines))
            brief_parts.append("=== COMMENT INTELLIGENCE ===\n" + "\n".join(f"- {s}" for s in sent_parts))
    except Exception:
        log.exception("sentiment context failed")

    # 10. COMPETITIVE INTELLIGENCE
    try:
        from app.services import competitors as comp_svc
        comp_pos = comp_svc.competitive_position(business_id)
        if comp_pos and comp_pos.get("leaderboard"):
            lb = comp_pos["leaderboard"]
            rank = comp_pos["rank"]
            total = comp_pos["total_tracked"]
            comp_lines = [f"Ozone ranks #{rank} of {total} tracked agencies by followers."]

            our = next((e for e in lb if e.get("is_us")), None)
            leader = lb[0] if lb else None
            if leader and not leader.get("is_us"):
                gap = leader["followers"] - (our["followers"] if our else 0)
                comp_lines.append(
                    f"Leader: {leader['name']} ({leader['followers']:,} followers) — "
                    f"gap of {gap:,} followers."
                )
            closest_above = None
            for i, e in enumerate(lb):
                if e.get("is_us") and i > 0:
                    closest_above = lb[i - 1]
                    break
            if closest_above:
                gap_above = closest_above["followers"] - (our["followers"] if our else 0)
                comp_lines.append(
                    f"Next above: {closest_above['name']} ({closest_above['followers']:,}) — "
                    f"close the {gap_above:,}-follower gap to move up."
                )
            most_active = max(
                (e for e in lb if e.get("talking_about")),
                key=lambda e: e["talking_about"], default=None,
            )
            if most_active and not most_active.get("is_us"):
                comp_lines.append(
                    f"Most buzz: {most_active['name']} ({most_active['talking_about']:,} talking about) — "
                    f"study what's driving their conversation."
                )
            brief_parts.append(
                "=== COMPETITIVE INTELLIGENCE ===\n" +
                "\n".join(f"- {l}" for l in comp_lines)
            )
    except Exception:
        log.exception("competitive intelligence failed")

    # 11. INDUSTRY BENCHMARKS (from external dataset analysis)
    try:
        benchmarks = _load_industry_benchmarks()
        if benchmarks:
            bm_parts = []
            overall = benchmarks.get("overall", {})
            travel = benchmarks.get("travel_content", {})
            eng = overall.get("engagement_rate", {})
            if eng:
                bm_parts.append(
                    f"General TikTok engagement rate: median {eng['median']}%, "
                    f"good (p75) {eng['p75']}%, excellent (p90) {eng['p90']}% "
                    f"(n={overall.get('sample_size', 0)})"
                )
            t_eng = travel.get("engagement_rate", {})
            if t_eng and travel.get("sample_size", 0) > 10:
                bm_parts.append(
                    f"Travel content specifically: median {t_eng['median']}%, "
                    f"good {t_eng.get('p75', '?')}%, excellent {t_eng.get('p90', '?')}%"
                )
            by_dur = benchmarks.get("by_duration", {})
            if by_dur:
                dur_lines = []
                for bucket in ("0-10s", "10-30s", "30-60s", "1-2min", "2min+"):
                    d = by_dur.get(bucket, {})
                    d_eng = d.get("engagement_rate", {})
                    if d_eng:
                        dur_lines.append(f"{bucket}: median {d_eng['median']}% (n={d.get('sample_size', 0)})")
                if dur_lines:
                    bm_parts.append("Engagement by video duration:\n  " + "\n  ".join(dur_lines))
            if bm_parts:
                brief_parts.append(
                    "=== INDUSTRY BENCHMARKS ===\n"
                    "Compare Ozone's metrics against these (source: 2000 TikTok videos, Shofo dataset):\n" +
                    "\n".join(f"- {b}" for b in bm_parts)
                )
    except Exception:
        log.exception("benchmark loading failed")

    return "\n\n".join(brief_parts)


def self_benchmark(business_id: str, period_days: int = 30) -> dict | None:
    """Compare Ozone's own recent period against the previous period of the same
    length. Detects seasonal patterns, improvement trends, and format shifts.

    Returns None if insufficient data (need posts in both periods)."""
    with db.get_conn() as conn:
        now = datetime.now(TUNIS)
        current_start = now - timedelta(days=period_days)
        previous_start = current_start - timedelta(days=period_days)

        def _period_stats(start, end):
            rows = conn.execute(
                """SELECT
                    COUNT(*) as posts,
                    COUNT(*) FILTER (WHERE type = 'reel') as reels,
                    COUNT(*) FILTER (WHERE type = 'static_post') as statics,
                    COALESCE(AVG(
                        COALESCE((analytics_features->>'like_count')::int,
                                 (analytics_features->>'reactions_count')::int, 0)
                        + COALESCE((analytics_features->>'comments_count')::int, 0)
                    ), 0) as avg_engagement,
                    COALESCE(SUM(
                        COALESCE((analytics_features->>'like_count')::int,
                                 (analytics_features->>'reactions_count')::int, 0)
                    ), 0) as total_likes,
                    COALESCE(SUM(
                        COALESCE((analytics_features->>'comments_count')::int, 0)
                    ), 0) as total_comments,
                    COUNT(DISTINCT platform) as platforms_used
                FROM content_analytics
                WHERE business_id = %s AND status = 'published'
                  AND COALESCE(published_at, created_at) BETWEEN %s AND %s""",
                (business_id, start, end),
            ).fetchone()
            posts = rows[0]
            if posts == 0:
                return None
            return {
                "posts": posts,
                "reels": rows[1],
                "statics": rows[2],
                "reel_pct": round(rows[1] / posts * 100),
                "avg_engagement": round(float(rows[3]), 1),
                "total_likes": int(rows[4]),
                "total_comments": int(rows[5]),
                "platforms_used": rows[6],
            }

        def _top_categories(start, end):
            cats = conn.execute(
                """SELECT content_category, COUNT(*),
                    AVG(COALESCE((analytics_features->>'like_count')::int,
                                 (analytics_features->>'reactions_count')::int, 0)
                        + COALESCE((analytics_features->>'comments_count')::int, 0))
                FROM content_analytics
                WHERE business_id = %s AND status = 'published'
                  AND content_category IS NOT NULL
                  AND COALESCE(published_at, created_at) BETWEEN %s AND %s
                GROUP BY content_category ORDER BY 3 DESC LIMIT 5""",
                (business_id, start, end),
            ).fetchall()
            return {r[0]: {"posts": r[1], "avg_eng": round(float(r[2]), 1)} for r in cats}

        def _lead_stats(start, end):
            row = conn.execute(
                """SELECT COUNT(*),
                    COUNT(*) FILTER (WHERE source_content_item_id IS NOT NULL)
                FROM leads
                WHERE business_id = %s AND created_at BETWEEN %s AND %s""",
                (business_id, start, end),
            ).fetchone()
            return {"total": row[0], "from_content": row[1]}

        current = _period_stats(current_start, now)
        previous = _period_stats(previous_start, current_start)

        if not current and not previous:
            return None

        def _delta(curr_val, prev_val):
            if prev_val == 0:
                return None
            return round((curr_val - prev_val) / prev_val * 100, 1)

        result = {
            "period_days": period_days,
            "current": {
                "start": current_start.date().isoformat(),
                "end": now.date().isoformat(),
                **(current or {"posts": 0, "reels": 0, "statics": 0, "reel_pct": 0,
                               "avg_engagement": 0, "total_likes": 0, "total_comments": 0,
                               "platforms_used": 0}),
            },
            "previous": {
                "start": previous_start.date().isoformat(),
                "end": current_start.date().isoformat(),
                **(previous or {"posts": 0, "reels": 0, "statics": 0, "reel_pct": 0,
                                "avg_engagement": 0, "total_likes": 0, "total_comments": 0,
                                "platforms_used": 0}),
            },
        }

        if current and previous:
            result["changes"] = {
                "posts_delta_pct": _delta(current["posts"], previous["posts"]),
                "engagement_delta_pct": _delta(current["avg_engagement"], previous["avg_engagement"]),
                "likes_delta_pct": _delta(current["total_likes"], previous["total_likes"]),
                "comments_delta_pct": _delta(current["total_comments"], previous["total_comments"]),
                "reel_pct_shift": current["reel_pct"] - previous["reel_pct"],
            }
            cur_cats = _top_categories(current_start, now)
            prev_cats = _top_categories(previous_start, current_start)
            result["current"]["top_categories"] = cur_cats
            result["previous"]["top_categories"] = prev_cats
            new_cats = set(cur_cats.keys()) - set(prev_cats.keys())
            dropped_cats = set(prev_cats.keys()) - set(cur_cats.keys())
            if new_cats:
                result["changes"]["new_categories"] = list(new_cats)
            if dropped_cats:
                result["changes"]["dropped_categories"] = list(dropped_cats)

            cur_leads = _lead_stats(current_start, now)
            prev_leads = _lead_stats(previous_start, current_start)
            result["current"]["leads"] = cur_leads
            result["previous"]["leads"] = prev_leads
            result["changes"]["leads_delta_pct"] = _delta(cur_leads["total"], prev_leads["total"])

            insights = []
            eng_d = result["changes"]["engagement_delta_pct"]
            if eng_d is not None:
                if eng_d > 15:
                    insights.append(f"Engagement up {eng_d}% — content quality is improving.")
                elif eng_d < -15:
                    insights.append(f"Engagement down {eng_d}% — review recent content quality and topics.")
            reel_shift = result["changes"]["reel_pct_shift"]
            if reel_shift > 10:
                insights.append(f"Reel ratio increased by {reel_shift}pp — good shift toward video.")
            elif reel_shift < -10:
                insights.append(f"Reel ratio dropped by {abs(reel_shift)}pp — consider more video content.")
            posts_d = result["changes"]["posts_delta_pct"]
            if posts_d is not None:
                if posts_d > 30:
                    insights.append(f"Posting volume up {posts_d}% — maintain consistency.")
                elif posts_d < -30:
                    insights.append(f"Posting volume down {posts_d}% — consistency is key for algorithms.")
            result["insights"] = insights

        return result


def recommend_next_content(business_id: str, platform: str = None) -> dict:
    """What should the next post be? A senior marketer's recommendation.

    Returns a recommendation dict with:
    - suggested_category: what topic area to cover
    - suggested_strategy_layer: pillar/distribution/engagement
    - suggested_format: static_post/reel/carousel
    - suggested_funnel_stage: awareness/consideration/conversion/retention
    - reasoning: why this recommendation
    """
    with db.get_conn() as conn:
        perf = _load_performance_context(conn, business_id)

    # What strategy layer is most needed?
    strategy = perf.get("strategy_balance", {})
    total = sum(d.get("posts", 0) for d in strategy.values()) or 1
    current_pcts = {k: d.get("posts", 0) / total * 100 for k, d in strategy.items()}
    target = {"pillar": 20, "distribution": 50, "engagement": 30}
    biggest_gap_layer = max(target.keys(), key=lambda k: target[k] - current_pcts.get(k, 0))

    # What category performs best?
    cat_perf = perf.get("category_performance", {})
    best_category = max(cat_perf.items(), key=lambda x: x[1].get("avg_engagement", 0))[0] if cat_perf else "destination"

    # What's the posting urgency?
    posting = perf.get("posting_status", {})
    urgent_platform = None
    for p, status in posting.items():
        if status["urgency"] in ("critical", "high"):
            urgent_platform = p
            break

    target_platform = urgent_platform or platform or "facebook"

    # What format works best on this platform?
    pb = PLATFORM_PLAYBOOKS.get(target_platform, {})
    if target_platform == "instagram":
        suggested_format = "reel" if biggest_gap_layer == "engagement" else "static_post"
    else:
        suggested_format = "static_post"

    # What funnel stage is most needed?
    pipeline = perf.get("pipeline", [])
    pipeline_layers = [p.get("strategy_layer") for p in pipeline]
    if "pillar" not in pipeline_layers and current_pcts.get("pillar", 0) < 15:
        funnel_stage = "consideration"
    elif "engagement" not in pipeline_layers and current_pcts.get("engagement", 0) < 25:
        funnel_stage = "awareness"
    else:
        funnel_stage = "consideration"

    reasoning_parts = []
    if urgent_platform:
        reasoning_parts.append(f"{urgent_platform} needs attention — hasn't posted in {posting[urgent_platform]['days_since_last_post']} days")
    reasoning_parts.append(f"Strategy gap: {biggest_gap_layer} content is at {current_pcts.get(biggest_gap_layer, 0):.0f}% vs {target[biggest_gap_layer]}% target")
    reasoning_parts.append(f"Best performing category: {best_category}")

    return {
        "platform": target_platform,
        "suggested_category": best_category,
        "suggested_strategy_layer": biggest_gap_layer,
        "suggested_format": suggested_format,
        "suggested_funnel_stage": funnel_stage,
        "reasoning": "; ".join(reasoning_parts),
        "platform_playbook": pb.get("content_rules", {}),
    }
