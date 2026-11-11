"""Comment sentiment analysis.

Pulls comments from our own FB/IG posts via Graph API, classifies each
through the AI model (sentiment + topics + buying intent), and stores
aggregated results that the learning engine and strategist can consume.

The key insight: likes tell you THAT something worked, but comments tell
you WHY.  A post with 50 likes and 3 "how much?" comments is worth more
than one with 200 likes and 3 "nice photo" comments."""
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app import config, db, security
from app.ai import prompts, run as ai_run
from app.services import graph_api

log = logging.getLogger("sentiment")
TUNIS = ZoneInfo("Africa/Tunis")

BATCH_SIZE = 15


def _get_credentials(conn, business_id: str) -> dict:
    rows = conn.execute(
        "SELECT platform, access_token, meta FROM business_credentials "
        "WHERE business_id = %s",
        (business_id,),
    ).fetchall()
    creds = {}
    for platform, token, meta in rows:
        creds[platform] = {"token": security.decrypt_token(token), "meta": meta or {}}
    return creds


def _posts_needing_analysis(conn, business_id: str, max_age_days: int = 0) -> list[dict]:
    """Find published posts that have comments we haven't analyzed yet (or recently).
    max_age_days=0 means no cutoff — analyze all historical posts."""
    query = """SELECT c.id, c.platform, c.meta_post_id, c.type,
                  c.content_category, c.strategy_layer,
                  COALESCE((c.analytics_features->>'like_count')::int,
                           (c.analytics_features->>'reactions_count')::int, 0) as likes,
                  COALESCE((c.analytics_features->>'comments_count')::int, 0) as comment_count,
                  ca.analyzed_at
           FROM content_analytics c
           LEFT JOIN comment_analysis ca ON ca.content_item_id = c.id
           WHERE c.business_id = %s
             AND c.status = 'published'
             AND c.meta_post_id IS NOT NULL
             AND COALESCE((c.analytics_features->>'comments_count')::int, 0) > 0"""
    params: list = [business_id]
    if max_age_days > 0:
        query += " AND COALESCE(c.published_at, c.created_at) > %s"
        params.append(datetime.now(TUNIS) - timedelta(days=max_age_days))
    query += " ORDER BY COALESCE((c.analytics_features->>'comments_count')::int, 0) DESC"
    rows = conn.execute(query, params).fetchall()
    posts = []
    now = datetime.now(TUNIS)
    for r in rows:
        analyzed_at = r[8]
        if analyzed_at and (now - analyzed_at.astimezone(TUNIS)).days < 3:
            continue
        posts.append({
            "id": str(r[0]), "platform": r[1], "meta_post_id": r[2],
            "type": r[3], "category": r[4], "strategy_layer": r[5],
            "likes": r[6], "comment_count": r[7],
        })
    return posts


def _fetch_comments(post: dict, creds: dict) -> list[dict]:
    platform = post["platform"]
    cred = creds.get(platform)
    if not cred:
        return []
    token = cred["token"]
    meta_id = post["meta_post_id"]
    if platform == "instagram":
        raw = graph_api.fetch_ig_media_comments(meta_id, token, limit=50)
        return [{"id": c["id"], "text": c.get("text", ""), "author": c.get("username", "")}
                for c in raw if c.get("text")]
    else:
        raw = graph_api.fetch_post_comments(meta_id, token, limit=50)
        return [{"id": c["id"], "text": c.get("message", ""),
                 "author": (c.get("from") or {}).get("name", "")}
                for c in raw if c.get("message")]


def _classify_batch(comments: list[dict]) -> list[dict]:
    """Classify a batch of comments via AI. Returns list of {id, sentiment, topics, intent}."""
    if not comments:
        return []
    texts = [{"id": c["id"], "text": c["text"][:300]} for c in comments]

    from app.ai.schemas import CommentSentimentOut
    result = ai_run(
        "sentiment", prompts.SENTIMENT.system,
        json.dumps(texts, ensure_ascii=False),
        CommentSentimentOut, model=config.CONTENT_MODEL,
        version=prompts.SENTIMENT.version,
    )
    return [item.model_dump() for item in result.comments]


def analyze_post_comments(business_id: str, post: dict, creds: dict) -> dict | None:
    """Fetch and analyze comments for a single post."""
    comments = _fetch_comments(post, creds)
    if not comments:
        return None

    classified = []
    for i in range(0, len(comments), BATCH_SIZE):
        batch = comments[i:i + BATCH_SIZE]
        try:
            classified.extend(_classify_batch(batch))
        except Exception:
            log.exception("sentiment classification failed for batch on post %s", post["id"])

    if not classified:
        return None

    sentiment_counts = defaultdict(int)
    all_topics = defaultdict(int)
    intent_scores = []
    for c in classified:
        sentiment_counts[c.get("sentiment", "neutral")] += 1
        for topic in c.get("topics", []):
            all_topics[topic] += 1
        intent_scores.append(c.get("buying_intent", 0))

    total = len(classified)
    analysis = {
        "total_comments": total,
        "sentiment": {
            "positive": sentiment_counts.get("positive", 0),
            "negative": sentiment_counts.get("negative", 0),
            "neutral": sentiment_counts.get("neutral", 0),
            "question": sentiment_counts.get("question", 0),
        },
        "sentiment_score": round(
            (sentiment_counts.get("positive", 0) - sentiment_counts.get("negative", 0))
            / total * 100
        ) if total else 0,
        "top_topics": dict(sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:10]),
        "avg_buying_intent": round(sum(intent_scores) / len(intent_scores)) if intent_scores else 0,
        "high_intent_count": sum(1 for s in intent_scores if s >= 60),
    }
    return analysis


def run_sentiment(business_id: str) -> dict:
    """Full sentiment analysis run for a business."""
    with db.get_conn() as conn:
        creds = _get_credentials(conn, business_id)
        posts = _posts_needing_analysis(conn, business_id)

    if not posts:
        return {"status": "no_posts", "detail": "no posts with unanalyzed comments"}

    results = []
    for post in posts[:30]:
        analysis = analyze_post_comments(business_id, post, creds)
        if analysis:
            analysis["content_item_id"] = post["id"]
            analysis["platform"] = post["platform"]
            analysis["type"] = post["type"]
            analysis["category"] = post.get("category")
            results.append(analysis)

    if not results:
        return {"status": "no_comments", "detail": "posts found but no comments to analyze"}

    with db.get_conn() as conn:
        for r in results:
            conn.execute(
                """INSERT INTO comment_analysis
                   (content_item_id, total_comments, sentiment, sentiment_score,
                    top_topics, avg_buying_intent, high_intent_count, analyzed_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                   ON CONFLICT (content_item_id) DO UPDATE SET
                     total_comments = EXCLUDED.total_comments,
                     sentiment = EXCLUDED.sentiment,
                     sentiment_score = EXCLUDED.sentiment_score,
                     top_topics = EXCLUDED.top_topics,
                     avg_buying_intent = EXCLUDED.avg_buying_intent,
                     high_intent_count = EXCLUDED.high_intent_count,
                     analyzed_at = now()""",
                (r["content_item_id"], r["total_comments"],
                 json.dumps(r["sentiment"]), r["sentiment_score"],
                 json.dumps(r["top_topics"]), r["avg_buying_intent"],
                 r["high_intent_count"]),
            )
        conn.commit()

    agg = _aggregate_sentiment(results)
    return {"status": "ok", "posts_analyzed": len(results), "aggregate": agg}


def _aggregate_sentiment(results: list[dict]) -> dict:
    """Roll up per-post sentiment into business-level insights."""
    total_comments = sum(r["total_comments"] for r in results)
    total_positive = sum(r["sentiment"]["positive"] for r in results)
    total_negative = sum(r["sentiment"]["negative"] for r in results)
    total_questions = sum(r["sentiment"]["question"] for r in results)
    all_topics = defaultdict(int)
    for r in results:
        for topic, count in r["top_topics"].items():
            all_topics[topic] += count

    high_intent_total = sum(r["high_intent_count"] for r in results)

    return {
        "total_comments_analyzed": total_comments,
        "overall_sentiment_score": round(
            (total_positive - total_negative) / total_comments * 100
        ) if total_comments else 0,
        "positive_pct": round(total_positive / total_comments * 100) if total_comments else 0,
        "negative_pct": round(total_negative / total_comments * 100) if total_comments else 0,
        "question_pct": round(total_questions / total_comments * 100) if total_comments else 0,
        "top_topics": dict(sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:10]),
        "high_intent_comments": high_intent_total,
        "posts_with_sentiment": len(results),
    }


def load_sentiment_context(business_id: str) -> dict | None:
    """Load the latest sentiment data for the strategist brief."""
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT ca.sentiment_score, ca.top_topics, ca.avg_buying_intent,
                      ca.high_intent_count, ca.total_comments,
                      c.content_category, c.type, c.platform
               FROM comment_analysis ca
               JOIN content_items c ON c.id = ca.content_item_id
               WHERE c.business_id = %s
               ORDER BY ca.analyzed_at DESC""",
            (business_id,),
        ).fetchall()
    if not rows:
        return None

    by_category = defaultdict(lambda: {"score": 0, "count": 0, "intent": 0})
    by_platform = defaultdict(lambda: {"score": 0, "count": 0})
    by_format = defaultdict(lambda: {"score": 0, "count": 0})
    all_topics = defaultdict(int)
    total_intent = 0
    total_high_intent = 0

    for score, topics, intent, high_intent, total, cat, fmt, platform in rows:
        score = score or 0
        intent = intent or 0
        high_intent = high_intent or 0
        total = total or 1
        if cat:
            by_category[cat]["score"] += score * total
            by_category[cat]["count"] += total
            by_category[cat]["intent"] += intent * total
        by_platform[platform]["score"] += score * total
        by_platform[platform]["count"] += total
        by_format[fmt]["score"] += score * total
        by_format[fmt]["count"] += total
        for topic, cnt in (topics or {}).items():
            all_topics[topic] += cnt
        total_intent += intent * total
        total_high_intent += high_intent

    total_comments = sum(r[4] for r in rows)
    cat_summary = {}
    for cat, d in by_category.items():
        if d["count"]:
            cat_summary[cat] = {
                "avg_sentiment": round(d["score"] / d["count"]),
                "avg_intent": round(d["intent"] / d["count"]),
                "comments": d["count"],
            }

    return {
        "overall_sentiment": round(sum(r[0] * r[4] for r in rows) / total_comments) if total_comments else 0,
        "avg_buying_intent": round(total_intent / total_comments) if total_comments else 0,
        "high_intent_comments": total_high_intent,
        "by_category": dict(sorted(cat_summary.items(), key=lambda x: x[1]["avg_sentiment"], reverse=True)),
        "top_topics": dict(sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:8]),
        "total_comments": total_comments,
    }
