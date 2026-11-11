"""Competitor intelligence — tracks public metrics of rival pages.

Scrapes public Facebook/Instagram page data (follower counts, activity)
via TinyFish search + fetch, stores periodic snapshots, and computes
competitive positioning for Ozone."""
import json
import logging
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app import db

log = logging.getLogger("competitors")
TUNIS = ZoneInfo("Africa/Tunis")


def list_competitors(business_id: str) -> list[dict]:
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT c.id, c.name, c.platform, c.page_url, c.page_id,
                      s.followers, s.talking_about, s.posts_last_30d,
                      s.avg_engagement, s.scraped_at
               FROM competitors c
               LEFT JOIN LATERAL (
                   SELECT * FROM competitor_snapshots cs
                   WHERE cs.competitor_id = c.id
                   ORDER BY cs.scraped_at DESC LIMIT 1
               ) s ON true
               WHERE c.business_id = %s
               ORDER BY s.followers DESC NULLS LAST""",
            (business_id,),
        ).fetchall()
    return [
        {
            "id": str(r[0]), "name": r[1], "platform": r[2],
            "page_url": r[3], "page_id": r[4],
            "latest": {
                "followers": r[5], "talking_about": r[6],
                "posts_last_30d": r[7],
                "avg_engagement": float(r[8]) if r[8] else None,
                "scraped_at": r[9].isoformat() if r[9] else None,
            } if r[5] is not None else None,
        }
        for r in rows
    ]


def add_competitor(business_id: str, name: str, page_url: str,
                   platform: str = "facebook", page_id: str | None = None) -> dict:
    with db.get_conn() as conn:
        row = conn.execute(
            """INSERT INTO competitors (business_id, name, platform, page_url, page_id)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (business_id, page_url) DO UPDATE SET name = EXCLUDED.name
               RETURNING id""",
            (business_id, name, platform, page_url, page_id),
        ).fetchone()
        conn.commit()
    return {"id": str(row[0]), "name": name, "platform": platform, "page_url": page_url}


def remove_competitor(business_id: str, competitor_id: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "DELETE FROM competitors WHERE id = %s AND business_id = %s RETURNING id",
            (competitor_id, business_id),
        ).fetchone()
        conn.commit()
    return row is not None


def record_snapshot(competitor_id: str, followers: int | None = None,
                    talking_about: int | None = None, posts_last_30d: int | None = None,
                    avg_engagement: float | None = None, extra: dict | None = None) -> str:
    with db.get_conn() as conn:
        row = conn.execute(
            """INSERT INTO competitor_snapshots
               (competitor_id, followers, talking_about, posts_last_30d, avg_engagement, extra)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (competitor_id, followers, talking_about, posts_last_30d,
             avg_engagement, json.dumps(extra or {})),
        ).fetchone()
        conn.commit()
    return str(row[0])


def _parse_fb_followers(snippet: str) -> tuple[int | None, int | None]:
    """Extract followers and talking_about from a Facebook search snippet."""
    followers = None
    talking = None
    m = re.search(r'([\d,]+)\s*followers', snippet.replace('\xa0', ' '))
    if m:
        followers = int(m.group(1).replace(',', ''))
    m = re.search(r'([\d,]+)\s*talking about', snippet.replace('\xa0', ' '))
    if m:
        talking = int(m.group(1).replace(',', ''))
    return followers, talking


def parse_fb_description(description: str) -> dict:
    """Parse a Facebook OG description to extract followers and talking_about.
    Facebook pages always include 'X followers · Y talking about this' in their
    meta description / OG description."""
    result = {}
    desc = description.replace('\xa0', ' ').replace(',', '')
    m = re.search(r'([\d]+)\s*followers', desc)
    if m:
        result["followers"] = int(m.group(1))
    m = re.search(r'([\d]+)\s*talking about', desc)
    if m:
        result["talking_about"] = int(m.group(1))
    m = re.search(r'([\d]+)\s*were here', desc)
    if m:
        result["were_here"] = int(m.group(1))
    return result


def refresh_competitor_from_snippet(competitor_id: str, snippet: str, description: str = "") -> dict | None:
    """Parse follower data from a search snippet or OG description and store a snapshot."""
    data = parse_fb_description(snippet)
    if not data.get("followers"):
        data = parse_fb_description(description)
    if not data.get("followers"):
        return None
    record_snapshot(
        competitor_id,
        followers=data.get("followers"),
        talking_about=data.get("talking_about"),
        extra={"were_here": data.get("were_here")},
    )
    return data


def refresh_all_from_search_results(business_id: str, search_results: list[dict]) -> dict:
    """Given TinyFish search results (list of {competitor_id, snippet, description}),
    update all competitor snapshots. Returns summary of updates."""
    updated = []
    failed = []
    for item in search_results:
        cid = item.get("competitor_id")
        snippet = item.get("snippet", "")
        desc = item.get("description", "")
        data = refresh_competitor_from_snippet(cid, snippet, desc)
        if data:
            updated.append({"competitor_id": cid, **data})
        else:
            failed.append(cid)
    return {"updated": len(updated), "failed": len(failed), "details": updated}


def competitive_position(business_id: str) -> dict | None:
    """Where does Ozone stand vs competitors?"""
    competitors = list_competitors(business_id)
    if not competitors:
        return None

    with db.get_conn() as conn:
        ozone = conn.execute(
            """SELECT
                COALESCE((SELECT metric_value FROM page_metrics
                          WHERE business_id = %s AND metric_name IN ('followers_count', 'fan_count')
                          ORDER BY fetched_at DESC LIMIT 1), 0),
                COALESCE((SELECT metric_value FROM page_metrics
                          WHERE business_id = %s AND metric_name = 'talking_about_count'
                          ORDER BY fetched_at DESC LIMIT 1), 0),
                (SELECT COUNT(*) FROM content_items
                 WHERE business_id = %s AND status = 'published'
                   AND COALESCE(published_at, created_at) > now() - interval '30 days'),
                COALESCE((SELECT AVG(
                    COALESCE((features->>'like_count')::int,
                             (features->>'reactions_count')::int, 0)
                    + COALESCE((features->>'comments_count')::int, 0))
                 FROM content_items
                 WHERE business_id = %s AND status = 'published'
                   AND COALESCE(published_at, created_at) > now() - interval '30 days'), 0)
            """,
            (business_id, business_id, business_id, business_id),
        ).fetchone()

    our_followers = int(ozone[0])
    our_talking = int(ozone[1])
    our_posts_30d = ozone[2]
    our_avg_eng = round(float(ozone[3]), 1)

    all_entries = []
    for c in competitors:
        if c["latest"]:
            all_entries.append({
                "name": c["name"],
                "followers": c["latest"]["followers"] or 0,
                "talking_about": c["latest"]["talking_about"] or 0,
                "posts_last_30d": c["latest"].get("posts_last_30d"),
                "is_us": False,
            })

    all_entries.append({
        "name": "Ozone Voyages",
        "followers": our_followers,
        "talking_about": our_talking,
        "posts_last_30d": our_posts_30d,
        "is_us": True,
    })

    all_entries.sort(key=lambda x: x["followers"], reverse=True)
    our_rank = next((i + 1 for i, e in enumerate(all_entries) if e["is_us"]), len(all_entries))

    return {
        "our_followers": our_followers,
        "our_talking_about": our_talking,
        "our_posts_30d": our_posts_30d,
        "our_avg_engagement": our_avg_eng,
        "rank": our_rank,
        "total_tracked": len(all_entries),
        "leaderboard": all_entries,
        "competitors": competitors,
    }
