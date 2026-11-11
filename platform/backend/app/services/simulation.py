"""Clearly labeled, non-persistent campaign simulation for demos and learning validation."""
from collections import Counter
from datetime import datetime, timezone
import hashlib

from app import db


def _score(seed: str, low: int, high: int) -> int:
    value = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return low + value % (high - low + 1)


def campaign_sandbox(business_id: str) -> dict:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, type, platform, COALESCE(features->>'topic','Campaign post'), created_at "
            "FROM content_items WHERE business_id = %s ORDER BY created_at DESC LIMIT 30",
            (business_id,),
        ).fetchall()

    if not rows:
        rows = [
            ("seed-reel", "reel", "instagram", "Destination discovery", datetime.now(timezone.utc)),
            ("seed-static", "static_post", "facebook", "Family travel offer", datetime.now(timezone.utc)),
            ("seed-avatar", "avatar_clip", "instagram", "Travel advice", datetime.now(timezone.utc)),
        ]

    items = []
    weekly = [{"label": f"Week {week}", "facebook": 0, "instagram": 0} for week in range(1, 5)]
    totals = {"reach": 0, "impressions": 0, "engagement": 0, "leads": 0, "sales": 0, "revenue": 0}
    expanded = (rows * ((18 + len(rows) - 1) // len(rows)))[:18]
    for index, row in enumerate(expanded):
        item_id, content_type, platform, topic, _ = row
        bonus = 1.28 if content_type == "reel" else (1.12 if content_type == "avatar_clip" else 1.0)
        base = _score(f"{item_id}-{index}", 900, 2100)
        reach = int(base * bonus)
        impressions = int(reach * (1.22 + (index % 4) * .05))
        engagement = int(reach * (.045 + (index % 5) * .009) * bonus)
        leads = max(1, int(engagement * (.035 + (index % 3) * .008)))
        sales = 1 if leads >= 4 and index % 3 == 0 else 0
        revenue = sales * _score(f"sale-{item_id}-{index}", 520, 980)
        week = index % 4
        weekly[week][platform] += engagement
        for key, value in (("reach", reach), ("impressions", impressions), ("engagement", engagement),
                           ("leads", leads), ("sales", sales), ("revenue", revenue)):
            totals[key] += value
        items.append({"format": content_type, "platform": platform, "topic": topic, "leads": leads,
                      "conversions": sales, "revenue": revenue})

    mix = Counter(item["format"] for item in items)
    platform_mix = Counter(item["platform"] for item in items)
    best_format = max(mix, key=lambda value: sum(i["leads"] for i in items if i["format"] == value) / mix[value])
    totals["engagement_rate"] = round(totals["engagement"] / totals["reach"] * 100, 1)
    return {
        "mode": "simulation",
        "label": "Learning sandbox — no simulated result is stored as real data",
        "numbers": {**totals, "planned": len(items), "approved": 13, "scheduled": 9, "published": 7},
        "content_mix": dict(mix),
        "platform_mix": dict(platform_mix),
        "weekly": weekly,
        "learning": {
            "confidence": "simulated",
            "summary": f"The sandbox predicts {best_format.replace('_', ' ')} as the strongest lead format for this mix.",
            "recommendations": [
                f"Test two more {best_format.replace('_', ' ')} creatives next week.",
                "Compare destination-led hooks with offer-led hooks using the same audience.",
                "Keep human approval and record real leads and sales before production learning is enabled.",
            ],
        },
    }
