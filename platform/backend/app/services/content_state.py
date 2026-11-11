"""Shared, versioned editing and final-artifact validation."""
import json

from app import db
from app.ai.rules import StyleRules


def media_version(content_id: str) -> int:
    """Invalidate review before doing slow media work; completion uses this version."""
    with db.get_conn() as conn:
        row = conn.execute(
            "UPDATE content_items SET status='pending', approved_version=NULL, approved_at=NULL "
            "WHERE id=%s AND status IN ('pending','approved','rejected','failed') RETURNING content_version",
            (content_id,),
        ).fetchone()
        conn.commit()
    if not row:
        raise ValueError("content is locked or not found; reload before generating media")
    return row[0]


def save_media(content_id: str, version: int, fields: dict):
    allowed = {'asset_refs','rendered_video_path','features','avatar_job_id','avatar_spoken_text','avatar_language'}
    if not fields or not set(fields) <= allowed:
        raise ValueError("unsupported media fields")
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{key}=%s")
        values.append(json.dumps(value) if key in ('asset_refs','features') else value)
    with db.get_conn() as conn:
        row = conn.execute(
            f"UPDATE content_items SET {','.join(sets)} WHERE id=%s AND content_version=%s "
            "AND status IN ('pending','rejected') RETURNING id", (*values, content_id, version),
        ).fetchone()
        conn.commit()
    if not row:
        raise ValueError("content changed during media generation; result was not attached; review and retry explicitly")


def validate_artifact(conn, content_id: str, caption: str | None = None):
    row = conn.execute(
        "SELECT c.caption,c.hashtags,c.platform,c.type,c.asset_refs,c.rendered_video_path,b.style_rules "
        "FROM content_items c JOIN businesses b ON b.id=c.business_id WHERE c.id=%s", (content_id,),
    ).fetchone()
    if not row:
        raise ValueError("content item not found")
    text = row[0] if caption is None else caption
    if not (text or '').strip():
        raise ValueError("cannot approve an empty caption")
    rules = StyleRules.model_validate(row[6] or {})
    # CTA belongs to the final message but is not the owner's closing sentence.
    editorial = (text or '').split('\n\nEnvoyez-nous un message : m.me/', 1)[0]
    problems = rules.violations(editorial, row[1] or [])
    if rules.max_caption_chars and len(text) > rules.max_caption_chars:
        problems.append("assembled caption exceeds the owner's character limit")
    full = text + '\n\n' + ' '.join('#' + tag.lstrip('#') for tag in row[1] or [])
    if row[2] == 'instagram' and len(full) > 2200:
        problems.append('Instagram caption and hashtags exceed 2200 characters')
    if row[3] in ('reel','avatar_clip') and not row[5]:
        problems.append('this item has no video yet - generate it first')
    if row[2] == 'instagram' and row[3] == 'static_post' and not row[4]:
        problems.append('Instagram static posts require a photo')
    if problems:
        raise ValueError('; '.join(problems))
