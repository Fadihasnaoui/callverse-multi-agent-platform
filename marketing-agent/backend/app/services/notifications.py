"""Notification service — stores in-app notifications and optionally pushes to Telegram."""
import json
import logging
import threading

import requests

from app import config, db

log = logging.getLogger("notifications")


def _send_telegram(title: str, body: str) -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    text = f"<b>{title}</b>\n{body}" if body else f"<b>{title}</b>"
    try:
        requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text[:4000], "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        log.exception("telegram send failed")


def notify(business_id: str, event_type: str, title: str, body: str = "", meta: dict | None = None) -> str | None:
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                "INSERT INTO notifications (business_id, event_type, title, body, meta) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (business_id, event_type, title, body or "", json.dumps(meta or {})),
            ).fetchone()
            conn.commit()
        threading.Thread(target=_send_telegram, args=(title, body), daemon=True).start()
        return str(row[0]) if row else None
    except Exception:
        log.exception("notify failed")
        return None


def list_notifications(business_id: str, limit: int = 50, unread_only: bool = False) -> list[dict]:
    clause = " AND read = false" if unread_only else ""
    with db.get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, event_type, title, body, meta, read, created_at "
            f"FROM notifications WHERE business_id = %s{clause} "
            f"ORDER BY created_at DESC LIMIT %s",
            (business_id, limit),
        ).fetchall()
    return [
        {"id": str(r[0]), "event_type": r[1], "title": r[2], "body": r[3],
         "meta": r[4], "read": r[5], "created_at": r[6].isoformat()}
        for r in rows
    ]


def mark_read(business_id: str, notification_id: str) -> bool:
    with db.get_conn() as conn:
        row = conn.execute(
            "UPDATE notifications SET read = true WHERE id = %s AND business_id = %s RETURNING id",
            (notification_id, business_id),
        ).fetchone()
        conn.commit()
    return row is not None


def mark_all_read(business_id: str) -> int:
    with db.get_conn() as conn:
        result = conn.execute(
            "UPDATE notifications SET read = true WHERE business_id = %s AND read = false",
            (business_id,),
        )
        conn.commit()
    return result.rowcount


def unread_count(business_id: str) -> int:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT count(*) FROM notifications WHERE business_id = %s AND read = false",
            (business_id,),
        ).fetchone()
    return row[0] if row else 0
