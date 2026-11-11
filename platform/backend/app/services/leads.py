"""Phase 6 - lead mining. Turns Meta webhook events (Messenger DMs, Page and
Instagram comments) into `leads` rows. The attribution link is the ref code:
a DM that arrives via a post's m.me/<page>?ref=<content_item_id> link carries
that id in its referral event, so the lead is tied to the exact content item
with no guessing. Later messages from the same person inherit that source,
because Meta only sends the ref on the first event of a conversation."""
import logging
import uuid

from app import config, db
from app.ai import guards, prompts, run as ai_run
from app.ai.schemas import IntentOut
from app.services import memory

log = logging.getLogger("leads")

def _classify(business_id: str, text: str) -> int | None:
    try:
        # Owner corrections ("this was / was not a real lead") steer the score.
        result = ai_run(
            "intent", prompts.INTENT.system, text[:1000] + memory.recall_block(business_id, "leads", text),
            IntentOut, timeout=30, version=prompts.INTENT.version,
        )
        return result.buying_intent
    except Exception:  # noqa: BLE001 - a classifier outage must never drop a lead
        log.exception("intent classification failed")
        return None


def _business_for(conn, platform: str, external_page_id: str) -> str | None:
    key = "page_id" if platform == "facebook" else "ig_user_id"
    row = conn.execute(
        "SELECT business_id FROM business_credentials WHERE platform = %s AND meta->>%s = %s",
        (platform, key, external_page_id),
    ).fetchone()
    return str(row[0]) if row else None


def _valid_content_item(conn, business_id: str, ref: str | None) -> str | None:
    if not ref:
        return None
    try:
        ref = str(uuid.UUID(ref))
    except ValueError:
        return None
    row = conn.execute(
        "SELECT 1 FROM content_items WHERE id = %s AND business_id = %s", (ref, business_id)
    ).fetchone()
    return ref if row else None


def _inherited_source(conn, business_id: str, platform: str, author_id: str) -> str | None:
    row = conn.execute(
        "SELECT source_content_item_id FROM leads WHERE business_id = %s AND platform = %s "
        "AND author_external_id = %s AND source_content_item_id IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 1",
        (business_id, platform, author_id),
    ).fetchone()
    return str(row[0]) if row else None


def _store(conn, business_id, source, platform, external_id, content_id, author_name, author_id, text, intent, force_new):
    status = "new" if (force_new or intent is None or intent >= config.LEAD_INTENT_THRESHOLD) else "ignored"
    inserted = conn.execute(
        "INSERT INTO leads (business_id, source, platform, external_id, source_content_item_id, "
        "author_name, author_external_id, message_text, intent_confidence, status) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (platform, external_id) DO NOTHING RETURNING id",
        (business_id, source, platform, external_id, content_id, author_name, author_id, text, intent, status),
    ).fetchone()
    conn.commit()
    return str(inserted[0]) if inserted else None


def _referral_ref(event: dict) -> str | None:
    for holder in (event, event.get("postback") or {}, event.get("message") or {}):
        ref = (holder.get("referral") or {}).get("ref")
        if ref:
            return ref
    return None


def ingest_event(payload: dict) -> list[str]:
    """Process one Meta webhook payload; returns ids of newly created leads."""
    created: list[str] = []
    platform = "instagram" if payload.get("object") == "instagram" else "facebook"

    for entry in payload.get("entry", []):
        page_id = str(entry.get("id"))
        with db.get_conn() as conn:
            business_id = _business_for(conn, platform, page_id)
            if not business_id:
                log.warning("webhook for unknown %s account %s", platform, page_id)
                continue

            for event in entry.get("messaging", []):
                sender = str((event.get("sender") or {}).get("id", ""))
                if not sender or sender == page_id or (event.get("message") or {}).get("is_echo"):
                    continue
                ref = _valid_content_item(conn, business_id, _referral_ref(event))
                message = event.get("message") or {}
                text = message.get("text")
                external_id = f"ref:{sender}:{ref}" if ref and not text else message.get('mid') or f"{sender}:{event.get('timestamp')}"
                if conn.execute('SELECT 1 FROM leads WHERE platform=%s AND external_id=%s',(platform,external_id)).fetchone():
                    continue
                intent = None

                if ref and not text:
                    # Pure "opened the chat from a post link" event.
                    lead = _store(conn, business_id, "dm", platform, f"ref:{sender}:{ref}", ref,
                                  None, sender, "(opened chat from post link)", None, True)
                elif text:
                    source_item = ref or _inherited_source(conn, business_id, platform, sender)
                    intent = _classify(business_id, text)
                    lead = _store(conn, business_id, "dm", platform, message.get("mid") or f"{sender}:{event.get('timestamp')}",
                                  source_item, None, sender, text, intent, bool(source_item))
                else:
                    continue
                if lead:
                    created.append(lead)
                    try:
                        from app.services.notifications import notify
                        notify(business_id, "lead_new",
                               f"New lead from DM ({platform})",
                               f"{sender}: {(text or '(opened chat)')[:200]}",
                               {"lead_id": lead, "platform": platform, "intent": intent})
                    except Exception:
                        pass

            for change in entry.get("changes", []):
                value = change.get("value") or {}
                is_fb_comment = change.get("field") == "feed" and value.get("item") == "comment" and value.get("verb") == "add"
                is_ig_comment = change.get("field") == "comments"
                if not (is_fb_comment or is_ig_comment):
                    continue
                author = value.get("from") or {}
                if str(author.get("id", "")) == page_id:
                    continue  # the page's own replies
                text = value.get("message") or value.get("text")
                comment_id = value.get("comment_id") or value.get("id")
                if not text or not comment_id:
                    continue
                if conn.execute('SELECT 1 FROM leads WHERE platform=%s AND external_id=%s',(platform,str(comment_id))).fetchone():
                    continue
                # Public-comment -> post attribution needs the post id; match it
                # against what we published (meta_post_id).
                post_id = value.get("post_id") or (value.get("media") or {}).get("id")
                # Comment webhooks carry "<pageid>_<postid>"; video uploads are stored as the bare
                # video id. Match either form.
                candidates = [post_id, post_id.split("_", 1)[-1]] if post_id else []
                row = conn.execute(
                    "SELECT id FROM content_items WHERE business_id = %s AND (meta_post_id = ANY(%s) "
                    "OR split_part(meta_post_id, '_', 2) = ANY(%s))",
                    (business_id, candidates, candidates),
                ).fetchone() if post_id else None
                author_name = author.get("name") or author.get("username")
                lead = _store(conn, business_id, "comment", platform, str(comment_id),
                              str(row[0]) if row else None, author_name,
                              str(author.get("id") or ""), text, _classify(business_id, text), False)
                if lead:
                    created.append(lead)
                    try:
                        from app.services.notifications import notify
                        notify(business_id, "lead_new",
                               f"New lead from comment ({platform})",
                               f"{author_name or 'Unknown'}: {text[:200]}",
                               {"lead_id": lead, "platform": platform})
                    except Exception:
                        pass
    return created
