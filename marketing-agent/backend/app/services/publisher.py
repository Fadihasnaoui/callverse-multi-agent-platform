"""Phase 5 - publishes approved content_items whose scheduled_for has come
due. Hard rule from the original brief: only status='approved' is ever
published - nothing the agent generates or schedules on its own skips the
human approval gate. An item is claimed atomically (approved -> publishing)
so overlapping scheduler ticks or a manual publish-now can never double-post."""
import logging
import json

from app import config, db, security
from app.services import graph_api, content_state

log = logging.getLogger("publisher")


def _message(caption: str, hashtags: list[str] | None) -> str:
    tags = " ".join("#" + h.lstrip("#") for h in (hashtags or []))
    return f"{caption}\n\n{tags}".strip()


def _credential(conn, business_id: str, platform: str) -> tuple[str, dict]:
    row = conn.execute(
        "SELECT access_token, meta FROM business_credentials WHERE business_id = %s AND platform = %s",
        (business_id, platform),
    ).fetchone()
    if not row:
        raise RuntimeError(f"no {platform} credential connected for this business")
    return security.decrypt_token(row[0]), (row[1] or {})


def _public_url(local_path: str) -> str:
    if not config.PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not set - Instagram needs publicly reachable media URLs")
    return f"{config.PUBLIC_BASE_URL.rstrip('/')}/renders/{config.render_filename(local_path)}"


def _publish(credential: tuple[str, dict], item: dict) -> str:
    token, meta = credential
    message = _message(item["caption"] or "", item["hashtags"])
    photos = list(item["asset_refs"] or [])
    video = item["rendered_video_path"]
    needs_video = item["type"] in {"reel", "avatar_clip"}

    if needs_video and not video:
        raise RuntimeError("video item has no rendered video yet - render it first")

    if item["platform"] == "facebook":
        page_id = meta.get("page_id")
        if not page_id:
            raise RuntimeError("facebook credential is missing meta.page_id")
        if needs_video:
            return graph_api.publish_facebook_video(page_id, token, video, message)
        if photos:
            return graph_api.publish_facebook_photo(page_id, token, photos[0], message)
        return graph_api.publish_facebook_text(page_id, token, message)

    if item["platform"] == "instagram":
        ig_user_id = meta.get("ig_user_id")
        if not ig_user_id:
            raise RuntimeError("instagram credential is missing meta.ig_user_id")
        if needs_video:
            return graph_api.publish_instagram(ig_user_id, token, message, video_url=_public_url(video),on_container=lambda container: _record_container(item.get('attempt_id'),container))
        if not photos:
            raise RuntimeError("Instagram static posts need at least one photo")
        return graph_api.publish_instagram(ig_user_id, token, message, image_url=photos[0],on_container=lambda container: _record_container(item.get('attempt_id'),container))

    raise RuntimeError(f"unsupported platform: {item['platform']}")


def _recover_and_expire():
    """Housekeeping before each tick: (1) an item stuck in 'publishing' was
    interrupted by a crash/restart - we cannot know if Meta received it, so it
    goes to 'failed' for the owner to check; (2) an approved item whose slot is
    long past (app was down) is not published late behind the owner's back."""
    with db.get_conn() as conn:
        conn.execute(
            "UPDATE content_items SET status = 'publish_unknown', publish_error = %s "
            "WHERE status = 'publishing' AND updated_at < now() - make_interval(mins => %s)",
            ("publishing was interrupted - check the Page before retrying, it may have gone out", config.PUBLISHING_TIMEOUT_MINUTES),
        )
        conn.execute(
            "UPDATE content_items SET status = 'failed', publish_error = %s "
            "WHERE status = 'approved' AND scheduled_for IS NOT NULL "
            "AND scheduled_for < now() - make_interval(hours => %s)",
            ("its time slot passed while the app could not publish - reschedule it", config.STALE_PUBLISH_HOURS),
        )
        conn.commit()


def publish_item(content_id: str) -> dict:
    """Publishes one approved item right now. Returns the new status. The claim,
    the network upload and the final write use separate short DB connections, so
    a multi-minute video upload never pins a pooled connection."""
    with db.get_conn() as conn:
        eligible = conn.execute("SELECT id FROM content_items WHERE id=%s AND status='approved' AND approved_version=content_version FOR UPDATE", (content_id,)).fetchone()
        if not eligible:
            raise ValueError("content item is not approved for its current version - nothing published")
        try:
            content_state.validate_artifact(conn, content_id)
        except ValueError as error:
            conn.execute("UPDATE content_items SET status='failed',publish_error=%s WHERE id=%s", (str(error),content_id))
            conn.commit()
            return {'id':content_id,'status':'failed','error':str(error)}
        row = conn.execute(
            "UPDATE content_items SET status = 'publishing', publish_error = NULL, updated_at = now() "
            "WHERE id = %s AND status = 'approved' AND approved_version=content_version "
            "RETURNING business_id, type, platform, caption, hashtags, asset_refs, rendered_video_path, content_version",
            (content_id,),
        ).fetchone()
        item = dict(zip(("business_id", "type", "platform", "caption", "hashtags", "asset_refs", "rendered_video_path", "content_version"), row))
        item['business_id'] = str(item['business_id'])
        attempt_id = str(conn.execute("INSERT INTO publish_attempts(content_id,content_version,snapshot) VALUES (%s,%s,%s) RETURNING id", (content_id, item['content_version'], json.dumps(item))).fetchone()[0])
        conn.commit()
    item['attempt_id']=attempt_id
    if not row:
        raise ValueError("content item is not in 'approved' status - nothing published")
    remote_started = False
    try:
        with db.get_conn() as conn:
            token_row = _credential(conn, item["business_id"], item["platform"])
        remote_started = True
        post_id = _publish(token_row, item)
    except Exception as error:  # noqa: BLE001 - any failure must release the claim
        log.exception("publish failed for %s", content_id)
        definite = not remote_started or isinstance(error,ValueError) or (isinstance(error,graph_api.GraphAPIError) and not error.outcome_unknown)
        result_status = 'failed' if definite else 'publish_unknown'
        _finish(content_id, result_status, publish_error=str(error)[:1000], attempt_id=attempt_id)
        try:
            from app.services.notifications import notify
            notify(item["business_id"], "content_failed",
                   f"Publish failed ({item['platform']})",
                   str(error)[:300],
                   {"content_id": content_id, "platform": item["platform"]})
        except Exception:
            pass
        return {"id": content_id, "status": result_status, "error": str(error)}

    try:
        _finish(content_id, "published", post_id=post_id, attempt_id=attempt_id)
    except Exception:  # the post IS live; make sure the id is never lost
        log.critical("POST IS LIVE but saving its id failed: content %s -> meta post %s", content_id, post_id)
        raise
    try:
        from app.services.notifications import notify
        notify(item["business_id"], "content_published",
               f"Post published on {item['platform']}",
               f"{item['type']} — {(item['caption'] or '')[:150]}",
               {"content_id": content_id, "platform": item["platform"], "meta_post_id": post_id})
    except Exception:
        pass
    return {"id": content_id, "status": "published", "meta_post_id": post_id}


def _finish(content_id: str, status: str, post_id: str | None = None, publish_error: str | None = None, attempt_id: str | None = None):
    with db.get_conn() as conn:
        if attempt_id:
            conn.execute("UPDATE publish_attempts SET status=%s,remote_post_id=COALESCE(%s,remote_post_id),error=%s,finished_at=now() WHERE id=%s", (status,post_id,publish_error,attempt_id))
        if status == "published":
            conn.execute(
                "UPDATE content_items SET status = 'published', published_at = now(), meta_post_id = %s WHERE id = %s",
                (post_id, content_id),
            )
        else:
            conn.execute(
                "UPDATE content_items SET status = %s, publish_error = %s WHERE id = %s",
                (status, publish_error, content_id),
            )
        conn.commit()


def _record_container(attempt_id, container_id):
    if attempt_id:
        with db.get_conn() as conn:
            conn.execute('UPDATE publish_attempts SET remote_container_id=%s WHERE id=%s',(container_id,attempt_id));conn.commit()


def reconcile(content_id: str, remote_post_id: str | None, confirmed_absent: bool) -> dict:
    """Explicit owner reconciliation. Unknown outcomes never go through normal retry."""
    if bool(remote_post_id) == bool(confirmed_absent):
        raise ValueError("provide a verified remote post ID or explicitly confirm the post is absent")
    with db.get_conn() as conn:
        row = conn.execute("SELECT status FROM content_items WHERE id=%s FOR UPDATE", (content_id,)).fetchone()
        if not row or row[0] != 'publish_unknown':
            raise ValueError("only an unknown publishing result can be reconciled")
        status = 'published' if remote_post_id else 'pending'
        conn.execute("UPDATE content_items SET status=%s,meta_post_id=%s,published_at=CASE WHEN %s::text IS NOT NULL THEN now() ELSE NULL END,approved_version=NULL,approved_at=NULL,publish_error=NULL WHERE id=%s", (status,remote_post_id,remote_post_id,content_id))
        conn.execute("UPDATE publish_attempts SET status=%s,remote_post_id=COALESCE(%s,remote_post_id),finished_at=now(),error='Owner reconciled uncertain outcome' WHERE content_id=%s AND status IN ('started','publish_unknown')", (status,remote_post_id,content_id))
        conn.commit()
    return {'id':content_id,'status':status}


def publish_due() -> int:
    """Scheduler tick: publish every approved item whose time has come."""
    _recover_and_expire()
    with db.get_conn() as conn:
        due = conn.execute(
            "SELECT id FROM content_items WHERE status = 'approved' "
            "AND scheduled_for IS NOT NULL AND scheduled_for <= now() ORDER BY scheduled_for"
        ).fetchall()
    for (content_id,) in due:
        try:
            publish_item(str(content_id))
        except Exception:  # noqa: BLE001 - one bad item must not stop the rest of the tick
            log.exception("publish tick failed for %s", content_id)
    return len(due)
