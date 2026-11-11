"""Phase 3 - renders a content_item into an actual Reel via the Remotion
composition in remotion/src/ReelComposition.tsx. Shells out to the Remotion
CLI as a subprocess (this is the one part of the project that isn't pure
Python - Remotion runs on Node) rather than reimplementing video rendering
in Python, same cost reasoning as the plan: self-hosted rendering, no
per-video vendor API fee."""
import json
import os
import re
import subprocess
import uuid

from app import config, db
from app.services import content_state


def _hook_text(caption: str) -> str:
    """First real sentence of the caption, short enough to read in ~1s on
    screen - not the whole caption, which is meant for the post body, not
    an on-screen overlay."""
    first_line = caption.split("\n")[0].strip()
    opening_clause = re.split(r"[,;.!?]", first_line, maxsplit=1)[0].strip()
    if 8 <= len(opening_clause) <= 52:
        return opening_clause
    shortened = first_line[:52].rsplit(" ", 1)[0].strip()
    if len(first_line) > len(shortened):
        return f"{shortened}…"
    return shortened
    return first_line[:70] + ("…" if len(first_line) > 70 else "")


def render_reel(content_id: str) -> dict:
    version = content_state.media_version(content_id)
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT business_id, caption, hashtags, asset_refs, features FROM content_items WHERE id = %s",
            (content_id,),
        ).fetchone()
        if not row:
            raise ValueError("content item not found")
        business_id, caption, hashtags, asset_refs, features = row

        business = conn.execute(
            "SELECT name FROM businesses WHERE id = %s", (business_id,)
        ).fetchone()
        business_name = business[0] if business else ""

    media_urls = list(asset_refs or [])
    if not media_urls:
        raise ValueError(
            "content item has no media attached - attach images or video clips first"
        )

    # Short on-screen highlight words, one per photo, cycling through
    # hashtags (stripped of '#') since those are already short, punchy, and
    # generated alongside the caption - not the full video_script prose,
    # which is a shooting-guide for a human editor, not on-screen text.
    caption_lines = (features or {}).get("overlay_lines") or [h.lstrip("#") for h in (hashtags or [])]
    music_path = (features or {}).get("music_path")
    music_url = ""
    if music_path and os.path.exists(music_path):
        music_url = f"http://127.0.0.1:8010/renders/{config.render_filename(music_path)}"
    logo_path = f"{config.RENDERS_DIR}/brand/ozone-logo.png"
    logo_url = ""
    if os.path.exists(logo_path):
        logo_url = "http://127.0.0.1:8010/renders/brand/ozone-logo.png"

    props = {
        "mediaUrls": media_urls,
        "hookText": _hook_text(caption or ""),
        "captionLines": caption_lines,
        "businessName": business_name,
        "musicUrl": music_url,
        "logoUrl": logo_url,
    }

    os.makedirs(config.RENDERS_DIR, exist_ok=True)
    output_filename = f"{content_id}-{uuid.uuid4().hex[:8]}.mp4"
    output_path = f"{config.RENDERS_DIR}/{output_filename}"
    props_path = f"{config.RENDERS_DIR}/{content_id}-props.json"

    with open(props_path, "w", encoding="utf-8") as f:
        json.dump(props, f)

    try:
        result = subprocess.run(
            [
                "npx",
                "remotion",
                "render",
                "src/index.ts",
                "ReelComposition",
                output_path,
                f"--props={props_path}",
                "--browser-executable=/usr/bin/chromium",
                "--concurrency=4",
            ],
            cwd=config.REMOTION_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Remotion render timed out after 10 minutes") from error
    if result.returncode != 0:
        raise RuntimeError(f"Remotion render failed: {result.stderr[-2000:]}")

    content_state.save_media(content_id, version, {"rendered_video_path": output_path})

    return {"content_id": content_id, "rendered_video_path": output_path}
