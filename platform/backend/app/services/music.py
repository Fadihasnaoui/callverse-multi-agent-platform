"""Generate one reusable instrumental soundtrack for a Reel."""
import json
import os
import uuid

import requests

from app import config, db
from app.services import content_state, costs

ELEVENLABS_MUSIC_URL = "https://api.elevenlabs.io/v1/music"


@costs.budgeted('elevenlabs-music',1.0)
def generate_reel_music(content_id: str, duration_ms: int = 10000) -> dict:
    if not config.ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    version = content_state.media_version(content_id)

    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT features FROM content_items WHERE id = %s", (content_id,)
        ).fetchone()
        if not row:
            raise ValueError("content item not found")
        features = row[0]

    features = dict(features or {})
    existing = features.get("music_path")
    if existing and os.path.exists(existing):
        return {"content_id": content_id, "music_path": existing, "reused": True}

    prompt = (
        "Instrumental Mediterranean travel reel soundtrack, elegant modern organic house, "
        "warm hand percussion, soft oud-inspired plucks, uplifting but premium, immediate "
        "rhythmic opening, one gentle lift halfway through, clean ending, no vocals, no lyrics."
    )
    costs.dispatched()
    response = requests.post(
        ELEVENLABS_MUSIC_URL,
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": config.ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "prompt": prompt,
            "music_length_ms": max(3000, min(duration_ms, 600000)),
            "model_id": config.ELEVENLABS_MUSIC_MODEL,
            "force_instrumental": True,
        },
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs music failed ({response.status_code}): {response.text[:500]}")

    os.makedirs(config.RENDERS_DIR, exist_ok=True)
    filename = f"{content_id}-music-{uuid.uuid4().hex[:8]}.mp3"
    path = f"{config.RENDERS_DIR}/{filename}"
    with open(path, "wb") as output:
        output.write(response.content)

    features["music_path"] = path
    features["music_prompt"] = prompt
    features["music_song_id"] = response.headers.get("song-id")
    content_state.save_media(content_id, version, {"features": features, "rendered_video_path": None})
    return {"content_id": content_id, "music_path": path, "reused": False}
