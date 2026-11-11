"""Phase 4 - Amira persona avatar clips. Pipeline: LLM turns the content
item's caption/video_script into a short spoken line -> ElevenLabs voices it
-> HeyGen lip-syncs the avatar to that audio -> MP4 downloaded into
RENDERS_DIR. HeyGen renders asynchronously, so start_avatar_clip() returns
immediately and refresh_avatar_clip() polls and downloads when ready."""
import os
import uuid

import requests

from app import config, db
from app.services import glossary, memory, costs, content_state
from app.ai import guards, prompts, run as ai_run
from app.ai.schemas import SpokenOut

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
HEYGEN_UPLOAD_URL = "https://upload.heygen.com/v1/asset"
HEYGEN_GENERATE_URL = "https://api.heygen.com/v2/video/generate"
HEYGEN_STATUS_URL = "https://api.heygen.com/v1/video_status.get"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def _require_keys():
    missing = [
        name
        for name, value in (
            ("ELEVENLABS_API_KEY", config.ELEVENLABS_API_KEY),
            ("ELEVENLABS_VOICE_ID", config.ELEVENLABS_VOICE_ID),
            ("HEYGEN_API_KEY", config.HEYGEN_API_KEY),
            ("HEYGEN_AVATAR_ID", config.HEYGEN_AVATAR_ID),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"avatar pipeline not configured, missing: {', '.join(missing)}")


def _spoken_text(business_id: str, caption: str, video_script: str) -> str:
    result = ai_run(
        "spoken", prompts.SPOKEN.system,
        f"Caption:\n{caption}\n\nVideo script notes:\n{video_script}{memory.recall_block(business_id, 'voice', caption)}",
        SpokenOut, model=config.CONTENT_MODEL, version=prompts.SPOKEN.version,
        guard=guards.all_of(guards.no_invented_numbers(caption, video_script, fields=("spoken_text",)),
                            guards.plain_speech("spoken_text")),
    )
    return result.spoken_text.strip()


def _voice_id(language: str) -> str:
    if language == "tn" and config.ELEVENLABS_TUNISIAN_VOICE_ID:
        return config.ELEVENLABS_TUNISIAN_VOICE_ID
    return config.ELEVENLABS_VOICE_ID


@costs.budgeted('elevenlabs-voice',1.0)
def _synthesize_voice(text: str, language: str = "fr") -> bytes:
    costs.dispatched()
    response = requests.post(
        f"{ELEVENLABS_TTS_URL}/{_voice_id(language)}",
        headers={"xi-api-key": config.ELEVENLABS_API_KEY, "Accept": "audio/mpeg"},
        json={"text": text, "model_id": config.ELEVENLABS_MODEL},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs failed ({response.status_code}): {response.text[:500]}")
    return response.content


def _upload_audio(audio: bytes) -> str:
    response = requests.post(
        HEYGEN_UPLOAD_URL,
        headers={"X-Api-Key": config.HEYGEN_API_KEY, "Content-Type": "audio/mpeg"},
        data=audio,
        timeout=120,
    )
    body = response.json() if response.content else {}
    asset_id = (body.get("data") or {}).get("id")
    if response.status_code != 200 or not asset_id:
        raise RuntimeError(f"HeyGen audio upload failed ({response.status_code}): {response.text[:500]}")
    return asset_id


@costs.budgeted('heygen',3.0)
def _start_heygen_video(audio_asset_id: str) -> str:
    costs.dispatched()
    response = requests.post(
        HEYGEN_GENERATE_URL,
        headers={"X-Api-Key": config.HEYGEN_API_KEY, "Content-Type": "application/json"},
        json={
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_id": config.HEYGEN_AVATAR_ID,
                        "avatar_style": "normal",
                    },
                    "voice": {"type": "audio", "audio_asset_id": audio_asset_id},
                }
            ],
            "dimension": {"width": 720, "height": 1280},
        },
        timeout=60,
    )
    body = response.json() if response.content else {}
    video_id = (body.get("data") or {}).get("video_id")
    if response.status_code != 200 or not video_id:
        raise RuntimeError(f"HeyGen generate failed ({response.status_code}): {response.text[:500]}")
    return video_id


@costs.budgeted('elevenlabs-transcription',0.50)
def transcribe_dictation(content_id: str, audio: bytes, content_type: str) -> dict:
    """Tunisian flow, step 1: the owner dictates the line by voice. The audio is
    transcribed (ElevenLabs Scribe) and stored as a *pending* transcript. Nothing
    is voiced or rendered until the owner approves it - speech-to-text on Derja
    is imperfect, so the owner is the check."""
    if not config.ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    if not audio:
        raise ValueError("empty audio")
    content_state.media_version(content_id)
    costs.dispatched()
    response = requests.post(
        ELEVENLABS_STT_URL,
        headers={"xi-api-key": config.ELEVENLABS_API_KEY},
        data={"model_id": config.ELEVENLABS_STT_MODEL, "language_code": config.DICTATION_LANGUAGE_CODE},
        files={"file": ("dictation", audio, content_type or "application/octet-stream")},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs transcription failed ({response.status_code}): {response.text[:500]}")
    raw = (response.json().get("text") or "").strip()
    if not raw:
        raise ValueError("no speech detected in the recording")
    with db.get_conn() as conn:
        owner = conn.execute("SELECT business_id FROM content_items WHERE id = %s", (content_id,)).fetchone()
    if not owner:
        raise ValueError("content item not found")
    business_id = str(owner[0])

    # raw speech-to-text -> confirmed glossary swaps -> one LLM proofreading pass
    swapped, fired = glossary.apply(business_id, raw)
    cleaned = glossary.proofread(business_id, swapped)
    _save_transcript(content_id, cleaned, "pending", raw=raw)
    return {
        "content_id": content_id, "transcript": cleaned, "transcript_status": "pending",
        "raw_transcript": raw, "glossary_swaps": [{"heard": h, "meant": m} for h, m in fired],
    }


def _save_transcript(content_id: str, transcript: str, status: str, raw: str | None = None):
    with db.get_conn() as conn:
        updated = conn.execute(
            "UPDATE content_items SET avatar_spoken_text = %s, avatar_language = 'tn', "
            "avatar_transcript_status = %s, avatar_raw_transcript = COALESCE(%s, avatar_raw_transcript) "
            "WHERE id = %s RETURNING id",
            (transcript, status, raw, content_id),
        ).fetchone()
        conn.commit()
    if not updated:
        raise ValueError("content item not found")


def resolve_transcript(content_id: str, transcript: str, approve: bool) -> dict:
    """Tunisian flow, step 2: the owner edits and/or approves the transcript."""
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("transcript is empty")
    status = "approved" if approve else "pending"
    with db.get_conn() as conn:
        prior = conn.execute(
            "SELECT business_id, avatar_raw_transcript, avatar_transcript_status FROM content_items WHERE id = %s",
            (content_id,),
        ).fetchone()
    _save_transcript(content_id, transcript, status)
    # Learn once, on the first approval of a dictated transcript: raw speech-to-text
    # vs. what the owner accepted. Untouched machine fixes count as confirmations.
    learned = 0
    if approve and prior and prior[2] == "pending" and prior[1]:
        learned = glossary.learn(str(prior[0]), prior[1], transcript)
    return {"content_id": content_id, "transcript": transcript, "transcript_status": status, "glossary_pairs_learned": learned}


def start_avatar_clip(content_id: str, language: str = "fr") -> dict:
    """language='fr': the agent writes the spoken line; the owner reviews the
    finished video through the normal approval flow. language='tn': the spoken
    line is the owner's dictation and must be transcript-approved first."""
    if language not in ("fr", "tn"):
        raise ValueError("language must be 'fr' or 'tn'")
    _require_keys()
    version = content_state.media_version(content_id)
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT caption, video_script, avatar_spoken_text, avatar_language, avatar_transcript_status, business_id, status "
            "FROM content_items WHERE id = %s",
            (content_id,),
        ).fetchone()
    if not row:
        raise ValueError("content item not found")
    caption, video_script, saved_text, saved_language, transcript_status, business_id, item_status = row
    if item_status not in ("pending", "rejected"):
        raise ValueError(f"item is {item_status} - a new avatar clip can only be generated for a pending item")

    if language == "tn":
        if saved_language != "tn" or not saved_text:
            raise ValueError("no dictated transcript yet - POST .../avatar/dictation first")
        if transcript_status != "approved":
            raise ValueError("transcript not approved - POST .../avatar/transcript with approve=true first")
        spoken = saved_text
    else:
        if not caption:
            raise ValueError("content item has no caption yet")
        spoken = _spoken_text(str(business_id), caption, video_script or "")

    audio = _synthesize_voice(spoken, language)
    video_id = _start_heygen_video(_upload_audio(audio))

    content_state.save_media(content_id,version,{'avatar_job_id':video_id,'avatar_spoken_text':spoken,'avatar_language':language,'rendered_video_path':None})
    return {
        "content_id": content_id, "avatar_job_id": video_id, "language": language,
        "spoken_text": spoken, "status": "processing",
    }


def refresh_avatar_clip(content_id: str) -> dict:
    """Polls HeyGen once; downloads the MP4 into RENDERS_DIR when finished."""
    _require_keys()
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT avatar_job_id, rendered_video_path,content_version FROM content_items WHERE id = %s", (content_id,)
        ).fetchone()
    if not row:
        raise ValueError("content item not found")
    job_id, existing_path, version = row
    if existing_path and os.path.exists(existing_path):
        return {'content_id':content_id,'status':'completed','rendered_video_path':existing_path}
    if not job_id:
        raise ValueError("no avatar job started - POST .../avatar first")

    response = requests.get(
        HEYGEN_STATUS_URL,
        headers={"X-Api-Key": config.HEYGEN_API_KEY},
        params={"video_id": job_id},
        timeout=30,
    )
    data = (response.json() or {}).get("data") or {}
    status = data.get("status")
    if status == "failed":
        return {"content_id": content_id, "status": "failed", "error": data.get("error")}
    if status != "completed" or not data.get("video_url"):
        return {"content_id": content_id, "status": status or "unknown"}

    os.makedirs(config.RENDERS_DIR, exist_ok=True)
    # One file per HeyGen job: refreshing again never re-downloads or orphans copies.
    output_path = f"{config.RENDERS_DIR}/{content_id}-avatar-{job_id}.mp4"
    if not os.path.exists(output_path):
        download = requests.get(data["video_url"], timeout=300)
        download.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(download.content)

    content_state.save_media(content_id,version,{'rendered_video_path':output_path})
    return {"content_id": content_id, "status": "completed", "rendered_video_path": output_path}
