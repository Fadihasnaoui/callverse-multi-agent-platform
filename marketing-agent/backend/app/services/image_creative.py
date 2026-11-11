"""OpenRouter Image API integration for vertical campaign creatives.

Structured prompt system: each content category gets a visual direction
that matches its marketing purpose, instead of one generic template."""
import base64
import json
import logging
import os
import time
import uuid

import requests

from app import config, db
from app.services import costs
from app.services import content_state

log = logging.getLogger("image_creative")
IMAGES_URL = "https://openrouter.ai/api/v1/images"

_CATEGORY_STYLES = {
    "destination": {
        "mood": "cinematic wanderlust, golden hour warmth, National Geographic editorial",
        "composition": "wide establishing shot showcasing the destination's most iconic view — "
                       "dramatic sky, leading lines, depth layers (foreground texture, landmark, sky)",
        "lighting": "golden hour or blue hour, warm color grading, long shadows",
        "subjects": "the PLACE is the star — the iconic landmark, the panoramic view, the dramatic "
                    "landscape. Show tiny silhouettes of people ONLY from far away for scale, "
                    "never close-up faces",
    },
    "promotion": {
        "mood": "aspirational luxury, magazine-cover quality, desire-inducing",
        "composition": "hero shot of the experience being sold — the pool, the view from the balcony, "
                       "the table set for dinner at sunset. Shallow depth of field, "
                       "clean space in upper and lower thirds for text overlay",
        "lighting": "bright, warm, high-contrast with saturated colors that pop on mobile screens",
        "subjects": "the EXPERIENCE being promoted — a beautifully set hotel terrace at sunset, "
                    "an infinity pool overlooking the sea, a lavish breakfast spread. "
                    "No close-up people. Props only (a hat on a chair, a drink on a ledge)",
    },
    "hotel": {
        "mood": "luxury editorial, Condé Nast Traveler aesthetic, aspirational comfort",
        "composition": "interior or exterior hero shot with depth layers — foreground detail "
                       "(a flower, a glass), main subject (pool/room/terrace), soft background (sky/garden)",
        "lighting": "warm interior glow mixed with natural window light, or twilight exterior "
                    "with lit windows and warm pool reflections",
        "subjects": "the property's signature feature — pool at golden hour, lobby architecture, "
                    "room with a stunning view, restaurant terrace. "
                    "Subtle life signs (placed book, glass of mint tea, draped towel) but NO people",
    },
    "engagement": {
        "mood": "warm, inviting, emotionally resonant — a scene that makes you want to BE there",
        "composition": "medium shot or close-up of a beautiful travel moment — a table of local food, "
                       "a doorway into a colorful medina, hands holding a glass of tea (cropped, no face)",
        "lighting": "soft natural light, warm golden tones, slightly shallow depth of field",
        "subjects": "sensory details of travel — food, architecture, textures, colors, details. "
                    "Hands or silhouettes OK but NEVER close-up faces. "
                    "The image should prompt 'I want to go there'",
    },
    "testimonial": {
        "mood": "authentic, warm, documentary-style travel moment",
        "composition": "a scenic POV shot — what a traveler SEES, not the traveler themselves. "
                       "A stunning view from a hotel balcony, feet dangling over a cliff edge, "
                       "a window view over a city at sunrise",
        "lighting": "natural available light, warm and slightly soft, early morning or late afternoon",
        "subjects": "first-person perspective travel moments — the view, the scene, the place. "
                    "Feet, hands, or silhouettes from behind are OK. NO faces, no posed portraits",
    },
    "event": {
        "mood": "festive, vibrant cultural richness, atmospheric",
        "composition": "wide or medium shot capturing atmosphere — decorations, lights, architecture, "
                       "food stalls. Depth with bokeh lights in background",
        "lighting": "mixed ambient — fairy lights, lanterns, sunset glow, warm and atmospheric",
        "subjects": "the EVENT atmosphere — decorations, lit venues, food displays, cultural details. "
                    "Distant crowds OK, but NO recognizable faces. Focus on the setting, not people",
    },
    "brand": {
        "mood": "professional, warm, curated travel expertise",
        "composition": "clean overhead flat-lay or styled vignette with negative space for overlay",
        "lighting": "even, professional, warm undertones, soft shadows",
        "subjects": "curated travel objects — a map with pins, passport with boarding passes, "
                    "vintage suitcase, compass, sunglasses on a guidebook. NO people",
    },
}

_PLATFORM_DIRECTIVES = {
    "instagram": "Instagram-optimized: rich saturated colors, strong central visual hook, "
                 "designed to earn saves. Vertical 9:16.",
    "facebook": "Facebook-optimized: warm tones that stand out in a text-heavy feed, "
                 "slightly more context in the frame. Vertical 9:16.",
}

_CONTENT_TYPE_DIRECTIVES = {
    "reel": "VIDEO COVER FRAME: must work as a compelling thumbnail. High visual energy, "
            "clear focal point, sense of motion (waves, clouds, fabric in wind).",
    "static_post": "FEED POST: must stop the scroll as a single image. Every element earns attention. "
                   "The scene alone tells a story.",
    "avatar_clip": "AVATAR BACKDROP: a beautiful, slightly blurred travel scene. "
                   "Rich colors but not overpowering. Clean center area.",
}

_GLOBAL_RULES = (
    "ABSOLUTE RULES (violating any of these makes the image unusable):\n"
    "- ZERO TEXT anywhere in the image — no words, letters, numbers, signs, labels, "
    "menus, shop names, street signs, license plates, banners, or watermarks. "
    "All surfaces must be blank or naturally blurred beyond readability.\n"
    "- ZERO LOGOS — no brand marks, no emblems, no symbols that look like logos.\n"
    "- NO CLOSE-UP HUMAN FACES — AI-generated faces look uncanny and ruin the image. "
    "Distant tiny silhouettes, hands, feet, or people seen from behind are OK.\n"
    "- NO AI ARTIFACTS — no extra fingers, merged limbs, floating objects, "
    "impossible architecture, or melted details.\n"
    "- Clean visual space in upper and lower 15% for app-added captions.\n"
    "- Aspect ratio: 9:16 vertical portrait.\n"
    "- Style: photorealistic editorial travel photography shot on a full-frame camera "
    "with a prime lens. NOT illustration, NOT 3D render, NOT painting."
)


def _build_image_prompt(business_name: str, visual_brief: str, category: str,
                        platform: str, content_type: str) -> str:
    style = _CATEGORY_STYLES.get(category, _CATEGORY_STYLES["destination"])
    platform_dir = _PLATFORM_DIRECTIVES.get(platform, _PLATFORM_DIRECTIVES["facebook"])
    type_dir = _CONTENT_TYPE_DIRECTIVES.get(content_type, _CONTENT_TYPE_DIRECTIVES["static_post"])

    return (
        f"Editorial travel photograph for {business_name}, a Tunisian travel agency.\n\n"
        f"SCENE: {visual_brief}\n\n"
        f"DIRECTION:\n"
        f"- Mood: {style['mood']}\n"
        f"- Composition: {style['composition']}\n"
        f"- Lighting: {style['lighting']}\n"
        f"- Subject: {style['subjects']}\n\n"
        f"PLATFORM: {platform_dir}\n"
        f"FORMAT: {type_dir}\n\n"
        f"{_GLOBAL_RULES}"
    )


@costs.budgeted('openrouter-images',0.50)
def generate_for_content(content_id: str) -> dict:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    costs.ensure_budget_available()
    version = content_state.media_version(content_id)
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT c.caption, c.features, c.asset_refs, c.platform, c.type, c.content_category, b.name "
            "FROM content_items c "
            "JOIN businesses b ON b.id = c.business_id WHERE c.id = %s",
            (content_id,),
        ).fetchone()
    if not row:
        raise ValueError("content item not found")
    caption, features, asset_refs, platform, content_type, content_category, business_name = row
    visual_brief = (features or {}).get("visual_brief") or (features or {}).get("topic") or caption
    category = content_category or _detect_category_from_caption(caption or "")
    prompt = _build_image_prompt(business_name, visual_brief, category, platform or "facebook",
                                content_type or "static_post")
    log.info("image prompt category=%s platform=%s type=%s", category, platform, content_type)
    started = time.time()
    costs.dispatched()
    response = requests.post(
        IMAGES_URL,
        headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": config.OPENROUTER_IMAGE_MODEL,
            "prompt": prompt,
            "resolution": config.OPENROUTER_IMAGE_RESOLUTION,
            "aspect_ratio": "9:16",
            "n": 1,
        },
        timeout=300,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            detail = response.text
        _log_image(False, int((time.time() - started) * 1000), {}, detail)
        raise RuntimeError(f"OpenRouter image generation failed: {detail[:500]}")
    payload = response.json()
    data = payload.get("data") or []
    if not data or not data[0].get("b64_json"):
        _log_image(False, int((time.time() - started) * 1000), payload.get("usage") or {}, "no image returned")
        raise RuntimeError("OpenRouter image generation returned no image")

    os.makedirs(config.RENDERS_DIR, exist_ok=True)
    media_type = data[0].get("media_type") or "image/png"
    extension = {"image/jpeg": "jpg", "image/webp": "webp", "image/svg+xml": "svg"}.get(media_type, "png")
    filename = f"{content_id}-creative-{uuid.uuid4().hex[:8]}.{extension}"
    path = os.path.join(config.RENDERS_DIR, filename)
    with open(path, "wb") as output:
        output.write(base64.b64decode(data[0]["b64_json"]))
    base = config.PUBLIC_BASE_URL.rstrip("/") if config.PUBLIC_BASE_URL else "http://127.0.0.1:8010"
    public_url = f"{base}/renders/{filename}"
    updated_assets = list(asset_refs or []) + [public_url]
    content_state.save_media(content_id, version, {"asset_refs": updated_assets, "rendered_video_path": None})
    usage = payload.get("usage") or {}
    _log_image(True, int((time.time() - started) * 1000), usage, None)
    return {"content_id": content_id, "image_url": public_url, "model": config.OPENROUTER_IMAGE_MODEL,
            "resolution": config.OPENROUTER_IMAGE_RESOLUTION, "cost_usd": usage.get("cost"), "prompt": prompt}


def _detect_category_from_caption(caption: str) -> str:
    import re
    patterns = {
        "promotion": r"(promo|offre|réduction|discount|prix|%\s*off|remise|تخفيض|عرض)",
        "destination": r"(istanbul|antalya|tunisie|bodrum|sharm|marrakech|dubai|maldives|zanzibar|"
                       r"bali|paris|rome|barcelone|hurghada|destination|voyage|سفر|رحلة)",
        "hotel": r"(hotel|hôtel|resort|spa|palace|all\s*inclusive|séjour|إقامة|فندق)",
        "engagement": r"(comment|avis|partagez|dites.nous|which|choisi|vote|sondage|\?$)",
        "event": r"(soirée|event|fête|gala|concert|célébr|anniversaire|حفل|مهرجان)",
        "testimonial": r"(témoignage|client|review|merci|شكر|تجربة)",
        "brand": r"(ozone|notre\s*équipe|notre\s*agence|bienvenue|about\s*us|qui\s*sommes)",
    }
    for cat, pat in patterns.items():
        if re.search(pat, caption, re.I):
            return cat
    return "destination"


def _log_image(ok: bool, latency_ms: int, usage: dict, error: str | None) -> None:
    costs.account(usage.get('cost'))
    try:
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO ai_calls (task, prompt_version, model, provider, ok, attempts, latency_ms, "
                "prompt_tokens, completion_tokens, cost_usd, usage_details, error,reservation_id) "
                "VALUES ('image', 'creative-v3', %s, 'openrouter', %s, 1, %s, %s, %s, %s, %s, %s,%s)",
                (config.OPENROUTER_IMAGE_MODEL, ok, latency_ms, usage.get("prompt_tokens"),
                 usage.get("completion_tokens"), usage.get("cost"), json.dumps(usage),
                 (error or "")[:500] or None,costs.reservation_id()),
            )
            conn.commit()
    except Exception:
        pass
