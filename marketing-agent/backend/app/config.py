"""Central env loading. Mirrors the Ozone project's pattern of reading .env
once at startup rather than scattering os.getenv() defaults everywhere."""
import ntpath
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "marketing_agent")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "marketing_agent")
POSTGRES_POOL_MAX_SIZE = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b")
# Tried once if the primary model keeps failing validation or its API is down (see app/ai/client.py).
AI_FALLBACK_MODEL = os.getenv("AI_FALLBACK_MODEL", "openai/gpt-4o-mini")
# Content generation specifically uses a real OpenAI model (via OpenRouter,
# same gateway/billing as everything else) rather than the default - set
# separately since different services can reasonably want different
# cost/quality tradeoffs.
CONTENT_MODEL = os.getenv("CONTENT_MODEL", "openai/gpt-4o-mini")
OPENROUTER_IMAGE_MODEL = os.getenv("OPENROUTER_IMAGE_MODEL", "openai/gpt-image-2")
OPENROUTER_IMAGE_RESOLUTION = os.getenv("OPENROUTER_IMAGE_RESOLUTION", "2K")
MONTHLY_AI_BUDGET_USD = float(os.getenv("MONTHLY_AI_BUDGET_USD", "10"))
OPENROUTER_TOTAL_CREDITS = float(os.getenv("OPENROUTER_TOTAL_CREDITS", "0"))
AI_STATS_SINCE = os.getenv("AI_STATS_SINCE", "")
# Weekly memory compaction runs rarely and must resolve contradictions, so it can use a stronger model.
COMPACTION_MODEL = os.getenv("COMPACTION_MODEL", "openai/gpt-4o")

META_GRAPH_API_VERSION = os.getenv("META_GRAPH_API_VERSION", "v21.0")
META_GRAPH_BASE_URL = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}"
META_APP_ID = os.getenv("META_APP_ID", "")

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY", "")
# Phase 4 - fixed avatar with separate French and Tunisian voices. Tunisian
# falls back to the primary voice until a dedicated voice is selected.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_TUNISIAN_VOICE_ID = os.getenv("ELEVENLABS_TUNISIAN_VOICE_ID", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_MUSIC_MODEL = os.getenv("ELEVENLABS_MUSIC_MODEL", "music_v2")
HEYGEN_AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID", "")
# Owner voice dictation (Tunisian flow). Scribe has no separate Derja code;
# "ara" is the closest, and the owner corrects the transcript anyway.
ELEVENLABS_STT_MODEL = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")
# Final Derja proofreading pass over each dictation (short text, so a stronger model is cheap).
PROOFREAD_MODEL = os.getenv("PROOFREAD_MODEL", "openai/gpt-4o")
DICTATION_LANGUAGE_CODE = os.getenv("DICTATION_LANGUAGE_CODE", "ara")

# Phase 5 - Instagram only fetches media from public URLs, so reels must be
# reachable at PUBLIC_BASE_URL/renders/<file>. Facebook uploads directly.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
PUBLISH_INTERVAL_SECONDS = int(os.getenv("PUBLISH_INTERVAL_SECONDS", "60"))

# Phase 6 - webhooks. App secret enables signature checking (leave unset
# only for local testing); comments/DMs scoring below the threshold are
# stored as 'ignored', never dropped.
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
LEAD_INTENT_THRESHOLD = int(os.getenv("LEAD_INTENT_THRESHOLD", "50"))

CREDENTIALS_ENCRYPTION_KEY = os.getenv("CREDENTIALS_ENCRYPTION_KEY", "")

# Phase 3 - Remotion render pipeline
REMOTION_DIR = os.getenv("REMOTION_DIR", str(ROOT / "remotion"))
RENDERS_DIR = os.getenv("RENDERS_DIR", str(ROOT / "renders"))


# Publishing safety
# A post whose slot is older than this is NOT auto-published (the app was down, say) -
# it is marked failed so the owner decides, instead of a week-old post going out late.
STALE_PUBLISH_HOURS = int(os.getenv("STALE_PUBLISH_HOURS", "6"))
# An item stuck in 'publishing' this long was interrupted (crash/restart); it is moved to 'failed'.
PUBLISHING_TIMEOUT_MINUTES = int(os.getenv("PUBLISHING_TIMEOUT_MINUTES", "15"))
# Webhooks are rejected unless META_APP_SECRET is set. Only set this for local testing.
ALLOW_UNSIGNED_WEBHOOKS = os.getenv("ALLOW_UNSIGNED_WEBHOOKS", "").lower() in ("1", "true", "yes")


def render_filename(path: str) -> str:
    """File name of a render regardless of / or \\ separators (Windows dev runs)."""
    return ntpath.basename(path)

# Staff login (app/auth.py). Protected routes answer 503 until AUTH_PASSWORD is set.
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")
AUTH_SECRET = os.getenv("AUTH_SECRET", "")
AUTH_SESSION_HOURS = int(os.getenv("AUTH_SESSION_HOURS", "168"))
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "").lower() in ("1", "true", "yes")  # set behind HTTPS
ALLOW_NO_AUTH = os.getenv("ALLOW_NO_AUTH", "").lower() in ("1", "true", "yes")  # local development only

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Keep the central platform on 8000. Workforce can publish separately on 8014.
WORKFORCE_API_URL = os.getenv('WORKFORCE_API_URL','http://127.0.0.1:8014').rstrip('/')

# Ozone Digital Twin orchestration. These services retain their own data and
# public contracts; the hub stores only the shared session/transcript envelope.
DIGITAL_TWIN_ADVISOR_URL = os.getenv(
    "DIGITAL_TWIN_ADVISOR_URL", "http://127.0.0.1:8011/v1/agents/customer-advisor/respond"
).rstrip("/")
DIGITAL_TWIN_SIMULATOR_URL = os.getenv("DIGITAL_TWIN_SIMULATOR_URL", "http://127.0.0.1:8012").rstrip("/")
DIGITAL_TWIN_QUALITY_URL = os.getenv(
    "DIGITAL_TWIN_QUALITY_URL", "http://127.0.0.1:8010/v1/quality-analyst"
).rstrip("/")
# Optional future Workforce webhook. Blank means the hub persists a routing
# event locally while the existing Workforce dashboard continues to operate.
WORKFORCE_ROUTING_WEBHOOK_URL = os.getenv("WORKFORCE_ROUTING_WEBHOOK_URL", "").rstrip("/")
WORKFORCE_EXTERNAL_ROUTING_URL = os.getenv(
    "WORKFORCE_EXTERNAL_ROUTING_URL", f"{WORKFORCE_API_URL}/external/route"
).rstrip("/")
# Host-side LiveKit voice transport (scripts/multi_call_voice_manager.py).
# Blank disables voiced calls; text sessions are unaffected.
DIGITAL_TWIN_VOICE_URL = os.getenv("DIGITAL_TWIN_VOICE_URL", "").rstrip("/")
# Written run reports: generated through the LLM gateway Amira already runs
# (/llm-chat), so no model key has to be copied into this app.
DIGITAL_TWIN_REPORT_LLM_URL = os.getenv(
    "DIGITAL_TWIN_REPORT_LLM_URL", DIGITAL_TWIN_ADVISOR_URL.split("/v1/", 1)[0] + "/llm-chat"
).rstrip("/")
DIGITAL_TWIN_REPORT_PROVIDERS = os.getenv("DIGITAL_TWIN_REPORT_PROVIDERS", "openrouter,groq,ollama")
# OpenRouter model for reports and per-call reviews: no hidden reasoning (which emptied
# gpt-oss replies on long transcripts) and about $0.025 for a 12-call report.
DIGITAL_TWIN_REPORT_MODEL = os.getenv("DIGITAL_TWIN_REPORT_MODEL", "openai/gpt-4.1-mini")
# Voice Lab rule-based client personas, mounted read-only (app/services/twin_clients.py).
DIGITAL_TWIN_SCRIPTED_PERSONAS_DIR = os.getenv("DIGITAL_TWIN_SCRIPTED_PERSONAS_DIR", "/voice-lab/personas")
DIGITAL_TWIN_SCRIPTED_ENGINE = os.getenv("DIGITAL_TWIN_SCRIPTED_ENGINE", "/voice-lab/call_orchestrator.py")
