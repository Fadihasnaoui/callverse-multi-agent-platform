"""Staff login for the interface and API.

One shared staff password (AUTH_PASSWORD) -> a signed, expiring session cookie. Per-customer logins can be
layered on later; for an agency managing customers this is the minimum that keeps the API from being open.

Fail closed: if AUTH_PASSWORD is not set, protected routes answer 503 (set ALLOW_NO_AUTH=true only for local
development). Webhooks, health and rendered videos stay public: Meta and Instagram must reach them."""
import base64
import hashlib
import hmac
import json
import time

from app import config

PROTECTED_PREFIXES = ("/businesses", "/assets", "/ai", "/workforce-api", "/digital-twin")
COOKIE = "agent_session"


def _secret() -> bytes:
    base = config.AUTH_SECRET or ("auth:" + config.CREDENTIALS_ENCRYPTION_KEY)
    return hashlib.sha256(base.encode()).digest()


def check_password(candidate: str) -> bool:
    if not config.AUTH_PASSWORD:
        return False
    return hmac.compare_digest(
        hashlib.sha256((candidate or "").encode()).digest(), hashlib.sha256(config.AUTH_PASSWORD.encode()).digest()
    )


def make_token(hours: int | None = None) -> str:
    exp = int(time.time()) + 3600 * (hours or config.AUTH_SESSION_HOURS)
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def valid_token(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    body, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()):
        return False
    try:
        exp = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))["exp"]
    except (ValueError, KeyError):
        return False
    return exp > time.time()


def is_protected(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES)


# crude brute-force brake: 8 wrong passwords per client per 5 minutes
_failures: dict[str, list[float]] = {}


def too_many_failures(client: str) -> bool:
    now = time.time()
    recent = [t for t in _failures.get(client, []) if now - t < 300]
    _failures[client] = recent
    return len(recent) >= 8


def record_failure(client: str):
    _failures.setdefault(client, []).append(time.time())
