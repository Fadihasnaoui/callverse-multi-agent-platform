import hashlib
import secrets

DEFAULT_USER_EMAIL = "admin@simulator.com"
DEFAULT_USER_PASSWORD = "Password123!"
DEFAULT_USER_NAME = "Conseiller Commercial"
DEFAULT_USER_ROLE = "admin"
# Login of the personas created without a chosen password (default, generated); twins pick their own
DEFAULT_CLIENT_PASSWORD = "Client123!"


def hash_password(password: str) -> str:
    """Hash password using salt and sha256."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed password."""
    try:
        salt, expected_hash = hashed_password.split("$", 1)
        actual_hash = hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest()
        return secrets.compare_digest(actual_hash, expected_hash)
    except Exception:
        return False
