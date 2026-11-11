"""
Client accounts: every persona (default, generated or "jumeau") gets a login so the client
can open their own space and read the simulations played with their persona.
"""
import re
import unicodedata
from typing import Optional
from sqlalchemy.orm import Session
from app.core.security import DEFAULT_CLIENT_PASSWORD, hash_password
from app.database.models import CustomerProfileModel, UserModel

CLIENT_EMAIL_DOMAIN = "client.jasmin.tn"


def default_client_email(db: Session, name: str) -> str:
    """'Leïla Ben Ammar' -> leila.ben.ammar@client.jasmin.tn, suffixed -2, -3... if taken."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", ".", ascii_name.lower()).strip(".") or "client"
    email, n = f"{slug}@{CLIENT_EMAIL_DOMAIN}", 1
    while db.query(UserModel).filter(UserModel.email == email).first():
        n += 1
        email = f"{slug}-{n}@{CLIENT_EMAIL_DOMAIN}"
    return email


def create_client_account(db: Session, profile: CustomerProfileModel,
                          email: Optional[str] = None, password: Optional[str] = None) -> UserModel:
    """Add (not commit) the login of this persona. Without email/password: generated email + default password."""
    user = UserModel(
        email=email or default_client_email(db, profile.name),
        hashed_password=hash_password(password or DEFAULT_CLIENT_PASSWORD),
        full_name=profile.name,
        role="client",
        profile_id=profile.id,
    )
    db.add(user)
    db.flush()
    return user


def backfill_client_accounts(db: Session) -> int:
    """Give a login to every persona that has none yet (existing databases, reseeded personas)."""
    owned = {pid for (pid,) in db.query(UserModel.profile_id).filter(UserModel.profile_id.isnot(None))}
    missing = [p for p in db.query(CustomerProfileModel).all() if p.id not in owned]
    for profile in missing:
        create_client_account(db, profile)
    db.commit()
    return len(missing)
