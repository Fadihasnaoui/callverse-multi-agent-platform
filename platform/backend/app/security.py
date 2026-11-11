"""Encrypt/decrypt business_credentials.access_token at rest. Fernet (symmetric,
authenticated) is enough here — the threat model is "don't store third-party
Page tokens as plaintext in the DB," not key rotation infrastructure."""
from cryptography.fernet import Fernet

from app import config


def _fernet() -> Fernet:
    if not config.CREDENTIALS_ENCRYPTION_KEY:
        raise RuntimeError(
            "CREDENTIALS_ENCRYPTION_KEY is not set. Generate one with: "
            "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(config.CREDENTIALS_ENCRYPTION_KEY.encode())


def encrypt_token(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()
