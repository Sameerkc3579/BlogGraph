"""Password hashing and session tokens (standard library only)."""
import hashlib
import hmac
import secrets

PBKDF2_ITERATIONS = 240_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, hash_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


def hash_token(token: str) -> str:
    """Only this hash is stored, so a leaked database can't be used to log in."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_session_token() -> tuple[str, str]:
    """Returns (token for the browser, hash for the database)."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


# Verified against when an email isn't registered, so login takes the same time either way
DUMMY_PASSWORD_HASH = hash_password(secrets.token_hex(16))
