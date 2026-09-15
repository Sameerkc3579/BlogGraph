"""Database layer: SQLite locally, PostgreSQL when DATABASE_URL is set (required on Render,
whose disk is wiped on every deploy/restart)."""
import argparse
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))
DB_FILE = Path(__file__).parent / "bloggraph.db"

if IS_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row

    IntegrityError = psycopg.IntegrityError
else:
    IntegrityError = sqlite3.IntegrityError


def _now(offset: timedelta = timedelta(0)) -> str:
    """UTC time as 'YYYY-MM-DD HH:MM:SS' — compares and sorts correctly as text in both databases."""
    return (datetime.now(timezone.utc) + offset).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _connect():
    """Yields a connection; commits on success, rolls back on error, always closes."""
    if IS_POSTGRES:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _sql(query: str) -> str:
    """Queries are written with SQLite '?' placeholders; psycopg uses '%s'."""
    return query.replace("?", "%s") if IS_POSTGRES else query


def _fetchone(query: str, params=()):
    with _connect() as conn:
        row = conn.execute(_sql(query), params).fetchone()
    return dict(row) if row else None


def _fetchall(query: str, params=()):
    with _connect() as conn:
        rows = conn.execute(_sql(query), params).fetchall()
    return [dict(r) for r in rows]


def _execute(query: str, params=()) -> int:
    """Runs a write and returns the number of affected rows."""
    with _connect() as conn:
        return conn.execute(_sql(query), params).rowcount


def init_db():
    """Creates tables if they don't exist."""
    pk = "BIGSERIAL PRIMARY KEY" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    blob = "BYTEA" if IS_POSTGRES else "BLOB"
    statements = [
        f"""CREATE TABLE IF NOT EXISTS users (
            id {pk},
            email TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT '',
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id BIGINT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        f"""CREATE TABLE IF NOT EXISTS blogs (
            id {pk},
            user_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            blog_title TEXT NOT NULL,
            markdown TEXT NOT NULL,
            mode TEXT NOT NULL,
            sections_count INTEGER NOT NULL,
            created_at TEXT
        )""",
        f"""CREATE TABLE IF NOT EXISTS images (
            filename TEXT PRIMARY KEY,
            content_type TEXT NOT NULL,
            data {blob} NOT NULL,
            created_at TEXT NOT NULL
        )""",
        f"""CREATE TABLE IF NOT EXISTS generations (
            id {pk},
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_blogs_user ON blogs (user_id)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)",
        "CREATE INDEX IF NOT EXISTS idx_generations_user ON generations (user_id, created_at)",
    ]
    with _connect() as conn:
        for statement in statements:
            conn.execute(statement)


# ── Users & sessions ──────────────────────────────────────────────
def create_user(email: str, name: str, password_hash: str):
    """Creates a user. Returns the user dict, or None if the email is already registered."""
    try:
        with _connect() as conn:
            row = conn.execute(
                _sql("INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?) "
                     "RETURNING id, email, name, created_at"),
                (email, name, password_hash, _now()),
            ).fetchone()
        return dict(row)
    except IntegrityError:
        return None

def get_user_by_email(email: str):
    """Includes password_hash — only for login verification."""
    return _fetchone("SELECT * FROM users WHERE email = ?", (email,))

def create_session(user_id: int, token_hash: str, days: int):
    _execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (token_hash, user_id, _now(timedelta(days=days)), _now()),
    )

def get_session_user(token_hash: str):
    """Returns the user for a valid, unexpired session token hash."""
    return _fetchone(
        "SELECT u.id, u.email, u.name, u.created_at FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token_hash = ? AND s.expires_at > ?",
        (token_hash, _now()),
    )

def delete_session(token_hash: str):
    _execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

def delete_expired_sessions():
    _execute("DELETE FROM sessions WHERE expires_at <= ?", (_now(),))


# ── Blogs ─────────────────────────────────────────────────────────
def save_blog(user_id: str, topic: str, blog_title: str, markdown: str, mode: str, sections_count: int):
    """Saves a generated blog and returns it."""
    with _connect() as conn:
        row = conn.execute(
            _sql("INSERT INTO blogs (user_id, topic, blog_title, markdown, mode, sections_count, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *"),
            (user_id, topic, blog_title, markdown, mode, sections_count, _now()),
        ).fetchone()
    return dict(row)

def get_user_blogs(user_id: str):
    """Fetches all blogs for a user, newest first."""
    return _fetchall(
        "SELECT * FROM blogs WHERE user_id = ? ORDER BY created_at DESC, id DESC", (user_id,)
    )

def delete_blog(blog_id: int, user_id: str) -> bool:
    """Deletes a blog owned by user_id, plus the images it references. Returns True if deleted."""
    with _connect() as conn:
        row = conn.execute(
            _sql("SELECT markdown FROM blogs WHERE id = ? AND user_id = ?"), (blog_id, user_id)
        ).fetchone()
        if not row:
            return False
        conn.execute(_sql("DELETE FROM blogs WHERE id = ? AND user_id = ?"), (blog_id, user_id))
        for filename in set(re.findall(r"/images/([A-Za-z0-9_.-]+)", dict(row)["markdown"])):
            conn.execute(_sql("DELETE FROM images WHERE filename = ?"), (filename,))
    return True


# ── Images (stored in the database so they survive redeploys) ─────
def save_image(filename: str, data: bytes, content_type: str):
    _execute(
        "INSERT INTO images (filename, content_type, data, created_at) VALUES (?, ?, ?, ?)",
        (filename, content_type, data, _now()),
    )

def get_image(filename: str):
    return _fetchone("SELECT content_type, data FROM images WHERE filename = ?", (filename,))


# ── Generation usage (per-user limit protects Hugging Face credits) ─
def record_generation(user_id: str):
    _execute("INSERT INTO generations (user_id, created_at) VALUES (?, ?)", (user_id, _now()))

def count_recent_generations(user_id: str, hours: int = 24) -> int:
    row = _fetchone(
        "SELECT COUNT(*) AS n FROM generations WHERE user_id = ? AND created_at > ?",
        (user_id, _now(-timedelta(hours=hours))),
    )
    return int(row["n"]) if row else 0


# ── Maintenance ───────────────────────────────────────────────────
def claim_orphan_blogs(email: str) -> int:
    """Assigns blogs that belong to no account (e.g. saved before login existed) to this user."""
    user = get_user_by_email(email.strip().lower())
    if not user:
        raise SystemExit(f"No account found for {email}. Sign up in the app first.")
    return _execute(
        "UPDATE blogs SET user_id = ? WHERE user_id NOT IN (SELECT CAST(id AS TEXT) FROM users)",
        (str(user["id"]),),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlogGraph database utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    claim = sub.add_parser("claim-blogs", help="Move blogs saved before accounts existed into your account")
    claim.add_argument("email")
    args = parser.parse_args()

    init_db()
    if args.command == "claim-blogs":
        print(f"Moved {claim_orphan_blogs(args.email)} blog(s) to {args.email}.")
