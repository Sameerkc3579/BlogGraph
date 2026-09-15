import logging
import os
import re
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import requests as http_requests

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import auth
import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bloggraph")

ON_RENDER = os.environ.get("RENDER", "").lower() == "true"
if ON_RENDER and not database.IS_POSTGRES:
    raise RuntimeError(
        "DATABASE_URL is not set. Render wipes the disk on every deploy and restart, so a PostgreSQL "
        "DATABASE_URL is required to keep accounts and blogs. Add it under Environment in the Render dashboard."
    )

DAILY_GENERATION_LIMIT = int(os.environ.get("DAILY_GENERATION_LIMIT", "5"))  # 0 = unlimited

# ── Google OAuth config ─────────────────────────────────────────
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
# Dynamically built so it works both locally and on Render
def _google_redirect_uri(request: Request) -> str:
    base = os.environ.get("APP_BASE_URL", "").rstrip("/")
    if not base:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host   = request.headers.get("x-forwarded-host", request.url.netloc)
        base   = f"{scheme}://{host}"
    return f"{base}/auth/google/callback"

# ── import the compiled LangGraph app ──────────────────────────
from bwa_backend import app as blog_app

# ── FastAPI setup ───────────────────────────────────────────────
api = FastAPI(title="BlogGraph API", version="1.1.0")

database.init_db()

# The frontend is served from this same app, so CORS is only needed for other origins
allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed_origins:
    api.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"])


@api.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a single readable message (the frontend shows `detail` as text)."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(p) for p in first.get("loc", [])[1:])
    message = first.get("msg", "Invalid request.")
    return JSONResponse(status_code=422, content={"detail": f"{field}: {message}" if field else message})


# ── Request models ──────────────────────────────────────────────
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}\.(?:jpg|png)$")

class SignupRequest(BaseModel):
    name: str = Field("", max_length=100)
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=128)

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=128)
    remember: bool = False

class GenerateRequest(BaseModel):
    topic: str = Field(..., max_length=500)
    as_of: str = Field(default_factory=lambda: str(date.today()))   # ISO date, defaults to today


# ── Auth helpers ────────────────────────────────────────────────
def _bearer_token(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None

def current_user(authorization: str | None = Header(default=None)) -> dict:
    token = _bearer_token(authorization)
    user = database.get_session_user(auth.hash_token(token)) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    return user

def _public_user(user: dict) -> dict:
    return {"id": user["id"], "email": user["email"], "name": user["name"]}

def _start_session(user: dict, days: int) -> dict:
    token, token_hash = auth.new_session_token()
    database.create_session(user["id"], token_hash, days)
    return {"token": token, "user": _public_user(user)}


# In-memory throttling of failed logins (per email and per IP)
_failed_logins: dict[str, deque] = defaultdict(deque)
_failed_lock = threading.Lock()
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_LIMITS = {"email": 10, "ip": 30}

def _login_blocked(key: str, limit: int) -> bool:
    with _failed_lock:
        attempts = _failed_logins[key]
        cutoff = time.time() - LOGIN_WINDOW_SECONDS
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        return len(attempts) >= limit

def _record_failed_login(*keys: str):
    with _failed_lock:
        for key in keys:
            _failed_logins[key].append(time.time())


# One generation at a time per user
_active_generations: set[str] = set()
_active_lock = threading.Lock()

def _generation_error(e: Exception) -> tuple[int, str]:
    """Maps agent/provider failures to a status code and a message a user can act on."""
    text = str(e)
    if "402 Payment Required" in text or "depleted your monthly included credits" in text:
        return 503, "The AI service has run out of Hugging Face credits. Please try again later."
    if "429 Too Many Requests" in text:
        return 503, "The AI service is busy (rate limited). Please try again in a few minutes."
    if "401 Unauthorized" in text and "huggingface" in text.lower():
        return 503, "The AI service rejected the server's Hugging Face token (check HF_TOKEN)."
    if "timed out" in text.lower() or "timeout" in type(e).__name__.lower():
        return 504, "The AI service took too long to respond. Please try again."
    return 500, f"Agent error: {text[:300]}"


# ── Endpoints ──────────────────────────────────────────────────
@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/auth/signup")
def signup(req: SignupRequest):
    email = req.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    user = database.create_user(email, req.name.strip(), auth.hash_password(req.password))
    if user is None:
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please log in.")
    log.info("New account created: user %s", user["id"])
    return _start_session(user, days=30)


@api.post("/auth/login")
def login(req: LoginRequest, request: Request):
    email = req.email.strip().lower()
    ip = request.client.host if request.client else "unknown"
    email_key, ip_key = f"email:{email}", f"ip:{ip}"
    if _login_blocked(email_key, LOGIN_LIMITS["email"]) or _login_blocked(ip_key, LOGIN_LIMITS["ip"]):
        raise HTTPException(status_code=429, detail="Too many failed sign-in attempts. Please wait 15 minutes and try again.")

    user = database.get_user_by_email(email)
    password_ok = auth.verify_password(req.password, user["password_hash"] if user else auth.DUMMY_PASSWORD_HASH)
    if not user or not password_ok:
        _record_failed_login(email_key, ip_key)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    database.delete_expired_sessions()
    return _start_session(user, days=30 if req.remember else 7)


@api.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    token = _bearer_token(authorization)
    if token:
        database.delete_session(auth.hash_token(token))
    return {"status": "success"}


# ── Google OAuth endpoints ──────────────────────────────────────
@api.get("/auth/google")
def google_login(request: Request):
    """Redirect the browser to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google login is not configured on this server.")
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  _google_redirect_uri(request),
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "online",
        "prompt":        "select_account",
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url)


@api.get("/auth/google/callback")
def google_callback(code: str | None = None, error: str | None = None, request: Request = None):
    """Google redirects here with ?code=... after the user approves."""
    if error or not code:
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/#login?error=google_denied")

    # 1. Exchange auth code for tokens
    token_resp = http_requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  _google_redirect_uri(request),
            "grant_type":    "authorization_code",
        },
        timeout=10,
    )
    if not token_resp.ok:
        log.error("Google token exchange failed: %s", token_resp.text)
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/#login?error=google_token")

    access_token = token_resp.json().get("access_token")

    # 2. Fetch the user's profile from Google
    profile_resp = http_requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if not profile_resp.ok:
        log.error("Google userinfo failed: %s", profile_resp.text)
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/#login?error=google_profile")

    profile  = profile_resp.json()
    email    = (profile.get("email") or "").strip().lower()
    name     = profile.get("name") or email.split("@")[0]

    if not email:
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/#login?error=google_no_email")

    # 3. Find or create the user (Google accounts have no password)
    user = database.get_user_by_email(email)
    if user is None:
        user = database.create_user(email, name, password_hash="")
        if user is None:          # race condition: just created by another request
            user = database.get_user_by_email(email)
        log.info("New Google account: user %s", user["id"] if user else "?")

    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/#login?error=google_db")

    # 4. Issue a session and redirect to the app with the token in the URL fragment
    session = _start_session(user, days=30)
    token   = session["token"]
    u       = session["user"]
    # Pass token + basic user info via URL fragment so the frontend can store it
    fragment = urllib.parse.urlencode({
        "token": token,
        "id":    u["id"],
        "email": u["email"],
        "name":  u["name"],
    })
    from fastapi.responses import RedirectResponse
    return RedirectResponse(f"/#google_auth:{fragment}")


@api.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return {
        "user": _public_user(user),
        "daily_generation_limit": DAILY_GENERATION_LIMIT,
        "generations_last_24h": database.count_recent_generations(str(user["id"])),
    }


@api.post("/generate")
def generate(req: GenerateRequest, user: dict = Depends(current_user)):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    try:
        date.fromisoformat(req.as_of)
    except ValueError:
        raise HTTPException(status_code=400, detail="as_of must be a date in YYYY-MM-DD format.")

    user_key = str(user["id"])
    if DAILY_GENERATION_LIMIT > 0 and database.count_recent_generations(user_key) >= DAILY_GENERATION_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"You've reached the limit of {DAILY_GENERATION_LIMIT} blogs per 24 hours. Please try again later.",
        )

    with _active_lock:
        if user_key in _active_generations:
            raise HTTPException(status_code=409, detail="A blog is already being generated for your account. Please wait for it to finish.")
        _active_generations.add(user_key)

    initial_state = {
        "topic": topic,
        "as_of": req.as_of,
        "mode": "closed_book",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "recency_days": 3650,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    try:
        # Counted before running so failed attempts (which still spend credits) count too
        database.record_generation(user_key)
        result = blog_app.invoke(initial_state)
    except Exception as e:
        log.exception("Generation failed for user %s", user_key)
        status, message = _generation_error(e)
        raise HTTPException(status_code=status, detail=message)
    finally:
        with _active_lock:
            _active_generations.discard(user_key)

    markdown = result.get("final") or result.get("merged_md") or ""
    if not markdown.strip():
        raise HTTPException(status_code=500, detail="Agent returned an empty blog.")
    plan = result.get("plan")
    blog_title = plan.blog_title if plan else topic
    mode = result.get("mode", "closed_book")
    # The reducer subgraph returns `sections` to the parent graph, where the operator.add reducer
    # appends them a second time, so count the planned sections instead of that list.
    sections_count = len(plan.tasks) if plan else len({tid for tid, _ in result.get("sections", [])})

    # Saved server-side under the logged-in account
    blog = database.save_blog(
        user_id=user_key,
        topic=topic,
        blog_title=blog_title,
        markdown=markdown,
        mode=mode,
        sections_count=sections_count,
    )

    return {
        "markdown": markdown,
        "mode": mode,
        "blog_title": blog_title,
        "sections_count": sections_count,
        "blog": blog,
    }


@api.get("/blogs")
def list_blogs(user: dict = Depends(current_user)):
    return {"status": "success", "blogs": database.get_user_blogs(str(user["id"]))}


@api.delete("/blogs/{blog_id}")
def delete_blog(blog_id: int, user: dict = Depends(current_user)):
    if not database.delete_blog(blog_id, str(user["id"])):
        raise HTTPException(status_code=404, detail="Blog not found.")
    return {"status": "success", "id": blog_id}


@api.get("/images/{filename}")
def get_image(filename: str):
    # Filenames contain a random suffix, so image URLs are unguessable
    if not IMAGE_NAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="Image not found.")
    image = database.get_image(filename)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found.")
    return Response(
        content=bytes(image["data"]),
        media_type=image["content_type"],
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


# ── Serve the static frontend ──────────────────────────────────
frontend_dir = Path(__file__).parent / "frontend"
api.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@api.get("/")
def root():
    return FileResponse(str(frontend_dir / "index.html"), headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:api",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=not ON_RENDER,
    )
