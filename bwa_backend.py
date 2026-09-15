from __future__ import annotations

import io
import json
import operator
import os
import logging
import re
import secrets
import hashlib
import threading
import time
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

import database

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR
# SAVE_MARKDOWN_FILES=1 also writes each finished blog as a .md file next to this script (local use)
SAVE_MARKDOWN_FILES = os.environ.get("SAVE_MARKDOWN_FILES", "0") == "1"
# ENABLE_IMAGES=0 skips image generation entirely (images use far more Hugging Face credits than text)
ENABLE_IMAGES = os.environ.get("ENABLE_IMAGES", "1") != "0"

log = logging.getLogger("bloggraph.agent")

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema ----
class ImageSpec(BaseModel):
    insert_after_heading: str = Field(
        ..., description='Exact text of the "## " section heading the image goes after, without the "## ".'
    )
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM — bring your own key
# Every request supplies the user's own keys through config["configurable"]["api_keys"].
# Nothing here reads the server's environment keys, so visitors can only spend their own quota.
# -----------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")


class MissingApiKey(RuntimeError):
    """Raised when a request lacks a key for a provider it needs."""


@dataclass(frozen=True)
class ApiKeys:
    """Per-request provider keys.

    Passed as an object (not plain strings) because LangChain copies plain string config values
    into trace metadata (e.g. LangSmith); repr=False keeps the values out of logs too.
    """
    google_api_key: str = field(default="", repr=False)
    hf_token: str = field(default="", repr=False)


def _api_keys(config: RunnableConfig | None) -> ApiKeys:
    keys = ((config or {}).get("configurable") or {}).get("api_keys")
    return keys if isinstance(keys, ApiKeys) else ApiKeys()


def get_llm(config: RunnableConfig | None) -> ChatGoogleGenerativeAI:
    api_key = _api_keys(config).google_api_key
    # Must be explicit: with an empty key the client would fall back to the server's GOOGLE_API_KEY
    if not api_key:
        raise MissingApiKey("A Google Gemini API key is required to generate blogs.")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        # gemini-2.5-flash spends part of this budget on thinking, so leave headroom
        max_output_tokens=8192,
        timeout=120,
        # call_llm() handles retries: the SDK's immediate retries would burn the per-minute quota
        max_retries=0,
    )


def _message_text(message) -> str:
    """Plain text of a chat reply (Gemini can return content as a list of parts)."""
    text = message.text
    # langchain-core 1.x: `.text` is already a str (calling it is deprecated); older versions: a method
    return (text if isinstance(text, str) else text()) or ""


# Free Gemini keys allow only a few requests per minute (5 for gemini-2.5-flash), while one blog
# needs ~7-11 calls. Calls are paced per key and per-minute 429s are waited out instead of failing.
GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "5"))  # 0 = no pacing (paid keys)
LLM_RETRIES = 4


class _PerMinuteLimiter:
    """Blocks until a call fits within `rpm` calls per rolling 60 seconds for that key."""

    def __init__(self):
        self._calls: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def wait(self, key_id: str, rpm: int):
        if rpm <= 0:
            return
        while True:
            with self._lock:
                calls = self._calls[key_id]
                now = time.monotonic()
                while calls and now - calls[0] >= 60:
                    calls.popleft()
                if len(calls) < rpm:
                    calls.append(now)
                    return
                delay = 60 - (now - calls[0]) + 0.5
            time.sleep(delay)


_limiter = _PerMinuteLimiter()


def _key_id(llm) -> str:
    """Hash of the key, so the limiter never holds raw keys."""
    secret = getattr(llm, "google_api_key", None)
    raw = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def call_llm(llm, messages: list) -> str:
    """Invokes the model with per-key pacing and retries for per-minute limits and transient errors."""
    key_id = _key_id(llm)
    for attempt in range(LLM_RETRIES + 1):
        _limiter.wait(key_id, GEMINI_RPM)
        try:
            return _message_text(llm.invoke(messages))
        except Exception as e:
            text = str(e)
            per_minute_limit = "RESOURCE_EXHAUSTED" in text and "PerDay" not in text
            transient = any(s in text for s in ("503", "UNAVAILABLE", "500 INTERNAL", "overloaded"))
            if attempt == LLM_RETRIES or not (per_minute_limit or transient):
                raise
            match = re.search(r"retry in ([\d.]+)s", text)
            delay = min(float(match.group(1)) + 1, 65.0) if (per_minute_limit and match) else (30.0 if per_minute_limit else 5.0)
            log.warning("Gemini %s; retrying in %.0fs (attempt %d/%d)",
                        "per-minute limit hit" if per_minute_limit else "temporarily unavailable",
                        delay, attempt + 1, LLM_RETRIES)
            time.sleep(delay)


def _extract_json(text: str) -> str:
    """Pulls the JSON object out of a model reply (handles ```json fences and surrounding prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start:end + 1]


def structured_invoke(schema: type[BaseModel], messages: list, llm: ChatGoogleGenerativeAI, retries: int = 1):
    """Structured output via prompting + Pydantic validation.

    ChatHuggingFace.with_structured_output() rejects Pydantic schemas for function calling,
    so the schema is sent in the prompt and the reply is parsed and validated here instead.
    """
    instructions = (
        "\n\nRespond with ONLY a single JSON object (no prose, no markdown fences) "
        "that validates against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    msgs = list(messages)
    msgs[0] = SystemMessage(content=msgs[0].content + instructions)

    last_error: Exception | None = None
    for _ in range(retries + 1):
        reply = call_llm(llm, msgs)
        try:
            return schema.model_validate_json(_extract_json(reply))
        except Exception as e:
            last_error = e
            msgs = msgs + [
                AIMessage(content=reply),
                HumanMessage(content=f"That output was invalid ({e}). Return ONLY the corrected JSON object."),
            ]
    raise ValueError(f"Model did not return valid {schema.__name__} JSON: {last_error}")

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State, config: RunnableConfig) -> dict:
    llm = get_llm(config)  # outside the try: a missing key must fail, not silently fall back
    try:
        decision = structured_invoke(
            RouterDecision,
            [
                SystemMessage(content=ROUTER_SYSTEM),
                HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
            ],
            llm,
        )
    except Exception:
        # If the router's structured output fails, fall back to writing without research
        decision = RouterDecision(needs_research=False, mode="closed_book", reason="router fallback")

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research and bool(decision.queries),
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (DuckDuckGo)
# -----------------------------
def _web_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        results = wrapper.results(query, max_results)
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("link") or "",
                    "snippet": r.get("snippet") or "",
                    "published_at": r.get("date"),
                    "source": r.get("source") or "DuckDuckGo",
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State, config: RunnableConfig) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_web_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    llm = get_llm(config)
    try:
        pack = structured_invoke(
            EvidencePack,
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"As-of date: {state['as_of']}\n"
                        f"Recency days: {state['recency_days']}\n\n"
                        f"Raw results:\n{raw[:40]}"
                    )
                ),
            ],
            llm,
        )
        evidence = pack.evidence
    except Exception:
        # Fall back to the raw search results if synthesis fails
        evidence = [EvidenceItem(**r) for r in raw if r["url"]]

    dedup = {}
    for e in evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = _iso_to_date(state["as_of"]) or date.today()
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        # Drop only items known to be stale; search results often have no date at all
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) is None or d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State, config: RunnableConfig) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = structured_invoke(
        Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence[:16]]}"
                )
            ),
        ],
        get_llm(config),
    )
    if not plan.tasks:
        raise ValueError("Planner returned no sections.")
    if forced_kind:
        plan.blog_kind = "news_roundup"

    # Make task ids unique and ordered so sections merge in plan order
    for i, task in enumerate(plan.tasks, start=1):
        task.id = i

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict, config: RunnableConfig) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    ) or "(none)"

    section_md = call_llm(get_llm(config),
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).strip()

    # Guarantee each section starts with its heading so image placement can find it
    if not section_md.startswith("#"):
        section_md = f"## {task.title}\n\n{section_md}"

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- For each image, set insert_after_heading to the exact text of one "## " heading from the blog.
- If no images are needed, return images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State, config: RunnableConfig) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    # Images need the user's own Hugging Face token; without one the blog is text-only
    if not ENABLE_IMAGES or not _api_keys(config).hf_token:
        return {"md_with_placeholders": merged_md, "image_specs": []}

    # Only the headings and a short excerpt are sent, so the model never has to echo the whole blog
    headings = re.findall(r"^##\s+(.+?)\s*$", merged_md, flags=re.M)
    if not headings:
        return {"md_with_placeholders": merged_md, "image_specs": []}

    try:
        image_plan = structured_invoke(
            GlobalImagePlan,
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Section headings:\n" + "\n".join(f"- {h}" for h in headings) + "\n\n"
                        f"Blog excerpt:\n{merged_md[:4000]}"
                    )
                ),
            ],
            get_llm(config),
        )
        specs = image_plan.images[:3]
    except Exception:
        specs = []

    md = merged_md
    image_specs: List[dict] = []
    for i, spec in enumerate(specs, start=1):
        placeholder = f"[[IMAGE_{i}]]"
        heading = spec.insert_after_heading.strip().lstrip("#").strip()
        # Place the image at the end of the chosen section (before the next heading)
        pattern = re.compile(rf"(^##\s+{re.escape(heading)}\s*$.*?)(?=^##\s|\Z)", flags=re.M | re.S)
        md, n = pattern.subn(lambda m: m.group(1).rstrip() + f"\n\n{placeholder}\n\n", md, count=1)
        if n:
            data = spec.model_dump()
            data["placeholder"] = placeholder
            image_specs.append(data)

    return {"md_with_placeholders": md, "image_specs": image_specs}


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _unique_image_filename(name: str) -> str:
    """Random suffix so blogs never share or overwrite each other's images (and URLs are unguessable)."""
    stem = _safe_slug(Path(name).stem)[:60]
    return f"{stem}-{secrets.token_hex(8)}.jpg"


def _huggingface_generate_image_bytes(prompt: str, token: str) -> bytes:
    from huggingface_hub import InferenceClient

    # Must be explicit: without a token the client would fall back to the server's HF_TOKEN / cached login
    if not token:
        raise MissingApiKey("A Hugging Face token is required to generate images.")
    client = InferenceClient(token=token, timeout=120)
    image = client.text_to_image(prompt, model=IMAGE_MODEL)
    buf = io.BytesIO()
    # JPEG is ~5-10x smaller than PNG, which matters since images are stored in the database
    image.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def generate_and_place_images(state: State, config: RunnableConfig) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    hf_token = _api_keys(config).hf_token

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = _unique_image_filename(spec["filename"])
        try:
            # Stored in the database (not on disk) so images survive redeploys
            database.save_image(filename, _huggingface_generate_image_bytes(spec["prompt"], hf_token), "image/jpeg")
        except Exception:
            # Leave the image out rather than showing a broken placeholder
            log.exception("Image generation failed for %s", filename)
            md = md.replace(placeholder + "\n\n", "").replace(placeholder, "")
            continue

        img_md = f"![{spec['alt']}](/images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    # Remove any placeholders that were not filled
    md = re.sub(r"\[\[IMAGE_\d+\]\]\n*", "", md)

    if SAVE_MARKDOWN_FILES:
        (OUTPUT_DIR / f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
