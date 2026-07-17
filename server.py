"""FastAPI backend for the Neural Map web UI.

Run:   ./.venv/bin/python server.py
Open:  http://127.0.0.1:8000

Thin HTTP layer over the exact same brain/ pipeline used by the CLI.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import re
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode, urlparse

import ollama
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import config
from brain import action_packages, attention, awareness, arbiter, cognitive_state, domains as domain_routing, emotion, evolution, graph, grounding as semantic_grounding, homeostasis, initiative, integrity, llm, log, observability, perspective, portable, presence, self_model, sensory, sleep as sleep_mod, store, tasks as tasks_mod, tools, will as will_mod, workspace as shared_workspace
from brain.build_info import source_fingerprint
from brain.pipeline import _INJECT_NOTE, _PREAMBLE, _TOOLS_NOTE
from brain.documents import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    extract_pdf_text,
    ingest_document_text,
    is_supported_document,
)
from brain.vision import VisionUnavailable, describe_image
from brain.embeddings import EmbeddingUnavailable, embed_document, text_only_embedding
from brain.execution import context_aware_execution
from brain.memory import MemoryUnit
from brain.pipeline import answer, capture, reinforce_recall
from brain import reflect as reflect_mod
from brain.retriever import retrieve_weighted
from brain.summarize import compact_old_episodes, summarize_category

app = FastAPI(title="Neural Map")
WEB_DIR = Path(__file__).resolve().parent / "web"
SOURCE_FINGERPRINT = source_fingerprint()

_TOKEN_COOKIE = "nm_token"


def _require_token_beyond_loopback() -> None:
    """Refuse to start a non-loopback server without a shared secret. The token
    middleware below is a no-op when NEURAL_MAP_TOKEN is empty, which is safe
    only while nothing but this machine can reach the port; on 0.0.0.0 it would
    hand memories, state-changing endpoints, and tool approvals to any LAN
    client."""
    if config.SERVER_TOKEN:
        return
    host = (config.SERVER_HOST or "").strip()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback:
        raise SystemExit(
            f"refusing to start: NEURAL_MAP_HOST={host or '<empty>'} accepts connections "
            "from other machines but NEURAL_MAP_TOKEN is empty, so every endpoint would "
            "be unauthenticated. Set NEURAL_MAP_TOKEN or bind to 127.0.0.1."
        )


_require_token_beyond_loopback()


@app.middleware("http")
async def require_body_size_limit(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
            except ValueError:
                return JSONResponse({"detail": "invalid content-length"}, status_code=400)
            if size > config.MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {"detail": f"request body exceeds {config.MAX_REQUEST_BODY_BYTES} bytes"},
                    status_code=413,
                )
        received = 0
        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await request._receive()
            if message.get("type") == "http.disconnect":
                break
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body") or b""
            received += len(chunk)
            if received > config.MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {"detail": f"request body exceeds {config.MAX_REQUEST_BODY_BYTES} bytes"},
                    status_code=413,
                )
            chunks.append(chunk)
            more_body = bool(message.get("more_body", False))
        body = b"".join(chunks)
        sent = False

        async def replay_body():
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = replay_body
    return await call_next(request)


def _split_host_port(host_header: str) -> tuple[str, int | None]:
    """(hostname, port) from a Host/authority value; port is None when omitted.
    '127.0.0.1:8000' -> ('127.0.0.1', 8000), '[::1]:8000' -> ('::1', 8000),
    'localhost' -> ('localhost', None)."""
    host = host_header.strip()
    port_str = ""
    if host.startswith("["):                       # bracketed IPv6, e.g. [::1]:8000
        end = host.find("]")
        if end != -1:
            name = host[1:end]
            rest = host[end + 1:]
            port_str = rest[1:] if rest.startswith(":") else ""
        else:
            name = host[1:]
    elif host.count(":") == 1:                      # host:port (a bare IPv6 has >1 colon)
        name, _, port_str = host.rpartition(":")
    else:                                           # bare host or bare IPv6
        name = host
    try:
        return name, (int(port_str) if port_str else None)
    except ValueError:
        return name, None


def _host_name(host_header: str) -> str:
    """The hostname from a Host header, with the optional :port (and IPv6 [..])
    stripped. '127.0.0.1:8000' -> '127.0.0.1', '[::1]:8000' -> '::1'."""
    return _split_host_port(host_header)[0]


def _name_allowed(name: str) -> bool:
    """Shared allow rule for Host and Origin hostnames: IP literals, 'localhost',
    '*.local' mDNS names, and NEURAL_MAP_ALLOWED_HOSTS. Everything the web UI is
    legitimately served from matches one of these; a registered third-party
    domain (the rebinding / CSRF vector) does not."""
    if name == "localhost" or name.endswith(".local") or name in config.SERVER_ALLOWED_HOSTS:
        return True
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _host_allowed(host_header: str | None) -> bool:
    """Anti-DNS-rebinding gate: accept only Host values that cannot be aimed at
    this machine by an external DNS zone — IP literals, 'localhost', '*.local'
    mDNS names, and any host explicitly listed in NEURAL_MAP_ALLOWED_HOSTS. A
    rebinding attack carries a registered domain name as Host, which fails here.

    An absent Host header (HTTP/1.0, raw sockets) is allowed: browsers — the only
    rebinding vector — always send one, so its absence isn't a rebind attempt."""
    if host_header is None:
        return True
    name = _host_name(host_header).lower()
    if not name:
        return True
    return _name_allowed(name)


@app.middleware("http")
async def require_allowed_host(request: Request, call_next):
    if not _host_allowed(request.headers.get("host")):
        return JSONResponse({"detail": "invalid host"}, status_code=421)
    return await call_next(request)


def _origin_allowed(origin_header: str | None, host_header: str | None = None) -> bool:
    """Anti-CSRF gate for every request on the (often token-less loopback)
    server. A cross-site "simple request" — e.g. a malicious page running
    fetch("http://127.0.0.1:8000/api/...", {method: "POST", headers:
    {"Content-Type": "text/plain"}, body: JSON...}) — skips the CORS preflight,
    and the Host allowlist doesn't help because the browser targets this server
    directly (Host is 127.0.0.1). Cross-site GETs matter too: the page can't
    read the response, but the handler still runs (export, model inspection,
    lazy collection creation). The browser always stamps the attacking page's
    OWN origin, which a web page cannot forge.

    A matching NAME is not enough. Unlike the Host gate — where accepting every
    IP literal is safe because a rebinding attacker can only wield a DNS name —
    a CSRF page can be served straight from a public IP literal (https://8.8.8.8)
    or another loopback port, and it stamps that origin honestly. So the Origin
    must be the server's OWN origin: an http(s) scheme, an allowed host name, AND
    a host:port equal to the request's Host (a header the page cannot forge).

    An absent Origin header is allowed: curl, launchd jobs, and non-browser
    clients don't send one, and they aren't confined by the browser sandbox
    anyway. "null" (sandboxed iframes, file:// pages) is rejected — the real UI
    is always served from an http origin."""
    if origin_header is None:
        return True
    origin = origin_header.strip()
    if not origin:
        return True
    if origin.lower() == "null":
        return False
    parsed = urlparse(origin)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:            # ftp://, file://, … are never the UI
        return False
    try:
        name, origin_port = (parsed.hostname or "").lower(), parsed.port
    except ValueError:                             # out-of-range port in the Origin
        return False
    if not name or not _name_allowed(name):
        return False
    # Same-origin check: the Origin's host:port must equal the request's Host.
    # This is what rejects https://8.8.8.8 and http://127.0.0.1:<other-port>,
    # both of which pass the name allowlist. Browsers always send Host; its
    # absence means a non-browser client (unconfined regardless), so skip.
    if host_header is None:
        return True
    host_name, host_port = _split_host_port(host_header)
    default_port = 443 if scheme == "https" else 80
    return name == host_name.lower() and (origin_port or default_port) == (host_port or default_port)


def _fetch_metadata_allowed(method: str, headers) -> bool:
    """Reject cross-site requests using Fetch Metadata when the browser sends it.

    The Origin gate cannot see no-cors subresource GETs (<img src=...>,
    <script src=...>): browsers omit the Origin header on those, yet the
    request still executes server-side and can trigger expensive or
    lazily-mutating work. Sec-Fetch-Site is stamped by the browser and cannot
    be forged by page script. "same-site" is rejected along with "cross-site"
    because another port on this host is a different origin, and the UI only
    ever talks to its own origin. Absent header (curl, older browsers, launchd
    jobs) and "none" (direct navigation) pass. The one cross-site shape allowed
    is a top-level GET document navigation: it lands as the user's own visible
    tab (links to the app keep working) and the initiating page can never read
    the response."""
    site = headers.get("sec-fetch-site")
    if site is None:
        return True
    site = site.strip().lower()
    if site in {"", "same-origin", "none"}:
        return True
    mode = (headers.get("sec-fetch-mode") or "").strip().lower()
    dest = (headers.get("sec-fetch-dest") or "").strip().lower()
    return method == "GET" and mode == "navigate" and dest == "document"


@app.middleware("http")
async def require_allowed_origin(request: Request, call_next):
    # Enforced on EVERY method, not just state-changing ones: on a tokenless
    # loopback server, a hostile page's cross-origin GET still executes the
    # handler (model inspection, export, stats, lazy collection creation) even
    # though the browser stops the page from reading the response.
    if not _origin_allowed(request.headers.get("origin"), request.headers.get("host")):
        return JSONResponse({"detail": "cross-site request blocked"}, status_code=403)
    if not _fetch_metadata_allowed(request.method, request.headers):
        return JSONResponse({"detail": "cross-site request blocked"}, status_code=403)
    return await call_next(request)


def _supplied_token(request: Request) -> str | None:
    """Pull the auth token from header, bearer, or cookie."""
    header = request.headers.get("X-Auth-Token")
    if header:
        return header
    authz = request.headers.get("Authorization", "")
    if authz.lower().startswith("bearer "):
        return authz[7:].strip()
    return request.cookies.get(_TOKEN_COOKIE)


def _clean_token_url(request: Request) -> str:
    params = [(k, v) for k, v in request.query_params.multi_items() if k != "token"]
    return str(request.url.replace(query=urlencode(params)))


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Shared-secret gate. No-op unless NEURAL_MAP_TOKEN is set (see config). When
    set, rejects any request that doesn't carry the token, and stamps a cookie so a
    browser that authenticated via ?token= stays logged in on later navigations."""
    token = config.SERVER_TOKEN
    if not token:
        # The import-time guard validates config.SERVER_HOST, but uvicorn can be
        # bound elsewhere directly (uvicorn server:app --host 0.0.0.0), which
        # never consults that setting. A request from a non-loopback peer while
        # no token is configured means the port is exposed; refuse it rather
        # than serve memories, mutations, and approvals unauthenticated. A peer
        # that isn't an IP literal (in-process test client) can't arrive over
        # TCP, so it stays confined to this machine and passes.
        peer = request.client.host if request.client else ""
        try:
            remote = not ipaddress.ip_address(peer).is_loopback
        except ValueError:
            remote = False
        if remote:
            return JSONResponse(
                {
                    "detail": "server has no NEURAL_MAP_TOKEN configured; refusing "
                    "non-loopback clients. Set NEURAL_MAP_TOKEN to serve the LAN."
                },
                status_code=403,
            )
        return await call_next(request)
    query_token = request.query_params.get("token")
    if query_token is not None:
        if not (query_token and hmac.compare_digest(query_token.encode(), token.encode())):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        if request.method != "GET":
            return JSONResponse({"detail": "query token is only accepted for browser bootstrap"}, status_code=400)
        response = RedirectResponse(_clean_token_url(request), status_code=303)
        response.set_cookie(
            _TOKEN_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 365
        )
        return response
    supplied = _supplied_token(request)
    if not (supplied and hmac.compare_digest(supplied.encode(), token.encode())):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    response = await call_next(request)
    if request.cookies.get(_TOKEN_COOKIE) != token:
        # Plain HTTP on the LAN, so no Secure flag (it would block the cookie over
        # http://). httponly keeps it out of reach of page scripts.
        response.set_cookie(
            _TOKEN_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 365
        )
    return response


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatHistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)


class RetrievalSettings(StrictModel):
    top_k: int = Field(default=config.TOP_K, ge=1, le=25)
    semantic: float = Field(default=config.W_SIM, ge=0, le=5)
    keyword: float = Field(default=config.W_KEYWORD, ge=0, le=5)
    recency: float = Field(default=config.W_RECENCY, ge=0, le=5)
    importance: float = Field(default=config.W_IMPORTANCE, ge=0, le=5)
    graph: float = Field(default=config.W_GRAPH, ge=0, le=5)
    sensory: float = Field(default=config.W_SENSORY, ge=0, le=5)
    category: str | None = Field(default=None, max_length=config.MAX_CATEGORY_CHARS)

    @field_validator("category", mode="before")
    @classmethod
    def _blank_category_is_all(cls, value):
        # The UI sends "" for "All categories"; treat it as no scope.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def require_signal(self) -> "RetrievalSettings":
        # Every signal now PRODUCES candidates: semantic, keyword, graph, and the
        # recency/importance pulls (most-recent / highest-importance memories). So
        # the only useless request is one where ALL weights are zero — no producer
        # runs and the retriever returns []. Reject just that up front.
        if self.semantic + self.keyword + self.graph + self.sensory + self.recency + self.importance <= 0:
            raise ValueError("at least one retrieval weight must be greater than zero")
        return self


class ChatReq(StrictModel):
    message: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)
    num_ctx: int | None = Field(default=config.DEFAULT_NUM_CTX, ge=config.MIN_NUM_CTX, le=config.MAX_NUM_CTX)
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=config.HISTORY_TURNS * 2,
    )
    activated_memory_ids: list[str] = Field(default_factory=list, max_length=10)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    # INITIATIVE LAYER: "quiet" restores the older reactive/reminder-only chat.
    initiative_mode: Literal["quiet", "helpful", "active", "coach"] | None = Field(default=None)
    memory_enabled: bool = True
    # Client-generated id that /api/chat/cancel can target to abort generation.
    request_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_complete_history_turns(self) -> "ChatReq":
        # ``message`` is the current user turn. ``history`` is exclusively the
        # completed transcript before it, so it must contain whole
        # user/assistant pairs. Keeping that contract explicit prevents a
        # client-side window slice from orphaning half a turn and silently
        # reducing the context that reaches the cognitive pipeline.
        for index, item in enumerate(self.history):
            expected = "user" if index % 2 == 0 else "assistant"
            if item.role != expected:
                raise ValueError(
                    "history must contain completed user/assistant pairs and "
                    "must not include the current message"
                )
        if len(self.history) % 2:
            raise ValueError(
                "history must contain completed user/assistant pairs and "
                "must not include the current message"
            )
        return self

    @field_validator("activated_memory_ids", mode="before")
    @classmethod
    def _clean_activated_memory_ids(cls, value):
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in value:
            memory_id = str(raw or "").strip()
            if not memory_id or len(memory_id) > 160 or memory_id in seen:
                continue
            seen.add(memory_id)
            cleaned.append(memory_id)
            if len(cleaned) >= 10:
                break
        return cleaned


class MemoryReq(StrictModel):
    text: str = Field(min_length=1, max_length=config.MAX_MEMORY_CHARS)
    category: str = Field(default="note", min_length=1, max_length=config.MAX_CATEGORY_CHARS)
    importance: int = Field(default=3, ge=1, le=5)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class PinReq(StrictModel):
    pinned: bool = True
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class EmotionReq(StrictModel):
    happiness: float | None = Field(default=None, ge=-1, le=1)
    stress: float | None = Field(default=None, ge=-1, le=1)
    confidence: float | None = Field(default=None, ge=-1, le=1)
    curiosity: float | None = Field(default=None, ge=-1, le=1)
    reason: str | None = Field(default=None, max_length=240)


class HomeostasisRuleReq(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=80)
    target: str = Field(min_length=1, max_length=80)
    weight: float = Field(ge=-2, le=2)
    threshold: float = Field(default=0.5, ge=0, le=1)
    release: float | None = Field(default=None, ge=0, le=1)
    note: str = Field(default="", max_length=240)


class HomeostasisRulesReq(StrictModel):
    rules: list[HomeostasisRuleReq] | None = None


class RecallReq(StrictModel):
    query: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class ActionPackageReq(StrictModel):
    initiative_id: str | None = Field(default=None, max_length=180)
    text: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)
    kind: str | None = Field(default=None, max_length=80)
    priority: int = Field(default=4, ge=1, le=5)
    source_ids: list[str] = Field(default_factory=list, max_length=10)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class TaskEnqueueReq(StrictModel):
    prompt: str = Field(min_length=1, max_length=config.MAX_MESSAGE_CHARS)


class TaskRunReq(StrictModel):
    id: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class SummReq(StrictModel):
    category: str = Field(min_length=1, max_length=config.MAX_CATEGORY_CHARS)
    replace: bool = False
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class IngestReq(StrictModel):
    filename: str = Field(min_length=1, max_length=config.MAX_FILENAME_CHARS)
    text: str | None = Field(default=None, min_length=1, max_length=config.MAX_DOCUMENT_CHARS)
    data_base64: str | None = Field(default=None, min_length=1, max_length=config.MAX_UPLOAD_BASE64_CHARS)
    category: str | None = Field(default=None, min_length=1, max_length=config.MAX_CATEGORY_CHARS)
    importance: int = Field(default=3, ge=1, le=5)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)
    vision_model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)

    @field_validator("filename")
    @classmethod
    def require_supported_filename(cls, value: str) -> str:
        name = Path(value).name
        if not is_supported_document(name):
            raise ValueError("filename must be a .md, .markdown, .txt, .pdf, image, or audio file")
        return name


class DeleteSourceReq(StrictModel):
    source: str = Field(min_length=1, max_length=config.MAX_DOCUMENT_CHARS)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class PruneReq(StrictModel):
    older_than_days: float | None = Field(default=None, ge=0)
    max_importance: int | None = Field(default=None, ge=1, le=5)
    dry_run: bool = False
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class CompactReq(StrictModel):
    older_than_days: float | None = Field(default=None, ge=0)
    max_episodes: int | None = Field(default=None, ge=1, le=500)
    dry_run: bool = False
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class SleepReq(StrictModel):
    older_than_days: float | None = Field(default=None, ge=0)
    max_archive: int | None = Field(default=None, ge=0, le=1000)
    max_dreams: int | None = Field(default=None, ge=0, le=20)
    dry_run: bool = False
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class WillReq(StrictModel):
    dry_run: bool = False
    mode: Literal["quiet", "helpful", "active", "coach"] | None = Field(default=None)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class WillOutcomeReq(StrictModel):
    initiative_id: str = Field(min_length=1, max_length=100)
    outcome: Literal["interesting", "useful", "neutral", "unhelpful"]
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class ReflectReq(StrictModel):
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class EvolutionProposalReq(StrictModel):
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class EvolutionDecisionReq(StrictModel):
    proposal_id: str = Field(min_length=1, max_length=80)
    reason: str | None = Field(default=None, max_length=1000)


class EvolutionSnapshotRestoreReq(StrictModel):
    snapshot_id: str = Field(min_length=1, max_length=80)


class EvolutionLogReq(StrictModel):
    entry: str = Field(min_length=1, max_length=5000)


class InitiativeDismissReq(StrictModel):
    ids: list[str] = Field(min_length=1, max_length=25)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class InitiativeContactReq(StrictModel):
    ids: list[str] = Field(default_factory=list, max_length=25)
    mode: Literal["quiet", "helpful", "active", "coach"] | None = Field(default=None)
    model: str | None = Field(default=None, min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)


class CognitiveGoalReq(StrictModel):
    id: str | None = Field(default=None, min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=1000)
    status: Literal["active", "blocked", "done", "dropped"] = "active"
    priority: int = Field(default=3, ge=1, le=5)
    due: str | None = Field(default=None, max_length=80)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    next_action: str | None = Field(default=None, max_length=1000)
    blocked_by: list[str] = Field(default_factory=list, max_length=20)


class CognitiveGoalStatusReq(StrictModel):
    status: Literal["active", "blocked", "done", "dropped"]
    blocked_by: list[str] | None = Field(default=None, max_length=20)


class CognitiveWorldReq(StrictModel):
    id: str | None = Field(default=None, min_length=1, max_length=80)
    key: str = Field(min_length=1, max_length=160)
    value: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(default=0.7, ge=0, le=1)
    source: str | None = Field(default=None, max_length=500)
    expires: str | None = Field(default=None, max_length=80)
    status: Literal["current", "stale", "archived"] = "current"


class CognitiveBeliefReq(StrictModel):
    id: str | None = Field(default=None, min_length=1, max_length=80)
    claim: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(default=0.5, ge=0, le=1)
    source: str | None = Field(default=None, max_length=500)
    last_verified: str | None = Field(default=None, max_length=80)
    expires: str | None = Field(default=None, max_length=80)
    status: Literal["active", "stale", "rejected"] = "active"


class PendingActionDecisionReq(StrictModel):
    action_id: str = Field(min_length=1, max_length=80)


class ToolPermissionReq(StrictModel):
    tool: str = Field(min_length=1, max_length=80)
    enabled: bool


def _ollama_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Ollama is unavailable: {exc}")


def _is_ollama_error(exc: Exception) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError, ollama.RequestError, ollama.ResponseError))


def _append_warning(current: str | None, message: str) -> str:
    return f"{current}; {message}" if current else message


def _backup_before(reason: str, model: str | None = None) -> str | None:
    if _store_count(model) <= 0:
        return None
    return str(portable.save_backup(reason, model=model))


def _normalize_history(req: ChatReq) -> list[dict]:
    # ChatReq validates the protocol at the boundary. Do not silently repair,
    # collapse, or truncate it here: every accepted prior turn reaches the
    # intelligence layer unchanged.
    return [item.model_dump() for item in req.history]


_APPROVE_PENDING_PREFIXES = (
    "approve",
    "approved",
    "i approve",
    "yes",
    "yep",
    "yeah",
    "ok",
    "okay",
    "continue",
    "proceed",
    "go ahead",
    "run it",
    "do it",
)
_REJECT_PENDING_PREFIXES = (
    "reject",
    "decline",
    "cancel",
    "no",
    "stop",
    "do not",
    "don't",
)
_PENDING_NEGATION_TERMS = ("don't", "do not", "not ", "never", "cancel", "reject", "stop", "wait", "hold")
_PENDING_ACTION_ID_RE = re.compile(r"\bact_[0-9a-f]{12}\b", re.IGNORECASE)
_APPROVAL_CORRECTION_RE = re.compile(
    r"\b(?:i\s+(?:already\s+|actually\s+)?approved|i\s+did\s+approve|clicked\s+approve)\b",
    re.IGNORECASE,
)


def _chat_pending_decision(message: str) -> Literal["approve", "reject"] | None:
    text = " ".join((message or "").strip().lower().split())
    if not text:
        return None
    # A user correcting the system ("no, I actually approved it") is affirming
    # the action, not rejecting it merely because the sentence begins with no.
    if _APPROVAL_CORRECTION_RE.search(text) and not re.search(r"\b(?:did not|didn't|never)\s+approve", text):
        return "approve"
    for prefix in _REJECT_PENDING_PREFIXES:
        if text == prefix or text.startswith(prefix + " ") or text.startswith(prefix + ".") or text.startswith(prefix + ","):
            return "reject"
    for prefix in _APPROVE_PENDING_PREFIXES:
        if text == prefix or text.startswith(prefix + " ") or text.startswith(prefix + ".") or text.startswith(prefix + ","):
            if any(term in text for term in _PENDING_NEGATION_TERMS):
                return None
            return "approve"
    return None


def _resolve_pending_from_chat(message: str) -> dict | None:
    decision = _chat_pending_decision(message)
    if decision is None:
        return None
    match = _PENDING_ACTION_ID_RE.search(message or "")
    pending_actions = tools.list_pending()
    if match is None:
        if not pending_actions:
            return None
        if len(pending_actions) > 1:
            return {
                "reply": (
                    f"I found {len(pending_actions)} pending actions, so that answer is ambiguous. "
                    "Open Actions and approve the intended item, or include its act_… ID."
                ),
                "tool_results": [],
                "tools_used": [],
                "pending_decision": "ambiguous",
            }
        action = pending_actions[0]
        action_id = str(action.get("id") or "").lower()
    else:
        action_id = match.group(0).lower()
        action = next((a for a in pending_actions if str(a.get("id") or "").lower() == action_id), None)
    if action is None:
        return None
    if decision == "reject":
        result = tools.reject_pending(action_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        resolved = result.get("action") or action
        return {
            "reply": f"Rejected {resolved.get('tool') or 'pending action'} ({action_id}).",
            "tool_results": [],
            "tools_used": [],
            "pending_decision": "rejected",
        }
    # A package whose claimed source initiative could not be verified against
    # the store has caller-asserted content: the blind one-word approval is
    # refused and the preview must be read in the panel. Checked before the
    # generic taint gate so the more specific explanation wins.
    if action_packages.UNVERIFIED_TAINT in (action.get("taint_sources") or []):
        return {
            "reply": (
                f"Not approving {action.get('tool')} ({action_id}) from chat: its claimed source "
                "initiative could not be verified against stored initiatives, so the content is "
                "unconfirmed. Review the full preview in the Actions panel and approve it there "
                "if it looks right."
            ),
            "tool_results": [],
            "tools_used": [],
            "pending_decision": "blocked_unverified_initiative",
        }
    # A tainted action means untrusted content (retrieved memories, web results,
    # file reads, calendar events) was in front of the model when it proposed
    # the action — exactly the prompt-injection shape. The one-word chat
    # approval never shows the preview, so route EVERY tainted action through
    # the Actions panel where it's visible. That includes local writes and
    # destructive memory/identity changes: audit and undo exist, but they do
    # not replace previewing a possibly injection-influenced action before it
    # runs.
    external_calendar_tools = {
        "create_google_calendar_event",
        "update_google_calendar_event",
        "delete_google_calendar_event",
    }
    if action.get("taint_sources"):
        sources = ", ".join(action["taint_sources"])
        if action.get("policy") == tools.EXTERNAL_SEND:
            decision = "blocked_tainted_external_send"
        elif action.get("policy") == tools.EXTERNAL_WRITE or action.get("tool") in external_calendar_tools:
            decision = "blocked_tainted_external_write"
        else:
            decision = "blocked_tainted_action"
        return {
            "reply": (
                f"Not approving {action.get('tool')} ({action_id}) from chat: it was proposed while "
                f"untrusted content was in context ({sources}). Review the full preview in the "
                "Actions panel and approve it there if it looks right."
            ),
            "tool_results": [],
            "tools_used": [],
            "pending_decision": decision,
        }
    result = tools.approve_pending(action_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    resolved = result.get("action") or action
    tool_name = str(resolved.get("tool") or "pending action")
    tool_result = result.get("result") or {}
    if result.get("execution_failed"):
        return {
            "reply": (
                f"Approval accepted for {tool_name} ({action_id}), but execution failed.\n"
                f"{json.dumps(tool_result, indent=2)}"
            ),
            "tool_results": [{"tool": tool_name, "result": tool_result}],
            "tools_used": [tool_name],
            "pending_decision": "execution_failed",
        }
    return {
        "reply": f"Approved {tool_name} ({action_id}).\n{json.dumps(tool_result, indent=2)}",
        "tool_results": [{"tool": tool_name, "result": tool_result}],
        "tools_used": [tool_name],
        "pending_decision": "approved",
    }


def _retrieve(
    question: str,
    settings: RetrievalSettings,
    model: str | None = None,
    domains: set[str] | None = None,
) -> list[dict]:
    return retrieve_weighted(
        question,
        top_k=settings.top_k,
        w_sim=settings.semantic,
        w_keyword=settings.keyword,
        w_recency=settings.recency,
        w_importance=settings.importance,
        w_graph=settings.graph,
        w_sensory=settings.sensory,
        model=model,
        category=settings.category,
        domains=domains,
    )


def _activated_memories(ids: list[str], model: str | None = None) -> list[dict]:
    if not ids:
        return []
    wanted = {memory_id: index for index, memory_id in enumerate(ids)}
    rows: list[dict] = []
    for rec in store.list_records_all(model):
        memory_id = rec.get("id")
        if memory_id not in wanted:
            continue
        meta = rec.get("meta") or {}
        # Same current-truth invariant as every retrieval producer: a memory
        # retired by mark_memory_wrong / merge / belief-updating cannot be
        # resurrected into the prompt by a stale activation chip.
        if meta.get("superseded"):
            continue
        rows.append(
            {
                "id": memory_id,
                "text": rec.get("text") or "",
                "category": meta.get("category"),
                "source": meta.get("source"),
                "date": meta.get("date"),
                "importance": meta.get("importance"),
                "source_type": meta.get("source_type", "legacy"),
                "pinned": bool(meta.get("pinned", False)),
                "score": 1.0,
                "similarity": 0.0,
                "keyword": 0.0,
                "recency": 0.0,
                "activated": True,
                "mode": rec.get("mode"),
            }
        )
    rows.sort(key=lambda row: wanted.get(row.get("id"), 0))
    return rows


def _merge_activated_memories(retrieved: list[dict], activated: list[dict]) -> list[dict]:
    if not activated:
        return retrieved
    seen = {m.get("id") for m in activated if m.get("id")}
    return activated + [m for m in retrieved if not m.get("id") or m.get("id") not in seen]


def _upload_source(filename: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"upload:{digest}:{filename}"


def _stale_upload_sources(filename: str, keep: str, model: str | None) -> list[str]:
    """Sources of PREVIOUS uploads of the same filename (different content hash).

    The upload source key is content-hashed, so an edited file re-uploaded under
    the same name gets a NEW source — without this sweep its old chunks would
    linger and retrieval could keep surfacing the outdated version. (The CLI keys
    on file path, so it replaces naturally; this mirrors that for web uploads.)"""
    suffix = f":{filename}"
    stale: set[str] = set()
    for rec in store.list_records_all(model):
        src = (rec.get("meta") or {}).get("source")
        if src and src != keep and src.startswith("upload:") and src.endswith(suffix):
            stale.add(src)
    return sorted(stale)


def _decode_upload(req: IngestReq, kind: str) -> bytes:
    if not req.data_base64:
        raise HTTPException(status_code=422, detail=f"{kind} uploads require data_base64")
    try:
        data = base64.b64decode(req.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="data_base64 is not valid base64") from exc
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{kind} upload exceeds {config.MAX_UPLOAD_BYTES} bytes")
    return data


def _ingest_payload(req: IngestReq) -> tuple[str, str, str | None, bytes | None]:
    suffix = Path(req.filename).suffix.lower()
    if suffix == ".pdf":
        data = _decode_upload(req, "PDF")
        text = extract_pdf_text(data)
        if len(text) > config.MAX_DOCUMENT_CHARS:
            raise HTTPException(status_code=413, detail=f"extracted PDF text exceeds {config.MAX_DOCUMENT_CHARS} characters")
        return text, _upload_source(req.filename, data), None, None
    if suffix in IMAGE_SUFFIXES:
        data = _decode_upload(req, "image")
        # Caption + OCR via a local vision model; raises VisionUnavailable (→400)
        # if no vision model is pulled. Keyed on the raw image bytes so the same
        # image re-uploaded cleanly replaces its description.
        text = describe_image(data, model=req.vision_model)
        return text, _upload_source(req.filename, data), "image", data
    if suffix in AUDIO_SUFFIXES:
        data = _decode_upload(req, "audio")
        return sensory.audio_summary(data, req.filename), _upload_source(req.filename, data), "audio", data
    if req.text is None:
        raise HTTPException(status_code=422, detail="text is required for text documents")
    return req.text, _upload_source(req.filename, req.text.encode("utf-8")), None, None


def _store_count(model: str | None = None) -> int:
    return store.count_all(model)


def _graph_collections(model: str | None) -> list[str | None]:
    return store.graph_collection_names(model)


def _delete_memory_all(memory_id: str, model: str | None = None) -> None:
    detail = _memory_detail(memory_id, model=model)
    source = (detail or {}).get("source")
    source_count = _source_count(source, model=model) if source else 0
    store.delete_all([memory_id], model=model)
    # A deletion must take the memory's graph edges with it; leaving them behind
    # turns every graph-backed delete into a dangling_graph_fact integrity error
    # that repair has to mop up after the fact.
    graph.delete_fact_ids([memory_id], collections=_graph_collections(model))
    if source and source_count <= 1:
        sensory.delete_source(source)


def _delete_source_all(source: str, model: str | None = None) -> None:
    ids = [
        rec["id"]
        for rec in store.list_records_all(model)
        if (rec.get("meta") or {}).get("source") == source
    ]
    store.delete_source_all(source, model=model)
    if ids:
        graph.delete_fact_ids(ids, collections=_graph_collections(model))
    sensory.delete_source(source)


def _memory_rows(model: str | None = None) -> list[dict]:
    rows = []
    for rec in store.list_records_all(model):
        meta = rec["meta"]
        rows.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "category": meta.get("category"),
                "source": meta.get("source"),
                "date": meta.get("date"),
                "importance": meta.get("importance"),
                "source_type": meta.get("source_type", "legacy"),
                "pinned": bool(meta.get("pinned", False)),
                "access_count": meta.get("access_count", 0),
                "last_accessed": meta.get("last_accessed"),
                "status": meta.get("status"),
                "due": meta.get("due"),
                "superseded": bool(meta.get("superseded", False)),
                "mode": rec["mode"],
            }
        )
    rows.sort(key=lambda row: (bool(row.get("pinned", False)), row.get("date") or ""), reverse=True)
    return rows


def _source_count(source: str, model: str | None = None) -> int:
    return sum(1 for rec in store.list_records_all(model) if (rec.get("meta") or {}).get("source") == source)


def _memory_detail(memory_id: str, model: str | None = None) -> dict | None:
    for rec in store.list_records_all(model):
        if rec["id"] == memory_id:
            meta = rec["meta"] or {}
            return {
                "id": rec["id"],
                "text": rec["text"],
                "mode": rec["mode"],
                "metadata": meta,
                "category": meta.get("category"),
                "source": meta.get("source"),
                "source_type": meta.get("source_type", "legacy"),
                "pinned": bool(meta.get("pinned", False)),
            }
    return None


def _default_prompt_parts() -> dict[str, str]:
    return {"preamble": _PREAMBLE, "tools_note": _TOOLS_NOTE, "inject_note": _INJECT_NOTE}


@app.get("/")
def index():
    # no-cache (revalidate, not "never store"): a stale cached copy means stale
    # UI until a hard refresh. With ETag revalidation an unchanged file still
    # answers 304, so this costs nothing.
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/app.js")
def app_js():
    return FileResponse(
        WEB_DIR / "app.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/models")
def models():
    """Dropdown source: every model Ollama can run, plus the configured default."""
    try:
        names = llm.list_models()
    except Exception as exc:
        if _is_ollama_error(exc):
            return {"models": [config.CHAT_MODEL], "default": config.CHAT_MODEL, "warning": str(exc)}
        raise
    if config.CHAT_MODEL not in names:
        names.append(config.CHAT_MODEL)
    return {"models": sorted(set(names)), "default": config.CHAT_MODEL}


@app.get("/api/build")
def build_info():
    return {"source_fingerprint": SOURCE_FINGERPRINT}


@app.get("/api/model_info")
def model_info(name: str = Query(min_length=1, max_length=config.MAX_MODEL_NAME_CHARS)):
    """Max context window for a model, so the UI can hint/clamp the ctx input."""
    return {"name": name, "max_context": llm.model_max_context(name)}


@app.get("/api/stats")
def stats(model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS)):
    return {"count": _store_count(model)}


@app.get("/api/export")
def export(
    format: Literal["json", "md"] = Query(default="json"),
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
    category: str | None = Query(default=None, max_length=config.MAX_CATEGORY_CHARS),
    source: str | None = Query(default=None, max_length=config.MAX_DOCUMENT_CHARS),
    pinned: bool | None = Query(default=None),
):
    """Download a portable backup of all memories + graph (re-importable via backup.py).

    Returned as a file attachment so the browser saves it. Embeddings are not
    included — they're recomputed on import (see brain/portable.py)."""
    if not isinstance(category, str):
        category = None
    if not isinstance(source, str):
        source = None
    if not isinstance(pinned, bool):
        pinned = None
    data = portable.export_data(model=model)
    if category or source or pinned is not None:
        payload = data["neural_map_export"]
        rows = payload["memories"]
        if category:
            rows = [m for m in rows if m.get("category") == category]
        if source:
            rows = [m for m in rows if m.get("source") == source]
        if pinned is not None:
            rows = [m for m in rows if bool(m.get("pinned", False)) is pinned]
        ids = {m.get("id") for m in rows}
        payload["memories"] = rows
        payload["count"] = len(rows)
        payload["graph"] = [e for e in payload.get("graph", []) if e.get("fact_id") in ids]
    if format == "md":
        body, media, ext = portable.to_markdown(data), "text/markdown", "md"
    else:
        import json as _json

        body, media, ext = _json.dumps(data, ensure_ascii=False, indent=2), "application/json", "json"
    stamp = data["neural_map_export"]["exported_at"][:10]
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="neural_map_{stamp}.{ext}"'},
    )


@app.get("/api/health")
def health(model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS)):
    """One-shot diagnostics: is the brain wired up and awake?

    Answers "why is my brain quiet today?" without grepping logs — Ollama
    reachability, whether the chat/embed models are actually pulled, how many
    memories live in each store (vector vs. text-only fallback), the graph size,
    and which optional features are on. Deliberately never raises: an unreachable
    Ollama is a reported state, not a 500, so the page still loads when it's down.
    `available` is None (unknown) rather than False when Ollama can't be reached.
    """
    chat_model = model or config.CHAT_MODEL
    embed_model = config.embedding_model(model)

    ollama_reachable = True
    ollama_error: str | None = None
    available: list[str] = []
    try:
        available = llm.list_models()
    except Exception as exc:
        if not _is_ollama_error(exc):
            raise
        ollama_reachable = False
        ollama_error = str(exc)

    chat_available = (chat_model in available) if ollama_reachable else None
    embed_available = (embed_model in available) if ollama_reachable else None

    vector_count = store.count(model, text_only=False)
    text_only_count = store.count(model, text_only=True)
    graph_stats = graph.stats()
    memories = store.list_records_all(model)
    pinned_count = sum(1 for rec in memories if (rec.get("meta") or {}).get("pinned"))
    source_types: dict[str, int] = {}
    for rec in memories:
        st = (rec.get("meta") or {}).get("source_type", "legacy")
        source_types[st] = source_types.get(st, 0) + 1
    edge_rows = graph.export_edges()
    ids = {rec["id"] for rec in memories}
    scoped_collections = {store.collection_name(model), store.collection_name(model, text_only=True), None}
    scoped_edges = [e for e in edge_rows if e.get("collection") in scoped_collections]
    resolving_edges = sum(1 for e in scoped_edges if e.get("fact_id") in ids)

    return {
        "ok": ollama_reachable and bool(chat_available) and bool(embed_available),
        "ollama": {"reachable": ollama_reachable, "error": ollama_error},
        "chat_model": {"name": chat_model, "available": chat_available},
        "embed_model": {"name": embed_model, "available": embed_available},
        "memories": {
            "vector": vector_count,
            "text_only": text_only_count,
            "total": vector_count + text_only_count,
            "pinned": pinned_count,
            "source_types": source_types,
        },
        "graph": {
            "edges": graph_stats["edges"],
            "entities": graph_stats["entities"],
            "scoped_edges": len(scoped_edges),
            "resolving_edges": resolving_edges,
        },
        "features": {
            "cognition_mode": config.COGNITION_MODE,
            "web_search": config.WEB_SEARCH_ENABLED,
            "web_search_mode": config.WEB_SEARCH_MODE,
            "web_search_strict_context_guard": config.WEB_SEARCH_STRICT_CONTEXT_GUARD,
            "web_search_monitor_unsourced_context": config.WEB_SEARCH_MONITOR_UNSOURCED_CONTEXT,
            "web_search_unsourced_context_guard": config.WEB_SEARCH_BLOCK_WITH_UNSOURCED_CONTEXT,
            "web_search_doc_guard": config.WEB_SEARCH_BLOCK_WITH_DOC_CONTEXT,
            "web_search_user_derived_with_context": config.WEB_SEARCH_ALLOW_USER_DERIVED_WITH_CONTEXT,
            "market_quotes": config.MARKET_QUOTES_ENABLED,
            "initiative": config.INITIATIVE_ENABLED,
            "action_packages": config.ACTION_PACKAGES_ENABLED,
            "arbiter": True,
            "arbiter_grounding": config.ARBITER_GROUNDING_ENABLED,
            "tool_calling": config.TOOL_CALLING_ENABLED,
            "action_tools": config.ACTION_TOOLS_ENABLED,
            "emotion": config.EMOTION_ENABLED,
            "awareness": config.AWARENESS_ENABLED,
            "attention": config.ATTENTION_ENABLED,
            "workspace": config.WORKSPACE_ENABLED,
            "semantic_grounding": config.GROUNDING_ENABLED,
            "presence_ledger": config.PRESENCE_ENABLED,
            "homeostasis": config.HOMEOSTASIS_ENABLED,
            "self_model": config.SELF_MODEL_ENABLED,
            "intuition": config.INTUITION_ENABLED,
            "metacognition": config.METACOGNITION_ENABLED,
            "inner_council": config.INNER_COUNCIL_ENABLED,
            "evolution": config.EVOLUTION_ENABLED,
            "cognitive_state": config.COGNITIVE_STATE_ENABLED,
            "integrity": True,
            "observability": True,
            "sensory": config.SENSORY_EMBEDDINGS_ENABLED,
            "sleep": config.SLEEP_ENABLED,
            "will": config.WILL_ENABLED,
            "will_perspective": config.WILL_PERSPECTIVE_ENABLED,
        },
        "emotion": emotion.state(),
        "awareness": awareness.state(),
        "attention": attention.state(),
        "workspace": shared_workspace.state(),
        "semantic_grounding": semantic_grounding.state(),
        "presence": presence.state(),
        "homeostasis": homeostasis.state(),
        "self_model": self_model.state(),
        "cognitive_state": cognitive_state.prompt_snapshot(),
        "sleep": sleep_mod.status(),
        "will": will_mod.state(),
        "will_perspective": perspective.status(),
    }


@app.get("/api/cognitive_state")
def get_cognitive_state(include_inactive: bool = Query(default=True)):
    return cognitive_state.state(include_inactive=include_inactive)


@app.post("/api/cognitive_state/goal")
def upsert_cognitive_goal(req: CognitiveGoalReq):
    try:
        return cognitive_state.upsert_goal(
            text=req.text,
            goal_id=req.id,
            status=req.status,
            priority=req.priority,
            due=req.due,
            evidence=req.evidence,
            next_action=req.next_action,
            blocked_by=req.blocked_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/cognitive_state/goal/{goal_id}/status")
def set_cognitive_goal_status(goal_id: str, req: CognitiveGoalStatusReq):
    if not goal_id or len(goal_id) > 80:
        raise HTTPException(status_code=422, detail="invalid goal id")
    try:
        updated = cognitive_state.set_goal_status(goal_id, req.status, blocked_by=req.blocked_by)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="goal not found")
    return updated


@app.post("/api/cognitive_state/world")
def upsert_cognitive_world(req: CognitiveWorldReq):
    try:
        record = cognitive_state.upsert_world_item(
            key=req.key,
            value=req.value,
            item_id=req.id,
            confidence=req.confidence,
            source=req.source,
            expires=req.expires,
            status=req.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        # A world-model change propagates through the homeostasis coupling
        # immediately instead of waiting for the next chat turn.
        homeostasis.update_from_turn()
    except Exception as exc:
        log.fallback("homeostasis_world_propagation_failed", exc)
    return record


@app.post("/api/cognitive_state/belief")
def upsert_cognitive_belief(req: CognitiveBeliefReq):
    try:
        return cognitive_state.upsert_belief(
            claim=req.claim,
            belief_id=req.id,
            confidence=req.confidence,
            source=req.source,
            last_verified=req.last_verified,
            expires=req.expires,
            status=req.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/integrity")
def integrity_check(
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
    fallback_limit: int = Query(default=25, ge=0, le=200),
):
    return integrity.run(model=model, fallback_limit=fallback_limit)


@app.post("/api/integrity/repair")
def integrity_repair(
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
    dry_run: bool = Query(default=True),
):
    return integrity.repair(model=model, dry_run=dry_run)


def _mediated_proactive_context(model: str | None = None, mode: str | None = None) -> dict:
    reminders = reflect_mod.active_reminders(model=model)
    initiatives = initiative.active_initiatives(model=model, mode=mode)
    return arbiter.mediate(reminders, initiatives)


@app.get("/api/reminders")
def reminders(
    mode: Literal["quiet", "helpful", "active", "coach"] | None = Query(default=None),
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
):
    """Active proactive reminders (cheap; no LLM) for the UI banner."""
    mediated = _mediated_proactive_context(model=model, mode=mode)
    return {
        "reminders": mediated["reminders"],
        "conflicts": mediated["conflicts"],
        "grounding_reviews": mediated.get("grounding_reviews", []),
    }


@app.get("/api/initiative")
def initiatives(
    mode: Literal["quiet", "helpful", "active", "coach"] | None = Query(default=None),
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
):
    """Optional soft outreach/open-loop candidates.

    INITIATIVE LAYER: returns [] when mode=quiet or NEURAL_MAP_INITIATIVE=0,
    preserving the old reminder-only UI behavior.
    """
    mediated = _mediated_proactive_context(model=model, mode=mode)
    return {
        "mode": initiative.normalize_mode(mode),
        "initiatives": mediated["initiatives"],
        "conflicts": mediated["conflicts"],
        "grounding_reviews": mediated.get("grounding_reviews", []),
    }


@app.post("/api/initiative/dismiss")
def dismiss_initiatives(req: InitiativeDismissReq):
    learned = will_mod.record_outcomes_for_initiatives(req.ids, "unhelpful", model=req.model)
    return {"dismissed": initiative.dismiss(req.ids, model=req.model), "will_outcomes": learned}


@app.post("/api/initiative/contact")
def contact_initiatives(req: InitiativeContactReq):
    # IDs are accepted for symmetry/debuggability; the budget only needs to know
    # that non-empty initiative was actually shown to the user.
    items = [{"id": i} for i in req.ids]
    initiative.note_shown(items, mode=req.mode)
    return {"state": initiative.state()}


@app.post("/api/action_packages/from_initiative")
def action_package_from_initiative(req: ActionPackageReq):
    result = action_packages.queue_from_initiative(
        {
            "id": req.initiative_id,
            "text": req.text,
            "kind": req.kind,
            "priority": req.priority,
            "source_ids": req.source_ids,
        },
        model=req.model,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/will")
def will_status():
    return {"state": will_mod.state()}


@app.post("/api/will")
def will_review(req: WillReq):
    try:
        result = will_mod.review(model=req.model, dry_run=req.dry_run, mode=req.mode)
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    mediated = _mediated_proactive_context(model=req.model, mode=req.mode)
    result["active_reminders"] = mediated["reminders"]
    result["active_initiatives"] = mediated["initiatives"]
    result["conflicts"] = mediated["conflicts"]
    result["grounding_reviews"] = mediated.get("grounding_reviews", [])
    return result


@app.post("/api/will/outcome")
def will_outcome(req: WillOutcomeReq):
    result = will_mod.record_outcome(req.initiative_id, req.outcome, model=req.model)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    dismissed = 0
    if req.outcome == "unhelpful":
        dismissed = initiative.dismiss([req.initiative_id], model=req.model)
    return {**result, "dismissed": dismissed}


@app.get("/api/tasks")
def tasks_list(include_done: bool = Query(default=True)):
    """List background tasks (newest first)."""
    return {"tasks": tasks_mod.list_tasks(include_done=include_done)}


@app.post("/api/tasks")
def tasks_enqueue(req: TaskEnqueueReq):
    """Queue a natural-language task for the background runner. Read-only tools
    will auto-run; any write/send/delete is parked for approval."""
    try:
        task = tasks_mod.enqueue(req.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return task


@app.post("/api/tasks/run")
def tasks_run(req: TaskRunReq):
    """Run one queued task (a specific id, or the oldest queued if omitted)."""
    try:
        result = tasks_mod.run_one(task_id=req.id, model=req.model)
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    if result is None:
        return {"ran": False, "reason": "nothing to run"}
    return {"ran": True, "task": result}


@app.get("/api/briefing")
def briefing(
    mode: Literal["quiet", "helpful", "active", "coach"] | None = Query(default=None),
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
):
    """Daily-style briefing: due reminders, open loops, conflicts, and preferences.

    INITIATIVE LAYER: deterministic and no-LLM, so it is safe to call on page
    load or from a schedule. mode=quiet returns a reminder-only shape with no
    initiative candidates or learned communication preferences.
    """
    return initiative.briefing(model=model, mode=mode)


@app.get("/api/emotion")
def emotion_endpoint():
    return emotion.state()


@app.post("/api/emotion")
def set_emotion(req: EmotionReq):
    return emotion.set_state(req.model_dump(exclude_none=True))


@app.post("/api/emotion/reset")
def reset_emotion():
    return emotion.reset()


@app.get("/api/awareness")
def awareness_endpoint():
    return awareness.state()


@app.get("/api/attention")
def attention_endpoint():
    return attention.state()


@app.get("/api/workspace")
def workspace_endpoint():
    return shared_workspace.state()


@app.post("/api/workspace/reset")
def reset_workspace():
    return shared_workspace.reset()


@app.get("/api/grounding")
def grounding_endpoint():
    return semantic_grounding.state()


@app.post("/api/grounding/reset")
def reset_grounding():
    return semantic_grounding.reset()


@app.get("/api/presence")
def presence_endpoint():
    return presence.state()


@app.post("/api/presence/reset")
def reset_presence():
    return presence.reset()


@app.get("/api/homeostasis")
def homeostasis_endpoint():
    return homeostasis.state()


@app.post("/api/homeostasis/reset")
def reset_homeostasis():
    return homeostasis.reset()


@app.get("/api/homeostasis/rules")
def homeostasis_rules():
    return {"rules": homeostasis.effective_rules(), "defaults": homeostasis.DEFAULT_RULES}


@app.post("/api/homeostasis/rules")
def set_homeostasis_rules(req: HomeostasisRulesReq):
    """Replace the coupling rules (rules=null restores the shipped defaults).

    This endpoint is the approval gate: rules only ever change through an
    explicit call here, and the replacement is recorded in the adjustments log."""
    try:
        return homeostasis.set_rules(
            [rule.model_dump() for rule in req.rules] if req.rules is not None else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/awareness/reset")
def reset_awareness():
    return awareness.reset()


@app.get("/api/self_model")
def self_model_endpoint():
    return self_model.state()


@app.get("/api/perspective")
def perspective_endpoint():
    return perspective.status()


@app.post("/api/perspective/reflect")
def perspective_reflect_endpoint(req: ReflectReq):
    try:
        return perspective.reflect_recent(model=req.model)
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise


@app.post("/api/self_model/reset")
def reset_self_model():
    return self_model.reset()


@app.get("/api/retrieval_defaults")
def retrieval_defaults():
    data = RetrievalSettings().model_dump()
    data.update(
        {
            "top_k": evolution.hyperparam("TOP_K", data["top_k"]),
            "semantic": evolution.hyperparam("W_SIM", data["semantic"]),
            "keyword": evolution.hyperparam("W_KEYWORD", data["keyword"]),
            "recency": evolution.hyperparam("W_RECENCY", data["recency"]),
            "importance": evolution.hyperparam("W_IMPORTANCE", data["importance"]),
            "graph": evolution.hyperparam("W_GRAPH", data["graph"]),
            "sensory": evolution.hyperparam("W_SENSORY", data["sensory"]),
        }
    )
    return data


@app.get("/api/evolution")
def evolution_status():
    current = evolution.state()
    return {
        "enabled": config.EVOLUTION_ENABLED,
        "log_path": str(config.EVOLUTION_LOG_PATH),
        "canary_areas": list(evolution.CANARY_AREAS),
        "active": current.get("active", {}),
        "snapshots": current.get("snapshots", []),
        "proposals": current.get("proposals", []),
    }


@app.post("/api/evolution/log")
def evolution_log(req: EvolutionLogReq):
    result = evolution.append_log(req.entry)
    if result.get("error"):
        raise HTTPException(status_code=413, detail=result["error"])
    return result


@app.post("/api/evolution/propose")
def evolution_propose(req: EvolutionProposalReq):
    result = evolution.propose(_default_prompt_parts(), model=req.model)
    if result.get("error"):
        if result.get("error_kind") == "ollama_unavailable":
            raise HTTPException(status_code=503, detail=f"Ollama is unavailable: {result['error']}")
        raise HTTPException(status_code=400, detail=result["error"])
    result["preview"] = evolution.proposal_preview(result, _default_prompt_parts())
    return result


@app.post("/api/evolution/approve")
def evolution_approve(req: EvolutionDecisionReq):
    result = evolution.approve(req.proposal_id, prompt_parts=_default_prompt_parts())
    if result.get("error"):
        if result["error"] == "canary review failed":
            raise HTTPException(status_code=409, detail=result)
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/evolution/snapshots")
def evolution_snapshots():
    return {"snapshots": evolution.snapshots()}


@app.post("/api/evolution/snapshots/restore")
def evolution_snapshot_restore(req: EvolutionSnapshotRestoreReq):
    result = evolution.restore_snapshot(req.snapshot_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/evolution/reject")
def evolution_reject(req: EvolutionDecisionReq):
    result = evolution.reject(req.proposal_id, reason=req.reason)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/evolution/rollback")
def evolution_rollback():
    return evolution.rollback()


@app.get("/api/categories")
def categories(model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS)):
    """Distinct category labels (both stores), to scope retrieval in the UI."""
    return {"categories": store.categories_all(model)}


@app.get("/api/logs")
def logs(limit: int = Query(default=100, ge=1, le=200)):
    return {"events": log.recent(limit)}


@app.get("/api/tools/pending")
def pending_actions(include_resolved: bool = Query(default=False)):
    return {"actions": tools.list_pending(include_resolved=include_resolved)}


@app.post("/api/tools/pending/approve")
def approve_pending_action(req: PendingActionDecisionReq):
    try:
        result = tools.approve_pending(req.action_id)
    except Exception as exc:
        log.fallback("pending_approval_failed", exc, action_id=req.action_id)
        raise HTTPException(
            status_code=500,
            detail=f"Approval failed safely: {type(exc).__name__}: {exc}",
        ) from exc
    if result.get("error"):
        if result.get("error_kind") == "stale_preview":
            raise HTTPException(status_code=409, detail=result)
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/tools/pending/reject")
def reject_pending_action(req: PendingActionDecisionReq):
    result = tools.reject_pending(req.action_id)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/tools/pending/undo")
def undo_pending_action(req: PendingActionDecisionReq):
    result = tools.undo_pending(req.action_id)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/tools/audit")
def tool_audit(limit: int = Query(default=100, ge=1, le=200)):
    return {"events": tools.list_audit(limit=limit)}


@app.get("/api/tools/permissions")
def tool_permissions():
    return tools.list_permissions()


@app.post("/api/tools/permissions")
def tool_permission_update(req: ToolPermissionReq):
    result = tools.set_permission(req.tool, req.enabled)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/traces")
def traces(limit: int = Query(default=50, ge=1, le=100)):
    return {"traces": observability.recent(limit)}


@app.get("/api/timeline")
def timeline(model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS)):
    buckets: dict[str, dict] = {}
    for row in _memory_rows(model):
        day = (row.get("date") or "unknown")[:10] if row.get("date") else "unknown"
        bucket = buckets.setdefault(day, {"date": day, "count": 0, "categories": {}, "source_types": {}})
        bucket["count"] += 1
        cat = row.get("category") or "uncategorized"
        st = row.get("source_type") or "legacy"
        bucket["categories"][cat] = bucket["categories"].get(cat, 0) + 1
        bucket["source_types"][st] = bucket["source_types"].get(st, 0) + 1
    return {"days": [buckets[k] for k in sorted(buckets, reverse=True)]}


@app.get("/api/graph")
def graph_view(model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS), limit: int = Query(default=200, ge=1, le=1000)):
    collections = {store.collection_name(model), store.collection_name(model, text_only=True), None}
    edges = [e for e in graph.export_edges() if e.get("collection") in collections]
    return {"edges": edges[:limit], "count": len(edges)}


@app.get("/api/memories")
def memories(
    q: str | None = Query(default=None, max_length=config.MAX_MESSAGE_CHARS),
    category: str | None = Query(default=None, max_length=config.MAX_CATEGORY_CHARS),
    source: str | None = Query(default=None, max_length=config.MAX_DOCUMENT_CHARS),
    model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS),
    limit: int = Query(default=200, ge=1, le=1000),
):
    rows = _memory_rows(model)
    all_rows = rows
    if category:
        rows = [row for row in rows if row.get("category") == category]
    if source:
        rows = [row for row in rows if row.get("source") == source]
    if q:
        needle = q.casefold()
        rows = [
            row
            for row in rows
            if needle in (row.get("text") or "").casefold()
            or needle in (row.get("category") or "").casefold()
            or needle in (row.get("source") or "").casefold()
        ]
    categories = sorted({row["category"] for row in all_rows if row.get("category")})
    sources = sorted({row["source"] for row in all_rows if row.get("source")})
    return {"memories": rows[:limit], "categories": categories, "sources": sources, "count": _store_count(model)}


@app.get("/api/memory/{memory_id}")
def memory_detail(memory_id: str, model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS)):
    detail = _memory_detail(memory_id, model=model)
    if detail is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return detail


@app.post("/api/memory/{memory_id}/pin")
def pin_memory(memory_id: str, req: PinReq):
    if not memory_id or len(memory_id) > 128:
        raise HTTPException(status_code=422, detail="invalid memory id")
    n = store.update_metadata_all([memory_id], {"pinned": req.pinned}, model=req.model)
    if n <= 0:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"id": memory_id, "pinned": req.pinned}


@app.delete("/api/memory/{memory_id}", status_code=204)
def delete_memory(memory_id: str, model: str | None = Query(default=None, max_length=config.MAX_MODEL_NAME_CHARS)):
    if not memory_id or len(memory_id) > 128:
        raise HTTPException(status_code=422, detail="invalid memory id")
    _backup_before("before-delete-memory", model=model)
    _delete_memory_all(memory_id, model=model)
    return Response(status_code=204)


@app.post("/api/memory/delete_source/preview")
def preview_delete_source(req: DeleteSourceReq):
    return {"source": req.source, "count": _source_count(req.source, model=req.model)}


@app.post("/api/memory/delete_source")
def delete_source(req: DeleteSourceReq):
    backup_path = _backup_before("before-delete-source", model=req.model)
    count = _source_count(req.source, model=req.model)
    _delete_source_all(req.source, model=req.model)
    return {"deleted": count, "count": _store_count(req.model), "backup": backup_path}


@context_aware_execution
def _capture_background(
    message: str,
    reply: str,
    model: str | None,
    mems: list[dict] | None = None,
    thinking: dict | None = None,
) -> None:
    """Post-response memory capture. Errors are logged, never surfaced — the
    reply has already been sent."""
    try:
        capture(message, reply, model=model, mems=mems, thinking=thinking)
    except Exception as exc:
        log.fallback("capture_failed", exc, phase="api", model=model)


@context_aware_execution(attach_usage=True)
def _execute_chat_request(
    req: ChatReq,
    *,
    model: str | None,
    num_ctx: int,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run retrieval and the answer pipeline inside one execution context."""
    llm.raise_if_cancelled()
    history = _normalize_history(req)
    if req.memory_enabled:
        # One turn-level domain decision, shared with prompt construction
        # (build_system recomputes it from the same message + history), so
        # retrieval and the calibration prompts cannot disagree.
        mems = _merge_activated_memories(
            _retrieve(
                req.message,
                req.retrieval,
                model=model,
                domains=domain_routing.active_domains(req.message, history, strict=False),
            ),
            _activated_memories(req.activated_memory_ids, model=model),
        )
        reinforce_recall(mems, model=model)
    else:
        mems = []
    result = answer(
        req.message,
        mems,
        model=model,
        history=history,
        num_ctx=num_ctx,
        initiative_mode=req.initiative_mode,
        memory_enabled=req.memory_enabled,
    )
    return {"result": result, "mems": mems}


# In-flight chat generations that /api/chat/cancel can abort, keyed by the
# client-generated request_id. Bounded so a client can't grow it unboundedly;
# entries are removed in the handler's finally regardless of outcome.
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_CANCEL: dict[str, threading.Event] = {}
_INFLIGHT_MAX = 32
_PENDING_CANCEL: dict[str, float] = {}
_PENDING_CANCEL_MAX = 64
_PENDING_CANCEL_TTL_SECONDS = 10.0


def _prune_pending_cancel_locked(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    expired = [
        request_id
        for request_id, stamp in _PENDING_CANCEL.items()
        if now - stamp > _PENDING_CANCEL_TTL_SECONDS
    ]
    for request_id in expired:
        _PENDING_CANCEL.pop(request_id, None)


class CancelReq(StrictModel):
    request_id: str = Field(min_length=1, max_length=64)
    allow_pending: bool = False


@app.post("/api/chat/cancel")
def cancel_chat(req: CancelReq):
    with _INFLIGHT_LOCK:
        event = _INFLIGHT_CANCEL.get(req.request_id)
        if event is None and req.allow_pending:
            _prune_pending_cancel_locked()
            if len(_PENDING_CANCEL) >= _PENDING_CANCEL_MAX:
                return {"cancelled": False, "reason": "too many pending cancels"}
            _PENDING_CANCEL[req.request_id] = time.monotonic()
            return {"cancelled": True, "pending": True}
    if event is None:
        return {"cancelled": False, "reason": "no such in-flight request"}
    event.set()
    return {"cancelled": True}


@app.post("/api/chat")
def chat(req: ChatReq, background: BackgroundTasks):
    capture_error = None
    mems: list[dict] = []
    pending_result = _resolve_pending_from_chat(req.message)
    if pending_result is not None:
        count = None
        try:
            count = _store_count(req.model)
        except Exception as exc:
            capture_error = _append_warning(capture_error, f"Memory count failed: {exc}")
        return {
            "reply": pending_result["reply"],
            "memories": mems,
            "count": count,
            "capture_error": capture_error,
            "live_data": None,
            "web_data": None,
            "tools_used": pending_result.get("tools_used", []),
            "tool_results": pending_result.get("tool_results", []),
            "reminders": [],
            "initiatives": [],
            "thinking": None,
            "trace": None,
            "council": None,
            "awareness": awareness.state(),
            "attention": attention.state(),
            "workspace": shared_workspace.state(),
            "grounding": semantic_grounding.state(),
            "presence": presence.state(),
            "self_model": self_model.state(),
            "pending_decision": pending_result.get("pending_decision"),
            "memory_enabled": req.memory_enabled,
        }
    cancel_event = None
    if req.request_id:
        cancel_event = threading.Event()
        with _INFLIGHT_LOCK:
            _prune_pending_cancel_locked()
            if req.request_id in _INFLIGHT_CANCEL:
                raise HTTPException(status_code=409, detail="chat request_id is already in flight")
            if len(_INFLIGHT_CANCEL) >= _INFLIGHT_MAX:
                raise HTTPException(status_code=429, detail="too many in-flight cancellable chat requests")
            _INFLIGHT_CANCEL[req.request_id] = cancel_event
            pending_cancel = _PENDING_CANCEL.pop(req.request_id, None) is not None
        if pending_cancel:
            cancel_event.set()
    normalized_num_ctx = llm.normalize_num_ctx(req.num_ctx, req.model)
    try:
        execution = _execute_chat_request(
            req,
            model=req.model,
            num_ctx=normalized_num_ctx,
            cancel_event=cancel_event,
        )
        mems = execution["mems"]
        result = execution["result"]
        result["context_usage"] = execution.get("context_usage")
    except llm.GenerationCancelled:
        # User hit Stop: Ollama was told to abort (the stream closed) and
        # nothing is captured — the turn simply never happened.
        return {"reply": None, "cancelled": True, "memories": mems}
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    finally:
        if req.request_id:
            with _INFLIGHT_LOCK:
                _INFLIGHT_CANCEL.pop(req.request_id, None)
    # Final boundary guard: even a future/custom pipeline path that bypasses
    # llm.chat cannot return or capture a history-poisoning oversized response.
    reply = llm.sanitize_reply(result["reply"], req.model)
    # Capture = episode embed + a consolidation LLM pass — inline it roughly
    # doubled per-turn latency for output the reply never uses. Run it after the
    # response is sent; the returned count therefore lags the new episode by one
    # turn. Failures land in the fallback log instead of the response.
    background.add_task(
        _capture_background,
        req.message,
        reply,
        req.model,
        mems,
        result.get("thinking"),
    )
    count = None
    try:
        count = _store_count(req.model)
    except Exception as exc:
        capture_error = _append_warning(capture_error, f"Memory count failed: {exc}")
    return {
        "reply": reply,
        "memories": mems,
        "count": count,
        "capture_error": capture_error,
        "live_data": result.get("live_data"),
        "web_data": result.get("web_data"),
        "tools_used": [c.get("tool") for c in result.get("tools_used", [])],
        "tool_results": result.get("tools_used", []),
        "reminders": result.get("reminders", []),
        "initiatives": result.get("initiatives", []),
        "thinking": result.get("thinking"),
        "trace": result.get("trace"),
        "council": result.get("council"),
        "awareness": result.get("awareness"),
        "attention": result.get("attention"),
        "workspace": result.get("workspace"),
        "grounding": result.get("grounding"),
        "grounding_gate": result.get("grounding_gate"),
        "claim_receipts": result.get("claim_receipts", []),
        "claim_guard": result.get("claim_guard"),
        "self_model": result.get("self_model"),
        "presence": result.get("presence"),
        "context_usage": result.get("context_usage"),
        "memory_enabled": req.memory_enabled,
    }


@app.post("/api/memory")
def add_memory(req: MemoryReq):
    unit = MemoryUnit(
        text=req.text,
        category=req.category,
        importance=req.importance,
        extra={"source_type": "user_entered"},
    )
    try:
        try:
            embedding = embed_document(unit.text, model=req.model)
        except EmbeddingUnavailable:
            store.add(unit, text_only_embedding(), model=req.model, text_only=True)
        else:
            store.add(unit, embedding, model=req.model)
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    return {"id": unit.id, "count": _store_count(req.model)}


@app.post("/api/recall")
def recall(req: RecallReq):
    """What WOULD be retrieved for a query — no LLM call, no capture."""
    try:
        return {"memories": _retrieve(req.query, req.retrieval, model=req.model)}
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise


@app.post("/api/prune")
def prune(req: PruneReq):
    """Forget old, low-importance, never-recalled episodes (human-like decay)."""
    backup_path = None
    if not req.dry_run:
        preview = store.prune_episodes(
            model=req.model,
            older_than_days=req.older_than_days,
            max_importance=req.max_importance,
            dry_run=True,
        )
        if preview.get("pruned", 0) > 0:
            backup_path = _backup_before("before-prune", model=req.model)
    result = store.prune_episodes(
        model=req.model,
        older_than_days=req.older_than_days,
        max_importance=req.max_importance,
        dry_run=req.dry_run,
    )
    result["count"] = _store_count(req.model)
    result["backup"] = backup_path
    return result


@app.post("/api/compact")
def compact(req: CompactReq):
    preview = compact_old_episodes(
        model=req.model,
        older_than_days=req.older_than_days,
        max_episodes=req.max_episodes,
        dry_run=True,
    )
    if req.dry_run:
        preview["count"] = _store_count(req.model)
        preview["backup"] = None
        return preview
    backup_path = _backup_before("before-compact-episodes", model=req.model) if preview.get("eligible", 0) > 0 else None
    try:
        result = compact_old_episodes(
            model=req.model,
            older_than_days=req.older_than_days,
            max_episodes=req.max_episodes,
            dry_run=False,
        )
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    result["count"] = _store_count(req.model)
    result["backup"] = backup_path
    return result


@app.post("/api/sleep")
def sleep_cycle(req: SleepReq):
    """Weekly-style maintenance: cold-archive weak episodes and synthesize dreams."""
    preview = sleep_mod.run(
        model=req.model,
        dry_run=True,
        max_archive=req.max_archive,
        max_dreams=req.max_dreams,
        older_than_days=req.older_than_days,
    )
    if req.dry_run:
        preview["count"] = _store_count(req.model)
        preview["backup"] = None
        return preview
    backup_path = (
        _backup_before("before-sleep-cycle", model=req.model)
        if preview.get("archivable", 0) > 0 or preview.get("dreamable", 0) > 0
        else None
    )
    try:
        result = sleep_mod.run(
            model=req.model,
            dry_run=False,
            max_archive=req.max_archive,
            max_dreams=req.max_dreams,
            older_than_days=req.older_than_days,
        )
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    result["count"] = _store_count(req.model)
    result["backup"] = backup_path
    return result


@app.post("/api/reflect")
def reflect_endpoint(req: ReflectReq):
    """Run the reflection pass: generate insights + regenerate proactive reminders."""
    try:
        result = reflect_mod.reflect(model=req.model)
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    result["count"] = _store_count(req.model)
    mediated = _mediated_proactive_context(model=req.model)
    result["active_reminders"] = mediated["reminders"]
    result["active_initiatives"] = mediated["initiatives"]
    result["conflicts"] = mediated["conflicts"]
    result["worldview"] = perspective.reflect_recent(model=req.model)
    return result


@app.post("/api/summarize")
def summarize(req: SummReq):
    try:
        backup_path = _backup_before("before-summarize-replace", model=req.model) if req.replace else None
        return {"summary": summarize_category(req.category, replace=req.replace, model=req.model), "backup": backup_path}
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise


@app.post("/api/ingest/preview")
def preview_ingest(req: IngestReq):
    suffix = Path(req.filename).suffix.lower()
    if suffix in {".pdf"} | IMAGE_SUFFIXES | AUDIO_SUFFIXES:
        data = _decode_upload(req, "file")
        source = _upload_source(req.filename, data)
    else:
        if req.text is None:
            raise HTTPException(status_code=422, detail="text is required for text documents")
        source = _upload_source(req.filename, req.text.encode("utf-8"))
    stale = _stale_upload_sources(req.filename, source, req.model)
    return {
        "filename": req.filename,
        "source": source,
        "stale_sources": stale,
        "replace_count": sum(_source_count(s, model=req.model) for s in stale),
    }


@app.post("/api/ingest")
def ingest_markdown(req: IngestReq):
    """Chunk an uploaded document and store it (exact re-upload replaces by content key)."""
    try:
        text, source, sensory_kind, sensory_data = _ingest_payload(req)
        if sensory_kind and sensory_data:
            # Media uploads are a two-store write (text chunks + sensory
            # vectors) with no transaction across them, so order the steps to
            # keep a failed request from changing stored memory: add the new
            # chunks WITHOUT replacing the old ones, attempt the sensory
            # write, and only then retire the replaced chunks. If the sensory
            # write fails, dropping the just-added chunks is a full rollback —
            # the prior upload's chunks, graph edges, and sensory record were
            # never touched.
            old_ids = {
                rec["id"]
                for rec in store.list_records_all(req.model)
                if (rec.get("meta") or {}).get("source") == source
            }
            n, cat = ingest_document_text(
                text,
                filename=req.filename,
                source=source,
                category=req.category,
                importance=req.importance,
                model=req.model,
                replace=False,
            )
            try:
                sensory.add_media(
                    sensory_kind,
                    sensory_data,
                    source,
                    req.filename,
                    text,
                    cat,
                    req.importance,
                )
            except Exception:
                new_ids = [
                    rec["id"]
                    for rec in store.list_records_all(req.model)
                    if (rec.get("meta") or {}).get("source") == source and rec["id"] not in old_ids
                ]
                if new_ids:
                    store.delete_all(new_ids, model=req.model)
                raise
            if old_ids:
                store.delete_all(sorted(old_ids), model=req.model)
                # Same as ingest_document_text's replace path: the retired
                # chunks' graph edges must go with them or they dangle.
                graph.delete_fact_ids(sorted(old_ids), collections=_graph_collections(req.model))
        else:
            # An identical re-upload keeps the same content-hashed source key,
            # so this REPLACES the existing records in place.
            n, cat = ingest_document_text(
                text,
                filename=req.filename,
                source=source,
                category=req.category,
                importance=req.importance,
                model=req.model,
            )
        # New chunks are safely stored; now drop chunks from earlier uploads of
        # the same filename with different content, so an edited re-upload
        # replaces instead of accumulating stale versions. Done AFTER the ingest
        # so a failure above leaves the previous upload intact.
        stale = _stale_upload_sources(req.filename, source, req.model)
        if stale:
            _backup_before("before-reingest-replace", model=req.model)
        for old in stale:
            # Shared deletion so the stale chunks' graph edges and sensory
            # records go with them instead of dangling.
            _delete_source_all(old, model=req.model)
    except HTTPException:
        raise
    except VisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if _is_ollama_error(exc):
            raise _ollama_unavailable(exc) from exc
        raise
    return {"chunks": n, "category": cat, "count": _store_count(req.model)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False)
