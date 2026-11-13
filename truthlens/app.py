"""TruthLens web app (FastAPI).

Local:   uv run uvicorn truthlens.app:app --reload
Render:  uvicorn truthlens.app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import ipaddress
import os
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from . import __version__, config, factcheck
from .model import FAKE_THRESHOLD, REAL_THRESHOLD, Predictor

HERE = config.ROOT / "truthlens"
MAX_BODY_BYTES = 64 * 1024  # a 20k-character article as JSON fits easily
SHOW_API_DOCS = os.environ.get("TRUTHLENS_API_DOCS", "0") == "1"
# Optional: name of a header your host sets to the real client IP
# (e.g. "cf-connecting-ip"). If empty, the last X-Forwarded-For entry is used.
CLIENT_IP_HEADER = os.environ.get("CLIENT_IP_HEADER", "").strip().lower()

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor.load()  # load once at start-up, not per request
    yield


app = FastAPI(
    title="TruthLens", version=__version__, lifespan=lifespan,
    docs_url="/docs" if SHOW_API_DOCS else None, redoc_url=None,
    openapi_url="/openapi.json" if SHOW_API_DOCS else None,
)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


# ------------------------------------------------------------- rate limiting
class RateLimiter:
    """Small in-memory sliding-window limiter (enough for one Render instance)."""

    MAX_KEYS = 10_000

    OVERFLOW = "__overflow__"
    OVERFLOW_FACTOR = 20  # the shared overflow bucket allows 20x one visitor's limit

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        # Ordered by last use, so the least-recently-seen visitor is first.
        self.hits: OrderedDict[str, deque] = OrderedDict()
        self.overflow: deque = deque()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if key not in self.hits and len(self.hits) >= self.MAX_KEYS:
            self._evict_stale(now)
        if key not in self.hits and len(self.hits) >= self.MAX_KEYS:
            # Still full of visitors whose 60 s window is open. Never erase their
            # history (that would reset their limits) and never lock newcomers
            # out: newcomers share one overflow bucket with its own limit.
            return self._take(self.overflow, now, self.per_minute * self.OVERFLOW_FACTOR)
        q = self.hits.setdefault(key, deque())
        self.hits.move_to_end(key)
        return self._take(q, now, self.per_minute)

    @staticmethod
    def _take(q: deque, now: float, limit: int) -> bool:
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    def _evict_stale(self, now: float) -> None:
        # Oldest-used first; stop at the first visitor still inside its window.
        while self.hits:
            k, q = next(iter(self.hits.items()))
            if q and now - q[-1] <= 60:
                break
            del self.hits[k]


# Per-visitor limits, plus a site-wide cap on fact-check look-ups so the API key's
# quota is protected however many addresses a caller uses.
LIMITERS = {
    "/api/analyze": RateLimiter(60),
    "/api/factcheck": RateLimiter(config.FACTCHECK_RATE_LIMIT_PER_MIN),
}
GLOBAL_LIMITERS = {"/api/factcheck": RateLimiter(int(os.environ.get("FACTCHECK_GLOBAL_PER_MIN", "120")))}

# Only trust proxy headers when a proxy we know about sits in front of the app.
# Render sets RENDER=true; elsewhere opt in with TRUST_PROXY=1.
TRUST_PROXY = os.environ.get("TRUST_PROXY", os.environ.get("RENDER", "")).lower() in ("1", "true", "yes")


def visitor_key(ip: str) -> str:
    """One rate-limit bucket per IPv4 address or per IPv6 /64 network
    (a single machine usually controls a whole /64, so per-address IPv6
    buckets would let it pose as millions of visitors)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 4:
        return str(addr)
    if addr.ipv4_mapped is not None:  # "::ffff:1.2.3.4" is really an IPv4 client
        return str(addr.ipv4_mapped)
    return str(ipaddress.ip_network(f"{addr}/64", strict=False))


def client_ip(request: Request) -> str:
    socket_ip = request.client.host if request.client else "unknown"
    if not TRUST_PROXY:
        return socket_ip  # headers could be set by anyone; use the real connection
    if CLIENT_IP_HEADER and request.headers.get(CLIENT_IP_HEADER):
        return request.headers[CLIENT_IP_HEADER].strip()
    # Behind the proxy the client IP is the entry the proxy appended, i.e. the
    # LAST one. Earlier entries are whatever the client sent and can be faked.
    fwd = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    return fwd[-1] if fwd else socket_ip


# ------------------------------------------------------------- middleware
def _error(status: int, msg: str) -> JSONResponse:
    resp = JSONResponse({"detail": msg}, status_code=status)
    for k, v in SECURITY_HEADERS.items():
        resp.headers[k] = v
    return resp


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.method == "POST":
        # 1. Rate-limit BEFORE reading or validating the body, so bad requests count too.
        limiter = LIMITERS.get(request.url.path)
        if limiter and not limiter.allow(visitor_key(client_ip(request))):
            return _error(429, "Too many requests. Wait a minute and try again.")
        overall = GLOBAL_LIMITERS.get(request.url.path)
        if overall and not overall.allow("all"):
            return _error(429, "This service is busy. Try again in a minute.")
        # 2. Size limit. Require Content-Length: a "chunked" upload has none and
        #    would otherwise be read into memory in full, whatever its size.
        length = request.headers.get("content-length")
        if length is None:
            return _error(411, "Content-Length header is required.")
        if not length.isdigit() or int(length) > MAX_BODY_BYTES:
            return _error(413, "Request is too large.")

    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    if request.headers.get("x-forwarded-proto") == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    return response


# ------------------------------------------------------------- schemas
class AnalyzeIn(BaseModel):
    text: str = Field(..., max_length=config.MAX_INPUT_CHARS)

    @field_validator("text")
    @classmethod
    def enough_words(cls, v: str) -> str:
        if len(v.split()) < config.MIN_INPUT_WORDS:
            raise ValueError(f"Enter at least {config.MIN_INPUT_WORDS} words.")
        return v


class FactCheckIn(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Default FastAPI errors echo the submitted text back; return a short message instead.
    err = exc.errors()[0] if exc.errors() else {}
    msg = str(err.get("msg", "Invalid input.")).removeprefix("Value error, ")
    if err.get("type") == "string_too_long":
        msg = f"Text is too long (limit {config.MAX_INPUT_CHARS:,} characters)."
    return JSONResponse({"detail": msg}, status_code=422)


# ------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    meta = request.app.state.predictor.meta
    return templates.TemplateResponse(request, "index.html", {
        "meta": meta,
        "factcheck_enabled": factcheck.is_enabled(),
        "max_chars": config.MAX_INPUT_CHARS,
        "fake_threshold": FAKE_THRESHOLD,
        "real_threshold": REAL_THRESHOLD,
        "version": __version__,
    })


@app.post("/api/analyze")
def analyze(body: AnalyzeIn, request: Request):
    # A plain "def" (not async): FastAPI runs it in a worker thread, so the
    # CPU work of a long article never blocks other visitors or /healthz.
    return asdict(request.app.state.predictor.predict(body.text))


@app.post("/api/factcheck")
async def lookup(body: FactCheckIn):
    if not factcheck.is_enabled():
        raise HTTPException(404, "Fact-check lookup is not configured on this server.")
    try:
        return {"results": await factcheck.search(body.query)}
    except factcheck.FactCheckUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "version": __version__}
