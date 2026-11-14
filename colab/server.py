"""XFND-CLIP model server: lets the website talk to the model running in Colab.

It wraps the project's own `explain.Explainer` (unchanged model code in ../model)
and returns everything the website shows as JSON:

  POST /api/analyse   form fields: text (the claim), image (optional file)
  GET  /api/health    is the model loaded? which GPU?

Every request must carry the access key the notebook prints (header
`X-Access-Key`), so only people with the connection code can use your GPU.

In Colab the notebook starts it with:
    XFND_ACCESS_KEY=... uvicorn server:app --port 8000
and exposes it with a temporary Cloudflare link.
"""

from __future__ import annotations

import base64
import io
import os
import secrets
import threading
import time
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

API_VERSION = "1"
MAX_TEXT_CHARS = 1000
MIN_WORDS = 3
MAX_WORDS = 60   # BERT reads 64 tokens; words beyond that would get no importance
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
PREVIEW_SIDE = 768
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


# ------------------------------------------------------------------ helpers
def load_upload(data: bytes) -> Image.Image:
    """Decode an uploaded image safely (format allow-list, pixel limit)."""
    try:
        img = Image.open(io.BytesIO(data))
        if (img.format or "").upper() not in ALLOWED_FORMATS:
            raise HTTPException(422, "Use a JPG, PNG, WEBP, BMP or GIF image.")
        # Check the size from the header BEFORE decoding (Pillow alone only warns
        # between 1x and 2x its limit).
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(422, "That image is too large.")
        img.load()
    except Image.DecompressionBombError:
        raise HTTPException(422, "That image is too large.") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(422, "This file couldn't be opened as an image.") from None
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 1600:  # the model works on 224 px; keep work and memory small
        img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    return img


def data_url(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=85)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def heat_overlay(cam: np.ndarray) -> Image.Image:
    """Grad-CAM values 0..1 -> transparent-to-hot RGBA image.
    Low values are see-through so the photo shows; high values go yellow -> red."""
    v = np.clip(cam.astype(np.float32), 0, 1)
    s = np.clip((v - 0.2) / 0.6, 0, 1)
    rgba = np.stack([np.full_like(s, 255), 230 - 190 * s, 60 * (1 - s), s * 210], axis=-1)
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def shrink(img: Image.Image, side: int) -> Image.Image:
    out = img.copy()
    out.thumbnail((side, side), Image.Resampling.LANCZOS)
    return out


# ------------------------------------------------------------------ app
def create_app(explainer_factory: Callable[[], object], access_key: str,
               allowed_origins: list[str] | None = None, device_info: dict | None = None) -> FastAPI:
    """Build the API. `explainer_factory` returns an object with the same
    methods as model/explain.py's Explainer (tests pass a fake one)."""
    if not access_key or len(access_key) < 16:
        raise ValueError("Set a long random access key.")

    app = FastAPI(title="XFND-CLIP model server", docs_url=None, redoc_url=None, openapi_url=None)
    state = {"explainer": None, "error": None}
    gpu_lock = threading.Lock()  # one analysis at a time on the GPU

    def explainer():
        if state["explainer"] is None:
            with gpu_lock:
                if state["explainer"] is None and state["error"] is None:
                    try:
                        state["explainer"] = explainer_factory()
                    except Exception as exc:  # e.g. missing checkpoint, download failure
                        state["error"] = f"{type(exc).__name__}: {exc}"
        if state["explainer"] is None:
            raise HTTPException(500, f"The model failed to load. {state['error']}")
        return state["explainer"]

    def check_key(key: str | None) -> None:
        if not key or not secrets.compare_digest(key, access_key):
            raise HTTPException(401, "Wrong or missing access key. Copy the connection code from the notebook again.")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Check the key before anything is read or parsed (CORS preflights carry no key).
        if request.url.path.startswith("/api/") and request.method != "OPTIONS":
            key = request.headers.get("x-access-key") or ""
            if not secrets.compare_digest(key, access_key):
                return JSONResponse({"detail": "Wrong or missing access key. Copy the connection code from the notebook again."}, 401)
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "Content-Length header is required."}, 411)
            if not length.isdigit() or int(length) > MAX_IMAGE_BYTES + 64 * 1024:
                return JSONResponse({"detail": "That upload is too large (10 MB limit)."}, 413)
        return await call_next(request)

    @app.get("/api/health")
    def health(x_access_key: str | None = Header(default=None)):
        check_key(x_access_key)
        ex = explainer()
        return {"status": "ok", "api_version": API_VERSION, "model": "xfnd_clip",
                "clip_sim_mean": round(float(ex.sim_mean), 4), **(device_info or {})}

    @app.post("/api/analyse")
    def analyse(text: str = Form(...), image: UploadFile | None = File(default=None),
                x_access_key: str | None = Header(default=None)):
        check_key(x_access_key)
        text = " ".join(text.split())
        if len(text) > MAX_TEXT_CHARS:
            raise HTTPException(422, f"Keep the claim under {MAX_TEXT_CHARS} characters.")
        if len(text.split()) < MIN_WORDS:
            raise HTTPException(422, f"Type the full claim (at least {MIN_WORDS} words).")
        if len(text.split()) > MAX_WORDS:
            raise HTTPException(422, f"Keep the claim to {MAX_WORDS} words or fewer (the model reads a headline, not an article).")

        img = None
        if image is not None and image.filename:
            raw = image.file.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "That image is over 10 MB.")
            if raw:
                img = load_upload(raw)

        ex = explainer()
        t0 = time.perf_counter()
        try:
            with gpu_lock:
                pred = ex.predict(text, img)
                words = ex.word_importance(text, img)
                cam = ex.gradcam(text, img) if img is not None else None
        except Exception as exc:  # e.g. GPU out of memory; answer properly so the site can show it
            raise HTTPException(500, f"The model failed on this post ({type(exc).__name__}). Try again or use a smaller image.") from None
        elapsed = int((time.perf_counter() - t0) * 1000)

        image_out = None
        if img is not None:
            view = shrink(img, PREVIEW_SIDE)
            heat = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize(view.size, Image.Resampling.BILINEAR)
            image_out = {
                "preview": data_url(view, "JPEG"),
                "heatmap": data_url(heat_overlay(np.asarray(heat, np.float32) / 255), "PNG"),
                "width": img.width, "height": img.height,
            }
        sim = pred.get("clip_sim")
        sim_mean = float(ex.sim_mean)
        return {
            "api_version": API_VERSION,
            "claim": text,
            "label": pred["label"],
            "prob_fake": round(float(pred["prob_fake"]), 4),
            "modality": {"text": round(float(pred["att_text"]), 4),
                         "image": round(float(pred["att_image"]), 4),
                         "text_image": round(float(pred["att_mm"]), 4)},
            "clip_sim": None if sim is None else round(float(sim), 4),
            "clip_sim_mean": round(sim_mean, 4),
            "clip_match": None if sim is None else ("match" if sim >= sim_mean else "weak"),
            "words": [{"word": w, "weight": round(float(v), 4)} for w, v in words],
            "image": image_out,
            "elapsed_ms": elapsed,
        }

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": "Send the claim as 'text' (and optionally an 'image' file)."}, 422)

    # Added last so it wraps everything, including the size-limit replies above:
    # otherwise the browser would see a CORS failure instead of the message.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["X-Access-Key"],
        max_age=600,
    )
    return app


def _colab_app() -> FastAPI:
    """The real app used in Colab (imports torch and the model code lazily)."""
    import torch

    from explain import Explainer

    info = {"device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    origins = [o.strip() for o in os.environ.get("XFND_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    return create_app(Explainer, os.environ.get("XFND_ACCESS_KEY", ""), origins, info)


if os.environ.get("XFND_ACCESS_KEY"):
    app = _colab_app()
