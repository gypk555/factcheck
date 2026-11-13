"""Image checks: is a picture AI-generated, and has it been edited?

Two independent signals:

1. AI-generated detector: a small CNN (trained in scripts/train_image.py,
   exported to ONNX) that outputs P(AI-generated) and a class activation map
   (CAM): which regions pushed the score up.
2. Edit detector: Error Level Analysis (ELA). The image is re-saved as JPEG at
   a known quality and compared with the original. Regions pasted in or edited
   after the last save usually compress differently, so they light up. An
   optional trained model turns the ELA pattern into P(edited).

The trained models are optional files in models/. Without them the app still
shows the ELA map and says the scores are unavailable.

Security: images are decoded with Pillow inside strict limits (file size is
capped by the server, pixel count here, formats allow-listed), and nothing is
written to disk.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError

from . import config

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}
MAX_PIXELS = 24_000_000           # refuse decompression bombs; keeps memory well under 512 MB
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
PREVIEW_MAX_SIDE = 768            # size of images sent back to the page
ELA_QUALITY = 90

AI_THRESHOLD_HIGH = 0.65
AI_THRESHOLD_LOW = 0.35


class ImageRejected(ValueError):
    """The upload is not an image we can safely analyse."""


@dataclass
class Signal:
    available: bool
    prob: float | None = None
    map_png: str | None = None   # data: URL of a heatmap the page overlays
    note: str = ""


@dataclass
class ImageResult:
    label: str                   # AI_GENERATED, EDITED, REAL, UNCERTAIN, UNKNOWN
    headline: str
    summary: str
    ai: Signal
    edit: Signal
    preview_png: str             # the analysed image, resized, as a data: URL
    width: int
    height: int
    format: str
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ helpers
def load_image(data: bytes) -> tuple[Image.Image, str]:
    """Decode an upload safely. Returns (RGB image, original format)."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise ImageRejected("Use a JPG, PNG, WEBP, BMP or GIF image.")
        img.load()  # full decode happens here, inside Pillow's pixel limit
    except Image.DecompressionBombError:
        raise ImageRejected("That image is too large to analyse.") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ImageRejected("This file couldn't be opened as an image.") from None
    img = ImageOps.exif_transpose(img)  # respect camera rotation
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, fmt


def shrink(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=85)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def colorize(values: np.ndarray) -> Image.Image:
    """Map 0..1 values to a transparent-to-hot overlay (RGBA).

    Low values are fully transparent so the photo shows through; high values
    go yellow -> orange -> red with increasing opacity.
    """
    v = np.clip(values, 0.0, 1.0)
    # Opacity ramps up quickly (from 0.08 to 0.5) so mid-level evidence is
    # clearly visible on the dimmed photo; colour goes yellow -> red.
    strength = np.clip((v - 0.2) / 0.5, 0, 1)
    r = np.full_like(v, 255.0)
    g = 235 - 190 * strength
    b = 70 * (1 - strength)
    a = strength * 235
    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# ------------------------------------------------------------------ ELA
def error_level_map(img: Image.Image) -> np.ndarray:
    """Error Level Analysis on the FULL-resolution image, one value per 8x8 block.

    JPEG compresses in 8x8 blocks, so ELA must run before any resizing (resizing
    scrambles the block grid). Averaging per block removes pixel noise; then the
    map is scaled 0..1 by a high percentile. Returns shape (height//8, width//8).
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=ELA_QUALITY)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    diff = np.asarray(ImageChops.difference(img, resaved)).max(axis=2)  # uint8, H x W
    bh, bw = diff.shape[0] // 8, diff.shape[1] // 8
    if bh == 0 or bw == 0:
        return np.zeros((1, 1), np.float32)
    blocks = diff[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8).mean(axis=(1, 3), dtype=np.float32)
    scale = max(float(np.percentile(blocks, 99.5)), 1.0)
    return np.clip(blocks / scale, 0, 1)


def resize_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Smoothly resize a 0..1 map to `size` (width, height)."""
    im = Image.fromarray((values * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(im, np.float32) / 255


# ------------------------------------------------------------------ models
class _OnnxModel:
    """Thin wrapper around an optional ONNX classifier exported by training."""

    def __init__(self, path: Path):
        import onnxruntime as ort  # imported lazily: optional dependency

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # small, predictable CPU use on Render
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.size = int(meta.get("input_size", 224))
        self.mean = np.array([float(x) for x in meta.get("mean", "0.485,0.456,0.406").split(",")], np.float32)
        self.std = np.array([float(x) for x in meta.get("std", "0.229,0.224,0.225").split(",")], np.float32)
        self.outputs = [o.name for o in self.session.get_outputs()]

    def run(self, img: Image.Image) -> tuple[float, np.ndarray | None]:
        x = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
        arr = (np.asarray(x, np.float32) / 255.0 - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.session.get_inputs()[0].name: arr})
        prob = float(1 / (1 + np.exp(-out[0].ravel()[0])))
        cam = out[1][0] if len(out) > 1 else None
        return prob, cam


class ImageChecker:
    def __init__(self, model_dir: Path = config.MODEL_DIR):
        self.ai_model = self._try_load(model_dir / "image_ai.onnx")
        self.edit_model = self._try_load(model_dir / "image_edit.onnx")

    @staticmethod
    def _try_load(path: Path):
        if not path.exists():
            return None
        try:
            return _OnnxModel(path)
        except Exception:  # missing onnxruntime or a broken file: degrade gracefully
            return None

    def check(self, data: bytes) -> ImageResult:
        img, fmt = load_image(data)
        w, h = img.size
        view = shrink(img, PREVIEW_MAX_SIDE)
        warnings: list[str] = []
        if min(w, h) < 128:
            warnings.append("Very small image: there is little detail to analyse.")

        # --- AI-generated signal
        if self.ai_model:
            prob, cam = self.ai_model.run(img)
            heat = None
            if cam is not None:
                cam = cam - cam.min()
                cam = cam / cam.max() if cam.max() > 0 else cam
                heat_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(view.size, Image.Resampling.BICUBIC)
                heat = to_data_url(colorize(np.asarray(heat_img, np.float32) / 255))
            ai = Signal(True, round(prob, 4), heat)
        else:
            ai = Signal(False, note="The AI-image model isn't installed on this server yet.")

        # --- edit signal (ELA always available; probability needs the model)
        ela = error_level_map(img)  # full resolution, before any resizing
        ela_url = to_data_url(colorize(resize_map(ela, view.size)))
        if self.edit_model:
            ela_img = Image.fromarray((ela * 255).astype(np.uint8)).convert("RGB")
            prob, _ = self.edit_model.run(ela_img)
            edit = Signal(True, round(prob, 4), ela_url)
        else:
            edit = Signal(False, None, ela_url, "Showing the error-level map only; the edit model isn't installed yet.")
        if fmt != "JPEG":
            warnings.append(
                f"This is a {fmt} file. Edit detection works best on JPEG photos straight from a camera or phone."
            )
        if max(w, h) <= 1080 and fmt == "JPEG":
            warnings.append("Images re-shared on social media are compressed, which hides many traces of editing.")

        label, headline, summary = self._verdict(ai, edit)
        return ImageResult(label, headline, summary, ai, edit, to_data_url(view, "JPEG"), w, h, fmt, warnings)

    @staticmethod
    def _verdict(ai: Signal, edit: Signal) -> tuple[str, str, str]:
        probs = [s.prob for s in (ai, edit) if s.available]
        if not probs:
            return ("UNKNOWN", "No verdict yet",
                    "The trained image models aren't installed, so only the error-level map is shown.")
        if ai.available and ai.prob >= AI_THRESHOLD_HIGH:
            return ("AI_GENERATED", "Likely AI-generated",
                    "Patterns in this image resemble pictures made by AI image generators.")
        if edit.available and edit.prob >= AI_THRESHOLD_HIGH:
            return ("EDITED", "Likely edited",
                    "Parts of this image compress differently from the rest, a common sign of editing.")
        if ai.available and all(p <= AI_THRESHOLD_LOW for p in probs):
            return ("REAL", "Likely a real photo",
                    "No strong signs of AI generation or editing were found.")
        return ("UNCERTAIN", "Uncertain",
                "The signals are mixed, so the model won't call it either way.")
