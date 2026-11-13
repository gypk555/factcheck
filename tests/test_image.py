"""Image checking: safe decoding, ELA, verdict logic and the API."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from truthlens import app as app_module
from truthlens.image_check import ImageChecker, ImageRejected, Signal, error_level_map, load_image


def png_bytes(size=(64, 48), color=(120, 160, 200), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def test_load_image_accepts_common_formats():
    for fmt in ("PNG", "JPEG", "WEBP", "BMP", "GIF"):
        img, f = load_image(png_bytes(fmt=fmt))
        assert img.mode == "RGB" and f == fmt


def test_load_image_rejects_non_images():
    with pytest.raises(ImageRejected):
        load_image(b"not an image at all")


def test_load_image_rejects_other_formats():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "TIFF")
    with pytest.raises(ImageRejected):
        load_image(buf.getvalue())


def test_decompression_bomb_rejected():
    # A tiny PNG that claims to be 10,000 x 10,000 pixels (100 MP > 40 MP limit).
    buf = io.BytesIO()
    Image.new("1", (10_000, 10_000)).save(buf, "PNG")
    assert len(buf.getvalue()) < 200_000
    with pytest.raises(ImageRejected):
        load_image(buf.getvalue())


def test_ela_highlights_a_pasted_region():
    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    buf = io.BytesIO()
    base.save(buf, "JPEG", quality=70)            # the "original" photo, saved once
    photo = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    patch = Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8))
    photo.paste(patch, (70, 70))                   # pasted in after saving
    ela = error_level_map(photo)
    assert ela.shape == (16, 16)                   # one value per 8x8 JPEG block
    assert ela[9:13, 9:13].mean() > ela[1:7, 1:7].mean() * 1.3


def test_verdict_without_models_is_unknown():
    label, *_ = ImageChecker._verdict(Signal(False), Signal(False, None, "x"))
    assert label == "UNKNOWN"


@pytest.mark.parametrize("ai,edit,expected", [
    (0.9, 0.1, "AI_GENERATED"),
    (0.1, 0.9, "EDITED"),
    (0.1, 0.2, "REAL"),
    (0.5, 0.2, "UNCERTAIN"),
])
def test_verdicts(ai, edit, expected):
    label, *_ = ImageChecker._verdict(Signal(True, ai), Signal(True, edit))
    assert label == expected


def test_real_requires_the_ai_check():
    label, *_ = ImageChecker._verdict(Signal(False), Signal(True, 0.1))
    assert label != "REAL"


# ----- API
@pytest.fixture
def client():
    for lim in app_module.LIMITERS.values():
        lim.hits.clear()
    with TestClient(app_module.app) as c:
        yield c


def test_image_endpoint_returns_maps(client):
    r = client.post("/api/image", files={"file": ("x.jpg", png_bytes((300, 200), fmt="JPEG"), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 300 and body["format"] == "JPEG"
    assert body["preview_png"].startswith("data:image/")
    assert body["edit"]["map_png"].startswith("data:image/png;base64,")
    assert body["label"] in {"AI_GENERATED", "EDITED", "REAL", "UNCERTAIN", "UNKNOWN"}


def test_image_endpoint_rejects_garbage(client):
    r = client.post("/api/image", files={"file": ("x.png", b"hello", "image/png")})
    assert r.status_code == 422
    assert "couldn't be opened" in r.json()["detail"]


def test_image_endpoint_size_limit(client):
    big = b"0" * (app_module.MAX_IMAGE_BYTES + 200_000)
    r = client.post("/api/image", files={"file": ("x.png", big, "image/png")})
    assert r.status_code == 413


def test_text_endpoint_keeps_small_limit(client):
    r = client.post("/api/analyze", content=b"x" * 70_000, headers={"content-type": "application/json"})
    assert r.status_code == 413
