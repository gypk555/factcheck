"""Tests for colab/server.py using a fake Explainer (no BERT/CLIP needed)."""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "colab"))
import server  # noqa: E402

KEY = "k" * 32


class FakeExplainer:
    """Same methods and return shapes as model/explain.py's Explainer."""
    sim_mean = 0.2793

    def __init__(self):
        self.calls = 0

    def predict(self, text, img=None):
        self.calls += 1
        fake = "breaking" in text.lower()
        return {"prob_fake": 0.91 if fake else 0.12, "label": "FAKE" if fake else "REAL",
                "att_text": 0.5, "att_image": 0.3 if img else 0.0, "att_mm": 0.2,
                "clip_sim": 0.31 if img is not None else None}

    def word_importance(self, text, img=None):
        return [(w, 0.2 if w.lower() == "breaking" else -0.01) for w in text.split()]

    def gradcam(self, text, img):
        w, h = img.size
        cam = np.zeros((h, w), np.float32)
        cam[: h // 2, : w // 2] = 1.0
        return cam


@pytest.fixture
def client():
    fake = FakeExplainer()
    app = server.create_app(lambda: fake, KEY, ["https://site.example"])
    with TestClient(app) as c:
        c.fake = fake
        yield c


def jpeg(size=(320, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 140, 200)).save(buf, "JPEG")
    return buf.getvalue()


H = {"X-Access-Key": KEY}


def test_rejects_short_key():
    with pytest.raises(ValueError):
        server.create_app(FakeExplainer, "short")


def test_health_requires_key(client):
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Access-Key": "wrong" * 8}).status_code == 401
    r = client.get("/api/health", headers=H)
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["clip_sim_mean"] == 0.2793


def test_analyse_text_only(client):
    r = client.post("/api/analyse", data={"text": "BREAKING  miracle cure found today"}, headers=H)
    assert r.status_code == 200
    b = r.json()
    assert b["label"] == "FAKE" and b["prob_fake"] == 0.91
    assert b["claim"] == "BREAKING miracle cure found today"     # spaces tidied
    assert b["image"] is None and b["clip_sim"] is None and b["clip_match"] is None
    assert b["words"][0] == {"word": "BREAKING", "weight": 0.2}
    assert set(b["modality"]) == {"text", "image", "text_image"}


def test_analyse_with_image(client):
    r = client.post("/api/analyse", data={"text": "Minister opens new bridge in Delhi"},
                    files={"image": ("p.jpg", jpeg(), "image/jpeg")}, headers=H)
    b = r.json()
    assert r.status_code == 200
    assert b["clip_sim"] == 0.31 and b["clip_match"] == "match"
    assert b["image"]["width"] == 320 and b["image"]["height"] == 200
    assert b["image"]["preview"].startswith("data:image/jpeg;base64,")
    heat = Image.open(io.BytesIO(base64.b64decode(b["image"]["heatmap"].split(",")[1])))
    assert heat.mode == "RGBA"
    alpha = np.asarray(heat)[..., 3]
    assert alpha[: heat.height // 4, : heat.width // 4].mean() > 150   # hot corner visible
    assert alpha[-10:, -10:].mean() < 5                                 # cold area see-through


def test_empty_image_field_is_text_only(client):
    r = client.post("/api/analyse", data={"text": "Minister opens new bridge"},
                    files={"image": ("", b"", "application/octet-stream")}, headers=H)
    assert r.status_code == 200 and r.json()["image"] is None


@pytest.mark.parametrize("text,msg", [("hi there", "at least 3 words"), ("word " * 300, "under 1000")])
def test_text_validation(client, text, msg):
    r = client.post("/api/analyse", data={"text": text}, headers=H)
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_missing_text_short_message(client):
    r = client.post("/api/analyse", data={}, headers=H)
    assert r.status_code == 422 and "claim" in r.json()["detail"]


def test_bad_image_rejected(client):
    r = client.post("/api/analyse", data={"text": "one two three four"},
                    files={"image": ("x.png", b"not an image", "image/png")}, headers=H)
    assert r.status_code == 422


def test_tiff_rejected(client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "TIFF")
    r = client.post("/api/analyse", data={"text": "one two three four"},
                    files={"image": ("x.tif", buf.getvalue(), "image/tiff")}, headers=H)
    assert r.status_code == 422


def test_decompression_bomb_rejected(client):
    buf = io.BytesIO()
    Image.new("1", (10_000, 10_000)).save(buf, "PNG")
    r = client.post("/api/analyse", data={"text": "one two three four"},
                    files={"image": ("x.png", buf.getvalue(), "image/png")}, headers=H)
    assert r.status_code == 422


def test_large_body_rejected_with_cors(client):
    r = client.post("/api/analyse", content=b"x" * (11 * 1024 * 1024),
                    headers={**H, "content-type": "application/octet-stream", "Origin": "https://site.example"})
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == "https://site.example"


def test_cors_preflight_allows_key_header(client):
    r = client.options("/api/analyse", headers={"Origin": "https://site.example",
                                                "Access-Control-Request-Method": "POST",
                                                "Access-Control-Request-Headers": "X-Access-Key"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://site.example"


def test_other_origins_not_allowed(client):
    r = client.get("/api/health", headers={**H, "Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_no_docs_exposed(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_heat_overlay_ranges():
    img = server.heat_overlay(np.array([[0.0, 1.0]]))
    a = np.asarray(img)
    assert a[0, 0, 3] == 0 and a[0, 1, 3] > 200


def test_pixel_limit_is_exact(client):
    # 35 MP: above the 24 MP limit but below Pillow's own hard stop (48 MP).
    buf = io.BytesIO()
    Image.new("1", (7000, 5000)).save(buf, "PNG")
    r = client.post("/api/analyse", data={"text": "one two three four"},
                    files={"image": ("x.png", buf.getvalue(), "image/png")}, headers=H)
    assert r.status_code == 422 and "too large" in r.json()["detail"]


def test_key_checked_before_upload_is_read(client):
    r = client.post("/api/analyse", data={"text": "one two three four"},
                    files={"image": ("x.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 401
    assert client.fake.calls == 0


def test_model_load_failure_is_reported():
    def broken():
        raise FileNotFoundError("checkpoints/xfnd_clip.pt")
    with TestClient(server.create_app(broken, KEY)) as c:
        r = c.get("/api/health", headers=H)
        assert r.status_code == 500 and "failed to load" in r.json()["detail"]
        assert "xfnd_clip.pt" in r.json()["detail"]


def test_model_crash_returns_message_with_cors(client):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
    client.fake.word_importance = boom
    r = client.post("/api/analyse", data={"text": "one two three four"},
                    headers={**H, "Origin": "https://site.example"})
    assert r.status_code == 500 and "failed on this post" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == "https://site.example"


def test_word_cap(client):
    r = client.post("/api/analyse", data={"text": "word " * 61}, headers=H)
    assert r.status_code == 422 and "60 words" in r.json()["detail"]
