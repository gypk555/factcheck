"""Local preview of the website with a FAKE model (no BERT/CLIP, nothing downloaded).

    uv run python dev/mock_server.py
    -> open the link it prints (already connected)

It uses the real colab/server.py API code, but the "model" invents plausible
numbers from keywords. Only for checking the UI; the results mean nothing.
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "colab"))
import server  # noqa: E402

KEY = "local-preview-key-0123456789"
PORT = 8080
FAKE_CUES = {"leaked": 0.22, "free": 0.18, "viral": 0.2, "shocking": 0.25, "breaking": 0.24, "reveals": 0.12,
             "telegram": 0.1, "scooters": 0.08, "trip": 0.07, "london": 0.05, "miracle": 0.2}
REAL_CUES = {"parliament": -0.08, "presents": -0.1, "minister": -0.05, "attend": -0.07, "championship": -0.06,
             "budget": -0.03, "stadium": -0.05, "said": -0.06}


class MockExplainer:
    sim_mean = 0.2793

    @staticmethod
    def _h(text):
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def _weights(self, text):
        return [FAKE_CUES.get(w.lower().strip(".,!?'\""), 0) + REAL_CUES.get(w.lower().strip(".,!?'\""), 0)
                + ((self._h(w) % 7) - 3) / 400 for w in text.split()]

    def predict(self, text, img=None):
        time.sleep(1.2)  # feel like a real GPU call
        z = sum(self._weights(text)) * 4 - 0.6
        p = float(1 / (1 + np.exp(-z)))
        h = self._h(text)
        if img is None:
            att = (0.86, 0.0, 0.14)
        else:
            a = np.array([0.5 + (h % 20) / 100, 0.22 + (h % 13) / 100, 0.18 + (h % 11) / 100])
            att = tuple(a / a.sum())
        return {"prob_fake": p, "label": "FAKE" if p >= 0.5 else "REAL", "att_text": att[0], "att_image": att[1],
                "att_mm": att[2], "clip_sim": None if img is None else 0.17 + (h % 200) / 1000}

    def word_importance(self, text, img=None):
        return list(zip(text.split(), self._weights(text)))

    def gradcam(self, text, img):
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w * (0.35 + (self._h(text) % 30) / 100), h * 0.45
        return np.exp(-(((xx - cx) / (0.18 * w)) ** 2 + ((yy - cy) / (0.2 * h)) ** 2)).astype(np.float32)


app = server.create_app(MockExplainer, KEY, ["*"], {"device": "cpu", "gpu": "Mock model (local preview)"})


@app.middleware("http")
async def csp(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("Content-Security-Policy",
                            "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
                            "img-src 'self' data: blob:; connect-src 'self' https://*.trycloudflare.com; "
                            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
    return resp


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    print(f"\n  Preview: http://127.0.0.1:{PORT}/\n  Connection code (paste it in the Connect box): http://127.0.0.1:{PORT}#key={KEY}\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
