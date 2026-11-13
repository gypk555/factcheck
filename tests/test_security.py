"""Regression tests for the issues found in the security and correctness reviews."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from truthlens import app as app_module
from truthlens import config
from truthlens.model import Predictor
from truthlens.text import clean_text, join_title_body

ADVERSARIAL = [
    "@" * 20_000,
    "a@" * 10_000,
    "a" * 20_000,
    "x" * 19_000 + "tmsnrt",
    "(" * 20_000,
    "read more " * 2_000,
    "https:// " * 2_000,
]


@pytest.mark.parametrize("text", ADVERSARIAL, ids=range(len(ADVERSARIAL)))
def test_cleaning_is_fast_on_adversarial_input(text):
    start = time.perf_counter()
    clean_text(text)
    assert time.perf_counter() - start < 0.5


def test_email_pattern_still_works():
    assert clean_text("contact bob@mail.example.com today") == "contact today"


def test_nfkc_expansion_is_capped():
    # U+FDFA expands to 18 characters under NFKC.
    assert len(clean_text("ﷺ" * 5_000, max_chars=config.MAX_INPUT_CHARS)) <= config.MAX_INPUT_CHARS


def test_title_survives_reuters_dateline():
    doc = join_title_body(clean_text("May Brexit offer would hurt"),
                          clean_text("LONDON (Reuters) - Britain said on Monday ..."))
    assert doc.startswith("May Brexit offer would hurt . Britain said")


def test_new_leak_patterns_removed():
    for s in ["This article was funded in part by SAP. It is independently created by the Reuters "
              "editorial staff. SAP had no editorial involvement in its creation or production.",
              "(This story corrects paragraph 3)", "21WIRE", "Zero Hedge", "Image credit: Flickr"]:
        assert clean_text(s).strip(" .") == "", s


# ----- model behaviour fixes
def tiny():
    return Predictor(["shocking", "video", "shocking video", "said"],
                     idf=np.ones(4), coef=np.array([2.0, 0.5, 1.0, -1.0]), bias=-0.8)


def test_unknown_words_are_uncertain():
    p = tiny().predict("qwzx vbnm plok")
    assert p.label == "UNCERTAIN"
    assert p.warnings


def test_only_links_and_handles_is_uncertain():
    assert tiny().predict("@alice @bob https://x.com/a").label == "UNCERTAIN"


def test_highlight_weights_sum_to_contributions():
    m = tiny()
    text = "Shocking video! " * 4 + "officials said"
    p = m.predict(text)
    vec = m.vectorize(clean_text(text))
    total = sum(v * m.coef[j] for j, v in vec.items())
    assert sum(s["weight"] for s in p.highlights) == pytest.approx(total, abs=1e-3)


# ----- server hardening
@pytest.fixture
def client():
    for lim in app_module.LIMITERS.values():
        lim.hits.clear()
    with TestClient(app_module.app) as c:
        yield c


def test_chunked_body_without_length_rejected(client):
    def gen():
        yield b'{"text": "'
        yield b"x" * 1000
        yield b'"}'
    r = client.post("/api/analyze", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 411


def test_rate_limit_counts_invalid_requests(client, monkeypatch):
    monkeypatch.setitem(app_module.LIMITERS, "/api/analyze", app_module.RateLimiter(3))
    codes = [client.post("/api/analyze", json={"text": "hi"}).status_code for _ in range(4)]
    assert codes == [422, 422, 422, 429]


def test_error_responses_carry_security_headers(client, monkeypatch):
    monkeypatch.setitem(app_module.LIMITERS, "/api/analyze", app_module.RateLimiter(0))
    r = client.post("/api/analyze", json={"text": "one two three"})
    assert r.status_code == 429
    assert "content-security-policy" in r.headers


def test_analyze_is_sync_so_it_runs_in_a_thread():
    import inspect
    assert not inspect.iscoroutinefunction(app_module.analyze)


def test_client_ip_header_override(monkeypatch):
    monkeypatch.setattr(app_module, "CLIENT_IP_HEADER", "cf-connecting-ip")
    monkeypatch.setattr(app_module, "TRUST_PROXY", True)

    class Req:
        headers = {"cf-connecting-ip": "9.9.9.9", "x-forwarded-for": "1.1.1.1"}
        client = None
    assert app_module.client_ip(Req()) == "9.9.9.9"
