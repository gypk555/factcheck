import httpx
import pytest
from fastapi.testclient import TestClient

from truthlens import app as app_module
from truthlens import config, factcheck


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_index_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "TruthLens" in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_no_inline_styles_or_scripts(client):
    html = client.get("/").text
    assert 'style="' not in html
    assert "<script>" not in html


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_api_docs_hidden_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_analyze_ok(client):
    r = client.post("/api/analyze", json={"text": "BREAKING shocking video you won't believe, watch now"})
    body = r.json()
    assert r.status_code == 200
    assert body["label"] in {"FAKE", "REAL", "UNCERTAIN"}
    assert 0 <= body["prob_fake"] <= 1
    assert body["highlights"]


def test_analyze_rejects_too_few_words(client):
    r = client.post("/api/analyze", json={"text": "hi"})
    assert r.status_code == 422
    assert r.json() == {"detail": "Enter at least 3 words."}


def test_analyze_rejects_too_long_without_echo(client):
    text = "word " * (config.MAX_INPUT_CHARS // 5 + 10)
    r = client.post("/api/analyze", json={"text": text})
    assert r.status_code == 422
    assert "too long" in r.json()["detail"]
    assert "word word" not in r.text  # the input is not echoed back


def test_oversized_body_rejected(client):
    r = client.post("/api/analyze", content=b"x" * 70_000, headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_html_in_input_is_returned_as_plain_data(client):
    r = client.post("/api/analyze", json={"text": "<script>alert(1)</script> breaking news today now"})
    assert r.status_code == 200  # the browser inserts it with textContent, so it is never executed


def test_factcheck_disabled_without_key(client, monkeypatch):
    monkeypatch.setattr(config, "FACTCHECK_API_KEY", "")
    assert client.post("/api/factcheck", json={"query": "some claim"}).status_code == 404


def test_rate_limiter():
    rl = app_module.RateLimiter(2)
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    assert rl.allow("b")


class _Req:
    def __init__(self, headers, host="10.0.0.1"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def test_client_ip_uses_proxy_appended_entry(monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_PROXY", True)
    assert app_module.client_ip(_Req({"x-forwarded-for": "6.6.6.6, 1.2.3.4"})) == "1.2.3.4"


def test_client_ip_ignores_headers_without_trusted_proxy(monkeypatch):
    monkeypatch.setattr(app_module, "TRUST_PROXY", False)
    assert app_module.client_ip(_Req({"x-forwarded-for": "6.6.6.6"}, host="5.5.5.5")) == "5.5.5.5"


def test_rate_limiter_full_table_evicts_least_recent(monkeypatch):
    rl = app_module.RateLimiter(1)
    monkeypatch.setattr(rl, "MAX_KEYS", 2)
    assert rl.allow("a") and rl.allow("b")
    assert not rl.allow("a")          # "a" is limited, and now the most recently seen
    assert rl.allow("c")              # newcomer is never locked out; "b" is forgotten
    assert "b" not in rl.hits and len(rl.hits) == 2
    assert not rl.allow("a")          # "a" keeps its history


def test_visitor_key_groups_ipv6_by_64():
    k = app_module.visitor_key
    assert k("2001:db8:1:2::1") == k("2001:db8:1:2:ffff::9") == "2001:db8:1:2::/64"
    assert k("2001:db8:1:3::1") != k("2001:db8:1:2::1")
    assert k("203.0.113.7") == "203.0.113.7"
    assert k("::ffff:203.0.113.7") == "203.0.113.7"
    assert k("unknown") == "unknown"


# ----- fact-check parsing and safety
def test_factcheck_parse_drops_unsafe_links():
    data = {"claims": [
        {"text": "Claim A", "claimReview": [{"url": "javascript:alert(1)", "textualRating": "False"}]},
        {"text": "Claim B", "claimant": "Someone", "claimReview": [{
            "url": "https://example.org/check", "textualRating": "False",
            "publisher": {"name": "Checker"}, "reviewDate": "2024-05-01T00:00:00Z"}]},
    ]}
    out = factcheck.parse_response(data)
    assert len(out) == 1
    assert out[0]["url"] == "https://example.org/check"
    assert out[0]["date"] == "2024-05-01"


@pytest.mark.anyio
async def test_factcheck_search_hides_upstream_errors(monkeypatch):
    monkeypatch.setattr(config, "FACTCHECK_API_KEY", "secret-key")

    def handler(request):
        return httpx.Response(500, text="boom secret-key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(factcheck.FactCheckUnavailable) as exc:
            await factcheck.search("claim text", client=c)
    assert "secret-key" not in str(exc.value)


@pytest.fixture
def anyio_backend():
    return "asyncio"
