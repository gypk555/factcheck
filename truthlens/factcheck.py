"""Optional lookup of published fact-checks (Google Fact Check Tools API).

This is what the ML model cannot do on its own: find out whether real
fact-checkers (PolitiFact, AFP, BOOM, ...) have already rated a claim.
It needs a free API key in GOOGLE_FACTCHECK_API_KEY; without one the app
simply hides this panel.

Security notes
- The key stays on the server; the browser only talks to our own /api route.
- Only http(s) links from the API are passed on, so a malicious
  "javascript:" URL can never reach the page.
- Short timeout so a slow external service can't hang our server.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from . import config

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
MAX_QUERY_CHARS = 200


class FactCheckUnavailable(RuntimeError):
    pass


def is_enabled() -> bool:
    return bool(config.FACTCHECK_API_KEY)


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return url if parsed.scheme in ("http", "https") and parsed.netloc else None


def _short(value, limit: int = 300) -> str:
    return str(value or "")[:limit]


def parse_response(data: dict) -> list[dict]:
    """Flatten the API response into a small, safe list for the page."""
    results = []
    for claim in data.get("claims", [])[:10]:
        for review in claim.get("claimReview", [])[:2]:
            url = _safe_url(review.get("url"))
            if not url:
                continue
            results.append({
                "claim": _short(claim.get("text")),
                "claimant": _short(claim.get("claimant"), 120),
                "rating": _short(review.get("textualRating"), 80),
                "publisher": _short((review.get("publisher") or {}).get("name"), 120),
                "title": _short(review.get("title")),
                "url": url,
                "date": _short(review.get("reviewDate"), 10),
            })
    return results[:6]


async def search(query: str, client: httpx.AsyncClient | None = None) -> list[dict]:
    if not is_enabled():
        raise FactCheckUnavailable("Fact-check lookup is not configured.")
    query = " ".join(query.split())[:MAX_QUERY_CHARS]
    if not query:
        return []
    params = {"query": query, "languageCode": "en", "pageSize": 10, "key": config.FACTCHECK_API_KEY}
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=config.FACTCHECK_TIMEOUT_S)
    try:
        resp = await client.get(API_URL, params=params)
        resp.raise_for_status()
        return parse_response(resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        # Never echo the upstream error (it can contain the request URL with the key).
        raise FactCheckUnavailable("The fact-check service did not respond. Try again later.") from exc
    finally:
        if owns_client:
            await client.aclose()
