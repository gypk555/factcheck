"""Checks on the static website and the generated notebook."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def test_no_inline_scripts_or_styles():
    html = (WEB / "index.html").read_text()
    assert "<script>" not in html and 'style="' not in html
    assert re.findall(r"<script[^>]*src=\"([^\"]+)\"", html) == ["assets/app.js"]


def test_all_local_assets_exist():
    html = (WEB / "index.html").read_text()
    for ref in re.findall(r'(?:href|src)="((?:assets|data)/[^"]+)"', html):
        assert (WEB / ref).exists(), ref
    css = (WEB / "assets" / "style.css").read_text()
    for ref in re.findall(r'url\("([^"]+)"\)', css):
        assert (WEB / "assets" / ref).exists(), ref


def test_no_external_hosts_in_site():
    for f in [WEB / "index.html", WEB / "assets" / "app.js", WEB / "assets" / "style.css"]:
        text = f.read_text()
        for host in re.findall(r"https?://([\w.-]+)", text):
            assert host in {"factcheck.pib.gov.in", "www.w3.org", "127.0.0.1"}, (f.name, host)


def test_app_js_never_uses_innerhtml():
    js = (WEB / "assets" / "app.js").read_text()
    # actual use, not the comment that says it's never used
    assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|\beval\(|document\.write|new Function", js)


def test_results_data_present_and_consistent():
    m = json.loads((WEB / "data" / "metrics.json").read_text())
    assert m["xfnd_clip"]["mean"]["accuracy"] > 0.9
    live = (WEB / "data" / "live_test.csv").read_text().strip().splitlines()
    assert live[0].startswith("#,claim,fact_check,model") and len(live) == 6
    for i in range(1, 6):
        assert (WEB / "data" / "figures" / f"live_{i}.png").exists()
    for i in range(1, 4):
        assert (WEB / "data" / "figures" / f"wrong_{i}.png").exists()


def test_notebook_embeds_current_server():
    nb = json.loads((ROOT / "colab" / "XFND_CLIP_Server.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    writer = next(c for c in cells if c.startswith("%%writefile"))
    assert writer.split("\n", 1)[1] == (ROOT / "colab" / "server.py").read_text().rstrip("\n") or \
        writer.split("\n", 1)[1].rstrip("\n") == (ROOT / "colab" / "server.py").read_text().rstrip("\n")


def test_notebook_cells_are_valid_python():
    import ast
    nb = json.loads((ROOT / "colab" / "XFND_CLIP_Server.ipynb").read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            ast.parse(src.split("\n", 1)[1] if src.startswith("%%") else src)


def test_csp_matches_between_render_and_dev():
    render = (ROOT / "render.yaml").read_text()
    assert "connect-src 'self' https://*.trycloudflare.com" in render
    assert "'unsafe-inline'" not in render and "'unsafe-eval'" not in render
