"""Central settings. Everything that can change between laptop and Render lives here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"

# Input limits (protect the server from huge requests).
MAX_INPUT_CHARS = 20_000
MIN_INPUT_WORDS = 3

# Optional: Google Fact Check Tools API key. The app works without it.
FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()
FACTCHECK_TIMEOUT_S = 6.0
FACTCHECK_RATE_LIMIT_PER_MIN = int(os.environ.get("FACTCHECK_RATE_LIMIT_PER_MIN", "20"))
