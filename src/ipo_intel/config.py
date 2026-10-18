"""Paths, thresholds and source URLs. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("IPO_HOME") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
DB_PATH = DATA / "ipo.db"
RAW_DIR = DATA / "raw"
MANUAL_DIR = DATA / "manual"
MODELS_DIR = DATA / "models"
PRETRAINED_MODELS_PATH = Path(__file__).with_name("pretrained_models.joblib")
SEED_PATH = ROOT / "seed" / "ipo_cases.json"

# Issues at or above this size (Rs crore) are treated as "large" by the ML layer.
LARGE_CUTOFF_CR = float(os.environ.get("IPO_LARGE_CUTOFF_CR", "5000"))
# Minimum number of small-tier IPOs with known outcomes before the gradient-boosted model is trained.
MIN_SMALL_TRAIN = 60

USER_AGENT = "ipo-intelligence/0.1 (personal research tool)"
MIN_REQUEST_DELAY_S = float(os.environ.get("IPO_REQUEST_DELAY_S", "2.5"))

# Source URLs. Chittorgarh's performance tracker endpoint returns JSON by calendar year;
# override IPO_PERF_URL if its endpoint changes.
URLS = {
    "gmp": os.environ.get("IPO_GMP_URL", "https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/"),
    "ipo_list": os.environ.get(
        "IPO_LIST_URL", "https://www.chittorgarh.com/report/ipo-in-india-list-main-board-sme/82/"
    ),
    "perf_tracker": os.environ.get(
        "IPO_PERF_URL",
        "https://webnodejs.chittorgarh.com/cloud/report/data-read/236/1/10/"
        "{year}/{financial_year}/0/mainboard/0?search=",
    ),
}


def scrapers_enabled() -> bool:
    return os.environ.get("IPO_ENABLE_SCRAPERS", "1") == "1"


def ensure_dirs() -> None:
    for d in (DATA, RAW_DIR, MANUAL_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
