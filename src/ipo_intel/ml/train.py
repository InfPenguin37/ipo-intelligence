"""Train, save, load and apply the two size-tier models."""
from __future__ import annotations

import joblib
import pandas as pd

from .. import config
from .estimators import LargeModel, SmallModel
from .evaluate import loo_large, summarize, walk_forward_small
from .features import build_features


def train_all(history: pd.DataFrame, cutoff: float | None = None) -> dict:
    cutoff = cutoff or config.LARGE_CUTOFF_CR
    h = history.copy()
    h["issue_size_cr"] = pd.to_numeric(h["issue_size_cr"], errors="coerce")
    h["listing_gain_pct"] = pd.to_numeric(h["listing_gain_pct"], errors="coerce")
    h = h.dropna(subset=["issue_size_cr", "listing_gain_pct"])
    small, large = h[h["issue_size_cr"] < cutoff], h[h["issue_size_cr"] >= cutoff]

    models: dict = {"cutoff": cutoff, "n_small": len(small), "n_large": len(large), "small": None, "large": None}
    report: dict = {"cutoff": cutoff, "n_small": len(small), "n_large": len(large)}

    if len(small) >= config.MIN_SMALL_TRAIN:
        models["small"] = SmallModel().fit(build_features(small), small["listing_gain_pct"])
        if len(small) >= config.MIN_SMALL_TRAIN + 20:
            wf = walk_forward_small(small, min_train=config.MIN_SMALL_TRAIN)
            report["small_walk_forward"] = summarize(wf, "q50", ridge=True)
    else:
        report["small_note"] = f"need >= {config.MIN_SMALL_TRAIN} small IPOs with known outcomes, have {len(small)}; heuristic is used instead"

    models["large"] = LargeModel().fit(large)  # works with 0 rows: falls back to the prior
    report["large_k"] = round(models["large"].k, 3)
    report["large_interval_sd"] = round(models["large"].sd, 2)
    if len(large) >= 3:
        report["large_loo"] = summarize(loo_large(large), "q50")
    else:
        report["large_note"] = f"only {len(large)} large IPOs with outcomes; model is essentially the prior with a wide band"

    config.ensure_dirs()
    joblib.dump(models, config.MODELS_DIR / "models.joblib")
    return report


def load_models() -> dict | None:
    path = config.MODELS_DIR / "models.joblib"
    return joblib.load(path) if path.exists() else None


def predict_one(models: dict | None, values: dict) -> dict | None:
    """Prediction for one IPO from a flat dict of standard fields. None if no applicable trained model."""
    if not models:
        return None
    size = values.get("issue_size_cr")
    if size is None:
        return None
    row = pd.DataFrame([values])
    if size >= models["cutoff"]:
        m, tier, n = models["large"], "large", models["n_large"]
    else:
        m, tier, n = models["small"], "small", models["n_small"]
    if m is None:
        return None
    X = row if tier == "large" else build_features(row)
    p = m.predict(X).iloc[0]
    if p[["q10", "q50", "q90"]].isna().any():
        return None
    return {
        "tier": tier,
        "n_train": int(n),
        "q10": round(float(p["q10"]), 1),
        "q50": round(float(p["q50"]), 1),
        "q90": round(float(p["q90"]), 1),
    }
