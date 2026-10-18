"""Train, save, load and apply the two size-tier models."""
from __future__ import annotations

import joblib
import pandas as pd

from .. import config
from .estimators import LargeModel, SmallModel
from .evaluate import loo_large, summarize, walk_forward_small
from .features import build_features
from .targets import TRAINING_TARGET

MIN_LARGE_GMP_TRAIN = 25


def _coverage(rows: pd.DataFrame, cutoff: float) -> dict:
    fields = (
        TRAINING_TARGET,
        "listing_close_gain_pct",
        "listing_gain_pct",
        "issue_size_cr",
        "price_band_high",
        "gmp_open",
        "gmp_peak",
        "gmp_final",
        "gmp_final_observed_at",
        "sub_overall",
        "sub_qib",
        "sub_nii",
        "sub_retail",
        "listing_date",
    )
    sizes = pd.to_numeric(rows["issue_size_cr"], errors="coerce")
    tiers = {
        "all": rows,
        "small": rows[sizes < cutoff],
        "large": rows[sizes >= cutoff],
    }
    coverage = {}
    for tier, frame in tiers.items():
        counts = {"rows": len(frame)}
        for field in fields:
            counts[f"with_{field}"] = int(frame[field].notna().sum()) if field in frame else 0
        usable = pd.Series(True, index=frame.index)
        for field in (TRAINING_TARGET, "issue_size_cr", "price_band_high", "gmp_final"):
            usable &= frame[field].notna() if field in frame else False
        counts["usable_gmp_model_rows"] = int(usable.sum())
        coverage[tier] = counts
    coverage["label_basis_counts"] = (
        rows["listing_gain_basis"].fillna("unknown").value_counts().to_dict()
        if "listing_gain_basis" in rows
        else {}
    )
    generic_label = rows.get("listing_gain_pct", pd.Series(index=rows.index, dtype=float)).notna()
    open_label = rows.get(TRAINING_TARGET, pd.Series(index=rows.index, dtype=float)).notna()
    close_label = rows.get("listing_close_gain_pct", pd.Series(index=rows.index, dtype=float)).notna()
    coverage["rows_with_ambiguous_only_label"] = int((generic_label & ~open_label & ~close_label).sum())
    reported_gain = pd.to_numeric(
        rows.get("listing_gain_pct", pd.Series(index=rows.index, dtype=float)),
        errors="coerce",
    )
    open_gain = pd.to_numeric(
        rows.get(TRAINING_TARGET, pd.Series(index=rows.index, dtype=float)),
        errors="coerce",
    )
    coverage["rows_where_reported_gain_differs_from_open_pct"] = int(
        ((reported_gain - open_gain).abs() > 0.15).sum()
    )
    return coverage


def train_all(history: pd.DataFrame, cutoff: float | None = None) -> dict:
    cutoff = cutoff or config.LARGE_CUTOFF_CR
    h = history.copy()
    h["issue_size_cr"] = pd.to_numeric(h["issue_size_cr"], errors="coerce")
    if TRAINING_TARGET not in h:
        h[TRAINING_TARGET] = float("nan")
    h[TRAINING_TARGET] = pd.to_numeric(h[TRAINING_TARGET], errors="coerce")
    coverage = _coverage(h, cutoff)
    h = h.dropna(subset=["issue_size_cr", TRAINING_TARGET])
    small, large = h[h["issue_size_cr"] < cutoff], h[h["issue_size_cr"] >= cutoff]

    large_gmp = large.dropna(subset=["price_band_high", "gmp_final"])
    models: dict = {
        "cutoff": cutoff,
        "training_target": TRAINING_TARGET,
        "n_small": len(small),
        "n_large": len(large),
        "n_large_gmp": len(large_gmp),
        "small": None,
        "large": None,
    }
    report: dict = {
        "cutoff": cutoff,
        "training_target": TRAINING_TARGET,
        "n_small": len(small),
        "n_large": len(large),
        "coverage": coverage,
    }

    if len(small) >= config.MIN_SMALL_TRAIN:
        models["small"] = SmallModel().fit(build_features(small), small[TRAINING_TARGET])
        if len(small) >= config.MIN_SMALL_TRAIN + 20:
            wf = walk_forward_small(small, min_train=config.MIN_SMALL_TRAIN)
            report["small_walk_forward"] = summarize(wf, "q50", ridge=True)
    else:
        report["small_note"] = f"need >= {config.MIN_SMALL_TRAIN} small IPOs with known outcomes, have {len(small)}; heuristic is used instead"

    if len(large_gmp) >= MIN_LARGE_GMP_TRAIN:
        models["large"] = LargeModel().fit(large)
        report["large_k"] = round(models["large"].k, 3)
        report["large_interval_sd"] = round(models["large"].sd, 2)
        report["large_gmp_training_rows"] = models["large"].n_train
        report["large_loo"] = summarize(loo_large(large_gmp), "q50")
    else:
        models["large"] = None
        report["large_note"] = (
            f"only {len(large_gmp)} large IPOs have an opening-gain label and usable GMP; "
            f"need at least {MIN_LARGE_GMP_TRAIN}. The heuristic is used instead."
        )

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
        m, tier = models["large"], "large"
    else:
        m, tier = models["small"], "small"
    if m is None:
        return None
    X = row if tier == "large" else build_features(row)
    p = m.predict(X).iloc[0]
    if p[["q10", "q50", "q90"]].isna().any():
        return None
    return {
        "tier": tier,
        "n_train": int(m.n_train),
        "q10": round(float(p["q10"]), 1),
        "q50": round(float(p["q50"]), 1),
        "q90": round(float(p["q90"]), 1),
    }
