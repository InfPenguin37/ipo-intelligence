"""Honest evaluation: walk-forward for the small tier, leave-one-out for the large tier.

Every model is compared with two baselines: the raw GMP and the rule-based heuristic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import LargeModel, SmallModel
from .features import build_features
from .targets import TRAINING_TARGET


def _mae(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")


def _sign_acc(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = ~(np.isnan(a) | np.isnan(p))
    return float(np.mean((a[m] > 0) == (p[m] > 0))) if m.any() else float("nan")


def walk_forward_small(df: pd.DataFrame, min_train: int = 60, step: int = 10) -> pd.DataFrame:
    """Train on the past, predict the next `step` IPOs, repeat. Needs a listing_date column."""
    d = df.dropna(subset=[TRAINING_TARGET]).copy()
    d["_date"] = pd.to_datetime(d["listing_date"], errors="coerce")
    d = d.sort_values("_date").reset_index(drop=True)
    X = build_features(d)
    y = pd.to_numeric(d[TRAINING_TARGET])
    parts = []
    for start in range(min_train, len(d), step):
        tr, te = slice(0, start), slice(start, min(start + step, len(d)))
        pred = SmallModel().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        pred["actual"] = y.iloc[te].to_numpy()
        pred["raw_gmp"] = X["gmp_pct"].iloc[te].to_numpy()
        pred["heuristic"] = X["heuristic"].iloc[te].to_numpy()
        pred["name"] = d["name"].iloc[te].to_numpy()
        parts.append(pred)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def loo_large(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=[TRAINING_TARGET]).reset_index(drop=True)
    rows = []
    for i in range(len(d)):
        model = LargeModel().fit(d.drop(index=i))
        p = model.predict(d.iloc[[i]]).iloc[0]
        x = build_features(d.iloc[[i]]).iloc[0]
        rows.append(
            {
                "name": d.loc[i, "name"],
                "actual": d.loc[i, TRAINING_TARGET],
                "q10": p["q10"],
                "q50": p["q50"],
                "q90": p["q90"],
                "raw_gmp": x["gmp_pct"],
                "heuristic": x["heuristic"],
                "k": model.k,
            }
        )
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, model_col: str = "q50", ridge: bool = False) -> dict:
    if res.empty:
        return {"n": 0}
    out = {
        "n": int(len(res)),
        "mae_raw_gmp": _mae(res["actual"], res["raw_gmp"]),
        "mae_heuristic": _mae(res["actual"], res["heuristic"]),
        f"mae_{model_col}": _mae(res["actual"], res[model_col]),
        "sign_acc_raw_gmp": _sign_acc(res["actual"], res["raw_gmp"]),
        f"sign_acc_{model_col}": _sign_acc(res["actual"], res[model_col]),
        "interval_coverage_80": float(((res["actual"] >= res["q10"]) & (res["actual"] <= res["q90"])).mean()),
    }
    if ridge and "ridge" in res:
        out["mae_ridge"] = _mae(res["actual"], res["ridge"])
    return out
