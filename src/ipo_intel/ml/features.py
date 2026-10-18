"""Feature construction shared by training and prediction. Missing values stay NaN (the models handle them)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scoring.short_term import est_listing_gain

SUB_COLS = ["sub_overall", "sub_qib", "sub_nii", "sub_retail"]
FEATURES = [
    "gmp_pct",
    "gmp_peak_pct",
    "gmp_trend",
    "log_size",
    "ofs_share",
    *[f"{c}_log" for c in SUB_COLS],
    "heuristic",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    price = _num(df, "price_band_high")
    gmp_final = _num(df, "gmp_final")
    gmp_open = _num(df, "gmp_open")
    out = pd.DataFrame(index=df.index)
    out["gmp_pct"] = gmp_final / price * 100
    out["gmp_peak_pct"] = _num(df, "gmp_peak") / price * 100
    out["gmp_trend"] = gmp_final / gmp_open.where(gmp_open > 0)
    out["log_size"] = np.log10(_num(df, "issue_size_cr").where(lambda s: s > 0))
    out["ofs_share"] = _num(df, "ofs_share")
    for c in SUB_COLS:
        out[f"{c}_log"] = np.log1p(_num(df, c).clip(lower=0))
    out["heuristic"] = [
        est_listing_gain(s, p, g, o, r)
        for s, p, g, o, r in zip(
            _num(df, "issue_size_cr"), price, gmp_final, gmp_open, _num(df, "sub_retail")
        )
    ]
    return out[FEATURES]
