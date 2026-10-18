"""Two estimators, one per size tier.

SmallModel  - enough history for gradient-boosted quantile regression plus a ridge baseline.
LargeModel  - only a handful of mega-issues exist, so it learns ONE number: how much of the GMP survives
              to listing, shrunk towards a prior. Wide, honest intervals instead of a fancy fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..scoring.short_term import demand_factor, gmp_pct, gmp_ratio, trend_factor
from .features import FEATURES

Z80 = 1.2816  # 80% central interval


class SmallModel:
    def __init__(self, quantiles=(0.1, 0.5, 0.9), random_state: int = 0):
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmallModel":
        keep = y.notna()
        X, y = X.loc[keep, FEATURES], y[keep]
        self.n_train = len(y)
        self.gbm = {
            q: HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_depth=3,
                learning_rate=0.05,
                max_iter=200,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=self.random_state,
            ).fit(X, y)
            for q in self.quantiles
        }
        self.ridge = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-2, 3, 12)),
        ).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES]
        q = np.column_stack([self.gbm[k].predict(X) for k in self.quantiles])
        q.sort(axis=1)  # quantiles must not cross
        out = pd.DataFrame(q, columns=["q10", "q50", "q90"], index=X.index)
        out["ridge"] = self.ridge.predict(X)
        return out


class LargeModel:
    """gain ~ k * (GMP% x demand x trend), k shrunk towards `prior_k` by `prior_strength` pseudo-observations."""

    def __init__(self, prior_k: float = 0.55, prior_strength: float = 3.0, min_sd: float = 5.0):
        self.prior_k, self.prior_strength, self.min_sd = prior_k, prior_strength, min_sd

    @staticmethod
    def x(df: pd.DataFrame) -> pd.Series:
        def num(c):
            return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)

        vals = []
        for price, final, open_, retail in zip(num("price_band_high"), num("gmp_final"), num("gmp_open"), num("sub_retail")):
            if np.isnan(price) or np.isnan(final):
                vals.append(np.nan)
            else:
                vals.append(gmp_pct(final, price) * demand_factor(retail) * trend_factor(gmp_ratio(final, open_)))
        return pd.Series(vals, index=df.index, dtype=float)

    def _k(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return self.prior_k
        lam = self.prior_strength * float(np.mean(x**2)) if np.mean(x**2) > 0 else 1.0
        return float((np.sum(x * y) + lam * self.prior_k) / (np.sum(x**2) + lam))

    def fit(self, df: pd.DataFrame) -> "LargeModel":
        d = df.copy()
        d["_x"] = self.x(d)
        d["_y"] = pd.to_numeric(d.get("listing_gain_pct"), errors="coerce")
        d = d.dropna(subset=["_x", "_y"])
        x, y = d["_x"].to_numpy(), d["_y"].to_numpy()
        self.n_train = len(x)
        self.k = self._k(x, y)
        if len(x) >= 3:  # leave-one-out error for the interval width
            res = [y[i] - self._k(np.delete(x, i), np.delete(y, i)) * x[i] for i in range(len(x))]
            self.sd = max(float(np.sqrt(np.mean(np.square(res)))), self.min_sd)
        else:
            self.sd = self.min_sd
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        mid = self.k * self.x(df)
        return pd.DataFrame({"q10": mid - Z80 * self.sd, "q50": mid, "q90": mid + Z80 * self.sd}, index=df.index)
