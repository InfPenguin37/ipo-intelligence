"""Rule-based listing-gain estimate (the baseline the ML layer is compared against).

The adjustments were chosen with knowledge of four past large IPOs (see `ipo backtest`), so
treat the backtest as a sanity check, not out-of-sample validation.
"""
from __future__ import annotations

import math


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def size_haircut(size_cr: float) -> float:
    """Large issues absorb more supply, so the GMP overstates the listing pop."""
    if size_cr < 1000:
        return 0.90
    if size_cr < 5000:
        return 0.75
    return 0.55


def demand_factor(retail_x: float | None) -> float:
    """Retail depth decides who is left holding the stock on listing day."""
    if _missing(retail_x):
        return 1.0
    if retail_x < 1:
        return 0.60
    if retail_x < 3:
        return 0.85
    return 1.0


def gmp_ratio(gmp_final: float | None, gmp_open: float | None) -> float:
    if _missing(gmp_final) or _missing(gmp_open) or gmp_open <= 0:
        return 1.0
    return gmp_final / gmp_open


def trend_factor(ratio: float) -> float:
    """A GMP that collapsed during the issue is a warning sign."""
    return 0.8 if ratio < 0.5 else 1.0


def gmp_pct(gmp_final: float, price_band_high: float) -> float:
    return gmp_final / price_band_high * 100


def est_listing_gain(
    size_cr,
    price_band_high,
    gmp_final,
    gmp_open=None,
    sub_retail=None,
    surge_rule: bool = True,
) -> float:
    """Estimated listing gain in %, or NaN if the essentials are missing."""
    if _missing(size_cr) or _missing(price_band_high) or _missing(gmp_final):
        return float("nan")
    pct = gmp_pct(gmp_final, price_band_high)
    ratio = gmp_ratio(gmp_final, gmp_open)
    hair = size_haircut(size_cr)
    if surge_rule and ratio >= 1.4:  # GMP still climbing during the issue: demand is real
        hair = max(hair, 0.90)
    return pct * hair * demand_factor(sub_retail) * trend_factor(ratio)


def short_verdict(gain: float) -> str:
    if _missing(gain):
        return "INSUFFICIENT DATA"
    if gain >= 15:
        return "STRONG APPLY"
    if gain >= 6:
        return "APPLY"
    if gain >= 0:
        return "NEUTRAL (flat listing likely)"
    return "AVOID"


def scenarios(size_cr, price_band_high, gmp_final, gmp_open=None) -> list[dict]:
    """Bear/base/bull listing-gain scenarios, used before final retail subscription is known."""
    spec = [
        ("Bear: retail under 1x, GMP fades by about a third", 0.5, 0.68),
        ("Base: retail 1-3x, GMP holds", 1.5, 1.0),
        ("Bull: retail over 3x, GMP rises by about a quarter", 4.0, 1.27),
    ]
    out = []
    for label, retail, mult in spec:
        g = gmp_final * mult
        gain = est_listing_gain(size_cr, price_band_high, g, gmp_open, retail)
        out.append({"scenario": label, "gmp": round(g, 1), "est_gain_pct": round(gain, 1), "call": short_verdict(gain)})
    return out
