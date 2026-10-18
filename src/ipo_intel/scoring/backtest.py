"""Backtest of the rule-based estimator on the hand-collected seed cases."""
from __future__ import annotations

import json

from .. import config
from .short_term import est_listing_gain, gmp_pct, short_verdict


def run() -> tuple[list[dict], dict[str, float]]:
    cases = json.loads(config.SEED_PATH.read_text())["backtest"]
    rows, err = [], {"raw_gmp": [], "v1": [], "v2": []}
    for name, c in cases.items():
        g, sub = c["gmp"], (c.get("subscription") or {})
        kw = dict(
            size_cr=c["issue_size_cr"],
            price_band_high=c["price_band_high"],
            gmp_final=g["final"],
            gmp_open=g.get("open"),
            sub_retail=sub.get("retail"),
        )
        raw = gmp_pct(g["final"], c["price_band_high"])
        v1 = est_listing_gain(**kw, surge_rule=False)
        v2 = est_listing_gain(**kw, surge_rule=True)
        actual = c["actual"]["listing_gain_pct"]
        err["raw_gmp"].append(abs(raw - actual))
        err["v1"].append(abs(v1 - actual))
        err["v2"].append(abs(v2 - actual))
        rows.append(
            {
                "ipo": name,
                "raw_gmp_pct": round(raw, 1),
                "v1_est": round(v1, 1),
                "v2_est": round(v2, 1),
                "actual": actual,
                "call": short_verdict(v2),
                "latest_pct": c["actual"].get("latest_pct"),
            }
        )
    mae = {k: round(sum(v) / len(v), 2) for k, v in err.items()}
    return rows, mae
