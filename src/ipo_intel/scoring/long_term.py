"""Rule-based long-term (1-3 year) score out of 100. Not ML: too few IPOs with long outcomes."""
from __future__ import annotations

from ..schema import FUNDAMENTAL_KEYS


class MissingFields(Exception):
    def __init__(self, missing: list[str]):
        super().__init__(f"missing: {', '.join(missing)}")
        self.missing = missing


WEIGHTS = {
    "growth": 20,
    "profitability": 15,
    "returns (ROCE)": 10,
    "valuation": 20,
    "balance sheet": 10,
    "moat": 15,
    "concentration / regulatory": 10,
}


def _band(x: float, cuts: list[float], scores: list[int]) -> int:
    for c, s in zip(cuts, scores):
        if x >= c:
            return s
    return scores[-1]


def long_term(f: dict) -> dict:
    missing = [k for k in FUNDAMENTAL_KEYS if f.get(k) is None]
    if missing:
        raise MissingFields(missing)

    blended = (f["pat_cagr_pct"] + f["latest_qtr_pat_growth_pct"]) / 2
    peg = f["pe"] / blended if blended > 0 else float("inf")
    peer_disc = (f["best_peer_pe"] - f["pe"]) / f["best_peer_pe"] * 100

    s: dict[str, int] = {}
    s["growth"] = _band(blended, [25, 18, 12, 8], [10, 8, 6, 4, 2])
    s["profitability"] = _band(f["ebitda_margin_pct"], [45, 30, 15], [9, 7, 5, 3])
    s["returns (ROCE)"] = _band(f["roce_pct"], [20, 15, 10], [10, 7, 4, 2])
    peg_score = 9 if peg < 1 else 7 if peg < 1.5 else 5 if peg < 2 else 3 if peg < 3 else 1
    disc_score = _band(peer_disc, [20, 10, 0], [8, 7, 5, 3])
    s["valuation"] = round((peg_score + disc_score) / 2)
    bs = 5
    if f["capex_intensity_start_pct"] - f["capex_intensity_end_pct"] > 10:
        bs += 2
    if f["proceeds_to_debt_repay_pct"] > 50:
        bs += 1
    s["balance sheet"] = bs
    s["moat"] = 9 if f["market_rank"] == 1 else 7 if f["market_rank"] == 2 else 5
    conc = _band(f["single_subsidiary_ebitda_share_pct"], [90, 75], [4, 6, 8])
    if f["tariff_hike_dependent"]:
        conc -= 1
    s["concentration / regulatory"] = conc

    score = sum(s[k] * WEIGHTS[k] for k in WEIGHTS) / 10
    if score >= 75:
        verdict = "STRONG LONG-TERM BUY"
    elif score >= 60:
        verdict = "ACCUMULATE / HOLD 3Y+"
    elif score >= 45:
        verdict = "NEUTRAL"
    else:
        verdict = "AVOID"
    return {
        "score": round(score, 1),
        "verdict": verdict,
        "parts": s,
        "blended_growth_pct": round(blended, 1),
        "peg": round(peg, 2),
        "peer_discount_pct": round(peer_disc, 1),
    }
