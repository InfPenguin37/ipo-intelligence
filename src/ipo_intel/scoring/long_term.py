"""Rule-based long-term (1-3 year) score out of 100. Not ML: too few IPOs with long outcomes."""
from __future__ import annotations

from ..schema import FUNDAMENTAL_KEYS


WEIGHTS = {
    "growth": 20,
    "profitability": 15,
    "returns (ROCE)": 10,
    "valuation": 20,
    "balance sheet": 10,
    "moat": 15,
    "concentration / regulatory": 10,
}

FIELD_DESCRIPTIONS = {
    "pat_cagr_pct": "PAT CAGR across positive annual profits",
    "latest_qtr_pat_growth_pct": "comparable-quarter PAT growth (not loss reduction)",
    "best_peer_pe": "listed-peer median P/E",
    "capex_intensity_start_pct": "starting-period capex / revenue",
    "capex_intensity_end_pct": "ending-period capex / revenue",
    "proceeds_to_debt_repay_pct": "offer proceeds allocated to debt repayment",
    "single_subsidiary_ebitda_share_pct": "largest subsidiary's share of group EBITDA",
    "market_rank": "industry rank with metric, geography and year",
    "tariff_hike_dependent": "assessment of reliance on future tariff/price increases",
}


def _band(x: float, cuts: list[float], scores: list[int]) -> int:
    for c, s in zip(cuts, scores):
        if x >= c:
            return s
    return scores[-1]


def long_term(f: dict) -> dict:
    missing = [key for key in FUNDAMENTAL_KEYS if f.get(key) is None]
    unusable: list[str] = []
    parts: dict[str, int] = {}
    blended = peg = peer_disc = None

    if f.get("pat_cagr_pct") is not None and f.get("latest_qtr_pat_growth_pct") is not None:
        blended = (f["pat_cagr_pct"] + f["latest_qtr_pat_growth_pct"]) / 2
        parts["growth"] = _band(blended, [25, 18, 12, 8], [10, 8, 6, 4, 2])

    if f.get("ebitda_margin_pct") is not None:
        parts["profitability"] = _band(f["ebitda_margin_pct"], [45, 30, 15], [9, 7, 5, 3])

    if f.get("roce_pct") is not None:
        parts["returns (ROCE)"] = _band(f["roce_pct"], [20, 15, 10], [10, 7, 4, 2])

    pe, peer_pe = f.get("pe"), f.get("best_peer_pe")
    if pe is not None and peer_pe is not None and pe > 0 and peer_pe > 0 and blended is not None:
        peg = pe / blended if blended > 0 else float("inf")
        peer_disc = (peer_pe - pe) / peer_pe * 100
        peg_score = 9 if peg < 1 else 7 if peg < 1.5 else 5 if peg < 2 else 3 if peg < 3 else 1
        disc_score = _band(peer_disc, [20, 10, 0], [8, 7, 5, 3])
        parts["valuation"] = round((peg_score + disc_score) / 2)
    elif pe is not None and pe <= 0:
        unusable.append("P/E is non-positive and is not meaningful for valuation.")
    if f.get("ipowatch_no_listed_peers"):
        unusable.append("IPOWatch reports no listed peer companies; peer valuation is unavailable.")

    balance_keys = (
        "capex_intensity_start_pct",
        "capex_intensity_end_pct",
        "proceeds_to_debt_repay_pct",
    )
    if all(f.get(key) is not None for key in balance_keys):
        bs = 5
        if f["capex_intensity_start_pct"] - f["capex_intensity_end_pct"] > 10:
            bs += 2
        if f["proceeds_to_debt_repay_pct"] > 50:
            bs += 1
        parts["balance sheet"] = bs

    if f.get("market_rank") is not None:
        parts["moat"] = 9 if f["market_rank"] == 1 else 7 if f["market_rank"] == 2 else 5

    if (
        f.get("single_subsidiary_ebitda_share_pct") is not None
        and f.get("tariff_hike_dependent") is not None
    ):
        conc = _band(f["single_subsidiary_ebitda_share_pct"], [90, 75], [4, 6, 8])
        if f["tariff_hike_dependent"]:
            conc -= 1
        parts["concentration / regulatory"] = conc

    covered_weight = sum(WEIGHTS[key] for key in parts)
    coverage_pct = covered_weight
    score = (
        round(sum(parts[key] * WEIGHTS[key] for key in parts) / covered_weight * 10, 1)
        if covered_weight
        else None
    )
    provisional = coverage_pct < 100
    if provisional:
        verdict = "PROVISIONAL — incomplete fundamental coverage"
    elif score >= 75:
        verdict = "STRONG LONG-TERM BUY"
    elif score >= 60:
        verdict = "ACCUMULATE / HOLD 3Y+"
    elif score >= 45:
        verdict = "NEUTRAL"
    else:
        verdict = "AVOID"

    evidence = []
    if f.get("total_income_growth_pct") is not None:
        evidence.append(f"Total income grew {f['total_income_growth_pct']:.1f}% year over year.")
    if f.get("ebitda_growth_pct") is not None:
        evidence.append(f"EBITDA grew {f['ebitda_growth_pct']:.1f}% year over year.")
    if f.get("annual_pat_loss_change_pct") is not None:
        change = f["annual_pat_loss_change_pct"]
        evidence.append(
            f"Annual PAT loss {'narrowed' if change > 0 else 'widened'} "
            f"by {abs(change):.1f}% year over year."
        )
    if f.get("quarterly_pat_loss_narrowing_pct") is not None:
        change = f["quarterly_pat_loss_narrowing_pct"]
        evidence.append(
            f"Comparable quarterly PAT loss {'narrowed' if change > 0 else 'widened'} "
            f"by {abs(change):.1f}% year over year."
        )
    if f.get("ebitda_margin_pct") is not None:
        evidence.append(f"Latest reported EBITDA margin is {f['ebitda_margin_pct']:.2f}%.")
    if f.get("roce_pct") is not None:
        evidence.append(f"Latest available ROCE is {f['roce_pct']:.1f}%.")
    if f.get("latest_total_borrowings_cr") is not None and f.get("latest_net_worth_cr") is not None:
        evidence.append(
            f"Latest reported borrowings are ₹{f['latest_total_borrowings_cr']:,.2f} Cr "
            f"and net worth is ₹{f['latest_net_worth_cr']:,.2f} Cr."
        )
    if (
        f.get("chittorgarh_debt_repayment_cr") is not None
        and f.get("ipowatch_debt_repayment_cr") is not None
        and abs(f["chittorgarh_debt_repayment_cr"] - f["ipowatch_debt_repayment_cr"]) > 1
    ):
        evidence.append(
            "Debt repayment allocation conflicts across Chittorgarh and IPOWatch; "
            "verify against the RHP before using it."
        )

    return {
        "score": score,
        "verdict": verdict,
        "parts": parts,
        "coverage_pct": coverage_pct,
        "provisional": provisional,
        "missing_fields": missing,
        "missing_field_descriptions": [
            f"{key}: {FIELD_DESCRIPTIONS.get(key, key)}" for key in missing
        ],
        "unusable_fields": unusable,
        "evidence": evidence,
        "blended_growth_pct": round(blended, 1) if blended is not None else None,
        "peg": round(peg, 2) if peg is not None else None,
        "peer_discount_pct": round(peer_disc, 1) if peer_disc is not None else None,
    }
