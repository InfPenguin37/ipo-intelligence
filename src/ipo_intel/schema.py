"""Standard field names and a snapshot object that remembers where every value came from."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

TEXT_KEYS = {"structure", "listing_date"}
BOOL_KEYS = {"tariff_hike_dependent"}

MARKET_KEYS = [
    "issue_size_cr",
    "price_band_high",
    "structure",  # "OFS", "Fresh" or "Mixed"
    "ofs_share",  # 0..1 share of the issue that is offer-for-sale
    "gmp_open",
    "gmp_peak",
    "gmp_final",  # latest GMP in Rs per share
    "sub_overall",
    "sub_qib",
    "sub_nii",
    "sub_retail",
    "listing_date",
]
FUNDAMENTAL_KEYS = [
    "pat_cagr_pct",
    "latest_qtr_pat_growth_pct",
    "ebitda_margin_pct",
    "roce_pct",
    "pe",
    "best_peer_pe",
    "capex_intensity_start_pct",
    "capex_intensity_end_pct",
    "proceeds_to_debt_repay_pct",
    "single_subsidiary_ebitda_share_pct",
    "market_rank",
    "tariff_hike_dependent",
]
FUNDAMENTAL_CONTEXT_KEYS = [
    "total_income_growth_pct",
    "ebitda_growth_pct",
    "annual_pat_loss_change_pct",
    "quarterly_pat_loss_narrowing_pct",
    "latest_total_income_cr",
    "prior_total_income_cr",
    "latest_annual_ebitda_cr",
    "prior_annual_ebitda_cr",
    "latest_annual_pat_cr",
    "prior_annual_pat_cr",
    "latest_quarter_pat_cr",
    "prior_quarter_pat_cr",
    "latest_total_borrowings_cr",
    "latest_net_worth_cr",
    "chittorgarh_debt_repayment_cr",
    "chittorgarh_net_proceeds_cr",
    "ipowatch_debt_repayment_cr",
    "ipowatch_no_listed_peers",
]
ALL_KEYS = MARKET_KEYS + FUNDAMENTAL_KEYS
REQUIRED_FOR_SHORT_TERM = ["issue_size_cr", "price_band_high", "gmp_final"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def coerce(key: str, raw: Any) -> Any:
    """Turn a string (from the CLI or UI) into the right Python type for `key`."""
    if key in TEXT_KEYS:
        return str(raw).strip()
    if key in BOOL_KEYS:
        return str(raw).strip().lower() in {"1", "true", "yes", "y"}
    return float(re.sub(r"[,\s]", "", str(raw)))


@dataclass
class Datum:
    value: Any
    source: str
    fetched_at: str = field(default_factory=now_iso)


class Snapshot:
    """Everything known about one IPO right now, with a source and timestamp per value."""

    def __init__(self, name: str):
        self.name = name
        self.data: dict[str, Datum] = {}
        self.warnings: list[str] = []

    def set(self, key: str, value: Any, source: str, overwrite: bool = False) -> None:
        if value is None:
            return
        if isinstance(value, float) and value != value:  # NaN
            return
        if key in self.data and not overwrite:
            return
        self.data[key] = Datum(value, source)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.data.get(key)
        return d.value if d is not None else default

    def values(self) -> dict[str, Any]:
        return {k: d.value for k, d in self.data.items()}

    def missing_short_term(self) -> list[str]:
        return [k for k in REQUIRED_FOR_SHORT_TERM if k not in self.data]

    def provenance(self) -> list[dict[str, Any]]:
        order = {k: i for i, k in enumerate(ALL_KEYS + FUNDAMENTAL_CONTEXT_KEYS)}
        rows = [
            {"field": k, "value": d.value, "source": d.source, "fetched_at": d.fetched_at}
            for k, d in self.data.items()
        ]
        return sorted(rows, key=lambda r: order.get(r["field"], 999))
