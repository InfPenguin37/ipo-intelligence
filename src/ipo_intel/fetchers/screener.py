"""Fundamental data from an explicitly supplied Screener company page."""
from __future__ import annotations

import math
import re
from datetime import date
from urllib.parse import urlparse

import pandas as pd

from . import tables as T
from .base import FetchError, Fetcher
from .http import PoliteSession

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"


def _row_values(
    tables: list[pd.DataFrame],
    label_pattern: str,
    date_pattern: str,
    *,
    annual: bool = False,
    quarterly: bool = False,
) -> list[tuple[date, float]]:
    out = []
    for table in tables:
        if table.empty:
            continue
        dated_columns = [
            re.fullmatch(date_pattern, T.norm(column), re.I)
            for column in table.columns[1:]
        ]
        if annual and any(
            match and match.group(1).lower() != "mar" for match in dated_columns
        ):
            continue
        if quarterly and not any(
            match and _month_number(match.group(1)) != 3 for match in dated_columns
        ):
            continue
        label_col = table.columns[0]
        for _, row in table.iterrows():
            if not re.search(label_pattern, T.norm(row[label_col]), re.I):
                continue
            for column, match in zip(table.columns[1:], dated_columns):
                value = T.to_num(row[column])
                if match and value is not None:
                    month, year = match.groups()
                    out.append((date(int(year), _month_number(month), 1), value))
    return sorted(out)


def _month_number(month: str) -> int:
    return {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }[month[:3].lower()]


def _last_positive_3y_cagr(values: list[tuple[date, float]]) -> float | None:
    if not values:
        return None
    latest_date = values[-1][0]
    start_date = date(latest_date.year - 3, latest_date.month, 1)
    yearly = {period: value for period, value in values}
    periods = [
        date(start_date.year + offset, start_date.month, 1)
        for offset in range(4)
    ]
    if any(period not in yearly or yearly[period] <= 0 for period in periods):
        return None
    return (yearly[latest_date] / yearly[start_date]) ** (1 / 3) * 100 - 100


def parse_fundamentals(tables: list[pd.DataFrame]) -> dict:
    """Extract supported metrics; growth rates are omitted where earnings are non-positive."""
    annual = r"(Mar)\s+(\d{4})"
    quarterly = rf"({_MONTHS})\s+(\d{{4}})"
    out: dict = {}

    annual_pat = _row_values(
        tables, r"^(?:net profit|profit after tax|pat)\b", annual, annual=True
    )
    pat_cagr = _last_positive_3y_cagr(annual_pat)
    if pat_cagr is not None and math.isfinite(pat_cagr):
        out["pat_cagr_pct"] = pat_cagr

    quarterly_pat = _row_values(
        tables,
        r"^(?:net profit|profit after tax|pat)\b",
        quarterly,
        quarterly=True,
    )
    if quarterly_pat:
        latest_date, latest = quarterly_pat[-1]
        prior_date = date(latest_date.year - 1, latest_date.month, 1)
        prior = next((value for period, value in quarterly_pat if period == prior_date), None)
        if prior is not None and prior > 0 and latest > 0:
            out["latest_qtr_pat_growth_pct"] = (latest / prior - 1) * 100

    annual_opm = _row_values(tables, r"^opm\s*%", annual, annual=True)
    if annual_opm:
        out["ebitda_margin_pct"] = annual_opm[-1][1]

    annual_roce = _row_values(tables, r"^roce\s*%", annual, annual=True)
    if annual_roce:
        out["roce_pct"] = annual_roce[-1][1]

    return out


class ScreenerFetcher(Fetcher):
    name = "screener.in"

    def __init__(self, http: PoliteSession | None = None):
        self.http = http or PoliteSession()

    def fetch(self, ipo_name: str, company_url: str | None = None) -> dict:
        if not company_url:
            raise FetchError("provide a direct HTTPS Screener company URL")
        parsed = urlparse(company_url)
        if (
            parsed.scheme != "https"
            or parsed.netloc not in {"screener.in", "www.screener.in"}
            or not parsed.path.startswith("/company/")
            or parsed.query
        ):
            raise FetchError("provide a direct HTTPS Screener company URL; search URLs are not supported")
        values = parse_fundamentals(T.read_tables(self.http.get(company_url)))
        if not values:
            raise FetchError(f"no supported fundamentals found for '{ipo_name}' at {company_url}")
        return values
