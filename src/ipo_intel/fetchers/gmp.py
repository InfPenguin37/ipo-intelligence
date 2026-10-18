"""Optional GMP scraper. GMP is an unofficial grey-market indicator, not exchange data."""
from __future__ import annotations

import pandas as pd

from .. import config
from . import tables as T
from .base import FetchError, Fetcher
from .http import PoliteSession


def parse_gmp(tables: list[pd.DataFrame], ipo_name: str) -> dict:
    """Find the row for `ipo_name` in any table that has a company column and a GMP column."""
    for df in tables:
        cols = {c: T.norm(c) for c in df.columns}
        c_name = next((c for c, n in cols.items() if any(k in n for k in ("company", "ipo", "name"))), None)
        c_gmp = next((c for c, n in cols.items() if n.startswith("gmp")), None)
        if c_name is None or c_gmp is None:
            continue
        c_band = next((c for c, n in cols.items() if "price band" in n or n == "ipo price"), None)
        for _, row in df.iterrows():
            if T.name_matches(ipo_name, str(row[c_name])):
                gmp = T.to_num(row[c_gmp])
                if gmp is None:
                    continue
                out = {"gmp_current": gmp}
                if c_band is not None:
                    band = T.numbers(row[c_band])
                    if band:
                        out["price_band_high"] = max(band)
                return out
    return {}


class GmpFetcher(Fetcher):
    name = "gmp-table"

    def __init__(self, http: PoliteSession | None = None):
        self.http = http or PoliteSession()

    def fetch(self, ipo_name: str) -> dict:
        found = parse_gmp(T.read_tables(self.http.get(config.URLS["gmp"])), ipo_name)
        if not found:
            raise FetchError(f"'{ipo_name}' not found in GMP table at {config.URLS['gmp']}")
        return found
