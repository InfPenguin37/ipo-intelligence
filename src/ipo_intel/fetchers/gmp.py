"""Optional GMP scraper. GMP is an unofficial grey-market indicator, not exchange data."""
from __future__ import annotations

import re
from datetime import datetime, time, timezone
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

from .. import config
from . import tables as T
from .base import FetchError, Fetcher
from .http import PoliteSession


def find_gmp_history_url(html: str, ipo_name: str) -> str | None:
    for link in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = link["href"]
        label = link.get_text(" ", strip=True)
        if "ipo-gmp-grey-market-premium" in href and T.name_matches(ipo_name, label):
            return urljoin(config.URLS["gmp"], href)
    return None


def _candidate_gmp_history_url(ipo_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", ipo_name.lower()).strip("-")
    return urljoin(config.URLS["gmp"], f"/{slug}-ipo-gmp-grey-market-premium/")


def _gmp_page_matches(html: str, ipo_name: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    details = [
        soup.title.get_text(" ", strip=True) if soup.title else "",
        *(meta.get("content", "") for meta in soup.find_all("meta", attrs={"name": "description"})),
        *(heading.get_text(" ", strip=True) for heading in soup.find_all(["h1", "h2"])),
    ]
    return any(T.name_matches(ipo_name, detail) for detail in details)


def find_ipowatch_overview_url(html: str, ipo_name: str) -> str | None:
    for link in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = link["href"]
        path = href.lower().split("?", 1)[0].rstrip("/")
        label = link.get_text(" ", strip=True)
        if path.endswith("-ipo") and T.name_matches(ipo_name, label):
            return urljoin(config.URLS["gmp"], href)
    return None


def parse_ipowatch_overview(tables: list[pd.DataFrame], html: str) -> dict:
    out: dict = {}
    for table in tables:
        if table.shape[1] < 2:
            continue
        columns = {column: T.norm(column) for column in table.columns}
        purpose_column = next(
            (
                column
                for column, label in columns.items()
                if label in {"purpose", "issue objects", "objects of the issue"}
            ),
            None,
        )
        amount_column = next(
            (
                column
                for column, label in columns.items()
                if label == "crores" or "amount" in label or "amt" in label
            ),
            None,
        )
        rows = table
        if purpose_column is None or amount_column is None:
            header_index = next(
                (
                    index
                    for index, row in table.iterrows()
                    if T.norm(row.iloc[0]) in {"purpose", "issue objects", "objects of the issue"}
                    and (
                        T.norm(row.iloc[1]) == "crores"
                        or "amount" in T.norm(row.iloc[1])
                        or "amt" in T.norm(row.iloc[1])
                    )
                ),
                None,
            )
            if header_index is None:
                continue
            purpose_column, amount_column = table.columns[:2]
            rows = table.iloc[header_index + 1 :]
        repayments = [
            T.to_num(row[amount_column])
            for _, row in rows.iterrows()
            if "repayment" in T.norm(row[purpose_column])
        ]
        values = [value for value in repayments if value is not None]
        if values:
            out["ipowatch_debt_repayment_cr"] = sum(values)
            break

    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    peer_section = re.search(
        r"peer group comparison(.{0,500}?)(?:ipo lead managers|company address)",
        text,
        re.I,
    )
    if peer_section and re.search(r"no listed companies", peer_section.group(1), re.I):
        out["ipowatch_no_listed_peers"] = 1.0
    return out


def parse_gmp_history(
    tables: list[pd.DataFrame],
    reference_year: int | None = None,
) -> list[tuple[str, float]]:
    """Parse day-wise GMP rows, using the IPO open date or page publication year."""
    issue_year = None
    for table in tables:
        if table.shape[1] < 2:
            continue
        for _, row in table.iterrows():
            if T.norm(row.iloc[0]) in {"ipo open date", "issue open date"}:
                open_date = pd.to_datetime(row.iloc[1], errors="coerce")
                if pd.notna(open_date):
                    issue_year = open_date.year
                    break
        if issue_year is not None:
            break

    if issue_year is None:
        issue_year = reference_year
    if issue_year is None:
        return []

    observations: dict[str, float] = {}
    for table in tables:
        if table.shape[1] < 2:
            continue
        header = [T.norm(value) for value in table.columns]
        data = table
        if not {"date", "ipo gmp"} <= set(header):
            header = [T.norm(value) for value in table.iloc[0].tolist()]
            data = table.iloc[1:]
        try:
            date_column = header.index("date")
            gmp_column = header.index("ipo gmp")
        except ValueError:
            continue
        updated_column = next(
            (index for index, column in enumerate(header) if "last updated" in column),
            None,
        )

        for _, row in data.iterrows():
            date_text = str(row.iloc[date_column]).strip()
            gmp = T.to_num(row.iloc[gmp_column])
            if not date_text or date_text.lower() == "nan" or gmp is None:
                continue
            date = pd.to_datetime(f"{date_text} {issue_year}", errors="coerce", dayfirst=True)
            if pd.isna(date):
                continue
            updated_time = None
            if updated_column is not None:
                parsed_time = pd.to_datetime(
                    str(row.iloc[updated_column]).strip(),
                    format="%H:%M",
                    errors="coerce",
                )
                if pd.notna(parsed_time):
                    updated_time = parsed_time.time()
            local_time = updated_time or time(0, 0)
            observed_at = datetime.combine(
                date.date(),
                local_time,
                tzinfo=ZoneInfo("Asia/Kolkata"),
            ).astimezone(timezone.utc)
            observations[observed_at.isoformat(timespec="seconds")] = gmp

    return sorted(observations.items())


def _parse_gmp_page_history(html: str) -> list[tuple[str, float]]:
    reference_year = None
    for property_name in ("dateModified", "datePublished"):
        match = re.search(rf'"{property_name}"\s*:\s*"(\d{{4}})-', html)
        if match:
            reference_year = int(match.group(1))
            break
    return parse_gmp_history(T.read_tables(html), reference_year)


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
        self.history_url: str | None = None
        self.overview_url: str | None = None

    def fetch(self, ipo_name: str) -> dict:
        html = self.http.get(config.URLS["gmp"])
        self.history_url = find_gmp_history_url(html, ipo_name)
        self.overview_url = find_ipowatch_overview_url(html, ipo_name)
        found = parse_gmp(T.read_tables(html), ipo_name)
        if found:
            return found

        self.history_url = self.history_url or _candidate_gmp_history_url(ipo_name)
        detail_html = self.http.get(self.history_url)
        if not _gmp_page_matches(detail_html, ipo_name):
            raise FetchError(
                f"'{ipo_name}' not found in the GMP table or matching IPOWatch detail page "
                f"at {config.URLS['gmp']}"
            )
        observations = _parse_gmp_page_history(detail_html)
        if not observations:
            raise FetchError(
                f"no dated GMP observations found on IPOWatch detail page {self.history_url}"
            )
        found = {
            "gmp_current": observations[-1][1],
            "gmp_current_observed_at": observations[-1][0],
        }
        return found

    def fetch_history(self, ipo_name: str) -> list[tuple[str, float]]:
        if self.history_url is None:
            raise FetchError(f"no IPOWatch GMP history page found for '{ipo_name}'")
        observations = _parse_gmp_page_history(self.http.get(self.history_url))
        if not observations:
            raise FetchError(f"no dated GMP observations found at {self.history_url}")
        return observations

    def fetch_overview(self, ipo_name: str) -> dict:
        if self.overview_url is None:
            raise FetchError(f"no IPOWatch overview page found for '{ipo_name}'")
        html = self.http.get(self.overview_url)
        return parse_ipowatch_overview(T.read_tables(html), html)
