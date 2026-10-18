"""Optional scraper for IPO details, subscription and listing history from Chittorgarh.

The page layouts here have NOT been verified against the live site (the development sandbox could
not reach it). Parsers match on row labels and keywords so they tolerate layout changes, but if a
field comes back empty run `ipo probe <url>` and adjust. Respect the site's terms of use.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

from .. import config
from . import tables as T
from .base import FetchError, Fetcher
from .http import PoliteSession

LINK_RE = re.compile(r'<a[^>]+href="(?P<href>[^"]*?/ipo/[^"#?]+)"[^>]*>(?P<text>[^<]{3,})</a>', re.I)


def find_ipo_links(html: str) -> list[tuple[str, str]]:
    seen, out = set(), []
    for m in LINK_RE.finditer(html):
        text, href = m["text"].strip(), m["href"]
        if href not in seen:
            seen.add(href)
            out.append((text, href))
    return out


def best_link(name: str, links: list[tuple[str, str]]) -> str | None:
    for text, href in links:
        if T.name_matches(name, text):
            return href
    texts = {T.norm(t): h for t, h in links}
    close = difflib.get_close_matches(T.norm(name), list(texts), n=1, cutoff=0.6)
    return texts[close[0]] if close else None


def parse_details(tables: list[pd.DataFrame]) -> dict:
    kv = T.kv_pairs(tables)
    out: dict = {}
    fresh = ofs = total = None
    for label, raw in kv.items():
        if "price band" in label:
            nums = T.numbers(raw)
            if nums:
                out["price_band_high"] = max(nums)
        elif "fresh issue" in label and "size" not in label:
            fresh = T.rupee_crore(raw) if T.rupee_crore(raw) is not None else fresh
        elif "offer for sale" in label:
            ofs = T.rupee_crore(raw) if T.rupee_crore(raw) is not None else ofs
        elif label in {"total issue size", "issue size"}:
            total = T.rupee_crore(raw)
    if total is None and (fresh is not None or ofs is not None):
        total = (fresh or 0) + (ofs or 0)
    if total is not None:
        out["issue_size_cr"] = total
    if fresh is not None or ofs is not None:
        f, o = fresh or 0.0, ofs or 0.0
        if f + o > 0:
            out["ofs_share"] = o / (f + o)
            out["structure"] = "OFS" if f == 0 else "Fresh" if o == 0 else "Mixed"
    return out


def parse_subscription(tables: list[pd.DataFrame]) -> dict:
    """Category subscription (times). Uses the last numeric cell in each category row."""
    for df in tables:
        if df.shape[1] < 2:
            continue
        labels = [T.norm(x) for x in df.iloc[:, 0]]
        if not any(re.search(r"qib|qualified", lab) for lab in labels):
            continue
        out: dict = {}
        for lab, (_, row) in zip(labels, df.iterrows()):
            val = T.last_numeric(row.iloc[1:])
            if val is None:
                continue
            if re.search(r"qib|qualified", lab):
                out.setdefault("sub_qib", val)
            elif re.search(r"\bnii\b|non.?institutional|\bhni\b", lab):
                out.setdefault("sub_nii", val)
            elif re.search(r"retail|\brii\b", lab):
                out.setdefault("sub_retail", val)
            elif lab.startswith("total"):
                out.setdefault("sub_overall", val)
        if out:
            return out
    return {}


def parse_perf_tracker(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Listing-day gain per IPO from the performance tracker table."""
    df = T.find_table(tables, ["company", "listing"])
    if df is None:
        return pd.DataFrame()
    cols = {c: T.norm(c) for c in df.columns}

    def pick(*keys):
        for c, n in cols.items():
            if all(k in n for k in keys):
                return c
        return None

    c_name = pick("company")
    c_list = pick("listing", "gain") or pick("listing", "day")
    c_cur = pick("current")
    c_price = pick("issue", "price")
    c_date = pick("listed") or pick("listing", "date")
    out = pd.DataFrame({"name": df[c_name].astype(str).str.strip()})
    if c_list:
        out["listing_gain_pct"] = df[c_list].map(T.to_num)
    if c_cur:
        out["current_gain_pct"] = df[c_cur].map(T.to_num)
    if c_price:
        out["price_band_high"] = df[c_price].map(T.to_num)
    if c_date:
        out["listing_date"] = df[c_date].astype(str)
    return out.dropna(subset=["listing_gain_pct"]) if "listing_gain_pct" in out else out


class ChittorgarhFetcher(Fetcher):
    name = "chittorgarh"

    def __init__(self, http: PoliteSession | None = None):
        self.http = http or PoliteSession()

    def _page(self, ipo_name: str) -> str:
        html = self.http.get(config.URLS["ipo_list"])
        href = best_link(ipo_name, find_ipo_links(html))
        if not href:
            raise FetchError(f"no IPO page found for '{ipo_name}' on {config.URLS['ipo_list']}")
        if href.startswith("/"):
            href = "https://www.chittorgarh.com" + href
        return self.http.get(href)

    def fetch(self, ipo_name: str) -> dict:
        tables = T.read_tables(self._page(ipo_name))
        out = {**parse_details(tables), **parse_subscription(tables)}
        if not out:
            raise FetchError("page fetched but no known fields could be parsed; try `ipo probe`")
        return out

    def collect_history(self, store, limit: int | None = None, progress=print) -> int:
        """Listing gains from the tracker, enriched with size and subscription from each IPO page."""
        html = self.http.get(config.URLS["perf_tracker"])
        perf = parse_perf_tracker(T.read_tables(html))
        if perf.empty:
            raise FetchError("could not parse the performance tracker table; try `ipo probe`")
        links = find_ipo_links(html)
        known = set(store.load_history()["name"]) if len(store.load_history()) else set()
        done = 0
        for _, r in perf.iterrows():
            if r["name"] in known:
                continue
            row = r.to_dict()
            row["source"] = "chittorgarh"
            href = best_link(r["name"], links)
            if href:
                try:
                    url = href if href.startswith("http") else "https://www.chittorgarh.com" + href
                    tables = T.read_tables(self.http.get(url))
                    row.update({**parse_details(tables), **parse_subscription(tables)})
                except FetchError as e:
                    progress(f"  skipped detail page for {r['name']}: {e}")
            store.upsert_history(pd.DataFrame([row]))
            done += 1
            progress(f"[{done}] {r['name']}")
            if limit and done >= limit:
                break
        return done
