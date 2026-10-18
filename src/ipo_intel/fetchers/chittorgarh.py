"""Optional scraper for IPO details, subscription and listing history from Chittorgarh.

Page layouts may change; run `ipo probe <url>` if a field comes back empty. Respect the site's terms of use.
"""
from __future__ import annotations

import difflib
import re
from datetime import date
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

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


def parse_active_ipo_names(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    names: list[str] = []
    seen: set[str] = set()
    for paragraph in soup.select(".content-text p"):
        description = T.norm(paragraph.get_text(" ", strip=True))
        if not (
            "upcoming mainboard ipos in india" in description
            or "current mainboard active ipos" in description
        ):
            continue
        for link in paragraph.find_all("a", href=True):
            if not urlparse(link["href"]).path.startswith("/ipo/"):
                continue
            label = link.get("title") or link.get_text(" ", strip=True)
            name = re.sub(r"\s+IPO$", "", label.strip(), flags=re.I)
            key = T.norm(name)
            if name and key not in seen:
                names.append(name)
                seen.add(key)
    return names


def _schedule_date(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed.date().isoformat() if pd.notna(parsed) else None


def parse_active_ipo_schedule(payload: dict, names: list[str]) -> list[dict[str, str | None]]:
    rows = payload.get("reportTableData") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return [{"name": name, "opening_date": None, "closing_date": None, "listing_date": None} for name in names]

    schedules: list[dict[str, str | None]] = []
    for name in names:
        row = next(
            (
                item
                for item in rows
                if isinstance(item, dict)
                and T.name_matches(name, str(item.get("~IPO") or item.get("Company") or ""))
            ),
            {},
        )
        schedules.append(
            {
                "name": name,
                "opening_date": _schedule_date(row.get("~Issue_Open_Date") or row.get("Opening Date")),
                "closing_date": _schedule_date(row.get("~IssueCloseDate") or row.get("Closing Date")),
                "listing_date": _schedule_date(row.get("~ListingDate") or row.get("Listing Date")),
            }
        )
    return schedules


def best_link(name: str, links: list[tuple[str, str]]) -> str | None:
    for text, href in links:
        if T.name_matches(name, text):
            return href
    texts = {T.norm(t): h for t, h in links}
    close = difflib.get_close_matches(T.norm(name), list(texts), n=1, cutoff=0.6)
    return texts[close[0]] if close else None


def find_subscription_url(html: str, ipo_name: str) -> str | None:
    for link in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = link["href"]
        path = urlparse(href).path.strip("/").split("/")
        if len(path) < 2 or path[0] != "ipo_subscription":
            continue
        slug = path[1].replace("-", " ")
        if T.name_matches(ipo_name, slug):
            return urljoin("https://www.chittorgarh.com", href)
    return None


def parse_details(tables: list[pd.DataFrame]) -> dict:
    kv = T.kv_pairs(tables)
    out: dict = {}
    fresh = ofs = total = None
    for label, raw in kv.items():
        if "price band" in label:
            nums = T.numbers(raw)
            if nums:
                out["price_band_high"] = max(nums)
        elif "listing date" in label:
            listing_date = pd.to_datetime(
                re.sub(r"\s*[A-Za-z]$", "", str(raw).strip()),
                errors="coerce",
            )
            if pd.notna(listing_date):
                out["listing_date"] = listing_date.date().isoformat()
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


def _dated_values(tables: list[pd.DataFrame], row_pattern: str) -> dict[date, float]:
    values: dict[date, float] = {}
    for table in tables:
        if table.empty:
            continue
        for _, row in table.iterrows():
            label = T.norm(row.iloc[0])
            if not re.search(row_pattern, label, re.I):
                continue
            for column, raw in row.iloc[1:].items():
                period = pd.to_datetime(str(column), errors="coerce", dayfirst=True)
                value = T.to_num(raw)
                if pd.notna(period) and value is not None:
                    values[period.date().replace(day=1)] = value
    return values


def parse_financials(tables: list[pd.DataFrame]) -> dict:
    """Extract reported financial trends without treating loss reductions as profit growth."""
    income = _dated_values(tables, r"^(?:total income|revenue)$")
    pat = _dated_values(tables, r"^(?:profit after tax|net profit|pat)$")
    ebitda = _dated_values(tables, r"^ebitda$")
    borrowings = _dated_values(tables, r"^total borrowing")
    net_worth = _dated_values(tables, r"^net worth$")

    annual_income = {period: value for period, value in income.items() if period.month == 3}
    annual_pat = {period: value for period, value in pat.items() if period.month == 3}
    annual_ebitda = {period: value for period, value in ebitda.items() if period.month == 3}
    quarterly_pat = {period: value for period, value in pat.items() if period.month != 3}
    out: dict = {}

    def latest_pair(values: dict[date, float]) -> tuple[date, float, date, float] | None:
        periods = sorted(values)
        if len(periods) < 2:
            return None
        latest_period = periods[-1]
        prior_period = date(latest_period.year - 1, latest_period.month, 1)
        if prior_period not in values:
            return None
        return latest_period, values[latest_period], prior_period, values[prior_period]

    income_pair = latest_pair(annual_income)
    if income_pair:
        _, latest, _, prior = income_pair
        out["latest_total_income_cr"] = latest
        out["prior_total_income_cr"] = prior
        if prior > 0:
            out["total_income_growth_pct"] = (latest / prior - 1) * 100

    ebitda_pair = latest_pair(annual_ebitda)
    if ebitda_pair:
        _, latest, _, prior = ebitda_pair
        out["latest_annual_ebitda_cr"] = latest
        out["prior_annual_ebitda_cr"] = prior
        if prior > 0:
            out["ebitda_growth_pct"] = (latest / prior - 1) * 100

    latest_annual_periods = sorted(annual_income)
    if latest_annual_periods:
        period = latest_annual_periods[-1]
        total_income = annual_income[period]
        ebitda_value = annual_ebitda.get(period)
        if total_income > 0 and ebitda_value is not None:
            out["ebitda_margin_pct"] = ebitda_value / total_income * 100

    pat_pair = latest_pair(annual_pat)
    if pat_pair:
        _, latest, _, prior = pat_pair
        out["latest_annual_pat_cr"] = latest
        out["prior_annual_pat_cr"] = prior
        if latest < 0 and prior < 0:
            out["annual_pat_loss_change_pct"] = (1 - abs(latest) / abs(prior)) * 100
        elif latest > 0 and prior > 0:
            periods = sorted(annual_pat)
            if len(periods) >= 4:
                start_period = periods[-4]
                consecutive = all(
                    date(start_period.year + offset, start_period.month, 1) in annual_pat
                    and annual_pat[date(start_period.year + offset, start_period.month, 1)] > 0
                    for offset in range(4)
                )
                if consecutive:
                    out["pat_cagr_pct"] = (
                        (latest / annual_pat[start_period]) ** (1 / 3) - 1
                    ) * 100

    quarter_pair = latest_pair(quarterly_pat)
    if quarter_pair:
        _, latest, _, prior = quarter_pair
        out["latest_quarter_pat_cr"] = latest
        out["prior_quarter_pat_cr"] = prior
        if latest > 0 and prior > 0:
            out["latest_qtr_pat_growth_pct"] = (latest / prior - 1) * 100
        elif latest < 0 and prior < 0:
            out["quarterly_pat_loss_narrowing_pct"] = (1 - abs(latest) / abs(prior)) * 100

    for name, values in (
        ("latest_total_borrowings_cr", borrowings),
        ("latest_net_worth_cr", net_worth),
    ):
        if values:
            out[name] = values[max(values)]
    return out


def parse_issue_objects(tables: list[pd.DataFrame]) -> dict:
    for table in tables:
        if table.shape[1] < 3:
            continue
        columns = [T.norm(column) for column in table.columns]
        purpose_headers = ("issue object", "objects of the issue")
        if not any(any(header in column for header in purpose_headers) for column in columns):
            continue
        purpose_column = next(
            (
                column
                for column in table.columns
                if any(header in T.norm(column) for header in purpose_headers)
            ),
            None,
        )
        amount_column = next(
            (column for column in table.columns if "amt" in T.norm(column) or "crore" in T.norm(column)),
            None,
        )
        if purpose_column is None or amount_column is None:
            continue
        repayments = [
            T.to_num(row[amount_column])
            for _, row in table.iterrows()
            if "repayment" in T.norm(row[purpose_column])
        ]
        amounts = [value for value in repayments if value is not None]
        total = next(
            (
                T.to_num(row[amount_column])
                for _, row in table.iterrows()
                if T.norm(row[purpose_column]).startswith("total")
            ),
            None,
        )
        if amounts and total is not None:
            return {
                "chittorgarh_debt_repayment_cr": sum(amounts),
                "chittorgarh_net_proceeds_cr": total,
            }
    return {}


def parse_valuation(tables: list[pd.DataFrame]) -> dict:
    for table in tables:
        if table.shape[1] < 2:
            continue
        label_column = table.columns[0]
        row = next(
            (
                row
                for _, row in table.iterrows()
                if re.fullmatch(r"p\s*/\s*e(?:\s*\([^)]*\))?(?:\s*ratio)?", T.norm(row[label_column]))
            ),
            None,
        )
        if row is None:
            continue
        values = [T.to_num(value) for value in row.iloc[1:]]
        pe_values = [value for value in values if value is not None]
        if pe_values:
            return {"pe": pe_values[0]}
    return {}


def parse_subscription(tables: list[pd.DataFrame]) -> dict:
    """Category subscription (times). Uses the last numeric cell in each category row."""
    for df in tables:
        if df.shape[1] < 2:
            continue
        headers = [T.norm(column) for column in df.columns]
        if not any(
            "subscription" in header and ("time" in header or "times" in header)
            for header in headers
        ):
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


def parse_perf_tracker_data(payload: dict) -> pd.DataFrame:
    """Parse the JSON rows returned by Chittorgarh's current performance tracker."""
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows = payload.get("reportTableData")
    if not isinstance(rows, list):
        return pd.DataFrame()

    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("Company")
        gain = T.to_num(row.get("Listing Day Gain (%)"))
        if not isinstance(name, str) or not name.strip() or gain is None:
            continue

        parsed_row = {
            "name": name.strip(),
            "listing_gain_pct": gain,
            "listing_gain_basis": "source_defined_ambiguous",
        }
        listing_date = row.get("~IPO_listing_date") or row.get("Listed On")
        if listing_date:
            parsed_row["listing_date"] = str(listing_date).strip()
        issue_price = T.to_num(row.get("Issue Price (Rs.)"))
        if issue_price is not None:
            parsed_row["price_band_high"] = issue_price
        issue_size = T.to_num(row.get("Issue Size (Rs.cr.)"))
        if issue_size is not None:
            parsed_row["issue_size_cr"] = issue_size
        parsed.append(parsed_row)
    return pd.DataFrame(parsed)


def performance_tracker_url(year: int) -> str:
    fiscal_year = f"{year}-{(year + 1) % 100:02d}"
    return config.URLS["perf_tracker"].format(year=year, financial_year=fiscal_year)


class ChittorgarhFetcher(Fetcher):
    name = "chittorgarh"

    def __init__(self, http: PoliteSession | None = None):
        self.http = http or PoliteSession()

    def fetch_active_ipo_names(self) -> list[str]:
        html = self.http.get(config.URLS["ipo_list"])
        names = parse_active_ipo_names(html)
        if not names:
            raise FetchError(
                "no active or upcoming mainboard IPO names found on "
                f"{config.URLS['ipo_list']}; the page layout may have changed"
            )
        return names

    def fetch_active_ipo_schedule(self) -> list[dict[str, str | None]]:
        names = self.fetch_active_ipo_names()
        today = date.today()
        year = today.year
        financial_year = f"{year}-{(year + 1) % 100:02d}"
        url = config.URLS["ipo_schedule"].format(year=year, financial_year=financial_year)
        payload = self.http.get(url, as_json=True)
        return parse_active_ipo_schedule(payload, names)

    def _page(self, ipo_name: str) -> str:
        html = self.http.get(config.URLS["ipo_list"])
        href = best_link(ipo_name, find_ipo_links(html))
        if not href:
            raise FetchError(f"no IPO page found for '{ipo_name}' on {config.URLS['ipo_list']}")
        if href.startswith("/"):
            href = "https://www.chittorgarh.com" + href
        return self.http.get(href)

    def fetch(self, ipo_name: str) -> dict:
        html = self._page(ipo_name)
        tables = T.read_tables(html)
        out = parse_details(tables)
        out.update(parse_financials(tables))
        out.update(parse_issue_objects(tables))
        out.update(parse_valuation(tables))
        subscription_url = find_subscription_url(html, ipo_name)
        subscription_tables = T.read_tables(self.http.get(subscription_url)) if subscription_url else tables
        out.update(parse_subscription(subscription_tables))
        if not out:
            raise FetchError("page fetched but no known fields could be parsed; try `ipo probe`")
        return out

    def collect_history(
        self, store, limit: int | None = None, from_year: int = 2004, progress=print
    ) -> int:
        """Import historical listing outcomes, newest first, back to `from_year`."""
        today = date.today()
        if not 2004 <= from_year <= today.year:
            raise FetchError(f"start year must be between 2004 and {today.year}, got {from_year}")

        history = store.load_history()
        known = set(history["name"]) if not history.empty else set()
        done = 0
        for year in range(today.year, from_year - 1, -1):
            payload = self.http.get(performance_tracker_url(year), as_json=True)
            perf = parse_perf_tracker_data(payload)
            if perf.empty:
                raise FetchError(
                    f"Chittorgarh performance tracker JSON had no usable rows for {year}; check IPO_PERF_URL "
                    "and the cached response under data/raw/"
                )

            for _, r in perf.iterrows():
                name = r["name"]
                if name in known:
                    continue
                row = r.to_dict()
                row["source"] = "chittorgarh"
                store.upsert_history(pd.DataFrame([row]))
                known.add(name)
                done += 1
                progress(f"[{done}] {name}")
                if limit is not None and done >= limit:
                    return done
        return done
