"""Parser tests use small SYNTHETIC HTML fixtures shaped like the real pages. They prove the parsing
logic, not that the live sites still look like this."""
from datetime import date

import pandas as pd

from ipo_intel.fetchers import tables as T
from ipo_intel.fetchers.chittorgarh import (
    ChittorgarhFetcher,
    best_link,
    find_ipo_links,
    parse_details,
    parse_perf_tracker,
    parse_perf_tracker_data,
    parse_subscription,
)
from ipo_intel.fetchers.gmp import find_gmp_history_url, parse_gmp, parse_gmp_history
from ipo_intel.history_import import (
    map_investorgain_csv,
    map_kaggle_workbook,
    merge_kaggle_history,
)
from ipo_intel.store import HISTORY_COLUMNS

DETAILS = """<table>
<tr><td>Price Band</td><td>₹1065 to ₹1119 per share</td></tr>
<tr><td>Fresh Issue</td><td>₹30,213.00 Cr</td></tr>
<tr><td>Offer for Sale</td><td>₹0 Cr</td></tr>
<tr><td>Total Issue Size</td><td>₹30,213.00 Cr</td></tr>
</table>"""

SUBSCRIPTION = """<table><thead><tr><th>Category</th><th>Subscription (times)</th></tr></thead><tbody>
<tr><td>QIB</td><td>12.68</td></tr>
<tr><td>Non Institutional Investors (NII)</td><td>6.55</td></tr>
<tr><td>bNII (bids above ₹10L)</td><td>7.78</td></tr>
<tr><td>Retail Investors (RII)</td><td>1.39</td></tr>
<tr><td>Total</td><td>5.71</td></tr></tbody></table>"""

GMP = """<table><thead><tr><th>Company</th><th>GMP*</th><th>Trend</th><th>Price Band</th><th>Est. Gain</th><th>Date</th></tr></thead><tbody>
<tr><td>Paras Healthcare Upcoming</td><td>₹95</td><td>x</td><td>₹693</td><td>₹788 (13.71%)</td><td>21-23 Oct</td></tr>
<tr><td>Jio Platforms Upcoming</td><td>₹220</td><td>x</td><td>₹1,119</td><td>₹1339 (19.66%)</td><td>21-23 Oct</td></tr>
</tbody></table>"""

PERF = """<table><thead><tr><th>Company</th><th>Listing Day Gain / Loss</th><th>Current Gain / Loss</th></tr></thead><tbody>
<tr><td>National Stock Exchange of India Ltd.</td><td>1.85%</td><td>-2.45%</td></tr>
<tr><td>Moneyview Ltd.</td><td>58.47%</td><td>58.91%</td></tr></tbody></table>"""


def test_read_tables_parses_html_table():
    tables = T.read_tables("<table><tr><th>Name</th></tr><tr><td>IPO</td></tr></table>")
    assert len(tables) == 1
    assert tables[0].iloc[0, 0] == "IPO"


def test_details_fresh_issue():
    out = parse_details(T.read_tables(DETAILS))
    assert out["price_band_high"] == 1119
    assert out["issue_size_cr"] == 30213
    assert out["structure"] == "Fresh" and out["ofs_share"] == 0


def test_subscription_picks_main_nii_row_not_bnii():
    out = parse_subscription(T.read_tables(SUBSCRIPTION))
    assert out == {"sub_qib": 12.68, "sub_nii": 6.55, "sub_retail": 1.39, "sub_overall": 5.71}


def test_subscription_ignores_share_allocation_tables():
    allocations = """<table>
    <tr><th>Category</th><th>Shares Offered</th><th>Amt (₹ Cr.)</th><th>Size (%)</th></tr>
    <tr><td>QIB</td><td>248480366</td><td>3752.05</td><td>74.84%</td></tr>
    <tr><td>Retail</td><td>33130716</td><td>500.27</td><td>9.98%</td></tr>
    </table>"""
    assert parse_subscription(T.read_tables(allocations)) == {}


def test_chittorgarh_fetches_subscription_from_dedicated_ipo_page():
    profile = """
    <a href="/ipo/zetwerk-ipo/3091/">Zetwerk IPO</a>
    <a href="/ipo_subscription/zetwerk-ipo/3091/">Subscription</a>
    <table>
      <tr><td>Price Band</td><td>₹145 to ₹151 per share</td></tr>
      <tr><td>Total Issue Size</td><td>₹5,012 Cr</td></tr>
    </table>
    """
    subscription = """<table>
    <thead><tr><th>Investor Category</th><th>Subscription (times)</th></tr></thead>
    <tbody>
      <tr><td>Qualified Institutional</td><td>0.37x</td></tr>
      <tr><td>Non Institutional</td><td>0.17x</td></tr>
      <tr><td>Retail Individual</td><td>0.37x</td></tr>
      <tr><td>Total Subscription</td><td>0.31x</td></tr>
    </tbody></table>"""

    class FakeHttp:
        def __init__(self):
            self.urls = []

        def get(self, url):
            self.urls.append(url)
            if url.endswith("/ipo-in-india-list-main-board-sme/82/"):
                return '<a href="/ipo/zetwerk-ipo/3091/">Zetwerk IPO</a>'
            if url.endswith("/ipo/zetwerk-ipo/3091/"):
                return profile
            if url.endswith("/ipo_subscription/zetwerk-ipo/3091/"):
                return subscription
            raise AssertionError(f"unexpected URL: {url}")

    http = FakeHttp()
    result = ChittorgarhFetcher(http).fetch("Zetwerk")

    assert result["sub_overall"] == 0.31
    assert result["sub_qib"] == 0.37
    assert result["sub_nii"] == 0.17
    assert result["sub_retail"] == 0.37
    assert http.urls[-1] == "https://www.chittorgarh.com/ipo_subscription/zetwerk-ipo/3091/"


def test_gmp_row_matching_ignores_status_suffix():
    out = parse_gmp(T.read_tables(GMP), "Jio Platforms")
    assert out == {"gmp_current": 220.0, "price_band_high": 1119.0}
    assert parse_gmp(T.read_tables(GMP), "Nonexistent Corp") == {}


def test_find_gmp_history_page_and_parse_daywise_history():
    current_page = """
    <a href="/zetwerk-ipo-gmp-grey-market-premium/">Zetwerk</a>
    <a href="/other-ipo-gmp-grey-market-premium/">Other Company</a>
    """
    assert find_gmp_history_url(current_page, "Zetwerk") == (
        "https://ipowatch.in/zetwerk-ipo-gmp-grey-market-premium/"
    )
    assert find_gmp_history_url(current_page, "Unknown IPO") is None

    history_tables = [
        pd.DataFrame(
            [
                ["Date", "IPO GMP", "GMP Trend", "Gain", "Last Updated"],
                ["17 October", "₹5", "down", "3.31%", "17:00"],
                ["16 October", "₹11", "down", "7.28%", "17:14"],
                ["10 October", "₹-", "-", "-", "17:39"],
            ]
        ),
        pd.DataFrame([["IPO Open Date", "October 15, 2026"]]),
    ]

    assert parse_gmp_history(history_tables) == [
        ("2026-10-16T11:44:00+00:00", 11.0),
        ("2026-10-17T11:30:00+00:00", 5.0),
    ]


def test_perf_tracker():
    df = parse_perf_tracker(T.read_tables(PERF))
    assert list(df["name"]) == ["National Stock Exchange of India Ltd.", "Moneyview Ltd."]
    assert list(df["listing_gain_pct"]) == [1.85, 58.47]


def test_perf_tracker_json():
    df = parse_perf_tracker_data(
        {
            "reportTableData": [
                {
                    "Company": "Moneyview Ltd.",
                    "~IPO_listing_date": "2026-10-01T00:00:00.000Z",
                    "Issue Price (Rs.)": 34,
                    "Issue Size (Rs.cr.)": "1,091.68",
                    "Listing Day Gain (%)": 58.47,
                },
                {"Company": "No outcome yet", "Listing Day Gain (%)": None},
            ]
        }
    )
    assert df.to_dict("records") == [
        {
            "name": "Moneyview Ltd.",
            "listing_gain_pct": 58.47,
            "listing_date": "2026-10-01T00:00:00.000Z",
            "price_band_high": 34.0,
            "issue_size_cr": 1091.68,
        }
    ]


def test_collect_history_uses_json_tracker_and_respects_limit():
    class FakeHttp:
        def __init__(self):
            self.args = None

        def get(self, url, as_json=False):
            self.args = (url, as_json)
            return {
                "reportTableData": [
                    {
                        "Company": "Already saved",
                        "Listing Day Gain (%)": 10,
                    },
                    {
                        "Company": "New IPO",
                        "Issue Price (Rs.)": 120,
                        "Issue Size (Rs.cr.)": "500.00",
                        "Listing Day Gain (%)": -2.5,
                    },
                    {
                        "Company": "Beyond limit",
                        "Listing Day Gain (%)": 5,
                    },
                ]
            }

    class FakeStore:
        def __init__(self):
            self.rows = []

        def load_history(self):
            return pd.DataFrame({"name": ["Already saved"]})

        def upsert_history(self, frame):
            self.rows.extend(frame.to_dict("records"))

    http = FakeHttp()
    store = FakeStore()
    assert ChittorgarhFetcher(http).collect_history(store, limit=1, progress=lambda _: None) == 1

    assert http.args is not None
    url, as_json = http.args
    assert "/cloud/report/data-read/236/1/10/" in url
    assert as_json is True
    assert [row["name"] for row in store.rows] == ["New IPO"]
    assert store.rows[0]["issue_size_cr"] == 500.0
    assert store.rows[0]["source"] == "chittorgarh"


def test_collect_history_fetches_each_year_back_to_from_year():
    class FakeHttp:
        def __init__(self):
            self.urls = []

        def get(self, url, as_json=False):
            self.urls.append((url, as_json))
            year = int(url.split("/data-read/236/1/10/", 1)[1].split("/", 1)[0])
            return {
                "reportTableData": [
                    {
                        "Company": f"IPO {year}",
                        "Listing Day Gain (%)": year % 10,
                    }
                ]
            }

    class FakeStore:
        def __init__(self):
            self.rows = []

        def load_history(self):
            return pd.DataFrame(columns=["name"])

        def upsert_history(self, frame):
            self.rows.extend(frame.to_dict("records"))

    current_year = date.today().year
    http = FakeHttp()
    store = FakeStore()
    assert ChittorgarhFetcher(http).collect_history(
        store, from_year=current_year - 1, progress=lambda _: None
    ) == 2

    assert [int(url.split("/data-read/236/1/10/", 1)[1].split("/", 1)[0]) for url, _ in http.urls] == [
        current_year,
        current_year - 1,
    ]
    assert all(as_json for _, as_json in http.urls)
    assert [row["name"] for row in store.rows] == [f"IPO {current_year}", f"IPO {current_year - 1}"]


def test_kaggle_history_merge_matches_rows_and_preserves_gmp():
    workbook = pd.DataFrame(
        [
            {
                "Date": "2025-01-02",
                "IPO_Name": "Truncated Company",
                "Issue_Size(crores)": 500,
                "Offer Price": 100,
                "Listing Gain": 15,
                "QIB": 4.2,
                "HNI": 3.1,
                "RII": 2.4,
                "Total": 3.2,
                "GMP Open": 10,
                "GMP Peak": 20,
                "GMP Final": 15,
            },
            {
                "Date": "2025-01-03",
                "IPO_Name": "Power Grid Corporation of India Ltd.",
                "Issue_Size(crores)": 100,
                "Offer Price": 50,
                "Listing Gain": 8,
            },
            {
                "Date": "2010-01-04",
                "IPO_Name": "Power Grid Corporation of India Ltd.",
                "Issue_Size(crores)": 200,
                "Offer Price": 25,
                "Listing Gain": 12,
            },
            {
                "Date": "2025-01-05",
                "IPO_Name": "Pending IPO",
                "Issue_Size(crores)": 100,
                "Offer Price": 100,
                "Listing Gain": None,
            },
        ]
    )
    incoming = map_kaggle_workbook(workbook, "kaggle:test.xlsx")
    existing = pd.DataFrame(
        [
            {
                "name": "Truncated Company Limited",
                "listing_date": "2025-01-02T00:00:00Z",
                "issue_size_cr": 504,
                "price_band_high": 100,
                "gmp_final": 18,
                "listing_gain_pct": 12,
                "source": "chittorgarh",
            }
        ],
        columns=HISTORY_COLUMNS,
    )

    merged = merge_kaggle_history(incoming, existing)

    assert (merged.added, merged.updated, merged.skipped) == (2, 1, 1)
    matched = merged.rows.loc[merged.rows["name"] == "Truncated Company Limited"].iloc[0]
    assert matched["listing_gain_pct"] == 12
    assert matched["issue_size_cr"] == 504
    assert matched["sub_qib"] == 4.2
    assert matched["gmp_open"] == 10
    assert matched["gmp_peak"] == 20
    assert matched["gmp_final"] == 18
    assert matched["source"] == "chittorgarh+kaggle:test.xlsx"
    assert "Power Grid Corporation of India Ltd. (listed 2010-01-04)" in set(merged.rows["name"])

    repeat = merge_kaggle_history(incoming, merged.rows)
    assert (repeat.added, repeat.updated, repeat.skipped) == (0, 3, 1)


def test_investorgain_csv_maps_gmp_and_listing_outcome():
    tracker = pd.DataFrame(
        [
            {
                "Year": 2025,
                "IPO▲▼": "Example IPO",
                "Symbol▲▼": "EXAMPLE",
                "Listing Dt▲▼": "14-Oct-19",
                "Size▲▼": "₹1,234.56 Cr",
                "Sub▲▼": "111.91x",
                "GMP▲▼": "₹215",
                "Price▲▼": "₹320.00",
                "Est Price▲▼": "₹535.00",
                "Listing Price▲▼": "₹626.00 (95.62%)",
                "Listing Day Close▲▼": "₹728.60 (127.69%)",
                "Closing Price (LTP)▲▼": "₹466.65 (629.14%)",
            },
            {
                "Year": 2025,
                "IPO▲▼": "Zero GMP IPO",
                "Listing Dt▲▼": "4-Dec-19",
                "Size▲▼": "₹409.68 Cr",
                "Sub▲▼": "86.91x",
                "GMP▲▼": "₹0",
                "Price▲▼": "₹195.00",
                "Listing Price▲▼": "₹275.00 (41.03%)",
            },
        ]
    )

    mapped = map_investorgain_csv(tracker, "investorgain:test.csv")

    assert mapped.loc[0, "listing_date"] == "2019-10-14"
    assert mapped.loc[0, "issue_size_cr"] == 1234.56
    assert mapped.loc[0, "price_band_high"] == 320
    assert mapped.loc[0, "sub_overall"] == 111.91
    assert mapped.loc[0, "gmp_final"] == 215
    assert mapped.loc[0, "listing_gain_pct"] == 95.62
    assert mapped.loc[1, "gmp_final"] == 0


def test_link_discovery():
    html = '<a href="/ipo/jio-platforms-ipo/3152/">Jio Platforms IPO</a> <a href="/ipo/nse-ipo/3151/">NSE IPO</a>'
    links = find_ipo_links(html)
    assert best_link("Jio Platforms", links) == "/ipo/jio-platforms-ipo/3152/"
    assert best_link("Unknown Co", links) is None


def test_numeric_helpers():
    assert T.to_num("₹1,119.00") == 1119.0
    assert T.to_num("5.71x") == 5.71
    assert T.to_num("-") is None
    assert T.rupee_crore("₹30,213.00 Cr") == 30213.0
