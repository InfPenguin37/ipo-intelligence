"""Parser tests use small SYNTHETIC HTML fixtures shaped like the real pages. They prove the parsing
logic, not that the live sites still look like this."""
from datetime import date

import pandas as pd
import pytest

from ipo_intel.fetchers import tables as T
from ipo_intel.fetchers import FetchError, PoliteSession
from ipo_intel.fetchers.chittorgarh import (
    ChittorgarhFetcher,
    best_link,
    find_ipo_links,
    parse_active_ipo_names,
    parse_active_ipo_schedule,
    parse_financials,
    parse_issue_objects,
    parse_valuation,
    parse_details,
    parse_perf_tracker,
    parse_perf_tracker_data,
    parse_subscription,
)
from ipo_intel.fetchers.gmp import (
    GmpFetcher,
    find_gmp_history_url,
    find_ipowatch_overview_url,
    parse_gmp,
    parse_gmp_history,
    parse_ipowatch_overview,
)
from ipo_intel.fetchers.screener import ScreenerFetcher, parse_fundamentals
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


def test_parse_active_ipo_names_only_returns_current_and_upcoming_issues():
    html = """
    <div class="content-text">
      <p>The upcoming mainboard IPOs in India this week are
        <a href="/ipo/paras-ipo/" title="Paras Healthcare IPO">Paras Healthcare IPO</a>
        <a href="/ipo/jio-ipo/" title="Jio Platforms IPO">Jio Platforms IPO</a>
      </p>
      <p>The current mainboard active IPOs are
        <a href="/ipo/sahajanand-ipo/" title="Sahajanand Medical Technologies IPO">
          Sahajanand Medical Technologies IPO
        </a>
      </p>
      <p>Recently listed IPOs:
        <a href="/ipo/old-ipo/" title="Old Company IPO">Old Company IPO</a>
      </p>
    </div>
    """

    assert parse_active_ipo_names(html) == [
        "Paras Healthcare",
        "Jio Platforms",
        "Sahajanand Medical Technologies",
    ]


def test_fetch_active_ipo_names_uses_mainboard_page_and_reports_layout_changes():
    class FakeHttp:
        def get(self, url):
            assert url.endswith("/82/mainboard/")
            return (
                '<div class="content-text"><p>The upcoming mainboard IPOs in India are '
                '<a href="/ipo/jio/" title="Jio Platforms IPO">Jio Platforms IPO</a></p></div>'
            )

    assert ChittorgarhFetcher(FakeHttp()).fetch_active_ipo_names() == ["Jio Platforms"]

    class EmptyHttp:
        def get(self, url):
            return "<p>Changed page</p>"

    with pytest.raises(FetchError, match="no active or upcoming mainboard IPO names"):
        ChittorgarhFetcher(EmptyHttp()).fetch_active_ipo_names()


def test_parse_active_ipo_schedule_extracts_dates_and_matches_names_without_ltd():
    payload = {
        "reportTableData": [
            {
                "~IPO": "Jio Platforms IPO",
                "~Issue_Open_Date": "2026-10-21T00:00:00.000Z",
                "~IssueCloseDate": "2026-10-23T00:00:00.000Z",
                "~ListingDate": "2026-10-28T00:00:00.000Z",
            },
            {
                "~IPO": "Paras Healthcare IPO",
                "Opening Date": "21-Oct-2026",
                "Closing Date": "23-Oct-2026",
                "Listing Date": "",
            },
        ]
    }

    assert parse_active_ipo_schedule(payload, ["Jio Platforms", "Paras Healthcare"]) == [
        {
            "name": "Jio Platforms",
            "opening_date": "2026-10-21",
            "closing_date": "2026-10-23",
            "listing_date": "2026-10-28",
        },
        {
            "name": "Paras Healthcare",
            "opening_date": "2026-10-21",
            "closing_date": "2026-10-23",
            "listing_date": None,
        },
    ]


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


def test_chittorgarh_financial_trends_keep_losses_distinct_from_growth():
    html = """<table>
    <tr><th>Particulars</th><th>30 Jun 2025</th><th>31 Mar 2025</th><th>30 Jun 2026</th><th>31 Mar 2026</th></tr>
    <tr><td>Total Income</td><td>—</td><td>11492.23</td><td>—</td><td>16100.63</td></tr>
    <tr><td>Profit After Tax</td><td>-1368.47</td><td>-370.71</td><td>-732.98</td><td>-1606.17</td></tr>
    <tr><td>EBITDA</td><td>—</td><td>322.59</td><td>—</td><td>421.34</td></tr>
    <tr><td>Total Borrowings</td><td>1100</td><td>1000</td><td>1250</td><td>1200</td></tr>
    <tr><td>Net Worth</td><td>1800</td><td>2000</td><td>2400</td><td>2500</td></tr>
    </table>"""

    result = parse_financials(T.read_tables(html))

    assert result["total_income_growth_pct"] == pytest.approx((16100.63 / 11492.23 - 1) * 100)
    assert result["ebitda_growth_pct"] == pytest.approx((421.34 / 322.59 - 1) * 100)
    assert result["ebitda_margin_pct"] == pytest.approx(421.34 / 16100.63 * 100)
    assert result["annual_pat_loss_change_pct"] == pytest.approx((1 - 1606.17 / 370.71) * 100)
    assert result["quarterly_pat_loss_narrowing_pct"] == pytest.approx((1 - 732.98 / 1368.47) * 100)
    assert "pat_cagr_pct" not in result
    assert "latest_qtr_pat_growth_pct" not in result
    assert result["latest_total_borrowings_cr"] == 1250
    assert result["latest_net_worth_cr"] == 2400


def test_chittorgarh_issue_objects_extract_repayment_total():
    html = """<table>
    <tr><th>Sr. No.</th><th>Objects of the Issue</th><th>Est Amt (₹ Cr.)</th></tr>
    <tr><td>1</td><td>Repayment of certain borrowings</td><td>1,300</td></tr>
    <tr><td>2</td><td>Repayment of certain borrowings</td><td>860</td></tr>
    <tr><td></td><td>Total</td><td>2,160</td></tr>
    </table>"""

    assert parse_issue_objects(T.read_tables(html)) == {
        "chittorgarh_debt_repayment_cr": 2160,
        "chittorgarh_net_proceeds_cr": 2160,
    }


def test_chittorgarh_parses_loss_making_pe_as_negative():
    table = """<table><tr><th>Valuation Metric</th><th>Pre IPO</th><th>Post IPO</th></tr>
    <tr><td>EPS (₹)</td><td>-10.66</td><td>-ve</td></tr>
    <tr><td>P/E (x)</td><td>-14.17</td><td>-ve</td></tr></table>"""
    assert parse_valuation(T.read_tables(table)) == {"pe": -14.17}


def test_ipowatch_overview_resolves_page_and_flags_peer_and_debt_data():
    tracker = '<a href="/zetwerk-ipo/">Zetwerk Manufacturing IPO</a>'
    assert find_ipowatch_overview_url(tracker, "Zetwerk") == "https://ipowatch.in/zetwerk-ipo/"

    overview = """<h2>Objects of the Issue</h2>
    <table><tr><td>Purpose</td><td>Crores</td></tr>
    <tr><td>Repayment of borrowings</td><td>1250</td></tr>
    <tr><td>Repayment of borrowings</td><td>550</td></tr></table>
    <h2>Peer Group Comparison</h2><p>There are no listed companies.</p>
    <h2>IPO Lead Managers</h2>"""
    parsed = parse_ipowatch_overview(T.read_tables(overview), overview)
    assert parsed == {
        "ipowatch_debt_repayment_cr": 1800,
        "ipowatch_no_listed_peers": 1.0,
    }


def test_screener_extracts_supported_fundamentals():
    html = """<table>
    <tr><th></th><th>Mar 2023</th><th>Mar 2024</th><th>Mar 2025</th><th>Mar 2026</th></tr>
    <tr><td>Net Profit +</td><td>100</td><td>120</td><td>144</td><td>172.8</td></tr>
    <tr><td>OPM %</td><td>10%</td><td>11%</td><td>12%</td><td>13%</td></tr>
    </table>
    <table>
    <tr><th></th><th>Mar 2025</th><th>Jun 2025</th><th>Mar 2026</th><th>Jun 2026</th></tr>
    <tr><td>Net Profit +</td><td>8</td><td>20</td><td>10</td><td>30</td></tr>
    </table>
    <table>
    <tr><th></th><th>Mar 2023</th><th>Mar 2024</th><th>Mar 2025</th><th>Mar 2026</th></tr>
    <tr><td>ROCE %</td><td>8%</td><td>11%</td><td>13%</td><td>15%</td></tr>
    </table>"""

    result = parse_fundamentals(T.read_tables(html))

    assert result["pat_cagr_pct"] == pytest.approx(20)
    assert result["latest_qtr_pat_growth_pct"] == pytest.approx(50)
    assert result["ebitda_margin_pct"] == 13
    assert result["roce_pct"] == 15


def test_screener_omits_growth_when_pat_values_are_not_positive():
    html = """<table>
    <tr><th></th><th>Mar 2023</th><th>Mar 2024</th><th>Mar 2025</th><th>Mar 2026</th></tr>
    <tr><td>Net Profit +</td><td>100</td><td>-20</td><td>10</td><td>20</td></tr>
    <tr><td>OPM %</td><td>-5%</td><td>0%</td><td>1%</td><td>2%</td></tr>
    </table>"""
    assert parse_fundamentals(T.read_tables(html)) == {"ebitda_margin_pct": 2}


def test_screener_fetch_requires_direct_company_url():
    class FakeHttp(PoliteSession):
        def get(self, url):
            raise AssertionError(f"must reject URL before fetching: {url}")

    fetcher = ScreenerFetcher(FakeHttp())
    for url in (
        "http://www.screener.in/company/123/",
        "https://www.screener.in/?q=zetwerk",
        "https://example.com/company/123/",
    ):
        with pytest.raises(FetchError, match="direct HTTPS Screener company URL"):
            fetcher.fetch("Zetwerk", url)


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
            if url.endswith("/ipo-in-india-list-main-board-sme/82/mainboard/"):
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
    assert parse_gmp(T.read_tables(GMP), "Jio Platforms Ltd.") == out
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


def test_gmp_fetch_falls_back_to_ipo_specific_history_page():
    detail_page = """<html>
    <head>
      <title>Sahajanand Medical IPO GMP Today</title>
      <meta name="description" content="Sahajanand Medical Technologies IPO GMP today">
      <script type="application/ld+json">
        {"datePublished":"2025-10-17T12:12:45+00:00","dateModified":"2026-10-17T12:12:49+00:00"}
      </script>
    </head>
    <body><table>
      <thead><tr><th>Date</th><th>IPO GMP</th><th>GMP Trend</th><th>Gain</th><th>Last Updated</th></tr></thead>
      <tbody>
        <tr><td>17 October</td><td>₹15</td><td>down</td><td>4.36%</td><td>17:00</td></tr>
        <tr><td>16 October</td><td>₹40</td><td>down</td><td>11.63%</td><td>17:14</td></tr>
      </tbody>
    </table></body></html>"""

    class FakeHttp:
        def __init__(self):
            self.urls = []

        def get(self, url):
            self.urls.append(url)
            if url.endswith("/ipo-grey-market-premium-latest-ipo-gmp/"):
                return "<html><body>No current GMP rows for this IPO.</body></html>"
            if url.endswith("/sahajanand-medical-technologies-ipo-gmp-grey-market-premium/"):
                return detail_page
            raise AssertionError(f"unexpected URL: {url}")

    http = FakeHttp()
    fetcher = GmpFetcher(http)

    assert fetcher.fetch("Sahajanand Medical Technologies") == {
        "gmp_current": 15.0,
        "gmp_current_observed_at": "2026-10-17T11:30:00+00:00",
    }
    assert fetcher.fetch_history("Sahajanand Medical Technologies") == [
        ("2026-10-16T11:44:00+00:00", 40.0),
        ("2026-10-17T11:30:00+00:00", 15.0),
    ]
    assert http.urls[1] == (
        "https://ipowatch.in/sahajanand-medical-technologies-ipo-gmp-grey-market-premium/"
    )


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
            "listing_gain_basis": "source_defined_ambiguous",
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
                "List Price": 110,
                "Listing Day Close": 130,
                "Listing Gain": 15,
                "CMP(BSE)": 140,
                "CMP(NSE)": 141,
                "Current Gains": 40,
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
                "listing_gain_basis": "source_defined_ambiguous",
                "source": "chittorgarh",
            }
        ],
        columns=HISTORY_COLUMNS,
    )

    merged = merge_kaggle_history(incoming, existing)

    assert (merged.added, merged.updated, merged.skipped) == (2, 1, 1)
    matched = merged.rows.loc[merged.rows["name"] == "Truncated Company Limited"].iloc[0]
    assert matched["listing_gain_pct"] == 12
    assert matched["listing_open_price"] == 110
    assert matched["listing_open_gain_pct"] == 10
    assert matched["listing_close_price"] == 130
    assert matched["listing_close_gain_pct"] == 30
    assert matched["listing_gain_basis"] == "opening_price_return"
    assert matched["issue_size_cr"] == 504
    assert matched["sub_qib"] == 4.2
    assert matched["gmp_open"] == 10
    assert matched["gmp_peak"] == 20
    assert matched["gmp_final"] == 18
    assert matched["source"] == "chittorgarh+kaggle:test.xlsx"
    assert not {"CMP(BSE)", "CMP(NSE)", "Current Gains"} & set(HISTORY_COLUMNS)
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
    assert mapped.loc[0, "listing_open_price"] == 626
    assert mapped.loc[0, "listing_open_gain_pct"] == 95.62
    assert mapped.loc[0, "listing_close_price"] == 728.6
    assert mapped.loc[0, "listing_close_gain_pct"] == 127.69
    assert mapped.loc[0, "listing_gain_basis"] == "opening_price_return"
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
