"""Parser tests use small SYNTHETIC HTML fixtures shaped like the real pages. They prove the parsing
logic, not that the live sites still look like this."""
from ipo_intel.fetchers import tables as T
from ipo_intel.fetchers.chittorgarh import (
    best_link,
    find_ipo_links,
    parse_details,
    parse_perf_tracker,
    parse_subscription,
)
from ipo_intel.fetchers.gmp import parse_gmp

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


def test_details_fresh_issue():
    out = parse_details(T.read_tables(DETAILS))
    assert out["price_band_high"] == 1119
    assert out["issue_size_cr"] == 30213
    assert out["structure"] == "Fresh" and out["ofs_share"] == 0


def test_subscription_picks_main_nii_row_not_bnii():
    out = parse_subscription(T.read_tables(SUBSCRIPTION))
    assert out == {"sub_qib": 12.68, "sub_nii": 6.55, "sub_retail": 1.39, "sub_overall": 5.71}


def test_gmp_row_matching_ignores_status_suffix():
    out = parse_gmp(T.read_tables(GMP), "Jio Platforms")
    assert out == {"gmp_current": 220.0, "price_band_high": 1119.0}
    assert parse_gmp(T.read_tables(GMP), "Nonexistent Corp") == {}


def test_perf_tracker():
    df = parse_perf_tracker(T.read_tables(PERF))
    assert list(df["name"]) == ["National Stock Exchange of India Ltd.", "Moneyview Ltd."]
    assert list(df["listing_gain_pct"]) == [1.85, 58.47]


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
