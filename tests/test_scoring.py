import json

import pytest

from ipo_intel import config
from ipo_intel.pipeline import flatten_case
from ipo_intel.scoring import MissingFields, est_listing_gain, long_term, short_verdict
from ipo_intel.scoring import backtest


def test_known_backtest_estimates():
    assert est_listing_gain(22562, 1785, 40, 145, 1.39) == pytest.approx(0.84, abs=0.02)  # NSE
    assert est_listing_gain(9813, 574, 105, 92, 3.45) == pytest.approx(10.06, abs=0.05)  # SBI AMC
    assert est_listing_gain(10603, 2165, 360, 240, None) == pytest.approx(14.97, abs=0.05)  # ICICI AMC, surge rule on
    assert est_listing_gain(10603, 2165, 360, 240, None, surge_rule=False) == pytest.approx(9.15, abs=0.05)


def test_missing_inputs_give_nan_and_insufficient_data():
    g = est_listing_gain(None, 100, 10)
    assert g != g
    assert short_verdict(g) == "INSUFFICIENT DATA"


def test_verdict_bands():
    assert short_verdict(20) == "STRONG APPLY"
    assert short_verdict(8) == "APPLY"
    assert short_verdict(2).startswith("NEUTRAL")
    assert short_verdict(-1) == "AVOID"


def test_backtest_errors_match_documented_results():
    rows, mae = backtest.run()
    assert len(rows) == 4
    assert mae == {"raw_gmp": pytest.approx(5.36, abs=0.02), "v1": pytest.approx(4.19, abs=0.02), "v2": pytest.approx(2.73, abs=0.02)}


def test_long_term_score_for_jio_seed():
    case = json.loads(config.SEED_PATH.read_text())["target"]["Jio Platforms (Oct 2026)"]
    res = long_term(flatten_case(case))
    assert res["score"] == 64.0
    assert res["verdict"] == "ACCUMULATE / HOLD 3Y+"
    assert res["parts"]["concentration / regulatory"] == 3


def test_long_term_reports_missing_fields():
    with pytest.raises(MissingFields) as e:
        long_term({"pe": 30})
    assert "roce_pct" in e.value.missing
