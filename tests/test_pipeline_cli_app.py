from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import make_history
from ipo_intel import cli, pipeline
from ipo_intel.store import Store

APP = str(Path(__file__).resolve().parents[1] / "src" / "ipo_intel" / "app.py")


def test_parse_overrides():
    assert pipeline.parse_overrides(["sub_retail=2.4", "", "gmp_final = 1,250"]) == {"sub_retail": 2.4, "gmp_final": 1250.0}
    with pytest.raises(ValueError, match="unknown field"):
        pipeline.parse_overrides(["bogus=1"])
    with pytest.raises(ValueError, match="key=value"):
        pipeline.parse_overrides(["nonsense"])
    with pytest.raises(ValueError, match="bad value"):
        pipeline.parse_overrides(["sub_retail=abc"])


def test_jio_run_from_seed_before_subscription():
    rep = pipeline.run("Jio Platforms")
    assert rep["missing_short_term"] == []
    assert rep["heuristic_gain"] == pytest.approx(9.2, abs=0.1)  # base scenario: retail assumed 1-3x
    assert len(rep["scenarios"]) == 3
    assert rep["long_term"]["score"] == 64.0
    assert any("seed data" in r["source"] for r in rep["provenance"])


def test_overrides_beat_seed_and_remove_scenarios():
    rep = pipeline.run("Jio Platforms", overrides={"sub_retail": 4.0, "gmp_final": 280.0})
    assert rep["headline_gain"] == pytest.approx(22.5, abs=0.2)
    assert rep["short_verdict"] == "STRONG APPLY"
    assert "scenarios" not in rep
    srcs = {r["field"]: r["source"] for r in rep["provenance"]}
    assert srcs["gmp_final"] == "user override"


def test_saved_overrides_persist():
    pipeline.run("Jio Platforms", overrides={"sub_retail": 0.8}, save=True)
    rep = pipeline.run("Jio Platforms")
    srcs = {r["field"]: r["source"] for r in rep["provenance"]}
    assert srcs["sub_retail"] == "manual file"


def test_own_gmp_history_feeds_open_peak_final():
    store = Store()
    for ts, g in [("2026-10-18T09:00:00", 150), ("2026-10-19T09:00:00", 240), ("2026-10-20T09:00:00", 200)]:
        store.record_gmp("Brand New Co", g, "test", ts)
    snap = pipeline.collect("Brand New Co", store=store)
    assert (snap.get("gmp_open"), snap.get("gmp_peak"), snap.get("gmp_final")) == (150, 240, 200)


def test_unknown_ipo_reports_what_is_missing():
    rep = pipeline.run("Totally Unknown Ltd")
    assert set(rep["missing_short_term"]) == {"issue_size_cr", "price_band_high", "gmp_final"}
    assert rep["long_term"] is None


def test_scrape_flag_is_refused_when_not_enabled(monkeypatch):
    monkeypatch.delenv("IPO_ENABLE_SCRAPERS", raising=False)
    rep = pipeline.run("Jio Platforms", scrape=True)
    assert any("IPO_ENABLE_SCRAPERS" in w for w in rep["warnings"])


def test_scraper_failures_become_warnings_not_crashes(monkeypatch):
    monkeypatch.setenv("IPO_ENABLE_SCRAPERS", "1")
    from ipo_intel.fetchers import FetchError, GmpFetcher, ChittorgarhFetcher

    def boom(self, name):
        raise FetchError("network down")

    monkeypatch.setattr(GmpFetcher, "fetch", boom)
    monkeypatch.setattr(ChittorgarhFetcher, "fetch", boom)
    rep = pipeline.run("Jio Platforms", scrape=True)
    assert len([w for w in rep["warnings"] if "network down" in w]) == 2
    assert rep["long_term"]["score"] == 64.0  # seed data still works


def test_cli_run_backtest_and_errors(capsys):
    assert cli.main(["run", "jio"]) == 0
    out = capsys.readouterr().out
    assert "SHORT TERM" in out and "LONG TERM" in out and "not investment advice" in out
    assert cli.main(["backtest"]) == 0
    assert cli.main(["run", "jio", "--set", "bogus=1"]) == 2
    assert cli.main(["collect-history"]) == 2  # scrapers disabled


def test_cli_import_and_train(tmp_path, capsys):
    csv = tmp_path / "h.csv"
    make_history().to_csv(csv, index=False)
    assert cli.main(["import-csv", str(csv)]) == 0
    assert cli.main(["train"]) == 0
    assert "small_walk_forward" in capsys.readouterr().out
    rep = pipeline.run("Jio Platforms")
    assert rep["ml"]["tier"] == "large"


def test_web_app_runs_end_to_end():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert "Short term" in [s.value for s in at.subheader]
    assert any("Score" in m.label for m in at.metric)
