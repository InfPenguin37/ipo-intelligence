import pandas as pd
import pytest

from ipo_intel.forecast_ledger import (
    FORECAST_COLUMNS,
    append_forecast,
    build_forecast_record,
    load_forecasts,
    merge_uploaded_forecasts,
)


def _report():
    return {
        "name": "Jio Platforms",
        "headline_gain": 12.5,
        "basis": "Rule-based estimate",
        "heuristic_gain": 12.5,
        "short_verdict": "APPLY",
        "ml": {"tier": "large", "n_train": 34, "q10": -8.0, "q50": 4.0, "q90": 18.0},
        "provenance": [
            {"field": "listing_date", "value": "2026-10-28", "source": "chittorgarh", "fetched_at": "2026-10-27T10:00:00+00:00"},
            {"field": "issue_size_cr", "value": 30213.0, "source": "chittorgarh", "fetched_at": "2026-10-27T10:00:00+00:00"},
            {"field": "price_band_high", "value": 1119.0, "source": "chittorgarh", "fetched_at": "2026-10-27T10:00:00+00:00"},
            {"field": "gmp_final", "value": 245.0, "source": "ipowatch", "fetched_at": "2026-10-27T10:00:00+00:00"},
        ],
    }


def test_build_forecast_record_keeps_prediction_inputs_and_sources():
    record = build_forecast_record(_report())

    assert set(record) == set(FORECAST_COLUMNS)
    assert record["ipo_name"] == "Jio Platforms"
    assert record["offer_price"] == 1119.0
    assert record["predicted_open_gain_pct"] == 12.5
    assert record["ml_q10"] == -8.0
    assert record["actual_listing_open_gain_pct"] is None
    assert '"source": "ipowatch"' in record["inputs_and_sources_json"]


def test_append_and_load_forecasts_preserves_each_prediction(tmp_path):
    path = tmp_path / "research" / "forecast_ledger.csv"

    first = append_forecast(_report(), path)
    second = append_forecast(_report(), path)
    history = load_forecasts(path)

    assert len(history) == 2
    assert first["forecast_id"] != second["forecast_id"]
    assert history["forecast_id"].tolist() == [first["forecast_id"], second["forecast_id"]]


def test_import_updates_actual_outcomes_without_losing_existing_forecasts(tmp_path):
    path = tmp_path / "forecast_ledger.csv"
    first = append_forecast(_report(), path)
    append_forecast({**_report(), "name": "Paras Healthcare"}, path)
    uploaded = load_forecasts(path)
    uploaded.loc[0, "actual_listing_open_price"] = 1230.9
    uploaded.loc[0, "actual_listing_close_price"] = 1190.0
    uploaded["outcome_source"] = uploaded["outcome_source"].astype(object)
    uploaded.loc[0, "outcome_source"] = "exchange listing data"

    updated = merge_uploaded_forecasts(uploaded, path)

    assert len(updated) == 2
    jio = updated.set_index("forecast_id").loc[first["forecast_id"]]
    assert jio["actual_listing_open_gain_pct"] == pytest.approx((1230.9 / 1119.0 - 1) * 100)
    assert jio["actual_listing_close_gain_pct"] == pytest.approx((1190.0 / 1119.0 - 1) * 100)
    assert jio["outcome_source"] == "exchange listing data"
    assert updated.loc[updated["ipo_name"] == "Paras Healthcare", "actual_listing_open_price"].isna().all()


def test_import_rejects_missing_or_duplicate_forecast_ids(tmp_path):
    path = tmp_path / "forecast_ledger.csv"

    with pytest.raises(ValueError, match="forecast_id"):
        merge_uploaded_forecasts(pd.DataFrame({"ipo_name": ["Jio Platforms"]}), path)

    duplicate = pd.DataFrame(
        [{column: None for column in FORECAST_COLUMNS} for _ in range(2)]
    )
    duplicate["forecast_id"] = ["same-id", "same-id"]
    with pytest.raises(ValueError, match="duplicate forecast_id"):
        merge_uploaded_forecasts(duplicate, path)
