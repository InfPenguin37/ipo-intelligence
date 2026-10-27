"""Append-only forecast records with CSV export and outcome import."""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

FORECAST_COLUMNS = [
    "forecast_id",
    "forecasted_at_utc",
    "ipo_name",
    "listing_date",
    "days_to_listing",
    "issue_size_cr",
    "offer_price",
    "structure",
    "ofs_share",
    "gmp_open",
    "gmp_peak",
    "gmp_latest",
    "gmp_final",
    "gmp_latest_observed_at",
    "gmp_final_observed_at",
    "sub_overall",
    "sub_qib",
    "sub_nii",
    "sub_retail",
    "predicted_open_gain_pct",
    "prediction_basis",
    "heuristic_gain_pct",
    "ml_tier",
    "ml_training_rows",
    "ml_q10",
    "ml_q50",
    "ml_q90",
    "short_verdict",
    "inputs_and_sources_json",
    "actual_listing_open_price",
    "actual_listing_open_gain_pct",
    "actual_listing_close_price",
    "actual_listing_close_gain_pct",
    "outcome_source",
    "outcome_recorded_at_utc",
]

DISPLAY_COLUMNS = [
    "forecasted_at_utc",
    "ipo_name",
    "listing_date",
    "predicted_open_gain_pct",
    "heuristic_gain_pct",
    "ml_q10",
    "ml_q50",
    "ml_q90",
    "actual_listing_open_gain_pct",
    "actual_listing_close_gain_pct",
]

OUTCOME_PRICE_COLUMNS = (
    ("actual_listing_open_price", "actual_listing_open_gain_pct"),
    ("actual_listing_close_price", "actual_listing_close_gain_pct"),
)


def build_forecast_record(report: dict) -> dict:
    if report.get("headline_gain") is None:
        raise ValueError("Cannot record a forecast without a predicted opening gain.")

    provenance = {
        row["field"]: {
            "value": row["value"],
            "source": row["source"],
            "fetched_at": row["fetched_at"],
        }
        for row in report.get("provenance", [])
    }
    inputs = {field: item["value"] for field, item in provenance.items()}
    forecasted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    listing_date = inputs.get("listing_date")
    days_to_listing = None
    if listing_date:
        try:
            days_to_listing = (
                date.fromisoformat(str(listing_date)[:10])
                - datetime.now(timezone.utc).date()
            ).days
        except ValueError:
            days_to_listing = None

    ml = report.get("ml") or {}
    record = {
        "forecast_id": str(uuid.uuid4()),
        "forecasted_at_utc": forecasted_at,
        "ipo_name": report["name"],
        "listing_date": listing_date,
        "days_to_listing": days_to_listing,
        "issue_size_cr": inputs.get("issue_size_cr"),
        "offer_price": inputs.get("price_band_high"),
        "structure": inputs.get("structure"),
        "ofs_share": inputs.get("ofs_share"),
        "gmp_open": inputs.get("gmp_open"),
        "gmp_peak": inputs.get("gmp_peak"),
        "gmp_latest": inputs.get("gmp_latest"),
        "gmp_final": inputs.get("gmp_final"),
        "gmp_latest_observed_at": inputs.get("gmp_latest_observed_at"),
        "gmp_final_observed_at": inputs.get("gmp_final_observed_at"),
        "sub_overall": inputs.get("sub_overall"),
        "sub_qib": inputs.get("sub_qib"),
        "sub_nii": inputs.get("sub_nii"),
        "sub_retail": inputs.get("sub_retail"),
        "predicted_open_gain_pct": report["headline_gain"],
        "prediction_basis": report.get("basis"),
        "heuristic_gain_pct": report.get("heuristic_gain"),
        "ml_tier": ml.get("tier"),
        "ml_training_rows": ml.get("n_train"),
        "ml_q10": ml.get("q10"),
        "ml_q50": ml.get("q50"),
        "ml_q90": ml.get("q90"),
        "short_verdict": report.get("short_verdict"),
        "inputs_and_sources_json": json.dumps(provenance, sort_keys=True, default=str),
    }
    return {column: record.get(column) for column in FORECAST_COLUMNS}


def load_forecasts(path: Path | str | None = None) -> pd.DataFrame:
    ledger_path = Path(path) if path is not None else config.FORECAST_LEDGER_PATH
    if not ledger_path.exists():
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    frame = pd.read_csv(ledger_path)
    missing = set(FORECAST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast ledger is missing columns: {', '.join(sorted(missing))}")
    return frame[FORECAST_COLUMNS]


def append_forecast(report: dict, path: Path | str | None = None) -> dict:
    ledger_path = Path(path) if path is not None else config.FORECAST_LEDGER_PATH
    record = build_forecast_record(report)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        frame = load_forecasts(ledger_path)
        frame.loc[len(frame)] = record
    else:
        frame = pd.DataFrame([record], columns=FORECAST_COLUMNS)
    frame.to_csv(ledger_path, index=False)
    return record


def merge_uploaded_forecasts(
    uploaded: pd.DataFrame,
    path: Path | str | None = None,
) -> pd.DataFrame:
    if "forecast_id" not in uploaded:
        raise ValueError("Uploaded CSV must include the forecast_id column.")
    if uploaded["forecast_id"].isna().any() or uploaded["forecast_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Every uploaded row must have a forecast_id.")
    if uploaded["forecast_id"].duplicated().any():
        raise ValueError("Uploaded CSV contains duplicate forecast_id values.")

    missing = set(FORECAST_COLUMNS) - set(uploaded.columns)
    if missing:
        raise ValueError(f"Uploaded CSV is missing columns: {', '.join(sorted(missing))}")

    ledger_path = Path(path) if path is not None else config.FORECAST_LEDGER_PATH
    existing = load_forecasts(ledger_path)
    imported = uploaded[FORECAST_COLUMNS].copy()
    combined = existing.set_index("forecast_id").astype(object)
    for _, row in imported.iterrows():
        forecast_id = str(row["forecast_id"])
        values = row.drop(labels="forecast_id")
        if forecast_id in combined.index:
            for column, value in values.items():
                if pd.notna(value):
                    combined.loc[forecast_id, column] = value
        else:
            combined.loc[forecast_id, values.index] = values

    result = combined.reset_index().reindex(columns=FORECAST_COLUMNS)
    for price_column, gain_column in OUTCOME_PRICE_COLUMNS:
        prices = pd.to_numeric(result[price_column], errors="coerce")
        offer_prices = pd.to_numeric(result["offer_price"], errors="coerce")
        known = prices.notna() & offer_prices.notna() & offer_prices.ne(0)
        result.loc[known, gain_column] = (prices[known] / offer_prices[known] - 1) * 100

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(ledger_path, index=False)
    return result
