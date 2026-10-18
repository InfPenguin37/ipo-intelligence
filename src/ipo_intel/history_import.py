"""Import external historical IPO datasets into the training history."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .store import HISTORY_COLUMNS

_KAGGLE_COLUMNS = {
    "date": "listing_date",
    "iponame": "name",
    "issuesizecrores": "issue_size_cr",
    "qib": "sub_qib",
    "hni": "sub_nii",
    "rii": "sub_retail",
    "total": "sub_overall",
    "offerprice": "price_band_high",
    "listinggain": "listing_gain_pct",
    "gmpopen": "gmp_open",
    "gmppeak": "gmp_peak",
    "gmpfinal": "gmp_final",
    "finalgmp": "gmp_final",
    "gmp": "gmp_final",
}
_NUMERIC = {
    "issue_size_cr",
    "price_band_high",
    "sub_qib",
    "sub_nii",
    "sub_retail",
    "sub_overall",
    "listing_gain_pct",
    "gmp_open",
    "gmp_peak",
    "gmp_final",
}


@dataclass(frozen=True)
class HistoryImport:
    rows: pd.DataFrame
    added: int
    updated: int
    skipped: int


def _date_key(value) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _name_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _column_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def map_kaggle_workbook(workbook: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map the Kaggle IPO workbook's columns to the training-history schema."""
    columns = {_column_key(column): column for column in workbook.columns}
    required_headers = {
        "name": "iponame",
        "listing_date": "date",
        "issue_size_cr": "issuesizecrores",
        "price_band_high": "offerprice",
        "listing_gain_pct": "listinggain",
    }
    missing = [source_key for source_key in required_headers.values() if source_key not in columns]
    if missing:
        raise ValueError(
            "Kaggle workbook is missing required columns: "
            + ", ".join(missing)
        )

    data: dict[str, object] = {}
    for source_key, field in _KAGGLE_COLUMNS.items():
        column = columns.get(source_key)
        if column is not None:
            data[field] = workbook[column]
    mapped = pd.DataFrame(data)
    mapped["listing_date"] = mapped["listing_date"].map(_date_key)
    mapped["name"] = mapped["name"].astype("string").str.strip()
    for field in _NUMERIC:
        if field in mapped:
            mapped[field] = pd.to_numeric(mapped[field], errors="coerce")
    mapped["source"] = source
    return mapped.reindex(columns=HISTORY_COLUMNS)


def _same_issue(candidate: dict, row: dict) -> bool:
    candidate_price = candidate.get("price_band_high")
    row_price = row.get("price_band_high")
    candidate_size = candidate.get("issue_size_cr")
    row_size = row.get("issue_size_cr")
    if (
        pd.isna(candidate_price)
        or pd.isna(row_price)
        or round(float(candidate_price), 2) != round(float(row_price), 2)
    ):
        return False
    if pd.isna(candidate_size) or pd.isna(row_size):
        return False
    return abs(float(candidate_size) - float(row_size)) <= max(0.05, abs(float(row_size)) * 0.01)


def _combine_sources(existing: object, imported: str) -> str:
    existing_sources = "" if pd.isna(existing) else str(existing)
    sources = {part for part in existing_sources.split("+") if part}
    sources.add(imported)
    return "+".join(sorted(sources))


def merge_kaggle_history(incoming: pd.DataFrame, existing: pd.DataFrame) -> HistoryImport:
    """Merge workbook rows with existing history, retaining one training row per IPO."""
    records = existing.reindex(columns=HISTORY_COLUMNS).to_dict("records")
    record_dates = [_date_key(row.get("listing_date")) for row in records]
    claimed: set[int] = set()
    output: list[dict] = []
    added = updated = skipped = 0

    for imported in incoming.reindex(columns=HISTORY_COLUMNS).to_dict("records"):
        name = imported.get("name")
        date_key = _date_key(imported.get("listing_date"))
        required_values = (name, date_key, imported.get("issue_size_cr"), imported.get("price_band_high"), imported.get("listing_gain_pct"))
        if (
            not isinstance(name, str)
            or not name.strip()
            or date_key is None
            or any(pd.isna(value) for value in required_values[2:])
        ):
            skipped += 1
            continue
        imported["listing_date"] = date_key

        same_date = [
            index
            for index, existing_date in enumerate(record_dates)
            if existing_date == date_key and index not in claimed
        ]
        exact_name = [index for index in same_date if _name_key(records[index].get("name")) == _name_key(name)]
        candidates = exact_name or [index for index in same_date if _same_issue(records[index], imported)]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous existing IPO match for '{name}' listed {date_key}")

        if candidates:
            index = candidates[0]
            combined = dict(records[index])
            for field, value in imported.items():
                if (
                    field in {"name", "source"}
                    or pd.isna(value)
                    or not pd.isna(combined.get(field))
                ):
                    continue
                combined[field] = value
            combined["source"] = _combine_sources(combined.get("source"), str(imported["source"]))
            output.append(combined)
            claimed.add(index)
            updated += 1
            continue

        name_collision = any(
            record.get("name") == name and existing_date != date_key
            for record, existing_date in zip(records, record_dates)
        )
        if name_collision:
            imported["name"] = f"{name} (listed {date_key})"
        records.append(imported)
        record_dates.append(date_key)
        output.append(imported)
        added += 1

    return HistoryImport(pd.DataFrame(output, columns=HISTORY_COLUMNS), added, updated, skipped)


def prepare_kaggle_import(path: Path | str, existing: pd.DataFrame) -> HistoryImport:
    path = Path(path)
    workbook = pd.read_excel(path)
    incoming = map_kaggle_workbook(workbook, f"kaggle:{path.name}")
    return merge_kaggle_history(incoming, existing)
