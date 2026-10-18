"""SQLite store: GMP time series, saved runs, and the historical training table."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .schema import now_iso

HISTORY_COLUMNS = [
    "name",
    "listing_date",
    "issue_size_cr",
    "price_band_high",
    "ofs_share",
    "gmp_open",
    "gmp_peak",
    "gmp_final",
    "sub_overall",
    "sub_qib",
    "sub_nii",
    "sub_retail",
    "listing_gain_pct",
    "day1_close_pct",
    "source",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gmp_history (ipo TEXT, ts TEXT, gmp REAL, source TEXT);
CREATE TABLE IF NOT EXISTS snapshots (ipo TEXT, ts TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS history (
  name TEXT PRIMARY KEY, listing_date TEXT, issue_size_cr REAL, price_band_high REAL, ofs_share REAL,
  gmp_open REAL, gmp_peak REAL, gmp_final REAL, sub_overall REAL, sub_qib REAL, sub_nii REAL,
  sub_retail REAL, listing_gain_pct REAL, day1_close_pct REAL, source TEXT
);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        config.ensure_dirs()
        self.path = Path(path) if path else config.DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(_SCHEMA)

    # -- GMP series -------------------------------------------------------------
    def record_gmp(self, ipo: str, gmp: float, source: str, ts: str | None = None) -> None:
        observed_at = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        timestamp = observed_at.astimezone(timezone.utc).isoformat(timespec="seconds")
        self.con.execute(
            """
            INSERT INTO gmp_history (ipo, ts, gmp, source)
            SELECT ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM gmp_history WHERE ipo = ? AND ts = ? AND source = ?
            )
            """,
            (ipo.lower(), timestamp, float(gmp), source, ipo.lower(), timestamp, source),
        )
        self.con.commit()

    def gmp_series(self, ipo: str) -> list[tuple[str, float]]:
        cur = self.con.execute("SELECT ts, gmp FROM gmp_history WHERE ipo=? ORDER BY ts", (ipo.lower(),))
        return [(r[0], r[1]) for r in cur.fetchall()]

    def gmp_observations(self, ipo: str) -> list[tuple[str, float, str]]:
        cur = self.con.execute(
            "SELECT ts, gmp, source FROM gmp_history WHERE ipo=? ORDER BY ts",
            (ipo.lower(),),
        )
        return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    def gmp_stats(self, ipo: str) -> dict[str, float] | None:
        s = self.gmp_series(ipo)
        if not s:
            return None
        vals = [v for _, v in s]
        return {"gmp_open": vals[0], "gmp_peak": max(vals), "gmp_final": vals[-1]}

    # -- runs ---------------------------------------------------------------------
    def save_snapshot(self, ipo: str, payload: dict) -> None:
        self.con.execute("INSERT INTO snapshots VALUES (?,?,?)", (ipo, now_iso(), json.dumps(payload, default=str)))
        self.con.commit()

    # -- training table -------------------------------------------------------------
    def upsert_history(self, df: pd.DataFrame) -> int:
        d = df.reindex(columns=HISTORY_COLUMNS).astype(object).where(df.reindex(columns=HISTORY_COLUMNS).notna(), None)
        rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in r) for r in d.itertuples(index=False)]
        self.con.executemany(
            f"INSERT OR REPLACE INTO history ({','.join(HISTORY_COLUMNS)}) VALUES ({','.join('?' * len(HISTORY_COLUMNS))})",
            rows,
        )
        self.con.commit()
        return len(rows)

    def load_history(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM history", self.con)
