"""Layout-tolerant HTML table helpers. Parsers match on labels and keywords, not on positions."""
from __future__ import annotations

import re
from io import StringIO

import pandas as pd

_STOP = {"ipo", "ltd", "limited", "the", "india", "pvt", "private"}


def read_tables(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html))
    except ValueError:  # no tables
        return []


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def to_num(x) -> float | None:
    """'₹1,119.00' -> 1119.0, '5.71x' -> 5.71, '19.9%' -> 19.9, '-' -> None."""
    m = re.search(r"-?\d[\d,]*\.?\d*", str(x))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def numbers(text) -> list[float]:
    out = []
    for m in re.finditer(r"\d[\d,]*\.?\d*", str(text)):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def rupee_crore(text) -> float | None:
    m = re.search(r"([\d,]+\.?\d*)\s*(?:cr\b|crore)", str(text), re.I)
    return float(m.group(1).replace(",", "")) if m else None


def tokens(name: str) -> set[str]:
    s = re.sub(r"\(.*?\)", " ", name.lower())
    return {t for t in re.findall(r"[a-z0-9]+", s) if t not in _STOP}


def name_matches(query: str, cell: str) -> bool:
    """All meaningful words of the query appear in the cell, e.g. 'jio platforms' in 'Jio Platforms Upcoming'."""
    q = tokens(query)
    return bool(q) and q <= tokens(cell)


def kv_pairs(tables: list[pd.DataFrame]) -> dict[str, str]:
    """Label -> value from any table with at least two columns (first column = label)."""
    out: dict[str, str] = {}
    for df in tables:
        if df.shape[1] < 2:
            continue
        for _, row in df.iterrows():
            label = norm(row.iloc[0])
            if label and label != "nan" and label not in out:
                out[label] = str(row.iloc[1])
    return out


def find_table(tables: list[pd.DataFrame], must_have: list[str]) -> pd.DataFrame | None:
    for df in tables:
        cols = [norm(c) for c in df.columns]
        if all(any(k in c for c in cols) for k in must_have):
            return df
    return None


def last_numeric(values) -> float | None:
    found = [to_num(v) for v in values]
    found = [f for f in found if f is not None]
    return found[-1] if found else None
