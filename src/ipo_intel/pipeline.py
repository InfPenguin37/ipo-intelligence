"""Collect everything known about an IPO, then analyse it. Used by the CLI and the web app."""
from __future__ import annotations

import json
import re

from . import config
from .fetchers import ChittorgarhFetcher, FetchError, GmpFetcher, PoliteSession
from .fetchers.tables import name_matches
from .ml import load_models, predict_one
from .schema import ALL_KEYS, FUNDAMENTAL_KEYS, Snapshot, coerce
from .scoring import MissingFields, est_listing_gain, gmp_pct, long_term, scenarios, short_verdict
from .store import Store


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_overrides(items: list[str]) -> dict:
    """['sub_retail=2.4', 'gmp_final=250'] -> {'sub_retail': 2.4, 'gmp_final': 250.0}. Blank lines are ignored."""
    out = {}
    for item in items:
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"expected key=value, got '{item}'")
        key, raw = (s.strip() for s in item.split("=", 1))
        if key not in ALL_KEYS:
            raise ValueError(f"unknown field '{key}'. Valid fields: {', '.join(ALL_KEYS)}")
        try:
            out[key] = coerce(key, raw)
        except ValueError as e:
            raise ValueError(f"bad value for {key}: '{raw}'") from e
    return out


# ------------------------------------------------------------------ seed + manual data
_SUB_MAP = {"overall": "sub_overall", "qib": "sub_qib", "nii": "sub_nii", "retail": "sub_retail"}


def flatten_case(case: dict) -> dict:
    out: dict = {"issue_size_cr": case.get("issue_size_cr"), "price_band_high": case.get("price_band_high")}
    s = (case.get("structure") or "").lower()
    if s:
        out["structure"] = "OFS" if s.startswith("ofs") else "Fresh" if s.startswith("fresh") else "Mixed"
        out["ofs_share"] = {"OFS": 1.0, "Fresh": 0.0}.get(out["structure"])
    g = case.get("gmp") or {}
    out.update(gmp_open=g.get("open"), gmp_peak=g.get("peak"), gmp_final=g.get("final"))
    for k, v in (case.get("subscription") or {}).items():
        if k in _SUB_MAP:
            out[_SUB_MAP[k]] = v
    for k, v in (case.get("fundamentals") or {}).items():
        out["pat_cagr_pct" if k == "pat_cagr_fy24_26_pct" else k] = v
    return {k: v for k, v in out.items() if v is not None and k in ALL_KEYS}


def find_seed(name: str) -> tuple[str, dict] | None:
    if not config.SEED_PATH.exists():
        return None
    data = json.loads(config.SEED_PATH.read_text())
    for group in ("target", "backtest"):
        for key, case in data.get(group, {}).items():
            if name_matches(name, key):
                return key, flatten_case(case)
    return None


def _manual_path(name: str):
    return config.MANUAL_DIR / f"{slug(name)}.json"


def load_manual(name: str) -> dict:
    p = _manual_path(name)
    return json.loads(p.read_text()) if p.exists() else {}


def save_manual(name: str, values: dict) -> None:
    config.ensure_dirs()
    merged = {**load_manual(name), **values}
    _manual_path(name).write_text(json.dumps(merged, indent=2))


# ------------------------------------------------------------------ collect
def collect(name: str, *, scrape: bool = False, overrides: dict | None = None, store: Store | None = None) -> Snapshot:
    """Merge sources in priority order: live scrapers, own GMP history, seed data, manual file, overrides."""
    store = store or Store()
    snap = Snapshot(name)

    if scrape:
        if not config.scrapers_enabled():
            snap.warnings.append("Live fetch requested but disabled by IPO_ENABLE_SCRAPERS=0; skipped.")
        else:
            http = PoliteSession()
            for fetcher in (GmpFetcher(http), ChittorgarhFetcher(http)):
                try:
                    data = fetcher.fetch(name)
                except FetchError as e:
                    snap.warnings.append(f"{fetcher.name}: {e}")
                    continue
                if "gmp_current" in data:
                    store.record_gmp(name, data.pop("gmp_current"), fetcher.name)
                for k, v in data.items():
                    snap.set(k, v, fetcher.name)
                if isinstance(fetcher, GmpFetcher):
                    try:
                        for observed_at, gmp in fetcher.fetch_history(name):
                            store.record_gmp(
                                name,
                                gmp,
                                "ipowatch-history",
                                ts=observed_at,
                            )
                    except FetchError as e:
                        snap.warnings.append(f"{fetcher.name} history: {e}")

    observations = store.gmp_observations(name)
    if len(observations) >= 2:
        peak = max(observations, key=lambda point: point[1])
        for key, point in (
            ("gmp_open", observations[0]),
            ("gmp_peak", peak),
            ("gmp_final", observations[-1]),
        ):
            snap.set(key, point[1], point[2])
    elif len(observations) == 1:
        snap.set("gmp_final", observations[0][1], observations[0][2])

    seed = find_seed(name)
    if seed:
        key, vals = seed
        for k, v in vals.items():
            snap.set(k, v, f"seed data '{key}' (web research, 2026-10-18)")

    for k, v in load_manual(name).items():
        snap.set(k, v, "manual file", overwrite=True)
    for k, v in (overrides or {}).items():
        snap.set(k, v, "user override", overwrite=True)
    return snap


# ------------------------------------------------------------------ analyse
def analyse(snap: Snapshot, models="auto") -> dict:
    v = snap.values()
    if models == "auto":
        models = load_models()
    rep: dict = {"name": snap.name, "warnings": list(snap.warnings), "provenance": snap.provenance(), "notes": []}

    missing = snap.missing_short_term()
    rep["missing_short_term"] = missing
    if not missing:
        size, price, final = v["issue_size_cr"], v["price_band_high"], v["gmp_final"]
        known = v.get("sub_retail") is not None
        # Until retail subscription is known, assume the base case (retail 1-3x) rather than "no information".
        heur = est_listing_gain(size, price, final, v.get("gmp_open"), v["sub_retail"] if known else 1.5)
        rep["gmp_pct"] = round(gmp_pct(final, price), 1)
        rep["heuristic_gain"] = round(heur, 1)
        rep["subscription_known"] = known
        ml = predict_one(models, v)
        rep["ml"] = ml
        use_ml = bool(ml) and (ml["tier"] == "small" or ml["n_train"] >= 5)
        rep["headline_gain"] = ml["q50"] if use_ml else rep["heuristic_gain"]
        rep["basis"] = (
            f"ML model ({ml['tier']} tier, trained on {ml['n_train']} IPOs)" if use_ml else "rule-based heuristic"
        )
        rep["short_verdict"] = short_verdict(rep["headline_gain"])
        if not rep["subscription_known"]:
            rep["scenarios"] = scenarios(size, price, final, v.get("gmp_open"))
            rep["notes"].append("Retail subscription is not known yet; scenarios show the plausible range.")
        if v.get("structure") == "OFS" and size >= config.LARGE_CUTOFF_CR:
            rep["notes"].append(
                "Large pure-OFS issues in the seed backtest: 2 of 3 with later prices (NSE, SBI AMC) traded below "
                "issue price within weeks of listing. Decide your exit before you apply."
            )

    fundamentals = {k: v.get(k) for k in FUNDAMENTAL_KEYS}
    try:
        rep["long_term"] = long_term(fundamentals)
    except MissingFields as e:
        rep["long_term"] = None
        rep["long_term_missing"] = e.missing
    return rep


def run(name: str, *, scrape: bool = False, overrides: dict | None = None, save: bool = False) -> dict:
    store = Store()
    if save and overrides:
        save_manual(name, overrides)
    snap = collect(name, scrape=scrape, overrides=overrides, store=store)
    rep = analyse(snap)
    rep["gmp_history"] = [
        {"observed_at": timestamp, "gmp": gmp, "source": source}
        for timestamp, gmp, source in store.gmp_observations(name)
    ]
    store.save_snapshot(name, rep)
    return rep
