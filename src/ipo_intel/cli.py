"""Command line: `ipo run "Jio Platforms"`, `ipo watch`, `ipo train`, ..."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import config, pipeline, report
from .fetchers import ChittorgarhFetcher, FetchError, PoliteSession
from .fetchers.tables import read_tables
from .history_import import prepare_investorgain_import, prepare_kaggle_import
from .ml import train_all
from .scoring import backtest
from .store import HISTORY_COLUMNS, Store


def _cmd_run(a) -> int:
    if a.screener_url and not a.scrape:
        print("error: --screener-url requires live fetching; remove --no-scrape.", file=sys.stderr)
        return 2
    try:
        overrides = pipeline.parse_overrides(a.set)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rep = pipeline.run(a.name, scrape=a.scrape, screener_url=a.screener_url, overrides=overrides, save=a.save)
    print(json.dumps(rep, indent=2, default=str) if a.json else report.to_text(rep))
    return 0


def _cmd_watch(a) -> int:
    print(f"Refreshing '{a.name}' every {a.every} minutes. Ctrl-C to stop.")
    try:
        while True:
            rep = pipeline.run(a.name, scrape=True)
            stamp = time.strftime("%H:%M:%S")
            if rep["missing_short_term"]:
                print(f"[{stamp}] missing {rep['missing_short_term']}; {rep['warnings']}")
            else:
                print(f"[{stamp}] GMP {rep['gmp_pct']:+.1f}%  headline {rep['headline_gain']:+.1f}%  {rep['short_verdict']}")
            time.sleep(a.every * 60)
    except KeyboardInterrupt:
        return 0


def _cmd_collect_history(a) -> int:
    if not config.scrapers_enabled():
        print("Live fetching is disabled; unset IPO_ENABLE_SCRAPERS or set it to 1 (see README on scraping).", file=sys.stderr)
        return 2
    try:
        n = ChittorgarhFetcher().collect_history(Store(), limit=a.limit, from_year=a.from_year)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Added {n} IPOs to the history table.")
    return 0


def _cmd_import_csv(a) -> int:
    df = pd.read_csv(a.path)
    if "name" not in df or "listing_gain_pct" not in df:
        print(f"error: CSV needs at least 'name' and 'listing_gain_pct'. Known columns: {', '.join(HISTORY_COLUMNS)}", file=sys.stderr)
        return 2
    df["source"] = df.get("source", f"csv:{Path(a.path).name}")
    print(f"Imported {Store().upsert_history(df)} rows.")
    return 0


def _cmd_import_excel(a) -> int:
    store = Store()
    imported = prepare_kaggle_import(a.path, store.load_history())
    store.upsert_history(imported.rows)
    print(
        f"Added {imported.added} IPOs, updated {imported.updated} matches, "
        f"skipped {imported.skipped} incomplete rows."
    )
    return 0


def _cmd_import_gmp_csv(a) -> int:
    store = Store()
    imported = prepare_investorgain_import(a.path, store.load_history())
    store.upsert_history(imported.rows)
    print(
        f"Added {imported.added} IPOs, updated {imported.updated} matches, "
        f"skipped {imported.skipped} incomplete rows."
    )
    return 0


def _cmd_train(a) -> int:
    hist = Store().load_history()
    if hist.empty:
        print("History table is empty. Run `ipo collect-history` or `ipo import-csv <file>` first.", file=sys.stderr)
        return 1
    print(json.dumps(train_all(hist), indent=2, default=str))
    return 0


def _cmd_backtest(a) -> int:
    rows, mae = backtest.run()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nMean absolute error (percentage points):", mae)
    print("Parameters were set knowing these outcomes: a sanity check, not out-of-sample validation.")
    return 0


def _cmd_probe(a) -> int:
    http = PoliteSession(respect_robots=not a.ignore_robots)
    try:
        body = http.get(a.url)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    tables = read_tables(body)
    print(f"{len(tables)} table(s) found. Raw response saved under {config.RAW_DIR}")
    for i, t in enumerate(tables):
        print(f"\n--- table {i}: shape {t.shape}\ncolumns: {list(t.columns)}\n{t.head(3).to_string()}")
    return 0


def _cmd_app(a) -> int:
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ipo", description="IPO Intelligence")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyse one IPO")
    r.add_argument("name")
    r.add_argument("--scrape", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--no-scrape", action="store_false", dest="scrape", help="skip live data fetching")
    r.set_defaults(scrape=True)
    r.add_argument("--screener-url", help="direct Screener company URL; requires live fetching")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a field, repeatable")
    r.add_argument("--save", action="store_true", help="persist --set values for this IPO")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_cmd_run)

    w = sub.add_parser("watch", help="refresh live GMP and re-analyse on a timer")
    w.add_argument("name")
    w.add_argument("--every", type=float, default=60, help="minutes between refreshes")
    w.set_defaults(fn=_cmd_watch)

    c = sub.add_parser("collect-history", help="build the training table from listing history")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--from-year", type=int, default=2004, help="oldest calendar year to import (default: 2004)")
    c.set_defaults(fn=_cmd_collect_history)

    i = sub.add_parser("import-csv", help="load historical IPOs from a CSV")
    i.add_argument("path")
    i.set_defaults(fn=_cmd_import_csv)

    x = sub.add_parser("import-excel", help="merge the Kaggle IPO history workbook")
    x.add_argument("path")
    x.set_defaults(fn=_cmd_import_excel)

    g = sub.add_parser("import-gmp-csv", help="merge an InvestorGain GMP tracker CSV")
    g.add_argument("path")
    g.set_defaults(fn=_cmd_import_gmp_csv)

    sub.add_parser("train", help="train ML models and print walk-forward results").set_defaults(fn=_cmd_train)
    sub.add_parser("backtest", help="rule-based estimator on the seed cases").set_defaults(fn=_cmd_backtest)

    pr = sub.add_parser("probe", help="show the tables found at a URL (to debug scrapers)")
    pr.add_argument("url")
    pr.add_argument("--ignore-robots", action="store_true")
    pr.set_defaults(fn=_cmd_probe)

    sub.add_parser("app", help="launch the web UI").set_defaults(fn=_cmd_app)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
