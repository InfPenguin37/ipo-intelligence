# IPO Intelligence

A local tool for Indian IPOs. Name an IPO, and it collects the latest data (GMP, subscription, issue size,
financials), scores it, and gives a **short-term** (listing and first weeks) and a **long-term** (1-3 years) read,
with the source and timestamp of every number shown.

> Research tool, not investment advice. GMP is an unofficial grey-market indicator and can be wrong.

## Setup (macOS, conda, Python 3.12)

```bash
git clone https://github.com/InfPenguin37/ipo-intelligence.git
cd ipo-intelligence
make setup                 # creates the `ipo-intel` conda env and installs the package
conda activate ipo-intel
make test                  # tests should pass
```

Without Conda, install the project and its dependencies with `python -m pip install -r requirements.txt`.

## Use it

```bash
ipo run "Jio Platforms"                          # analyse using seed data + anything saved
ipo run jio --set sub_retail=2.4 --set gmp_final=250   # plug in new numbers
ipo run jio --set sub_retail=2.4 --save          # ...and remember them for next time
ipo app                                          # web UI with a Run button (same as `make run`)
ipo backtest                                     # rule-based estimator on the 4 seed IPOs
```

Fields you can set: see the list in the web UI's "Manual overrides" box (`gmp_final`, `sub_qib`, `sub_nii`,
`sub_retail`, `sub_overall`, `issue_size_cr`, `price_band_high`, `pe`, `roce_pct`, ...).

### During the Jio issue (21-23 Oct)
1. Enable live fetching (below), then `ipo watch "Jio Platforms" --every 120` to record GMP every two hours.
   Each refresh adds a point to the app's own GMP history, which is what gives the model `gmp_open`/`gmp_peak`.
2. After bidding closes on 23 Oct, enter the final category subscription:
   `ipo run jio --set sub_retail=<x> --set sub_nii=<x> --set sub_qib=<x> --set sub_overall=<x> --save`
3. Read the verdict. Without subscription numbers it shows bear/base/bull scenarios instead of a single number.

## Live fetchers (on by default)

```bash
ipo probe https://ipowatch.in/ipo-grey-market-premium-latest-ipo-gmp/   # see what tables the page exposes
ipo run jio
```

The app and `ipo run` fetch current public GMP and subscription data by default. Uncheck **Fetch live data** in
the app or use `ipo run jio --no-scrape` to skip fetching for one analysis. To disable live fetching entirely,
set `IPO_ENABLE_SCRAPERS=0` before starting the app or CLI.

* The fetchers read public web pages. They honour `robots.txt`, wait between requests and cache raw responses
  in `data/raw/`. Check each site's terms of use before relying on them. Don't lower `IPO_REQUEST_DELAY_S`.
* `ipo collect-history` reads Chittorgarh's JSON performance-tracker endpoint. The `ipo probe` command inspects
  HTML tables, so it reports zero tables for that JavaScript-rendered tracker page.
* The GMP and IPO-detail scrapers may need tuning as those sites change. Parser tests use synthetic HTML and JSON.
  If a field comes back empty, `ipo probe <url>` shows what the page exposes; adjust the URL via the documented
  environment variables or update the parser.
* With live fetching enabled, the app reads current GMP from IPOWatch and its linked day-wise GMP page, then
  reads current category subscription ratios from Chittorgarh's dedicated subscription page. The dashboard
  shows reported subscription as a changing snapshot and charts the dated GMP observations with their source.
* A failed fetcher only produces a warning; the app falls back to seed data, your saved overrides, and the GMP
  history it recorded itself. Official NSE/BSE subscription pages are not scraped; read them and use `--set`.

## Building the training data and ML models

```bash
ipo collect-history            # all available calendar years since 2004 -> data/ipo.db
ipo collect-history --from-year 2015  # only 2015 through the current year
ipo collect-history --limit 20        # optional cap for a small test run
ipo import-csv my_history.csv  # or load your own CSV (needs name, listing_gain_pct; see store.HISTORY_COLUMNS)
ipo import-excel "data/Initial Public Offering (Updated).xlsx"  # merge the Kaggle IPO workbook
ipo import-gmp-csv data/investorgain_mainboard_ipo_gmp_2018_2026.csv  # merge GMP and tracker outcomes
ipo train                      # trains both tiers, prints walk-forward / leave-one-out results
```

History collection queries the performance tracker once per calendar year, from the current year back to 2004
by default. It skips IPO names already in the local history database.
The Excel importer maps the Kaggle workbook columns, merges likely matches by listing date and company name
or issue price/size, fills missing fields without replacing existing values, and preserves existing GMP values.
If you have GMP data, add `GMP Open`, `GMP Peak`, and `GMP Final` columns to the workbook; a single `GMP`
column is treated as the final GMP. It skips incomplete/unlisted rows and keeps repeated issuer names on distinct
IPO dates as separate records.
The InvestorGain CSV importer maps the tracker’s `GMP` column to `gmp_final`, reads listing gain from the
listing-price percentage, and merges by listing date plus IPO name or matching issue price and size. It fills
missing values without replacing populated history.

Historical **GMP** is the hard part: public pages rarely keep it. Rows without GMP still train (missing values are
handled), but the GMP-to-listing-gain relationship is learned only from rows that have it. Supplying a CSV with
`gmp_open`, `gmp_peak`, `gmp_final` columns will improve the models the most. See `docs/ML_PLAN.md`.

## How the verdict is produced

* **Short term:** rule-based estimate (GMP, discounted for issue size, retail demand and GMP trend). Once models are
  trained, the ML estimate (with an 80% band) becomes the headline for the tier it applies to. Large IPOs (>= 5,000 cr by
  default) use a heavily shrunk model because only about 15 exist.
* **Long term:** weighted 0-100 score on growth, margins, ROCE, valuation versus peers, balance sheet, moat and
  concentration/regulatory risk. Rule-based on purpose; there are too few IPOs with long outcomes for ML.

## Layout

```
src/ipo_intel/  scoring/  rule-based short and long-term scores
                fetchers/ polite HTTP client, GMP and Chittorgarh scrapers, table parsing helpers
                ml/       features, small/large estimators, walk-forward evaluation, training
                pipeline.py  merge sources (scrapers > own GMP history > seed > manual file > overrides)
                cli.py, app.py, store.py (SQLite), schema.py (fields + provenance)
seed/           hand-collected facts: 4 past IPOs and Jio (web research, 2026-10-18)
docs/ML_PLAN.md  the modelling plan and its limits
tests/          tests (synthetic data for ML and parsers)
```

## Known limits

* The 4-IPO backtest is a sanity check: its parameters were set knowing the outcomes.
* The seed Jio numbers are a snapshot from 2026-10-18; update them with `--set` as bidding progresses.
* Not modelled: market regime on listing day, anchor-book quality, lock-in expiries, index-inclusion flows.
