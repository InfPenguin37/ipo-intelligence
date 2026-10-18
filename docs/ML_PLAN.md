# ML plan

## What is predicted
Listing-day gain (%) over the issue price, with an 80% band. Also reported: sign (up or down). Not predicted:
long-term returns (too few IPOs with multi-year outcomes, and market regime dominates).

## Why two tiers
| | Small (< 5,000 cr) | Large (>= 5,000 cr) |
|---|---|---|
| Count in India | hundreds since 2021 | about 15 ever |
| Model | gradient-boosted quantile regression (q10/q50/q90) + ridge baseline | one shrunk coefficient `k`: gain ~ k x (GMP% x demand x trend) |
| Why | enough rows for non-linear effects (size, subscription mix, GMP trend) | any richer model would memorise a dozen points |
| Uncertainty | quantile heads | leave-one-out RMSE, floored at 5 points |

The cutoff is `IPO_LARGE_CUTOFF_CR` (default 5,000) and should be revisited once the data is in: check whether the
GMP-to-gain ratio actually changes around that size.

## Features
GMP % of issue price, GMP peak %, GMP trend (final / open), log issue size, OFS share, log1p of overall / QIB /
NII / retail subscription, and the rule-based estimate (so the model starts from the heuristic).
Missing values stay missing: the gradient-boosted model handles NaN natively, the ridge model imputes the median and
adds missing-indicators.

## Large-tier estimator
`k` is the ridge-style posterior with prior `k0 = 0.55` worth 3 pseudo-observations:
`k = (sum(x*y) + lam*k0) / (sum(x^2) + lam)`, `lam = 3 * mean(x^2)`. With zero large IPOs it is exactly the prior;
each new large IPO moves it. Intervals use leave-one-out error so they widen if the fit is poor.

## Evaluation rules (do not skip)
1. Small tier: **walk-forward** by listing date (train on the past, predict the next 10, repeat). Never random k-fold.
2. Large tier: leave-one-out.
3. Always report against two baselines: raw GMP and the rule-based heuristic. A model that doesn't beat the heuristic
   on MAE and 80%-interval coverage (target ~0.8) should not become the headline.
4. Report n. A result on 20 IPOs is an anecdote.
5. Re-run after every few IPOs; regimes change (2026 median listing gains for 1,000-5,000 cr issues were ~4%
   against ~16% for 500-1,000 cr).

## Known gaps and how to close them
* **Historical GMP** is the scarce input. The app records its own GMP series during live issues; a CSV import path
  exists for anything you can source. Until enough rows have GMP, the small model leans on subscription and size.
* **Market regime** (Nifty return over the prior month, number of concurrent IPOs) is not in the features yet. Add it
  once there is a price-history source.
* **Post-listing drift** (does the gain hold for 2-6 weeks?) is a separate target; seed data hints large pure-OFS
  issues fade, but n is 3.
