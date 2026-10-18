# Short-term ML techniques

This document describes the short-term listing-gain models currently implemented in IPO Intelligence. These estimates
are research aids, not investment advice. The model is deliberately withheld until a live IPO has a qualifying final
GMP observation; the rule-based estimate remains available before then.

## Prediction target

The training target is `listing_open_gain_pct`: the percentage change from the IPO offer price to the opening listing
price. It is calculated as:

```text
(listing open price - offer price) / offer price * 100
```

The Kaggle workbook's `List Price` is treated as the opening price. InvestorGain's `Listing Price` is also the opening
price, and its `Listing Day Close` is stored as a separate closing outcome. Generic historical `listing_gain_pct`
values are retained for reference, but rows without a verified opening-price label are excluded from training.

The model does not predict long-term returns. Long-term analysis is a separate, rule-based fundamentals score.

## Size tiers

The default cutoff is ₹5,000 crore, configurable with `IPO_LARGE_CUTOFF_CR`.

| Tier | Current technique | Rationale |
| --- | --- | --- |
| Small, below ₹5,000 crore | Three `HistGradientBoostingRegressor` quantile models estimate the 10th, 50th and 90th percentiles. A `RidgeCV` model is also fit as an evaluation baseline. | The larger historical sample can support nonlinear relationships between GMP, issue characteristics and subscription. |
| Large, ₹5,000 crore and above | A single-coefficient, prior-shrunk estimator maps a demand- and trend-adjusted GMP return to listing gain. | There are fewer large IPOs; a compact estimator limits overfitting compared with a richer model. |

The large-tier model is trained only when at least 25 large IPOs have an opening-gain label, offer price and final GMP.
Otherwise it is disabled and the app falls back to the rule-based estimate.

## Inputs

The small-tier feature builder derives:

- Final GMP as a percentage of offer price.
- Peak GMP as a percentage of offer price.
- GMP trend, calculated as final GMP divided by opening GMP.
- Logarithm of issue size.
- Offer-for-sale share.
- `log1p` of overall, QIB, NII and retail subscription multiples.
- The rule-based listing-gain estimate, giving the model a heuristic reference point.

The large-tier model uses final GMP return, retail-demand factor and GMP-trend factor to construct one adjusted GMP
signal. Missing historical feature values remain missing; the gradient-boosted model handles missing values, while the
Ridge baseline median-imputes and adds missingness indicators.

The feature definitions are in [`features.py`](../src/ipo_intel/ml/features.py), and the heuristic inputs are in
[`short_term.py`](../src/ipo_intel/scoring/short_term.py).

## Estimators and ranges

### Small tier

The small-tier model fits independent quantile gradient-boosting regressors for quantiles 0.1, 0.5 and 0.9, with
maximum depth 3, learning rate 0.05, up to 200 boosting iterations, 10 minimum samples per leaf and L2 regularization
of 1.0. Predicted quantiles are sorted per IPO to prevent crossing.

The displayed q10-to-q90 range is a nominal central 80% quantile range. It is **not currently conformal-calibrated**
and is not a guarantee of 80% coverage on future IPOs.

### Large tier

The large-tier estimate is `k × adjusted GMP return`. The coefficient is shrunk toward a prior of `k = 0.55`, with
prior strength equivalent to three pseudo-observations. The uncertainty scale is the root-mean-square leave-one-out
residual, floored at 5 percentage points. The displayed range is the estimate plus or minus `1.2816 ×` that scale.

This range is also an approximation, not a coverage guarantee. Leave-one-out results are a diagnostic for the small
large-IPO sample, not evidence of forward performance.

Estimator implementations are in [`estimators.py`](../src/ipo_intel/ml/estimators.py).

## Training and availability

Run `ipo train` to train and persist the models under the local data directory and print sample counts, feature
coverage, and available evaluation summaries. The model target is defined in [`targets.py`](../src/ipo_intel/ml/targets.py).

For a live IPO, `gmp_latest` is promoted to `gmp_final` only when its recorded observation date in India is exactly one
calendar day before the listing date. The app withholds the ML estimate and range until that condition is met. This
policy uses the listing date and observed GMP timestamp; it does not forecast when the final reading will be published.
If either value is unavailable, the model remains withheld. A user can explicitly provide `gmp_final` as a manual
override; doing so marks it final and allows prediction, so the user is responsible for supplying a genuinely final
reading.

## Evaluation and present limitations

Small-tier evaluation sorts labeled IPOs by listing date and uses expanding-window walk-forward validation, training on
the past and predicting later IPOs. It reports sample count, MAE against raw GMP, the heuristic, q50 and Ridge, sign
accuracy, and q10-to-q90 interval coverage. Large-tier leave-one-out evaluation is reported separately.

On the locally backfilled history available on 18 October 2026, the small-tier walk-forward evaluation reported:

| Measure | Result |
| --- | ---: |
| Evaluation IPOs (`n`) | 638 |
| q50 model MAE | 11.58 percentage points |
| Raw-GMP baseline MAE | 9.82 percentage points |
| Rule-based heuristic MAE | 10.43 percentage points |
| q10-to-q90 interval coverage | 64.6% |

These results show the current ML headline has **not** demonstrated an advantage over the raw-GMP baseline, and the
nominal 80% interval is under-covering on this evaluation. Treat both point estimates and uncertainty ranges
cautiously; the app's final-GMP availability gate is not an accuracy acceptance gate.

The imported history contained 745 verified opening labels among 746 rows, and 34 large IPOs with usable final GMP.
Imported historical GMP did not have observation timestamps, so these rows cannot establish which GMP values were
available at a consistent prediction time. That timing limitation can make historical evaluations more optimistic
than a strictly point-in-time test.

The following planned improvements are **not implemented**: rolling-market-regime features, fixed-time historical
GMP snapshots, a recent rolling-mean baseline gate for promoting ML to headline, calibrated prediction intervals,
and a chronological acceptance test for the large tier. See [`ML_PLAN.md`](./ML_PLAN.md) for the broader roadmap.

To reproduce the reported metrics against the current local database, run:

```bash
ipo train
```

Results can change as the local history is updated.
