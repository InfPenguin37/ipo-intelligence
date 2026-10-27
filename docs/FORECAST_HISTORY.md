# Forecast history

Every analysis that produces a short-term opening-gain estimate is recorded in
[`research/forecast_ledger.csv`](../research/forecast_ledger.csv). The ledger keeps one row per run, including the
forecast timestamp, IPO and listing date, estimated gain, heuristic and ML outputs, and the inputs with their source
and observation timestamps. Re-running an analysis creates another forecast row; it does not overwrite prior
predictions.

## Use the UI

1. Run an IPO analysis. Its forecast is appended to the ledger.
2. Review **Prediction history** near the bottom of the app, or download the full CSV there. The table shows the
   predicted and heuristic opening gains, ML range, and actual open/close gains once recorded.
3. After listing, fill in `actual_listing_open_price` and, if available, `actual_listing_close_price` in the downloaded
   CSV. Also fill in `outcome_source` and `outcome_recorded_at_utc`.
4. Upload that CSV under **Import and update history**. The app calculates the open and close gains from the actual
   prices and the recorded `offer_price`, then merges rows by `forecast_id`.

The importer requires every forecast row to retain its `forecast_id`. It preserves existing rows when an uploaded
CSV omits them and only replaces values with non-empty uploaded values for matching IDs.

## Persistence

The local app writes the ledger to the tracked `research/forecast_ledger.csv`; commit and push changes to preserve
them in GitHub. Streamlit Community Cloud's filesystem can be reset and does not automatically push file changes back
to GitHub. Download the ledger regularly from the UI, and upload the latest copy after an app restart if necessary.

The ledger is evaluation data, not training data: adding forecasts does not retrain or change the model. Compare
out-of-sample predictions with outcomes by IPO and forecast timestamp before considering any model changes.
