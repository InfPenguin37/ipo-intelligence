import joblib
import pandas as pd
import pytest

from ipo_intel import config
from conftest import make_history
from ipo_intel.ml import load_models, predict_one, train_all
from ipo_intel.ml.estimators import LargeModel, SmallModel
from ipo_intel.ml.evaluate import loo_large, summarize, walk_forward_small
from ipo_intel.ml.features import build_features

# NOTE: these use SYNTHETIC data. They check the mechanics (shapes, ordering, learning direction),
# not how well the models will do on real IPOs.


def _split(df, cutoff=5000):
    return df[df.issue_size_cr < cutoff].reset_index(drop=True), df[df.issue_size_cr >= cutoff].reset_index(drop=True)


def test_small_model_quantiles_are_ordered():
    small, _ = _split(make_history())
    X, y = build_features(small), small["listing_open_gain_pct"]
    pred = SmallModel().fit(X, y).predict(X)
    assert list(pred.columns) == ["q10", "q50", "q90", "ridge"]
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all()


def test_small_model_handles_missing_gmp_and_subscription():
    small, _ = _split(make_history())
    X = build_features(small)
    model = SmallModel().fit(X, small["listing_open_gain_pct"])
    blank = build_features(pd.DataFrame([{"issue_size_cr": 800, "price_band_high": 300}]))
    assert model.predict(blank).notna().all(axis=None)


def test_small_model_drops_features_missing_for_every_training_row():
    small, _ = _split(make_history())
    X = build_features(small)
    X.loc[:, [column for column in X if column != "log_size"]] = float("nan")

    model = SmallModel().fit(X, small["listing_open_gain_pct"])

    assert model.predict(X).notna().all(axis=None)
    assert model.gbm_features == ["log_size"]


def test_walk_forward_beats_raw_gmp_when_gmp_is_biased():
    small, _ = _split(make_history(n_small=200))
    res = walk_forward_small(small, min_train=60, step=20)
    s = summarize(res, "q50", ridge=True)
    assert s["n"] == 140
    assert s["mae_q50"] < s["mae_raw_gmp"]
    assert 0.5 < s["interval_coverage_80"] <= 1.0


def test_large_model_with_no_data_is_the_prior():
    m = LargeModel().fit(pd.DataFrame(columns=["price_band_high", "gmp_final", "listing_gain_pct"]))
    assert m.k == 0.55 and m.sd == 5.0


def test_large_model_learns_towards_truth_but_stays_shrunk():
    _, large = _split(make_history(n_large=12))
    m = LargeModel().fit(large)
    assert abs(m.k - 0.40) < abs(0.55 - 0.40)  # moved from the prior towards the true 0.40
    assert m.k > 0.40 - 0.02  # shrinkage keeps it at or above the truth here
    assert m.sd >= 5.0


def test_large_loo_runs():
    _, large = _split(make_history(n_large=12))
    res = loo_large(large)
    assert len(res) == 12
    assert {"q10", "q50", "q90", "k"} <= set(res.columns)


def test_train_all_saves_models_and_predicts_both_tiers():
    hist = make_history(n_large=30)
    report = train_all(hist)
    assert report["n_small"] == 150 and report["n_large"] == 30
    assert report["training_target"] == "listing_open_gain_pct"
    assert report["coverage"]["all"]["with_gmp_final_observed_at"] == 0
    assert report["coverage"]["large"]["usable_gmp_model_rows"] == 30
    assert "small_walk_forward" in report and "large_loo" in report
    models = load_models()
    assert models is not None

    small_vals = {"issue_size_cr": 900, "price_band_high": 400, "gmp_final": 60, "gmp_open": 50, "sub_retail": 8}
    large_vals = {"issue_size_cr": 30000, "price_band_high": 1119, "gmp_final": 221, "gmp_open": 172}
    for vals, tier in ((small_vals, "small"), (large_vals, "large")):
        p = predict_one(models, vals)
        assert p["tier"] == tier
        assert p["q10"] <= p["q50"] <= p["q90"]


def test_bundled_pretrained_models_work_without_local_data():
    assert not (config.MODELS_DIR / "models.joblib").exists()

    models = load_models()

    assert models is not None
    assert models["small"] is not None
    prediction = predict_one(
        models,
        {
            "issue_size_cr": 900,
            "price_band_high": 400,
            "gmp_final": 60,
            "gmp_open": 50,
            "sub_retail": 8,
        },
    )
    assert prediction is not None
    assert prediction["tier"] == "small"
    assert prediction["q10"] <= prediction["q50"] <= prediction["q90"]


def test_local_models_override_bundled_pretrained_models():
    local_path = config.MODELS_DIR / "models.joblib"
    local_path.parent.mkdir(parents=True)
    joblib.dump({"local_override": True}, local_path)

    assert load_models() == {"local_override": True}


def test_training_ignores_ambiguous_and_closing_gain_labels():
    history = make_history(n_small=100, n_large=30)
    history["listing_gain_pct"] = history["listing_open_gain_pct"] + 30
    history["listing_close_gain_pct"] = history["listing_open_gain_pct"] - 20

    report = train_all(history)
    models = load_models()

    assert report["training_target"] == "listing_open_gain_pct"
    assert models["small"].n_train == 100
    assert models["large"].n_train == 30


def test_large_model_is_disabled_below_gmp_coverage_threshold():
    history = make_history(n_small=100, n_large=30)
    large_indexes = history.index[history["issue_size_cr"] >= 5000][:6]
    history.loc[large_indexes, "gmp_final"] = float("nan")

    report = train_all(history)
    models = load_models()

    assert report["coverage"]["large"]["usable_gmp_model_rows"] == 24
    assert models["large"] is None
    assert "need at least 25" in report["large_note"]


def test_too_little_small_data_falls_back_gracefully():
    hist = make_history(n_small=10, n_large=4)
    report = train_all(hist)
    assert "small_note" in report
    models = load_models()
    assert predict_one(models, {"issue_size_cr": 900, "price_band_high": 400, "gmp_final": 60}) is None
    assert predict_one(models, {"issue_size_cr": 30000, "price_band_high": 1119, "gmp_final": 221}) is None
