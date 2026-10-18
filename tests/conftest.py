import numpy as np
import pandas as pd
import pytest

from ipo_intel import config


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Every test gets its own empty data directory."""
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA", d)
    monkeypatch.setattr(config, "DB_PATH", d / "ipo.db")
    monkeypatch.setattr(config, "RAW_DIR", d / "raw")
    monkeypatch.setattr(config, "MANUAL_DIR", d / "manual")
    monkeypatch.setattr(config, "MODELS_DIR", d / "models")
    return d


def make_history(n_small=150, n_large=12, seed=0) -> pd.DataFrame:
    """SYNTHETIC IPO history with a known structure, for testing mechanics only (not real data)."""
    from ipo_intel.ml.estimators import LargeModel

    rng = np.random.default_rng(seed)

    def rows(n, size_lo, size_hi):
        price = rng.uniform(100, 1500, n)
        pct = rng.uniform(-5, 40, n)
        final = price * pct / 100
        return pd.DataFrame(
            {
                "issue_size_cr": rng.uniform(size_lo, size_hi, n),
                "price_band_high": price,
                "ofs_share": rng.uniform(0, 1, n),
                "gmp_final": final,
                "gmp_open": final * rng.uniform(0.7, 1.3, n),
                "gmp_peak": final * 1.2,
                "sub_overall": rng.uniform(1, 80, n),
                "sub_qib": rng.uniform(1, 120, n),
                "sub_nii": rng.uniform(1, 60, n),
                "sub_retail": rng.uniform(0.5, 30, n),
            }
        )

    small = rows(n_small, 200, 4000)
    small["listing_gain_pct"] = 0.7 * (small["gmp_final"] / small["price_band_high"] * 100) + rng.normal(0, 4, n_small)
    small["listing_open_gain_pct"] = small["listing_gain_pct"]
    large = rows(n_large, 5000, 30000)
    large["sub_retail"] = rng.uniform(0.5, 5, n_large)
    large["listing_gain_pct"] = 0.40 * LargeModel.x(large) + rng.normal(0, 2, n_large)
    large["listing_open_gain_pct"] = large["listing_gain_pct"]
    df = pd.concat([small, large], ignore_index=True)
    df["name"] = [f"IPO {i}" for i in range(len(df))]
    df["listing_date"] = pd.date_range("2022-01-01", periods=len(df), freq="5D").astype(str)
    df["source"] = "synthetic"
    return df.sample(frac=1, random_state=1).reset_index(drop=True)
