import sys, pathlib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import ml_utils  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    df["y"] = (df["a"] + 0.5 * df["b"] > 0).astype(int)
    return df


def test_missing_data_summary_runs():
    df = _toy()
    df.loc[:9, "a"] = np.nan
    ml_utils.missing_data_summary(df)


def test_random_forest_learns_signal(capsys):
    ml_utils.train_random_forest_with_split(_toy(), "y")
    out = capsys.readouterr().out.lower()
    assert "accuracy" in out
