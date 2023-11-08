import pandas as pd
import numpy as np
from pathlib import Path
from eis2img.preprocessing.percentile import PercentileNormalizer

def _write_series(dir: Path, name: str, rvals, xvals):
    p = dir / name; p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"R(ohm)": rvals, "X(ohm)": xvals}).to_csv(p/"a.csv", index=False)

def test_percentile_fit_and_apply(tmp_path: Path):
    train = tmp_path / "Train"
    _write_series(train, "B1", np.arange(0, 10), np.arange(-5, 5))
    _write_series(train, "B2", np.arange(10, 20), np.arange(5, 15))

    src_dir = tmp_path / "Val" / "B3"; src_dir.mkdir(parents=True)
    pd.DataFrame({"R(ohm)": np.linspace(5, 15, 21),
                  "X(ohm)": np.linspace(0, 10, 21)}).to_csv(src_dir/"b.csv", index=False)

    out_dir = tmp_path / "out"; out_dir.mkdir()
    norm = PercentileNormalizer(n_points=16, p_low=20, p_high=80)
    norm.fit_from_dir(train)
    norm.normalize_file(src_dir/"b.csv", out_dir/"b.csv")

    got = pd.read_csv(out_dir/"b.csv")
    assert len(got) == 16
    assert (got["Normalized_R(ohm)"] <= 1).all()
    assert (got["Normalized_R(ohm)"] >= -1).all()
