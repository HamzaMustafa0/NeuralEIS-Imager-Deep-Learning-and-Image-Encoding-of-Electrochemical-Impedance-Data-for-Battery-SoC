import pandas as pd
import numpy as np
from pathlib import Path
from eis2img.preprocessing.normalization import Normalizer

def test_local_normalizer(tmp_path: Path):
    df = pd.DataFrame({"R(ohm)": np.linspace(10, 20, 51),
                       "X(ohm)": np.linspace(-5, 5, 51)})
    src = tmp_path / "in.csv"; df.to_csv(src, index=False)
    out = tmp_path / "out.csv"

    Normalizer(n_points=32).normalize_file(src, out)
    got = pd.read_csv(out)
    assert len(got) == 32
    assert got["Normalized_R(ohm)"].between(0, 1).all()
    assert got["Normalized_X(ohm)"].between(0, 1).all()
