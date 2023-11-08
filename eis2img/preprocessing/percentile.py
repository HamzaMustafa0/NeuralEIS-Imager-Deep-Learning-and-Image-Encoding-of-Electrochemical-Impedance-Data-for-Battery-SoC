"""
eis2img.preprocessing.percentile
--------------------------------
Percentile-based global normalization. Percentiles are computed over the
TRAIN split of each fold and then applied to Train/Validation/Test.

Summary:
- Concatenates all R/X from TRAIN split to compute (p_low, p_high)
- Resamples each curve to N
- Applies custom (affine) normalization formula, then clips to [-1, 1]
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

@dataclass(frozen=True)
class PercentileStats:
    minR: float
    maxR: float
    minX: float
    maxX: float

class PercentileNormalizer:
    def __init__(self, n_points: int = 100, p_low: float = 20.0, p_high: float = 80.0, eps: float = 1e-12):
        self.N = n_points
        self.p_low = p_low
        self.p_high = p_high
        self.eps = eps
        self.stats: PercentileStats | None = None

    def _resample(self, df: pd.DataFrame) -> np.ndarray:
        r = df['R(ohm)'].to_numpy()
        x = df['X(ohm)'].to_numpy()
        idx = np.arange(len(r))
        eq = np.linspace(0, len(r) - 1, self.N)
        fr = interp1d(idx, r, kind='linear', fill_value='extrapolate')
        fx = interp1d(idx, x, kind='linear', fill_value='extrapolate')
        return np.column_stack((fr(eq), fx(eq)))

    @staticmethod
    def _all_csvs(root: Path) -> Iterable[Path]:
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            yield from sorted(sub.glob("*.csv"))

    def fit_from_dir(self, dir_with_soc_csvs: Path):
        """Aggregate TRAIN curves to compute percentiles."""
        real_vals: List[float] = []
        imag_vals: List[float] = []
        for csv in self._all_csvs(dir_with_soc_csvs):
            df = pd.read_csv(csv)
            real_vals.extend(df['R(ohm)'].tolist())
            imag_vals.extend(df['X(ohm)'].tolist())

        r_series = pd.Series(real_vals)
        x_series = pd.Series(imag_vals)
        minR = r_series.quantile(self.p_low / 100.0)
        maxR = r_series.quantile(self.p_high / 100.0)
        minX = x_series.quantile(self.p_low / 100.0)
        maxX = x_series.quantile(self.p_high / 100.0)
        from typing import cast
        self.stats = PercentileStats(float(minR), float(maxR), float(minX), float(maxX))
        return self.stats

    def normalize_file(self, in_csv: Path, out_csv: Path) -> None:
        """Normalize one CSV using previously fitted percentile stats."""
        if self.stats is None:
            raise RuntimeError("PercentileNormalizer not fitted. Call fit_from_dir(train_dir) first.")
        df = pd.read_csv(in_csv)
        res = self._resample(df)

        minR, maxR, minX, maxX = self.stats.minR, self.stats.maxR, self.stats.minX, self.stats.maxX
        denomR = (maxR - minR) if (maxR - minR) > self.eps else self.eps
        denomX = (maxX - minX) if (maxX - minX) > self.eps else self.eps

        # Affine normalization formula then clipped for GAF stability
        norm_R = (res[:, 0] - maxR) + (res[:, 0] - minR) / denomR
        norm_X = (res[:, 1] - maxX) + (res[:, 1] - minX) / denomX

        out = pd.DataFrame({
            'Resampled_R(ohm)': res[:, 0],
            'Resampled_X(ohm)': res[:, 1],
            'Normalized_R(ohm)': np.clip(norm_R, -1, 1),
            'Normalized_X(ohm)': np.clip(norm_X, -1, 1),
        })
        out.to_csv(out_csv, index=False)
