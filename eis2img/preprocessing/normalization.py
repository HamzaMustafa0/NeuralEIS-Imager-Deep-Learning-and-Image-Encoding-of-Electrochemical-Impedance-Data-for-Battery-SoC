"""
eis2img.preprocessing.normalization
-----------------------------------
Local (per-file) min–max normalization after resampling to N points.

Summary:
- Interpolates R(ohm) and X(ohm) to fixed N
- Normalizes each file independently to [0,1]
- Writes a traceable CSV with resampled and normalized columns
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from typing import Tuple

class Normalizer:
    def __init__(self, n_points: int = 100, eps: float = 1e-12):
        self.N = n_points
        self.eps = eps

    @staticmethod
    def _find_minmax(df: pd.DataFrame) -> Tuple[float, float, float, float]:
        minR, maxR = df['R(ohm)'].min(), df['R(ohm)'].max()
        minX, maxX = df['X(ohm)'].min(), df['X(ohm)'].max()
        return minR, maxR, minX, maxX

    def _resample(self, df: pd.DataFrame) -> np.ndarray:
        """Linear interpolation on index to get exactly N points."""
        real, imag = df['R(ohm)'].to_numpy(), df['X(ohm)'].to_numpy()
        x = np.arange(len(real))
        eq = np.linspace(0, len(real) - 1, self.N)
        f_r = interp1d(x, real, kind='linear', fill_value='extrapolate')
        f_i = interp1d(x, imag, kind='linear', fill_value='extrapolate')
        return np.column_stack((f_r(eq), f_i(eq)))

    def normalize_file(self, in_csv: Path, out_csv: Path) -> None:
        """Normalize one CSV and save a new CSV with resampled+normalized columns."""
        df = pd.read_csv(in_csv)
        res = self._resample(df)
        minR, maxR, minX, maxX = self._find_minmax(df)
        denomR = (maxR - minR) if (maxR - minR) > self.eps else self.eps
        denomX = (maxX - minX) if (maxX - minX) > self.eps else self.eps

        norm_R = (res[:, 0] - minR) / denomR
        norm_X = (res[:, 1] - minX) / denomX

        out = pd.DataFrame({
            'Resampled_R(ohm)': res[:, 0],
            'Resampled_X(ohm)': res[:, 1],
            'Normalized_R(ohm)': norm_R,
            'Normalized_X(ohm)': norm_X,
        })
        out.to_csv(out_csv, index=False)
