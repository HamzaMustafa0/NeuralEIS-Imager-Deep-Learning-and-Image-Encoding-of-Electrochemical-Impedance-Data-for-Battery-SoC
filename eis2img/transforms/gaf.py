"""
eis2img.transforms.gaf
----------------------
Gramian Angular Field (GAF) RGB encoder.

Summary:
- Convert normalized R/X in [-1, 1] to polar angles φ=arccos(v)
- Build channels:
  red   = cos(φ_imag + φ_imag)
  green = sin(φ_imag − φ_imag)
  blue  = cos(φ_real + φ_real)
- Map from [-1,1] to uint8 [0,254] via (v+1)*127
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

class GAFTransformer:
    @staticmethod
    def _polar(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Clip to valid arccos domain for numerical safety
        v = np.clip(v, -1.0, 1.0)
        phi = np.arccos(v)
        r = np.arange(len(v)) / len(v)
        return phi, r

    @staticmethod
    def _tabulate(x: np.ndarray, y: np.ndarray, f) -> np.ndarray:
        # Vectorize over a sparse mesh for memory efficiency
        X, Y = np.meshgrid(x, y, sparse=True)
        return np.vectorize(f)(X, Y)

    @staticmethod
    def _cos_sum(a, b): return np.cos(a + b)
    @staticmethod
    def _sin_diff(a, b): return np.sin(a - b)

    def encode_rgb(self, norm_R: np.ndarray, norm_X: np.ndarray) -> np.ndarray:
        """Create an (N,N,3) uint8 GAF image from normalized R/X."""
        phi_r, _ = self._polar(norm_R)
        phi_i, _ = self._polar(norm_X)

        red   = self._tabulate(phi_i, phi_i, self._cos_sum)
        green = self._tabulate(phi_i, phi_i, self._sin_diff)
        blue  = self._tabulate(phi_r, phi_r, self._cos_sum)

        rgb = np.zeros((len(red), len(red), 3), dtype=np.uint8)
        rgb[:, :, 0] = ((red   + 1) * 127).astype(np.uint8)
        rgb[:, :, 1] = ((green + 1) * 127).astype(np.uint8)
        rgb[:, :, 2] = ((blue  + 1) * 127).astype(np.uint8)
        return rgb
