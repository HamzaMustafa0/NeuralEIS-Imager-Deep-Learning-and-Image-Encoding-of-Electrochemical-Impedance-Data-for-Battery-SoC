"""
eis2img.pipeline.encoder
------------------------
Walks folds to produce normalized CSVs and GAF PNGs.

Summary:
- For each Fold*/{Train,Validation,Test}/{Battery}/*.csv:
  1) normalize (local or percentile)
  2) GAF encode to RGB
  3) save PNGs under images_root/FoldX/Split/Battery/SoC/
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Protocol, Optional
import pandas as pd
from PIL import Image

from eis2img.preprocessing.normalization import Normalizer
from eis2img.preprocessing.percentile import PercentileNormalizer
from eis2img.transforms.gaf import GAFTransformer

class NormalizerProtocol(Protocol):
    def normalize_file(self, in_csv: Path, out_csv: Path) -> None: ...

class EISGAFEncoder:
    def __init__(self, n_points: int = 100, normalizer: Optional[NormalizerProtocol] = None):
        # Default local normalization; can be swapped for PercentileNormalizer
        self.normalizer = normalizer or Normalizer(n_points)
        self.gaf = GAFTransformer()

    def process_split(self, split_root: Path, normalized_out: Path, images_out: Path) -> None:
        """Normalize all CSVs under `split_root` and emit PNGs."""
        for battery_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            out_csv_dir = normalized_out / battery_dir.name
            out_csv_dir.mkdir(parents=True, exist_ok=True)

            for csv in sorted(battery_dir.glob("*.csv")):
                # 1) normalize ➜ CSV
                out_csv = out_csv_dir / csv.name
                self.normalizer.normalize_file(csv, out_csv)

                # 2) read normalized series
                df = pd.read_csv(out_csv)
                r = df['Normalized_R(ohm)'].values
                x = df['Normalized_X(ohm)'].values

                # 3) GAF ➜ RGB image
                rgb = self.gaf.encode_rgb(r, x)

                # SoC derivation from filename: e.g. "...SoC_30..."
                soc_match = re.search(r'(?:SoC_)?(\d+)', csv.name)
                soc = f"SoC_{soc_match.group(1)}" if soc_match else "SoC_unk"

                # Output path: images_out/Battery/SoC/xxx.png
                img_dir = images_out / battery_dir.name / soc
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"GAF_{soc}_{csv.stem}.png"
                Image.fromarray(rgb).save(img_path)

    def process_fold(self, fold_dir: Path, normalized_root: Path, images_root: Path) -> None:
        """Process Train/Validation/Test using current normalizer."""
        for split in ("Train", "Validation", "Test"):
            split_path = fold_dir / split
            norm_out = normalized_root / fold_dir.name / split
            img_out = images_root / fold_dir.name / split
            norm_out.mkdir(parents=True, exist_ok=True)
            img_out.mkdir(parents=True, exist_ok=True)
            self.process_split(split_path, norm_out, img_out)

    def process_fold_with_percentiles(self, fold_dir: Path, normalized_root: Path,
                                      images_root: Path, p_low: float = 20, p_high: float = 80,
                                      n_points: int = 100) -> None:
        """
        Fit percentile stats on TRAIN of this fold, then process all splits.
        """
        train_dir = fold_dir / "Train"
        pnorm = PercentileNormalizer(n_points=n_points, p_low=p_low, p_high=p_high)
        pnorm.fit_from_dir(train_dir)
        self.normalizer = pnorm
        self.process_fold(fold_dir, normalized_root, images_root)
