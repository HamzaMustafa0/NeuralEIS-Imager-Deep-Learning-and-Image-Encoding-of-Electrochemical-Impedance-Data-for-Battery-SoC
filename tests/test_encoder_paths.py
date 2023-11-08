from pathlib import Path
import pandas as pd
import numpy as np
from eis2img.pipeline.encoder import EISGAFEncoder

def test_encoder_writes_images(tmp_path: Path):
    fold = tmp_path / "Fold1" / "Train" / "BattA"; fold.mkdir(parents=True)
    pd.DataFrame({"R(ohm)": np.linspace(1,2,25), "X(ohm)": np.linspace(0,1,25)}).to_csv(fold/"cell_SoC_30.csv", index=False)

    norm_root = tmp_path / "NormOut"
    img_root = tmp_path / "ImgOut"
    enc = EISGAFEncoder(n_points=16)
    enc.process_fold(tmp_path / "Fold1", norm_root, img_root)

    imgs = list((img_root / "Fold1" / "Train" / "BattA" / "SoC_30").glob("*.png"))
    assert len(imgs) == 1
