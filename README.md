# eis2img — EIS → Gramian Angular Field (GAF) images

This package encodes Electrochemical Impedance Spectroscopy (EIS) SoC curves into RGB Gramian Angular Field images.

## Pipeline
1. **Folds:** Create Train/Validation/Test folds by copying per-battery folders.
2. **Normalize:** 
   - `local` — Per-file min–max (0..1), robust for heterogeneous samples.  
   - `percentile` — Global percentiles (default 20th–80th) computed on the **TRAIN** split, then applied to all splits. Reduces outlier impact.
3. **Transform:** Convert normalized R/X to polar angles and build three GAF channels:
   - Red = cos(φ_imag + φ_imag)
   - Green = sin(φ_imag − φ_imag)
   - Blue = cos(φ_real + φ_real)
4. **Export:** PNG images per SoC and split.

## CLI usage
```bash
# Install for development
pip install -e .

# 1) Make folds
python -m eis2img.cli make-folds --source /path/to/Measurements --dest /path/to/folds

# 2a) Encode with local min-max
python -m eis2img.cli encode   --folds-root /path/to/folds   --normalized-out /path/to/Normalized   --images-out /path/to/GAF_EncodedImageOutput   --n-points 100 --norm local

# 2b) Encode with percentile normalization (fit on TRAIN per fold)
python -m eis2img.cli encode   --folds-root /path/to/folds   --normalized-out /path/to/Normalized   --images-out /path/to/GAF_EncodedImageOutput   --n-points 100 --norm percentile --p-low 20 --p-high 80
```



## Train a classifier on the generated images
```bash
# Example: train on Fold1 images produced by the encoder
python -m eis2img.cli train \
  --base-dir /path/to/GAF_EncodedImageOutput/Fold1 \  --model InceptionResNetV2 \  --epochs 10 --batch-size 32 --patience 5 --frozen-layers 0
```


