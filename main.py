#!/usr/bin/env python3
"""
main.py — One-touch launcher

Workflow on any option:
  - If no folds exist: create EXACTLY 11 folds (7 train / 3 val / 1 test)
  - Normalize using TRAIN ONLY (percentile fit per fold)
  - Generate GAF images for ALL splits in ALL folds

Options:
 [1] Generate GAF images only
 [2] Train / Evaluate DL model only
 [3] Full pipeline (ensure folds -> encode -> train/eval)

Paths are project-relative under eis2img/data/.
"""
from __future__ import annotations
import sys, subprocess
from pathlib import Path

# ---------- project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "eis2img" / "data"
SOURCE_MEASUREMENTS = DATA_DIR / "Measurements"
FOLDS_DIR          = DATA_DIR / "Folds"
NORMALIZED_DIR     = DATA_DIR / "Normalized"
IMAGES_DIR         = DATA_DIR / "GAF_EncodedImageOutput"
for d in (FOLDS_DIR, NORMALIZED_DIR, IMAGES_DIR): d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "InceptionResNetV2"

# ---------- selective deps ----------
ENCODE_PKGS = ["numpy", "pandas", "scipy", "pillow"]
DL_PKGS     = ["scikit-learn", "tensorflow>=2.10"]
def pip_install(pkgs: list[str]): 
    if pkgs: subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

# ---------- folds (exactly 11 if none exist) ----------
from eis2img.data.folds import FoldBuilder

def ensure_eleven_folds():
    existing = [p for p in FOLDS_DIR.iterdir() if p.is_dir() and p.name.startswith("Fold")]
    if existing:
        print(f"[folds] Found {len(existing)} existing fold(s). Skipping creation.")
        return
    print("[folds] No folds found. Creating EXACTLY 11 folds (7/3/1)…")
    FoldBuilder(SOURCE_MEASUREMENTS, FOLDS_DIR, 7, 3, 1).build_first_n(11)

# ---------- encode (percentile => fit on Train only, images for all splits) ----------
def do_encode_all_folds(n_points=100, p_low=20.0, p_high=80.0):
    pip_install(ENCODE_PKGS)
    from eis2img.pipeline.encoder import EISGAFEncoder
    ensure_eleven_folds()

    enc = EISGAFEncoder(n_points=n_points)
    print(f"[encode] Starting (percentile, n_points={n_points}, p_low={p_low}, p_high={p_high}) …")
    for fold in sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir() and p.name.startswith("Fold")):
        # NOTE: process_fold_with_percentiles FITS on Train of this fold, then processes Train/Val/Test
        enc.process_fold_with_percentiles(
            fold_dir=fold,
            normalized_root=NORMALIZED_DIR,
            images_root=IMAGES_DIR,
            p_low=p_low,
            p_high=p_high,
            n_points=n_points
        )
        print(f"[encode] {fold.name} → done")
    print(f"[encode] ✅ Images saved under: {IMAGES_DIR}")

# ---------- train/eval ----------
def do_train_eval_all_folds(model_name=MODEL_NAME, epochs=10, batch_size=32, patience=5, frozen_layers=0,
                            transfer_learning=True, fine_tune=True, eval_after=True):
    pip_install(ENCODE_PKGS + DL_PKGS)
    from eis2img.models.trainer import Trainer, TrainConfig

    # Ensure images exist
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        sys.exit("❌ No encoded images found. Run option [1] or [3] first.")

    for fold in sorted(p for p in IMAGES_DIR.iterdir() if p.is_dir() and p.name.startswith("Fold")):
        cfg = TrainConfig(
            base_dir=fold,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            frozen_layers=frozen_layers,
            transfer_learning=transfer_learning,
            fine_tune=fine_tune,
        )
        trn = Trainer(cfg)
        model, ft_path, tl_path = trn.train()
        print(f"[train] {fold.name}: saved TL={tl_path}  FT={ft_path}")
        if eval_after:
            weights = ft_path if fine_tune else tl_path
            cm, report = trn.evaluate(weights)
            print(f"[eval] {fold.name} — Confusion Matrix:\n{cm}\n\nClassification Report:\n{report}")

# ---------- menu ----------
def main():
    if not SOURCE_MEASUREMENTS.exists():
        sys.exit("❌ Missing data: eis2img/data/Measurements not found (per-battery folders expected).")

    print("Choose:")
    print(" [1] Generate GAF images only")
    print(" [2] Train / Evaluate DL model only")
    print(" [3] Full pipeline (ensure 11 folds -> encode -> train/eval)")
    choice = input("\nEnter 1/2/3: ").strip()

    if choice == "1":
        do_encode_all_folds()
    elif choice == "2":
        do_train_eval_all_folds()
    elif choice == "3":
        ensure_eleven_folds()
        do_encode_all_folds()
        do_train_eval_all_folds()
        print("\n🎉 Full pipeline complete.")
    else:
        sys.exit("Invalid choice. Please run again and enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
