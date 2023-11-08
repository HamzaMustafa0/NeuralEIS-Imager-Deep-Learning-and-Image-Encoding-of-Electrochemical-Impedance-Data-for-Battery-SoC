"""
Command-line interface
- make-folds: copy per-battery folders into Fold*, with Train/Val/Test
- encode: normalize (local|percentile) and GAF-encode all folds
"""

import argparse
from pathlib import Path
from eis2img.data.folds import FoldBuilder
from eis2img.pipeline.encoder import EISGAFEncoder

from eis2img.models.trainer import Trainer, TrainConfig

def main():
    p = argparse.ArgumentParser(prog="eis2img", description="EIS→GAF encoder")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Training / Evaluation
    tr = sub.add_parser("train", help="Train a classifier on GAF images")
    tr.add_argument("--base-dir", required=True, help="FoldX directory with Train/Val/Test")
    tr.add_argument("--model", default="InceptionResNetV2", help="Backbone name")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--patience", type=int, default=5)
    tr.add_argument("--frozen-layers", type=int, default=0)
    tr.add_argument("--no-tl", action="store_true", help="Disable transfer learning")
    tr.add_argument("--no-ft", action="store_true", help="Disable fine tuning")
    ev = sub.add_parser("eval", help="Evaluate a trained model on Test")
    ev.add_argument("--base-dir", required=True)
    ev.add_argument("--model", default="InceptionResNetV2")
    ev.add_argument("--weights", required=True, help="Path to .h5 weights file")


    mk = sub.add_parser("make-folds", help="Create Fold*/Train|Validation|Test")
    mk.add_argument("--source", required=True, help="Folder containing per-battery subfolders with CSVs")
    mk.add_argument("--dest", required=True, help="Destination folder to create Fold*")
    mk.add_argument("--n-train", type=int, default=7)
    mk.add_argument("--n-val", type=int, default=3)
    mk.add_argument("--n-test", type=int, default=1)

    enc = sub.add_parser("encode", help="Normalize + GAF encode all folds")
    enc.add_argument("--folds-root", required=True)
    enc.add_argument("--normalized-out", required=True)
    enc.add_argument("--images-out", required=True)
    enc.add_argument("--n-points", type=int, default=100)
    enc.add_argument("--norm", choices=["local", "percentile"], default="local",
                     help="Normalization strategy")
    enc.add_argument("--p-low", type=float, default=20.0, help="Low percentile (percentile mode)")
    enc.add_argument("--p-high", type=float, default=80.0, help="High percentile (percentile mode)")

    args = p.parse_args()
    if args.cmd == "make-folds":
        FoldBuilder(args.source, args.dest, args.n_train, args.n_val, args.n_test).build_all()
    elif args.cmd == "train":
        cfg = TrainConfig(
            base_dir=Path(args.base_dir), model_name=args.model,
            batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
            frozen_layers=args.frozen_layers, transfer_learning=not args.no_tl, fine_tune=not args.no_ft,
        )
        trn = Trainer(cfg)
        model, ft_path, tl_path = trn.train()
        print(f"Saved weights: TL={tl_path} FT={ft_path}")
    elif args.cmd == "eval":
        cfg = TrainConfig(base_dir=Path(args.base_dir), model_name=args.model)
        trn = Trainer(cfg)
        cm, report = trn.evaluate(Path(args.weights))
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
    else:
        if args.norm == "local":
            enc = EISGAFEncoder(args.n_points)
            folds_root = Path(args.folds_root)
            for fold in sorted([p for p in folds_root.iterdir() if p.is_dir() and p.name.startswith("Fold")]):
                enc.process_fold(fold, Path(args.normalized_out), Path(args.images_out))
        else:
            for fold in sorted([p for p in Path(args.folds_root).iterdir() if p.is_dir() and p.name.startswith("Fold")]):
                enc = EISGAFEncoder(args.n_points)
                enc.process_fold_with_percentiles(
                    fold_dir=fold,
                    normalized_root=Path(args.normalized_out),
                    images_root=Path(args.images_out),
                    p_low=args.p_low, p_high=args.p_high,
                    n_points=args.n_points
                )

if __name__ == "__main__":
    main()
