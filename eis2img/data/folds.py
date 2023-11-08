# eis2img/data/folds.py
from __future__ import annotations
import shutil
from itertools import combinations, islice
from pathlib import Path

class FoldBuilder:
    def __init__(self, source_root: str | Path, dest_root: str | Path,
                 n_train: int = 7, n_val: int = 3, n_test: int = 1):
        self.source_root = Path(source_root)
        self.dest_root = Path(dest_root)
        self.n_train, self.n_val, self.n_test = n_train, n_val, n_test

    def _batteries(self) -> list[str]:
        # deterministic order
        return sorted([p.name for p in self.source_root.iterdir() if p.is_dir()])

    def _write_fold(self, fold_name: str, train: list[str], val: list[str], test: list[str]) -> None:
        fold_dir = self.dest_root / fold_name
        train_dir, val_dir, test_dir = fold_dir / "Train", fold_dir / "Validation", fold_dir / "Test"
        for d in (train_dir, val_dir, test_dir):
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        for b in train: shutil.copytree(self.source_root / b, train_dir / b)
        for b in val:   shutil.copytree(self.source_root / b, val_dir / b)
        for b in test:  shutil.copytree(self.source_root / b, test_dir / b)
        print(f"[{fold_name}] Train={train} | Val={val} | Test={test}")

    def build_first_n(self, n_folds: int) -> None:
        """
        Create EXACTLY n_folds by taking the first n combinations (sorted deterministically)
        of size (n_train + n_val + n_test) from the battery list.
        """
        bats = self._batteries()
        total = self.n_train + self.n_val + self.n_test
        if len(bats) < total:
            raise ValueError(
                f"Not enough batteries: found {len(bats)}, need {total} "
                f"({self.n_train} train + {self.n_val} val + {self.n_test} test)"
            )

        combo_iter = combinations(bats, total)
        took_any = False
        for idx, combo in enumerate(islice(combo_iter, n_folds), start=1):
            took_any = True
            train = list(combo[:self.n_train])
            val   = list(combo[self.n_train:self.n_train + self.n_val])
            test  = list(combo[-self.n_test:])
            self._write_fold(f"Fold{idx}", train, val, test)

        if not took_any:
            print("[FoldBuilder] No combinations produced. Check your split sizes or battery count.")
