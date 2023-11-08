"""High-level trainer for transfer learning + fine-tuning on eis2img datasets."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

from .backbones import build_model
from ..utils.metrics import confusion_and_report

@dataclass
class TrainConfig:
    base_dir: Path
    model_name: str = 'InceptionResNetV2'
    batch_size: int = 32
    epochs: int = 10
    patience: int = 5
    frozen_layers: int = 0          # how many layers to keep frozen in fine-tune stage
    transfer_learning: bool = True
    fine_tune: bool = True
    lr_tl: float = 1e-3
    lr_ft: float = 1e-4
    augment: bool = True
    logs_dir: Path = Path('logs')

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_dir = cfg.base_dir / 'Train'
        self.val_dir = cfg.base_dir / 'Val'
        self.test_dir = cfg.base_dir / 'Test'
        self.nb_classes = len([p for p in self.train_dir.iterdir() if p.is_dir()])
        self.model, self.img_w, self.img_h = build_model(cfg.model_name, self.nb_classes)

    def _generators(self):
        aug = dict(rescale=1./255, horizontal_flip=True, vertical_flip=True, fill_mode='nearest') if self.cfg.augment else dict(rescale=1./255)
        train_gen = ImageDataGenerator(**aug).flow_from_directory(self.train_dir, target_size=(self.img_h, self.img_w),
                                                                  batch_size=self.cfg.batch_size, class_mode='categorical')
        val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(self.val_dir, target_size=(self.img_h, self.img_w),
                                                                         batch_size=self.cfg.batch_size, class_mode='categorical')
        test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(self.test_dir, target_size=(self.img_h, self.img_w),
                                                                          shuffle=False, target_size=(self.img_h, self.img_w),
                                                                          class_mode='categorical', batch_size=1)
        return train_gen, val_gen, test_gen

    def _compile(self, lr: float):
        opt = SGD(learning_rate=lr, momentum=0.9)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def _freeze_base(self, n_keep_frozen: Optional[int] = None):
        if n_keep_frozen is None:
            for layer in self.model.layers: layer.trainable = False
        else:
            for i, layer in enumerate(self.model.layers):
                layer.trainable = i >= n_keep_frozen

    def train(self) -> Tuple[tf.keras.Model, Path, Path]:
        train_gen, val_gen, _ = self._generators()
        tl_path = Path(f"{self.cfg.model_name}_{self.nb_classes}cls_tl.h5")
        ft_path = Path(f"{self.cfg.model_name}_{self.nb_classes}cls_ft.h5")

        # Transfer learning
        if self.cfg.transfer_learning:
            ckpt = ModelCheckpoint(str(tl_path), monitor='val_loss', save_best_only=True, verbose=1)
            early = EarlyStopping(monitor='val_loss', patience=self.cfg.patience, verbose=1)
            tb = TensorBoard(log_dir=str(self.cfg.logs_dir / 'transfer'))
            self._freeze_base(None)  # freeze all base
            self._compile(self.cfg.lr_tl)
            self.model.fit(train_gen, epochs=self.cfg.epochs, validation_data=val_gen, callbacks=[ckpt, early, tb])

        # Fine tuning
        if self.cfg.fine_tune:
            ckpt = ModelCheckpoint(str(ft_path), monitor='val_loss', save_best_only=True, verbose=1)
            early = EarlyStopping(monitor='val_loss', patience=self.cfg.patience, verbose=1)
            tb = TensorBoard(log_dir=str(self.cfg.logs_dir / 'fine'))
            self.model.load_weights(str(tl_path))
            self._freeze_base(self.cfg.frozen_layers)
            self._compile(self.cfg.lr_ft)
            self.model.fit(train_gen, epochs=self.cfg.epochs, validation_data=val_gen, callbacks=[ckpt, early, tb])

        return self.model, ft_path, tl_path

    def evaluate(self, weights_path: Path) -> Tuple[np.ndarray, str]:
        _, _, test_gen = self._generators()
        self.model.load_weights(str(weights_path))
        probs = self.model.predict(test_gen, steps=test_gen.samples)
        preds = np.argmax(probs, axis=1)
        cm, report = confusion_and_report(test_gen.classes, preds, target_names=list(test_gen.class_indices.keys()))
        return cm, report
