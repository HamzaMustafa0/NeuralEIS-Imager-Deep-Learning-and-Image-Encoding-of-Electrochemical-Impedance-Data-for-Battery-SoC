"""Keras backbones and simple CNNs with a clean factory API."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG19, ResNet50, InceptionV3, InceptionResNetV2, NASNetLarge, MobileNet, EfficientNetV2L
)

@dataclass(frozen=True)
class BackboneSpec:
    name: str
    input_size: Tuple[int, int]
    include_top: bool = False

def simple_cnn(input_shape=(224,224,3), n_classes=10) -> tf.keras.Model:
    m = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=input_shape),
        layers.MaxPool2D(),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return m

def alexnet_like(input_shape=(224,224,3), n_classes=10) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2
    m = models.Sequential([
        layers.Conv2D(96, 11, padding='same', kernel_regularizer=l2(0.0005), input_shape=input_shape),
        layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPool2D(),
        layers.Conv2D(256, 5, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPool2D(),
        layers.Conv2D(512, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPool2D(),
        layers.Conv2D(1024, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(1024, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'), layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(1024), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.5),
        layers.Dense(4096), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    return m

def keras_application(spec: BackboneSpec, n_classes: int, weights: Optional[str]='imagenet') -> tf.keras.Model:
    w = weights
    input_shape = (*spec.input_size, 3)
    if spec.name == 'VGG19':
        base = VGG19(weights=w, include_top=False, input_shape=input_shape)
    elif spec.name == 'ResNet50':
        base = ResNet50(weights=w, include_top=False, input_shape=input_shape)
    elif spec.name == 'InceptionV3':
        base = InceptionV3(weights=w, include_top=False, input_shape=input_shape)
    elif spec.name == 'InceptionResNetV2':
        base = InceptionResNetV2(weights=w, include_top=False, input_shape=input_shape)
    elif spec.name == 'NASNetLarge':
        base = NASNetLarge(weights=w, include_top=False, input_shape=input_shape)
    elif spec.name == 'MobileNet':
        base = MobileNet(weights=w, include_top=False, input_shape=input_shape)
    elif spec.name == 'EfficientNetV2L':
        base = EfficientNetV2L(weights=w, include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone {spec.name}")

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)

def build_model(model_name: str, n_classes: int) -> tuple[tf.keras.Model, int, int]:
    name = model_name
    if name == 'SimpleCNN':
        m = simple_cnn((224,224,3), n_classes); return m, 224, 224
    if name == 'AlexNet':
        m = alexnet_like((224,224,3), n_classes); return m, 224, 224
    if name == 'VGG19':
        return keras_application(BackboneSpec('VGG19', (256,256)), n_classes), 256, 256
    if name == 'ResNet50':
        return keras_application(BackboneSpec('ResNet50', (224,224)), n_classes), 224, 224
    if name == 'InceptionResNetV2':
        return keras_application(BackboneSpec('InceptionResNetV2', (299,299)), n_classes), 299, 299
    if name == 'InceptionV3':
        return keras_application(BackboneSpec('InceptionV3', (299,299)), n_classes), 299, 299
    if name == 'NASNetLarge':
        return keras_application(BackboneSpec('NASNetLarge', (331,331)), n_classes), 331, 331
    if name == 'MobileNet':
        return keras_application(BackboneSpec('MobileNet', (224,224)), n_classes), 224, 224
    if name == 'EfficientNetV2L':
        return keras_application(BackboneSpec('EfficientNetV2L', (600,600)), n_classes), 600, 600
    raise ValueError(f"Unsupported model_name: {model_name}")
