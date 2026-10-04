"""Image loading, model-specific preprocessing and augmentation.

Two facts drive the design of this module, both of them checked against the
installed Keras rather than copied from a blog post:

* ``tensorflow.keras.applications.mobilenet_v2.preprocess_input`` **does**
  rescale (``(x - 127.5) / 127.5``) and the ``MobileNetV2`` body contains **no**
  preprocessing layers, so the caller must apply it.
* ``tensorflow.keras.applications.efficientnet.preprocess_input`` is a
  **no-op placeholder** in Keras 3 because ``EfficientNetB0`` already contains
  ``Rescaling -> Normalization -> Rescaling`` internally.  Applying it changes
  nothing; assuming it normalises would silently corrupt training.

Rather than hard-coding either behaviour, :func:`model_expects_internal_preprocessing`
inspects the built network and the pipeline adapts.  The decision is written to
the run metadata so the thesis can state exactly what was fed to each model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Preprocessing dispatch
# --------------------------------------------------------------------------
PreprocessFn = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class PreprocessingPlan:
    """How images must be prepared for one specific architecture."""

    model_name: str
    #: attribute name under ``tensorflow.keras.applications``
    module_name: str
    #: True when the pretrained body normalises internally -> feed [0, 255].
    internal_preprocessing: bool

    @property
    def description(self) -> str:
        if self.internal_preprocessing:
            return (
                "images resized to 224x224, float32 in [0,255]; normalisation is "
                "performed by preprocessing layers inside the model"
            )
        return (
            "images resized to 224x224, float32 in [0,255], then "
            f"keras.applications.{self.module_name}.preprocess_input "
            "((x-127.5)/127.5) applied in the input pipeline"
        )

    @property
    def keras_preprocess_input(self) -> PreprocessFn:
        module = getattr(tf.keras.applications, self.module_name)
        return module.preprocess_input


def model_expects_internal_preprocessing(model: tf.keras.Model, _depth: int = 0) -> bool:
    """Detect whether *model* normalises its own input.

    Keras application models that bake preprocessing in start with
    ``Rescaling``/``Normalization`` layers before the first convolution.  This
    keeps the pipeline correct across TF/Keras major versions, where the
    location of that logic has moved between the model and ``preprocess_input``.

    A composed model (``Input -> base Functional -> head``) hides those layers
    inside the nested ``Functional``, so this descends into it.  Without that,
    detection on a *saved, reloaded* classifier would report "no internal
    preprocessing" for EfficientNet and could double-normalise its input.
    """
    layers_list = list(model.layers)
    start = 1 if layers_list and layers_list[0].__class__.__name__ == "InputLayer" else 0
    if start < len(layers_list) and isinstance(layers_list[start], tf.keras.Model):
        if _depth < 5:
            return model_expects_internal_preprocessing(layers_list[start], _depth + 1)
    head = [layer.__class__.__name__ for layer in layers_list[start : start + 6]]
    markers = {"Rescaling", "Normalization", "PreprocessingLayer", "_PreprocessingLayer"}
    return any(name in markers for name in head)


def build_plan(model_name: str, module_name: str, base_model: tf.keras.Model) -> PreprocessingPlan:
    """Create the :class:`PreprocessingPlan` for a freshly built base model."""
    return PreprocessingPlan(
        model_name=model_name,
        module_name=module_name,
        internal_preprocessing=model_expects_internal_preprocessing(base_model),
    )


def apply_preprocessing(images: tf.Tensor, plan: PreprocessingPlan) -> tf.Tensor:
    """Finish preprocessing a batch of ``[0, 255]`` float images for *plan*."""
    if plan.internal_preprocessing:
        # The model normalises internally; passing the Keras 3 no-op through
        # would be harmless but explicit is clearer for a reader.
        return images
    fn = plan.keras_preprocess_input
    return tf.ensure_shape(tf.convert_to_tensor(fn(images)), images.shape)


# --------------------------------------------------------------------------
# Decoding / resizing
# --------------------------------------------------------------------------
def decode_and_resize(path: tf.Tensor, image_size: int) -> tf.Tensor:
    """Read one file, decode it and return a ``image_size x image_size`` float32
    tensor in ``[0, 255]`` with three channels.

    ``grayscale=True`` is deliberately *not* used: both backbones are
    ImageNet-pretrained on RGB and lesion colour carries signal (rust pustules
    versus blight lesions).
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, dtype=tf.uint8, expand_animations=False)
    image = tf.image.resize(
        tf.cast(image, tf.float32),
        (image_size, image_size),
        method="bilinear",
        antialias=True,
    )
    return tf.clip_by_value(image, 0.0, 255.0)


def _decode_and_resize_one(path: tf.Tensor, label: tf.Tensor, image_size: int):
    """Map helper that also reports a readable error if a file is unreadable."""
    try:
        image = decode_and_resize(path, image_size)
    except tf.errors.InvalidArgumentError as exc:  # pragma: no cover
        raise tf.errors.InvalidArgumentError(
            node_def=None,
            op=None,
            message=(
                f"Could not decode image {path}. The dataset inspection step "
                f"should have removed unreadable files. Original error: {exc}"
            ),
        )
    return image, label


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
def build_augmentation(image_size: int) -> tf.keras.Sequential:
    """Training-only geometric + photometric augmentation.

    Choices and their justification (also in the README):

    * horizontal flip -- leaf orientation in a photo is arbitrary.
      *Vertical* flipping is omitted: dorsiventral leaf structure and lesion
      gradients are not vertically symmetric.
    * +/-3.6 deg rotation, 10% zoom, 10% translation -- mild framing changes
      that mimic a hand-held camera.
    * brightness/contrast +/-10% -- sun/cloud variation.
    * hue and saturation shifts are **omitted on purpose**: rust pustules and
      blight lesions are partly identified by colour, so changing colour
      statistics could destroy the class signal.
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(image_size, image_size, 3)),
            layers.RandomFlip(mode="horizontal"),
            layers.RandomRotation(factor=0.10, fill_mode="nearest"),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10, fill_mode="nearest"),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10, fill_mode="nearest"),
            layers.RandomBrightness(factor=0.10, value_range=(0.0, 255.0)),
            layers.RandomContrast(factor=0.10),
        ],
        name="train_time_augmentation",
    )


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------
def build_dataset(
    frame: pd.DataFrame,
    plan: PreprocessingPlan,
    config,
    *,
    augment: bool,
    shuffle: bool | None = None,
    repeat: bool = False,
    augment_layer: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    """Build a ``tf.data`` pipeline for one split.

    Augmentation is attached **only** when ``augment=True`` (the training
    split); validation and test sets are decoded and resized but never
    perturbed, so they measure the model rather than the augmentation.

    No ``.cache()`` is used: with ~2700 training images at 224x224x3 float32
    the pixel data alone would be ~1.6 GB, which is too much for a Colab CPU
    runtime.  Decoding on the fly plus ``prefetch`` keeps memory bounded.
    """
    paths = tf.constant(frame["path"].astype(str).to_numpy())
    labels = tf.constant(frame["label"].astype(int).to_numpy())

    def _autotune(value):
        return tf.data.AUTOTUNE if value is None else int(value)

    parallel = _autotune(getattr(config, "tf_data_num_parallel_calls", None))
    prefetch = _autotune(getattr(config, "tf_data_prefetch", None))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    image_size = int(config.image_size)
    ds = ds.map(
        lambda p, y: _decode_and_resize_one(p, y, image_size),
        num_parallel_calls=parallel,
    )
    if shuffle is None:
        shuffle = augment
    if shuffle:
        ds = ds.shuffle(int(config.shuffle_buffer), seed=config.seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(int(config.batch_size))

    aug = augment_layer if augment_layer is not None else (build_augmentation(image_size) if augment else None)

    @tf.function
    def _finish(x, y):
        if augment and aug is not None:
            x = aug(x, training=True)
            x = tf.clip_by_value(x, 0.0, 255.0)
        x = apply_preprocessing(x, plan)
        return x, y

    ds = ds.map(_finish, num_parallel_calls=parallel)
    return ds.prefetch(prefetch)


def dataset_size(ds: tf.data.Dataset) -> int:
    """Number of *images* in a batched dataset, counted (not assumed)."""
    total = 0
    for _, y in ds:
        total += int(tf.size(y))
    return total


def iter_predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities, labels)`` for a dataset -- used by evaluation."""
    probs = model.predict(ds, verbose=0)
    labels = np.concatenate([y.numpy() for _, y in ds])
    return np.asarray(probs), np.asarray(labels).astype(int)
