"""Model construction, the two-stage training strategy and callbacks.

Both models follow the same required skeleton::

    Input(224,224,3)
      -> pretrained ImageNet base (MobileNetV2 | EfficientNetB0), top layer removed
      -> GlobalAveragePooling2D
      -> Dropout
      -> Dense(128, relu, L2)
      -> Dense(4, softmax)

Training is two-stage (classic transfer learning):

* **Stage 1 - feature extraction.**  The whole pretrained base is frozen; only
  the new head is trained, at a relatively high learning rate.
* **Stage 2 - fine-tuning.**  Only the top ``finetune_unfreeze_fraction`` of the
  base is unfrozen and trained with a ~10x smaller learning rate, which limits
  catastrophic forgetting of the ImageNet features.

Which layers become trainable is not left as a mystery: :func:`freeze_for_stage`
returns a record that the training script writes to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
BaseFactory = Callable[..., tf.keras.Model]


@dataclass(frozen=True)
class BackboneSpec:
    """Everything that differs between the two candidate architectures."""

    key: str
    display_name: str
    #: attribute name inside ``tensorflow.keras.applications``
    module_name: str
    #: first image size accepted by the architecture
    min_input_size: int = 32
    #: EfficientNet uses BN/RMS-friendly weight decay only in the head; keep 0.0
    recommended_head_lr: float = 1e-3
    recommended_finetune_lr: float = 1e-4

    def build_base(self, weights: str = "imagenet", **kwargs: Any) -> tf.keras.Model:
        module = getattr(tf.keras.applications, self.module_name)
        cls = getattr(module, self.display_name.replace("-", ""))
        return cls(weights=weights, include_top=False, **kwargs)


BACKBONES: dict[str, BackboneSpec] = {
    "mobilenetv2": BackboneSpec(
        key="mobilenetv2",
        display_name="MobileNetV2",
        module_name="mobilenet_v2",
    ),
    "efficientnetb0": BackboneSpec(
        key="efficientnetb0",
        display_name="EfficientNetB0",
        module_name="efficientnet",
    ),
}


def get_backbone(key: str) -> BackboneSpec:
    try:
        return BACKBONES[key.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{key}'. Available: {', '.join(sorted(BACKBONES))}"
        ) from exc


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_model(
    key: str,
    num_classes: int,
    config,
    weights: str = "imagenet",
) -> tuple[tf.keras.Model, Any]:
    """Build the full classifier and its :class:`~src.preprocessing.PreprocessingPlan`.

    The base is instantiated first so that the preprocessing strategy can be
    *detected from the built graph* (see :mod:`src.preprocessing`) instead of
    being assumed per architecture.
    """
    from .preprocessing import build_plan

    spec = get_backbone(key)
    size = int(config.image_size)
    if size < spec.min_input_size:
        raise ValueError(
            f"{spec.display_name} needs images of at least {spec.min_input_size}px; "
            f"config.image_size is {size}."
        )

    base = spec.build_base(
        weights=weights,
        input_shape=(size, size, 3),
    )
    plan = build_plan(spec.key, spec.module_name, base)

    inputs = tf.keras.Input(shape=(size, size, 3), name="maize_leaf")
    # ``training=False`` is passed on purpose and kept for BOTH stages: it stops
    # BatchNormalization inside the pretrained base from updating its moving
    # statistics while fine-tuning (Keras updates those based on the training
    # flag, not on layer trainability).  Frozen BN + small learning rate is the
    # standard recipe for fine-tuning a shallow head onto deep features.
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="head_dropout")(x)
    x = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_weight),
        name="head_dense",
    )(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"maize_{spec.key}")
    # Attached for the training loop; *not* stored in the saved file, which is
    # why the same information is also written to models/inference_config.json.
    model.pretrained_base = base
    model.preprocessing_plan = plan
    return model, plan


def compile_model(model: tf.keras.Model, config, learning_rate: float) -> tf.keras.Model:
    """Compile with Adam + sparse categorical cross-entropy + monitoring metrics.

    ``SparseCategoricalCrossentropy`` is used because the pipeline yields
    integer labels rather than one-hot vectors -- that keeps the ``tf.data``
    pipeline small and the code obvious.

    Only accuracy-style metrics are attached.  ``Precision``/``Recall`` are
    deliberately *not* used here: with sparse integer labels Keras 3 raises a
    shape error inside its confusion-matrix update.  Precision and recall are
    instead computed properly (per class, macro and weighted) with
    scikit-learn in :mod:`src.evaluate`, which is what the report needs anyway.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )
    return model


# --------------------------------------------------------------------------
# Two-stage freezing
# --------------------------------------------------------------------------
def freeze_for_stage(model: tf.keras.Model, config, stage: str) -> dict[str, Any]:
    """Configure trainability of the pretrained base for *stage*.

    ``"feature_extraction"`` -> base entirely frozen.
    ``"fine_tuning"``        -> top ``config.finetune_unfreeze_fraction`` of the
    base unfrozen; BatchNormalization layers are kept non-trainable, which is
    the standard Keras recommendation when fine-tuning with a small batch.

    Returns a record describing the decision (written to JSON by ``train.py``).
    """
    base = getattr(model, "pretrained_base", None)
    if base is None:
        raise AttributeError(
            "This model has no '.pretrained_base' attribute; build it with "
            "src.models.build_model() so the two-stage strategy can freeze layers."
        )
    if stage not in {"feature_extraction", "fine_tuning"}:
        raise ValueError(f"Unknown stage '{stage}'.")

    base.trainable = True
    for layer in base.layers:
        layer.trainable = True  # reset, then apply the stage policy

    if stage == "feature_extraction":
        base.trainable = False
        for layer in base.layers:
            layer.trainable = False
        unfrozen_names: list[str] = []
    else:
        n_layers = len(base.layers)
        keep_frozen = int(round(n_layers * (1.0 - float(config.finetune_unfreeze_fraction))))
        unfrozen_names = []
        for i, layer in enumerate(base.layers):
            if i < keep_frozen:
                layer.trainable = False
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                # Frozen BN keeps ImageNet statistics -> stable fine-tuning.
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen_names.append(layer.name)

    record = {
        "stage": stage,
        "backbone_layers": len(base.layers),
        "base_layers_frozen": len(base.layers) - len(unfrozen_names),
        "base_layers_trainable": len(unfrozen_names),
        "first_trainable_base_layer": unfrozen_names[0] if unfrozen_names else None,
        "last_trainable_base_layer": unfrozen_names[-1] if unfrozen_names else None,
        "batch_normalization_kept_frozen": stage == "fine_tuning",
        "trainable_layer_names_in_base": unfrozen_names,
    }
    total, trainable = count_parameters(model)
    record.update({"parameters": total, "trainable_parameters": trainable})
    return record


def count_parameters(model: tf.keras.Model) -> tuple[int, int]:
    """Return (total, trainable) parameter counts, summed from the weights."""
    total = int(sum(int(np_prod(w.shape)) for w in model.weights))
    trainable = int(sum(int(np_prod(w.shape)) for w in model.trainable_weights))
    return total, trainable


def np_prod(shape) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def model_summary_text(model: tf.keras.Model, max_width: int = 120) -> str:
    """Capture ``model.summary()`` as text for the reports folder."""
    import io

    buf = io.StringIO()
    try:
        model.summary(print_fn=buf.write, expand_nested=False)
    except TypeError:  # older Keras signature
        model.summary(print_fn=buf.write)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
def build_callbacks(
    config,
    checkpoint_dir: Path,
    run_tag: str,
    monitor: str = "val_loss",
    mode: str = "min",
) -> tuple[list[tf.keras.callbacks.Callback], Path]:
    """EarlyStopping + ModelCheckpoint + ReduceLROnPlateau.

    All three watch **validation** metrics.  The test set is never consulted
    for model selection -- that is the point of holding it out.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{run_tag}_best.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=int(config.early_stopping_patience),
            min_delta=float(config.early_stopping_min_delta),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=float(config.lr_reduce_factor),
            patience=int(config.lr_reduce_patience),
            min_lr=float(config.lr_reduce_min_lr),
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks, best_path


def write_history(history: tf.keras.callbacks.History, path: Path) -> Path:
    """Dump a Keras ``History`` to CSV so curves can be re-plotted later."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({k: list(v) for k, v in history.history.items()})
    frame.insert(0, "epoch", range(1, len(frame) + 1))
    frame.to_csv(path, index=False)
    return path
