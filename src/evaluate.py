"""Test-set evaluation: classification metrics and figures, from real predictions.

The test set is used **once per model**, after training and model selection are
finished.  Nothing in here influences which epoch's weights are kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sys

from .config import DISPLAY_NAMES, ProjectConfig, seed_everything


def metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Accuracy, per-class and averaged precision/recall/F1, plus the confusion matrix.

    scikit-learn does the arithmetic; ``labels=range(k)`` keeps the matrix
    aligned with the model's output neurons even if a class never appears in
    the test split.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions.")
    k = len(class_names)

    report = {
        "num_test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": None,
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
        "class_names": list(class_names),
    }
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), zero_division=0
    )
    for i, name in enumerate(class_names):
        tn = int(report["confusion_matrix"][i][i])
        row_total = int(sum(report["confusion_matrix"][i]))
        report["per_class"][name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "correct": tn,
            "misclassified": row_total - tn,
            "recall_percent": 100.0 * float(rec[i]),
        }
    # Balanced accuracy = mean recall; computed explicitly so the number in the
    # thesis has a definition, not a surprise.
    recalls = [v["recall"] for v in report["per_class"].values()]
    report["balanced_accuracy"] = float(np.mean(recalls)) if recalls else None
    return report


def classification_report_frame(metrics: dict) -> pd.DataFrame:
    """scikit-learn-style classification report as a DataFrame (real class names)."""
    rows = []
    for name, values in metrics["per_class"].items():
        rows.append(
            {
                "class": name,
                "precision": round(values["precision"], 4),
                "recall": round(values["recall"], 4),
                "f1-score": round(values["f1"], 4),
                "support": values["support"],
                "correct": values["correct"],
                "misclassified": values["misclassified"],
            }
        )
    for label, key in (
        ("macro avg", "macro"),
        ("weighted avg", "weighted"),
    ):
        rows.append(
            {
                "class": label,
                "precision": round(metrics[f"precision_{key}"], 4),
                "recall": round(metrics[f"recall_{key}"], 4),
                "f1-score": round(metrics[f"f1_{key}"], 4),
                "support": metrics["num_test_images"],
                "correct": "",
                "misclassified": "",
            }
        )
    rows.append(
        {
            "class": "accuracy",
            "precision": "",
            "recall": "",
            "f1-score": "",
            "support": metrics["num_test_images"],
            "correct": int(round(metrics["accuracy"] * metrics["num_test_images"])),
            "misclassified": int(metrics["num_test_images"] - round(metrics["accuracy"] * metrics["num_test_images"])),
        }
    )
    return pd.DataFrame(rows)


def evaluate_model(
    model,
    dataset,
    class_names: Sequence[str],
    config=None,
    tag: str = "model",
    write_artifacts: bool = True,
) -> dict:
    """Predict on an (unseen) dataset and write the report/figure artefacts.

    Returns the metrics dict; also attaches numpy arrays under keys starting
    with ``_`` for the caller's convenience (not serialised).
    """
    from . import visualize as viz
    from .preprocessing import iter_predictions

    probs, y_true = iter_predictions(model, dataset)
    y_pred = np.argmax(probs, axis=1)
    metrics = metrics_from_predictions(y_true, y_pred, class_names)
    metrics["model"] = tag
    metrics["mean_confidence"] = float(np.mean(probs.max(axis=1)))
    metrics["low_confidence_predictions"] = int(
        np.sum(probs.max(axis=1) < float(getattr(config, "low_confidence_threshold", 0.60)))
    )

    if write_artifacts and config is not None:
        config.ensure_directories()
        frame = classification_report_frame(metrics)
        frame.to_csv(config.result(f"classification_report_{tag}.csv"), index=False)
        cm = np.array(metrics["confusion_matrix"], dtype=int)
        pd.DataFrame(cm, index=list(class_names), columns=list(class_names)).to_csv(
            config.result(f"confusion_matrix_{tag}.csv")
        )
        viz.plot_confusion_matrix(
            cm,
            list(class_names),
            config.figure(f"confusion_matrix_{tag}.png"),
            title=f"{DISPLAY_NAMES.get(tag, tag.replace('_', ' ').title())} - confusion matrix (test set, n={metrics['num_test_images']})",
            subtitle=f"accuracy {metrics['accuracy'] * 100:.2f}%  |  macro-F1 {metrics['f1_macro'] * 100:.2f}%",
        )
        viz.plot_confusion_matrix(
            cm,
            list(class_names),
            config.figure(f"confusion_matrix_{tag}_normalised.png"),
            title=f"{tag} - row-normalised confusion matrix",
            normalize=True,
        )
        serialisable = {k: v for k, v in metrics.items() if not k.startswith("_")}
        config.result(f"test_metrics_{tag}.json").write_text(
            json.dumps(serialisable, indent=2), encoding="utf-8"
        )
    metrics["_probabilities"] = probs
    metrics["_y_true"] = y_true
    metrics["_y_pred"] = y_pred
    return metrics


def sample_predictions_frame(
    y_true: Sequence[int],
    probs: np.ndarray,
    class_names: Sequence[str],
    paths: Sequence[str] | None = None,
    low_confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    """Per-image predictions, for ``outputs/predictions/sample_predictions.csv``."""
    y_true = np.asarray(y_true, dtype=int)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    frame = pd.DataFrame(
        {
            "image_path": list(paths) if paths is not None else [f"index_{i}" for i in range(len(y_true))],
            "true_label_index": y_true,
            "true_class": [class_names[i] for i in y_true],
            "predicted_label_index": pred,
            "predicted_class": [class_names[i] for i in pred],
            "confidence_percent": np.round(conf * 100.0, 2),
            "correct": pred == y_true,
            "low_confidence_flag": conf < low_confidence_threshold,
        }
    )
    for i, name in enumerate(class_names):
        frame[f"prob_{name}"] = np.round(probs[:, i], 5)
    return frame


def evaluate_saved_model(
    model_key: str,
    config: ProjectConfig,
    model_path: Path | None = None,
    write_artifacts: bool = True,
) -> dict:
    """Re-evaluate a saved model on the test split, without training again.

    Used by ``python -m src.evaluate`` and handy after moving weights between
    machines: the split file is reused, so the test set is exactly the same one
    the model was trained against (verified through the dataset fingerprint).
    """
    from . import data_loader as dl
    from . import visualize as viz
    from .preprocessing import build_dataset

    config.ensure_directories()
    seed_everything(config.seed, config.deterministic_ops)

    root = dl.find_dataset_root(config)
    mapping, _ = dl.build_class_mapping(root)
    splits_path = config.result("splits.csv")
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found, so the original split cannot be reconstructed.\n"
            "Run 'python -m src.inspect_dataset' first (it writes the split file)."
        )
    frames = dl.load_splits(splits_path, root)
    fingerprint = dl.dataset_fingerprint(frames)
    meta_path = config.result(f"train_meta_{model_key}.json")
    if meta_path.exists():
        recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("dataset", {}).get("fingerprint")
        if recorded and recorded != fingerprint:
            print(
                f"[evaluate] WARNING: dataset fingerprint changed ({recorded} -> {fingerprint}); "
                "the test set is no longer the one this model was trained/evaluated on."
            )

    path = Path(model_path) if model_path else config.model_artifact_path(model_key)
    model, plan = load_trained_model(path, config.inference_config_path, config)
    # The exported weights are the *inference* graph: they carry no frozen/trainable
    # distinction, but the preprocessing plan must match what training used.
    test_df = frames[frames["split"] == "test"]
    ds = build_dataset(test_df, plan, config, augment=False, shuffle=False)
    metrics = evaluate_model(
        model, ds, mapping.class_names, config, tag=model_key, write_artifacts=write_artifacts
    )
    frame = classification_report_frame(metrics)
    print(f"\n[evaluate] {model_key} ({path.name}) on {metrics['num_test_images']} test images")
    print(frame.to_string(index=False))
    samples = sample_predictions_frame(
        metrics["_y_true"],
        metrics["_probabilities"],
        mapping.class_names,
        paths=list(test_df["path"].astype(str)),
        low_confidence_threshold=config.low_confidence_threshold,
    )
    if write_artifacts:
        samples.to_csv(config.prediction(f"test_predictions_{model_key}.csv"), index=False)
        viz.plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            list(mapping.class_names),
            config.figure(f"confusion_matrix_{model_key}.png"),
            title=f"{model_key} - confusion matrix (test set, n={metrics['num_test_images']})",
            subtitle=f"accuracy {metrics['accuracy'] * 100:.2f}%  |  macro-F1 {metrics['f1_macro'] * 100:.2f}%",
        )
    metrics["model_path"] = str(path)
    metrics["dataset_fingerprint"] = fingerprint
    return metrics


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate a trained model on the untouched test split (no training).",
    )
    parser.add_argument("--model", required=True, choices=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--weights", default=None, help="Path to a .keras file (default: models/<model>_maize.keras)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--no-write", action="store_true", help="Print metrics without overwriting artefacts")
    args = parser.parse_args(argv)

    config = ProjectConfig().override(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
    )
    try:
        evaluate_saved_model(
            args.model,
            config,
            Path(args.weights) if args.weights else None,
            write_artifacts=not args.no_write,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def load_trained_model(
    model_path: Path,
    inference_config_path: Path | None = None,
    config=None,
):
    """Load a saved ``.keras`` model plus the preprocessing metadata it needs.

    Raises a clear error instead of an ``FileNotFoundError`` traceback when the
    project has not been trained yet.
    """
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        candidates = sorted(p.name for p in model_path.parent.glob("*.keras")) if model_path.parent.exists() else []
        raise FileNotFoundError(
            f"No trained model found at {model_path}.\n"
            "Please train the model first: run the Google Colab notebook, or\n"
            "    python -m src.train --model mobilenetv2\n"
            + (f"Existing files in {model_path.parent}: {candidates}" if candidates else "")
        )
    model = tf.keras.models.load_model(model_path, compile=False)

    from .preprocessing import PreprocessingPlan

    plan = None
    meta_path = Path(inference_config_path) if inference_config_path else model_path.parent / "inference_config.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        key = meta.get("preprocessing", {})
        plan = PreprocessingPlan(
            model_name=meta.get("model", "unknown"),
            module_name=key.get("module_name", "mobilenet_v2"),
            internal_preprocessing=bool(key.get("internal_preprocessing", False)),
        )
    if plan is None:
        print(
            "[evaluate] WARNING: models/inference_config.json not found; falling back to "
            "runtime detection of the preprocessing strategy. Re-run 'python -m src.compare' "
            "to regenerate it."
        )
        from .preprocessing import model_expects_internal_preprocessing

        module_name = "efficientnet" if "efficientnet" in model_path.stem.lower() else "mobilenet_v2"
        plan = PreprocessingPlan(
            model_name=model_path.stem,
            module_name=module_name,
            internal_preprocessing=model_expects_internal_preprocessing(model),
        )
    return model, plan


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
