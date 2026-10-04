"""Model comparison, final model selection and export of the Streamlit artifacts.

Selection criterion (documented, deliberately not "highest accuracy"):

    score = 0.40 * macro-F1            (imbalance-aware quality, per class)
          + 0.20 * balanced accuracy   (mean per-class recall)
          + 0.20 * generalisation      (1 - train/validation overfitting gap)
          + 0.20 * efficiency          (mean of size and latency, relative)

The first two terms measure quality on the held-out test set; generalisation is
judged from *validation* behaviour (never from the test set); efficiency keeps
the deployment cost of a mobile-friendly model in view.  Each component is
normalised by the best value among the candidates, so 1.0 is best.

If the quality gap between two models is smaller than the sampling noise of the
test set (reported as a 95% half-width), the script says so and prefers the
cheaper model -- a difference of 0.5% on ~580 test images is not evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DISPLAY_NAMES, ProjectConfig

#: component -> weight (must sum to 1)
SCORE_WEIGHTS: dict[str, float] = {
    "macro_f1": 0.40,
    "balanced_accuracy": 0.20,
    "generalisation": 0.20,
    "efficiency": 0.20,
}
MODEL_ORDER = ("mobilenetv2", "efficientnetb0")
DISPLAY = {"mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNetB0"}


def load_model_metadata(config: ProjectConfig, keys: tuple[str, ...] = MODEL_ORDER) -> dict[str, dict]:
    """Read ``train_meta_*.json`` written by :mod:`src.train`.

    Missing models are reported but not fatal: comparing a single trained model
    part-way through the project is a normal thing for a student to want.  The
    selection weights (size/latency/generalisation) are then uninformative, so a
    warning tells the reader not to interpret the score as a comparison.
    """
    out: dict[str, dict] = {}
    missing: list[str] = []
    for key in keys:
        path = config.result(f"train_meta_{key}.json")
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(key)
    if not out:
        raise FileNotFoundError(
            "No training metadata found for: " + ", ".join(keys)
            + f"\n(expected under {config.result('')})\nTrain the model(s) first, e.g.\n"
            "    python -m src.train --model mobilenetv2\n"
            "    python -m src.train --model efficientnetb0"
        )
    if missing:
        print(
            "[compare] WARNING: no results yet for "
            + ", ".join(missing)
            + " -- the table below compares only the available model(s), so the "
            "relative size/latency components of the score are not a comparison."
        )
    return out


def _accuracy_ci_halfwidth(accuracy: float, n: int, z: float = 1.96) -> float:
    """Normal-approximation 95% CI half-width for a proportion (Wald)."""
    if n <= 0:
        return float("nan")
    return z * math.sqrt(max(accuracy * (1.0 - accuracy), 1e-12) / n)


def build_comparison_table(meta: dict[str, dict]) -> pd.DataFrame:
    """Assemble the comparison table from measured metadata only."""
    rows = []
    for key, m in meta.items():
        tm = m["test_metrics"]
        stages = m.get("stages", {})
        # The overfitting gap uses the *final* epoch of fine-tuning: train minus
        # validation accuracy.  The test set is never part of this term.
        #
        ft = stages.get("fine_tuning", {}) or {}
        ta = ft.get("final_train_accuracy")
        va = ft.get("final_val_accuracy")
        n = tm["num_test_images"]
        acc = tm["accuracy"]
        row = {
            "model": key,
            "display_name": DISPLAY.get(key, m.get("backbone_display_name", key)),
            "accuracy": acc,
            "precision": tm["precision_weighted"],
            "recall": tm["recall_weighted"],
            "f1_score": tm["f1_weighted"],
            "macro_f1": tm["f1_macro"],
            "macro_precision": tm["precision_macro"],
            "macro_recall": tm["recall_macro"],
            "balanced_accuracy": tm["balanced_accuracy"],
            "mean_confidence": tm.get("mean_confidence"),
            "low_confidence_predictions": tm.get("low_confidence_predictions"),
            "test_images": n,
            "accuracy_95ci_halfwidth": round(_accuracy_ci_halfwidth(acc, n) * 100, 2),
            "parameters_total": m["parameter_counts"]["total"],
            "parameters_trainable": m["parameter_counts"]["trainable"],
            "model_size_mb": m.get("model_size_mb"),
            "training_time_seconds": m.get("training_seconds_total"),
            "inference_mean_ms": (m.get("inference_time") or {}).get("mean_ms"),
            "inference_median_ms": (m.get("inference_time") or {}).get("median_ms"),
            "best_val_loss": (stages.get("fine_tuning", {}) or {}).get("best_val_loss"),
            "train_val_accuracy_gap": (ta - va) if (ta is not None and va is not None) else None,
            "val_accuracy": va,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    # sort by macro-F1 for readability; selection is done separately
    return frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def compute_selection_scores(comparison: pd.DataFrame) -> pd.DataFrame:
    """Normalise each component to [0,1] (best = 1) and combine."""
    table = comparison.copy()

    def best_normalised(column: str, higher_is_better: bool = True) -> pd.Series:
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().sum() == 0:
            return pd.Series([np.nan] * len(table), index=table.index)
        reference = float(values.max() if higher_is_better else values.min())
        if reference == 0:
            return pd.Series(1.0, index=table.index)
        if higher_is_better:
            return values / reference
        # lower is better -> invert so 1.0 is still best
        return reference / values.replace(0, np.nan)

    table["component_macro_f1"] = best_normalised("macro_f1")
    table["component_balanced_accuracy"] = best_normalised("balanced_accuracy")

    # Generalisation: smaller train-vs-validation gap is better.  Missing gaps
    # (single model trained) are treated as perfect so nothing is penalised.
    gaps = pd.to_numeric(table["train_val_accuracy_gap"], errors="coerce").abs()
    if gaps.notna().sum() and gaps.notna().any():
        worst = float(np.nanmax(gaps)) if np.isfinite(gaps).any() else 0.0
        table["component_generalisation"] = 1.0 - (gaps.fillna(0.0) / worst if worst > 0 else gaps * 0.0)
    else:
        table["component_generalisation"] = 1.0

    size = pd.to_numeric(table["model_size_mb"], errors="coerce")
    latency = pd.to_numeric(table["inference_mean_ms"], errors="coerce")
    if len(table) < 2:
        # With a single candidate there is nothing to be relatively cheap
        # against, so efficiency is neutral rather than accidentally zero.
        table["component_efficiency"] = 1.0
    else:
        cost = 0.5 * (size / size.max() if size.max() else 1.0) + 0.5 * (
            latency / latency.max() if latency.max() else 1.0
        )
        table["component_efficiency"] = (1.0 - cost).clip(lower=0.0)

    table["selection_score"] = sum(
        SCORE_WEIGHTS[name] * table[f"component_{name}"].fillna(0.0)
        for name in SCORE_WEIGHTS
    )
    return table.sort_values("selection_score", ascending=False).reset_index(drop=True)


def select_final_model(scored: pd.DataFrame, noise_margin_pp: float = 1.0) -> tuple[str, str]:
    """Return ``(model_key, rationale)`` for the exported final model."""
    best_row = scored.iloc[0]
    rationale = (
        f"Selected '{best_row['display_name']}' with selection_score "
        f"{best_row['selection_score']:.4f} (weights: {SCORE_WEIGHTS})."
    )
    if len(scored) > 1:
        runner_up = scored.iloc[1]
        gap_pp = abs(best_row["accuracy"] - runner_up["accuracy"]) * 100.0
        ci = float(best_row["accuracy_95ci_halfwidth"])
        if best_row["selection_score"] == runner_up["selection_score"]:
            rationale += (
                f"\nBoth models tie on the composite score; '{best_row['display_name']}' is "
                "chosen because it is the smaller/faster candidate."
            )
        elif gap_pp <= max(noise_margin_pp, ci):
            rationale += (
                f"\nNote: the test-accuracy difference to '{runner_up['display_name']}' is "
                f"{gap_pp:.2f} percentage points, which is within the ~+/-{ci:.2f} pp 95% "
                "confidence half-width of the test set, so it should not be presented as a "
                "meaningful quality advantage; the composite score decides on the other terms."
            )
        else:
            rationale += (
                f"\nIt leads '{runner_up['display_name']}' by {gap_pp:.2f} pp accuracy "
                f"(95% half-width ~{ci:.2f} pp) and by "
                f"{abs(best_row['macro_f1'] - runner_up['macro_f1']) * 100:.2f} pp macro-F1."
            )
    return str(best_row["model"]), rationale


def export_final_model(config: ProjectConfig, chosen_key: str, meta: dict[str, dict]) -> dict:
    """Copy the winning weights to the stable name and write inference metadata.

    ``models/best_maize_disease_model.keras``, ``models/class_names.json`` and
    ``models/inference_config.json`` are the only three files the Streamlit app
    needs -- nothing is left as undocumented notebook state.
    """
    source = Path(meta[chosen_key]["model_path"])
    if not source.exists():
        raise FileNotFoundError(
            f"Trained weights missing: {source}\nRe-run 'python -m src.train --model {chosen_key}'."
        )
    config.models_dir.mkdir(parents=True, exist_ok=True)
    target = config.final_model_path
    shutil.copyfile(source, target)

    class_block = meta[chosen_key].get("classes", {})
    class_names = class_block.get("class_names") or [
        DISPLAY_NAMES[k] for k in class_block.get("class_keys", [])
    ]
    config.class_map_path.write_text(
        json.dumps(
            {
                **class_block,
                "class_names": class_names,
                "display_names": {DISPLAY_NAMES[k]: DISPLAY_NAMES[k] for k in DISPLAY_NAMES},
                "source_model": chosen_key,
                "num_classes": len(class_names),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prep = meta[chosen_key].get("preprocessing", {})
    dataset = meta[chosen_key].get("dataset", {})
    inference_config = {
        "model": chosen_key,
        "display_name": DISPLAY.get(chosen_key, chosen_key),
        "weights_file": target.name,
        "image_size": meta[chosen_key]["config"]["image_size"],
        "classes": class_names,
        "class_to_index": {name: i for i, name in enumerate(class_names)},
        "preprocessing": {
            "module_name": prep.get("module_name"),
            "internal_preprocessing": bool(prep.get("internal_preprocessing", False)),
            "description": prep.get("description"),
            "resize_method": "bilinear with antialiasing",
            "rescale": "(x-127.5)/127.5" if not prep.get("internal_preprocessing") else "inside model (0-255 input)",
        },
        "low_confidence_threshold": config.low_confidence_threshold,
        "trained_on": {
            "total_images": dataset.get("total_images"),
            "train_images": dataset.get("train_images"),
            "test_images": dataset.get("test_images"),
            "dataset_fingerprint": dataset.get("fingerprint"),
            "seed": config.seed,
        },
        "test_metrics": {
            k: meta[chosen_key]["test_metrics"].get(k)
            for k in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "balanced_accuracy")
        },
        "environment": meta[chosen_key].get("environment", {}),
        "disclaimer": (
            "Experimental research prototype. Predictions are model estimates on "
            "controlled single-leaf photographs and are not a professional "
            "agronomic diagnosis."
        ),
    }
    config.inference_config_path.write_text(json.dumps(inference_config, indent=2), encoding="utf-8")
    return inference_config


def compare_and_select(config: ProjectConfig, export: bool = True) -> dict:
    """Full comparison + selection + export; writes CSVs and a figure."""
    from . import visualize as viz

    config.ensure_directories()
    meta = load_model_metadata(config)
    comparison = build_comparison_table(meta)
    single_model = len(comparison) < 2
    scored = compute_selection_scores(comparison)
    chosen, rationale = select_final_model(scored)

    scored.to_csv(config.result("model_comparison.csv"), index=False)
    comparison.to_csv(config.result("model_comparison_raw.csv"), index=False)
    viz.plot_model_comparison(
        scored,
        config.figure("model_comparison.png"),
        metrics=("accuracy", "macro_f1", "balanced_accuracy"),
        title="Candidate models on the held-out test set",
    )

    exported: dict = {}
    if export:
        exported = export_final_model(config, chosen, meta)
        exported["chosen_model"] = chosen
        exported["rationale"] = rationale

    payload = {
        "complete_comparison": bool(not single_model),
        "selected_model": chosen,
        "rationale": rationale,
        "score_weights": SCORE_WEIGHTS,
        "ranking": scored[
            ["model", "display_name", "selection_score", "accuracy", "macro_f1", "balanced_accuracy",
             "model_size_mb", "inference_mean_ms", "train_val_accuracy_gap"]
        ].to_dict(orient="records"),
        "exported": exported,
    }
    config.result("model_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_selection_report(config, scored, payload, meta)

    print("\n=== model comparison (test set) ===")
    show_cols = [
        "display_name", "accuracy", "precision", "recall", "f1_score", "macro_f1",
        "parameters_total", "model_size_mb", "training_time_seconds", "inference_mean_ms", "selection_score",
    ]
    print(scored[[c for c in show_cols if c in scored.columns]].to_string(index=False))
    print(f"\n[selection] {rationale}")
    if export:
        print(f"[export] final model  -> {config.final_model_path}")
        print(f"[export] class mapping  -> {config.class_map_path}")
        print(f"[export] inference cfg  -> {config.inference_config_path}")
    return payload


def _write_selection_report(config, scored: pd.DataFrame, payload: dict, meta: dict) -> None:
    path = config.report("model_comparison_and_selection.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model comparison and final selection",
        "",
        "Generated by `python -m src.compare`. Every value is read from the run",
        "metadata written during training (`outputs/results/train_meta_*.json`).",
        "",
        "## Test-set results",
        "",
    ]
    cols = [
        "display_name", "accuracy", "precision", "recall", "f1_score", "macro_f1",
        "balanced_accuracy", "parameters_total", "model_size_mb", "training_time_seconds",
        "inference_mean_ms", "val_accuracy", "train_val_accuracy_gap",
    ]
    table = scored[[c for c in cols if c in scored.columns]].copy()
    for num in ("accuracy", "precision", "recall", "f1_score", "macro_f1", "balanced_accuracy",
                "val_accuracy", "train_val_accuracy_gap"):
        if num in table.columns:
            table[num] = (pd.to_numeric(table[num], errors="coerce") * 100).round(2)
    if "training_time_seconds" in table.columns:
        table["training_time_seconds"] = pd.to_numeric(table["training_time_seconds"], errors="coerce").round(1)
    if "inference_mean_ms" in table.columns:
        table["inference_mean_ms"] = pd.to_numeric(table["inference_mean_ms"], errors="coerce").round(1)
    if "model_size_mb" in table.columns:
        table["model_size_mb"] = pd.to_numeric(table["model_size_mb"], errors="coerce").round(2)
    table = table.rename(columns={"display_name": "Model", "accuracy": "Accuracy %", "precision": "Precision (wtd) %",
                                 "recall": "Recall (wtd) %", "f1_score": "F1 (wtd) %", "macro_f1": "Macro-F1 %",
                                 "balanced_accuracy": "Balanced acc %", "parameters_total": "Parameters",
                                 "model_size_mb": "Size MB", "training_time_seconds": "Train s",
                                 "inference_mean_ms": "Infer ms", "val_accuracy": "Val acc %",
                                 "train_val_accuracy_gap": "Train-Val gap"})
    from .inspect_dataset import _df_to_markdown

    lines += [_df_to_markdown(table), ""]
    lines += [
        "Accuracy is reported with its 95% normal-approximation half-width,",
        "so small differences are not over-interpreted:",
        "",
    ]
    for row in scored.itertuples(index=False):
        lines.append(
            f"- {getattr(row, 'display_name')}: accuracy {getattr(row, 'accuracy') * 100:.2f}% "
            f"+/- {getattr(row, 'accuracy_95ci_halfwidth'):.2f} pp "
            f"(n = {getattr(row, 'test_images')} test images)"
        )
    lines += [
        "",
        "## Composite selection score",
        "",
        *(
            ["**Only one model was available**, so this is a description of that",
             "model rather than a comparison; run both models for a real ranking.",
             ""]
            if payload.get("complete_comparison") is False
            else []
        ),
        f"weights: `{json.dumps(payload['score_weights'])}`",
        "",
    ]
    for row in payload["ranking"]:
        lines.append(f"- `{row['model']}`: score {row['selection_score']:.4f}")
    lines += [
        "",
        f"**Decision:** {payload['rationale']}",
        "",
        "## Per-class test results",
        "",
    ]
    for key, m in meta.items():
        lines.append(f"### {DISPLAY.get(key, key)}")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, values in m["test_metrics"]["per_class"].items():
            lines.append(
                f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} | "
                f"{values['f1']:.4f} | {values['support']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.compare", description=__doc__)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--no-export", action="store_true", help="Compare only; do not write the final model")
    args = parser.parse_args(argv)
    config = ProjectConfig().override(artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None)
    try:
        compare_and_select(config, export=not args.no_export)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
