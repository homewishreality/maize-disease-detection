"""Two-stage transfer-learning training.

Functions are deliberately split so that the notebook can present each step as
its own section while the CLI runs the same steps in sequence:

``prepare_experiment`` -> data (already-split frames)
``build_pipelines``    -> three ``tf.data`` pipelines (augment: train only)
``stage_one``          -> feature extraction with a frozen base
``stage_two``          -> fine-tuning the top of the base
``finalize``           -> save weights, evaluate on the test split, write metadata
``train_one_model``    -> the orchestrator used by ``python -m src.train``

Model selection inside training uses **validation** loss only (checkpoint +
early stopping).  The test set is touched once, in ``finalize``, for reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ProjectConfig,
    configure_gpu_memory_growth,
    environment_report,
    seed_everything,
)
from . import data_loader as dl


def peak_memory_mb() -> float | None:
    """Peak resident set size of this process in MB (Linux/macOS via ``resource``).

    Recorded so the thesis can state the memory the experiment actually needed,
    and so an ``Exit code 137 / Killed`` failure can be diagnosed as memory
    exhaustion instead of guessed at.  ``None`` on Windows (no ``resource``).
    """
    try:
        import resource
        import sys as _sys

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        mb = raw / (1024.0 * 1024.0) if _sys.platform == "darwin" else raw / 1024.0
        return round(mb, 1)
    except Exception:  # noqa: BLE001 - informational only
        return None


def measure_inference_ms(model, paths: list[str], plan, config, n: int = 16) -> dict:
    """Time single-image (batch size 1) prediction on real images."""
    import tensorflow as tf

    from .preprocessing import apply_preprocessing, decode_and_resize

    subset = list(paths[:n])
    if not subset:
        return {"n_images": 0, "mean_ms": None}

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
    def prepare(path):
        """One image -> a batch of size 1, preprocessed for this architecture."""
        x = decode_and_resize(path, int(config.image_size))[tf.newaxis, ...]
        return apply_preprocessing(x, plan)

    first = prepare(tf.constant(subset[0]))
    for _ in range(3):  # warm up graph tracing + allocators before timing
        model.predict(first, verbose=0)

    latencies: list[float] = []
    for path in subset:
        x = prepare(tf.constant(path))
        t0 = time.perf_counter()
        model.predict(x, verbose=0)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(latencies, dtype=float)
    return {
        "n_images": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
    }


def load_split_frames(config: ProjectConfig, root: Path, mapping: dl.ClassMapping):
    """Load (or create) the split assignment so both models share one split.

    The dataset fingerprint is re-checked, so a split file produced from a
    different download -- or a different ``--max-images-per-class`` -- is
    rejected and regenerated instead of being silently used.
    """
    splits_path = config.result("splits.csv")
    frames = None
    if splits_path.exists():
        try:
            frames = dl.load_splits(splits_path, root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[train] ignoring unusable split file ({exc}); re-inspecting the dataset")
            frames = None
        if frames is not None:
            all_images = dl.collect_images(root, mapping, config)
            if dl.dataset_fingerprint(all_images) != dl.dataset_fingerprint(frames):
                print("[train] split file does not match the current dataset; re-inspecting")
                frames = None
    if frames is None:
        from .inspect_dataset import inspect_dataset

        print("[train] running the dataset inspection stage first ...")
        inspect_dataset(config, write=True)
        frames = dl.load_splits(splits_path, root)
    return frames, mapping


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------
def prepare_experiment(model_key: str, config: ProjectConfig, weights: str = "imagenet") -> dict:
    """Seed RNGs, check memory, locate data, build the model and its pipelines.

    Returns a context dict consumed by :func:`stage_one`, :func:`stage_two` and
    :func:`finalize`.
    """
    from . import models as model_lib

    seed_everything(config.seed, config.deterministic_ops)
    gpu = configure_gpu_memory_growth()
    config.ensure_directories()

    print(f"\n{'=' * 78}\nTRAINING {model_key.upper()}   (seed={config.seed})\n{'=' * 78}")
    if not gpu["gpu_detected"]:
        print("[train] no GPU detected -> running on CPU. Expect long epochs; the README explains "
              "how to enable a T4 runtime in Colab.")

    # ---- memory guard ----------------------------------------------------
    # TensorFlow plus a 224x224 input pipeline needs well over a gigabyte.
    # Rather than let the kernel kill the run silently ("Killed", exit 137),
    # shrink the pipeline on small machines and say so out loud.
    from .config import available_memory_mb

    memory = available_memory_mb()
    if memory.get("available_mb") is not None:
        print(f"[train] memory: {memory['available_mb']:.0f} MB available of "
              f"{memory.get('total_mb', '?')} MB total ({memory.get('source')})")
        if memory["available_mb"] < 2500:
            shrink = []
            if config.tf_data_num_parallel_calls is None:
                config.tf_data_num_parallel_calls = 1
                shrink.append("tf.data parallel calls -> 1")
            if config.tf_data_prefetch is None:
                config.tf_data_prefetch = 1
                shrink.append("prefetch -> 1 batch")
            if config.batch_size > 16:
                config.batch_size = 16
                shrink.append("batch size -> 16")
            if shrink:
                print("[train] low-memory guard active: " + ", ".join(shrink)
                      + ". If the process is still killed, use --low-memory "
                        "--max-images-per-class 400, or run in Google Colab.")

    # ---- data ------------------------------------------------------------
    root = dl.find_dataset_root(config)
    mapping, _ = dl.build_class_mapping(root)
    frames, mapping = load_split_frames(config, root, mapping)
    train_df = frames[frames["split"] == "train"].reset_index(drop=True)
    val_df = frames[frames["split"] == "validation"].reset_index(drop=True)
    test_df = frames[frames["split"] == "test"].reset_index(drop=True)
    print(f"[data] train={len(train_df)}  validation={len(val_df)}  test={len(test_df)}  "
          f"(total {len(frames)}, fingerprint {dl.dataset_fingerprint(frames)})")

    # ---- model -----------------------------------------------------------
    model, plan = model_lib.build_model(model_key, mapping.num_classes, config, weights=weights)
    print(f"[model] {plan.model_name} | input {config.image_size}x{config.image_size} | "
          f"{mapping.num_classes} classes")
    print(f"[preprocessing] {plan.description}")
    config.result(f"preprocessing_{model_key}.json").write_text(
        json.dumps({"model": model_key, "plan": plan.__dict__}, indent=2), encoding="utf-8"
    )

    return {
        "model_key": model_key,
        "config": config,
        "model": model,
        "plan": plan,
        "mapping": mapping,
        "root": root,
        "frames": frames,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "memory": memory,
        "weights_requested": weights,
        "started": time.perf_counter(),
    }


def build_pipelines(ctx: dict) -> dict[str, object]:
    """Create the three ``tf.data`` pipelines: augmentation on the train split only."""
    from .preprocessing import build_dataset

    config, plan = ctx["config"], ctx["plan"]
    return {
        "train": build_dataset(ctx["train_df"], plan, config, augment=True),
        "validation": build_dataset(ctx["val_df"], plan, config, augment=False, shuffle=False),
        "test": build_dataset(ctx["test_df"], plan, config, augment=False, shuffle=False),
    }


def _fit_stage(model, ctx, ds_train, ds_val, learning_rate: float, epochs: int, run_tag: str,
               stage: str, quiet: bool) -> tuple[dict, dict]:
    """Shared implementation of the two training stages."""
    from . import models as model_lib

    config = ctx["config"]
    record = model_lib.freeze_for_stage(model, config, stage)
    model_lib.compile_model(model, config, learning_rate)

    class_weight = None
    if config.class_weights:
        weights_map = dl.compute_class_weights(ctx["train_df"])
        class_weight = {int(k): float(v) for k, v in weights_map.items()}
        names = ctx["mapping"].class_names
        print(f"[{stage}] class weights (from the train split): "
              f"{ {names[k]: round(v, 3) for k, v in class_weight.items()} }")

    if stage == "feature_extraction":
        print(f"[stage 1] feature extraction: {record['base_layers_frozen']}/"
              f"{record['backbone_layers']} base layers frozen; "
              f"trainable params {record['trainable_parameters']:,}; lr={learning_rate:g}")
    else:
        print(f"[stage 2] fine-tuning: unfroze the top {config.finetune_unfreeze_fraction:.0%} of the "
              f"base -> {record['base_layers_trainable']} layers trainable "
              f"({record['first_trainable_base_layer']} .. {record['last_trainable_base_layer']}); "
              f"lr={learning_rate:g}; trainable params {record['trainable_parameters']:,}")

    callbacks, best_path = model_lib.build_callbacks(
        config, config.artifacts_dir / "checkpoints", run_tag
    )
    t0 = time.perf_counter()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=int(epochs),
        callbacks=callbacks,
        class_weight=class_weight,
        # The Dataset already shuffles; telling Keras not to re-shuffle avoids a
        # warning and a redundant pass over the input pipeline.
        shuffle=False,
        verbose=0 if quiet else 2,
    )
    seconds = time.perf_counter() - t0
    records = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    meta = {
        **record,
        "learning_rate": float(learning_rate),
        "epochs_requested": int(epochs),
        "epochs_run": len(records.get("loss", [])),
        "final_train_accuracy": records["accuracy"][-1] if records.get("accuracy") else None,
        "final_val_accuracy": records["val_accuracy"][-1] if records.get("val_accuracy") else None,
        "best_val_loss": float(min(records["val_loss"])) if records.get("val_loss") else None,
        "wall_clock_seconds": round(seconds, 2),
        "checkpoint": str(best_path),
        "class_weights": class_weight,
    }
    print(f"[{stage}] {meta['epochs_run']} epoch(s) in {seconds / 60:.1f} min, "
          f"best val_loss={meta['best_val_loss']:.4f}")
    return records, meta


def stage_one(ctx: dict, ds_train, ds_val, quiet: bool = False) -> tuple[dict, dict]:
    """Stage 1: feature extraction (pretrained base entirely frozen)."""
    config = ctx["config"]
    return _fit_stage(
        ctx["model"], ctx, ds_train, ds_val,
        learning_rate=config.head_learning_rate,
        epochs=config.feature_extraction_epochs,
        run_tag=f"{ctx['model_key']}_stage1",
        stage="feature_extraction",
        quiet=quiet,
    )


def stage_two(ctx: dict, ds_train, ds_val, quiet: bool = False) -> tuple[dict, dict]:
    """Stage 2: fine-tuning (top ``finetune_unfreeze_fraction`` of the base)."""
    config = ctx["config"]
    return _fit_stage(
        ctx["model"], ctx, ds_train, ds_val,
        learning_rate=config.finetune_learning_rate,
        epochs=config.finetune_epochs,
        run_tag=f"{ctx['model_key']}_stage2",
        stage="fine_tuning",
        quiet=quiet,
    )


def finalize(ctx: dict, histories: list[dict], stage_metas: dict[str, dict],
             ds_test, *, run_note: str | None = None) -> dict:
    """Save weights, evaluate on the test split, write figures/metadata, return meta."""
    from . import evaluate as eval_lib
    from . import models as model_lib
    from . import visualize as viz

    config = ctx["config"]
    model_key = ctx["model_key"]
    mapping = ctx["mapping"]
    model = ctx["model"]

    # ---- combined training history --------------------------------------
    combined: dict[str, list[float]] = {}
    for record in histories:
        for metric, values in record.items():
            combined.setdefault(metric, []).extend(values)
    if not combined:
        raise RuntimeError("No training history was supplied; nothing to record.")
    longest = max(len(v) for v in combined.values())
    combined = {k: v + [float("nan")] * (longest - len(v)) for k, v in combined.items()}
    combined["epoch"] = list(range(1, longest + 1))
    frame = pd.DataFrame(combined)
    history_csv = config.result(f"training_history_{model_key}.csv")
    frame.to_csv(history_csv, index=False)
    viz.plot_training_history(
        frame,
        config.figure(f"training_history_{model_key}.png"),
        title=f"{model_key}: stage 1 (epochs 1-{stage_metas['feature_extraction']['epochs_run']}) "
              f"then stage 2 fine-tuning (validation loss dips at the stage boundary because the "
              f"learning rate and the trainable set change)",
    )

    # ---- save ------------------------------------------------------------
    model_path = config.model_artifact_path(model_key)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    model_size_mb = round(model_path.stat().st_size / 1e6, 2)
    total_params, trainable_params = model_lib.count_parameters(model)

    # ---- test-set evaluation (exactly once, after selection) ------------
    metrics = eval_lib.evaluate_model(model, ds_test, mapping.class_names, config, tag=model_key)
    report = eval_lib.classification_report_frame(metrics)
    print(f"\n[eval] {model_key} on the untouched test set (n={metrics['num_test_images']}):")
    print(report.to_string(index=False))

    samples = eval_lib.sample_predictions_frame(
        metrics["_y_true"],
        metrics["_probabilities"],
        mapping.class_names,
        paths=list(ctx["test_df"]["path"].astype(str)),
        low_confidence_threshold=config.low_confidence_threshold,
    )
    samples.to_csv(config.prediction(f"test_predictions_{model_key}.csv"), index=False)

    inference = measure_inference_ms(
        model, list(ctx["test_df"]["path"].astype(str)), ctx["plan"], config
    )
    elapsed = time.perf_counter() - ctx["started"]

    meta = {
        "model": model_key,
        "backbone_display_name": model_lib.get_backbone(model_key).display_name,
        "config": config.to_dict(),
        "environment": environment_report(config),
        "run_note": run_note,
        "weights_requested": ctx["weights_requested"],
        "preprocessing": {
            "internal_preprocessing": ctx["plan"].internal_preprocessing,
            "module_name": ctx["plan"].module_name,
            "description": ctx["plan"].description,
        },
        "parameter_counts": {"total": total_params, "trainable": trainable_params},
        "model_path": str(model_path),
        "model_size_mb": model_size_mb,
        "stages": stage_metas,
        "training_seconds_total": round(
            sum(v.get("wall_clock_seconds", 0.0) for v in stage_metas.values()), 2
        ),
        "wall_clock_seconds_total": round(elapsed, 2),
        "peak_memory_mb": peak_memory_mb(),
        "test_metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "inference_time": inference,
        "classes": mapping.to_dict(),
        "history_csv": str(history_csv),
        "figures": {
            "history": str(config.figure(f"training_history_{model_key}.png")),
            "confusion_matrix": str(config.figure(f"confusion_matrix_{model_key}.png")),
        },
        "dataset": {
            "root": str(ctx["root"]),
            "total_images": int(len(ctx["frames"])),
            "fingerprint": dl.dataset_fingerprint(ctx["frames"]),
            "train_images": int(len(ctx["train_df"])),
            "validation_images": int(len(ctx["val_df"])),
            "test_images": int(len(ctx["test_df"])),
        },
    }
    config.result(f"train_meta_{model_key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save] weights -> {model_path} ({model_size_mb} MB); metadata -> "
          f"{config.result(f'train_meta_{model_key}.json')}")
    return meta


def train_one_model(
    model_key: str,
    config: ProjectConfig,
    weights: str = "imagenet",
    quiet: bool = False,
    run_note: str | None = None,
) -> dict:
    """Run the whole experiment for one candidate model."""
    ctx = prepare_experiment(model_key, config, weights=weights)
    pipelines = build_pipelines(ctx)
    h1, m1 = stage_one(ctx, pipelines["train"], pipelines["validation"], quiet=quiet)
    h2, m2 = stage_two(ctx, pipelines["train"], pipelines["validation"], quiet=quiet)
    return finalize(
        ctx,
        [h1, h2],
        {"feature_extraction": m1, "fine_tuning": m2},
        pipelines["test"],
        run_note=run_note,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=["mobilenetv2", "efficientnetb0", "both"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="stage 1 (feature extraction) epochs")
    parser.add_argument("--finetune-epochs", type=int, default=None, help="stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--unfreeze-fraction", type=float, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--group-aware-split", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Cap tf.data parallelism and prefetch (for machines with <4 GB RAM, or when a run "
        "is killed by the OOM killer)",
    )
    parser.add_argument("--weights", default="imagenet",
                        help="'imagenet' (default) or 'None' for the from-scratch ablation")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--note", default=None, help="Free-text note stored with the run metadata")
    args = parser.parse_args(argv)

    weights = None if str(args.weights).lower() in {"none", "null"} else args.weights
    config = ProjectConfig().override(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        seed=args.seed,
        feature_extraction_epochs=args.epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        head_learning_rate=args.head_lr,
        finetune_learning_rate=args.finetune_lr,
        early_stopping_patience=args.patience,
        finetune_unfreeze_fraction=args.unfreeze_fraction,
        max_images_per_class=args.max_images_per_class,
        group_aware_split=args.group_aware_split or None,
        class_weights=False if args.no_class_weights else None,
        tf_data_num_parallel_calls=2 if args.low_memory else None,
        tf_data_prefetch=1 if args.low_memory else None,
    )
    keys = ["mobilenetv2", "efficientnetb0"] if args.model == "both" else [args.model]

    try:
        for key in keys:
            train_one_model(key, config, weights=weights, quiet=args.quiet, run_note=args.note)
        if len(keys) > 1:
            from .compare import compare_and_select

            compare_and_select(config)
    except (dl.DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[train] interrupted; nothing was saved for the unfinished stage.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
