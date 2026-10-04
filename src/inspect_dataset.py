"""Dataset inspection stage: measure the data that is actually on disk.

Produces, for a real dataset directory:

* ``outputs/results/dataset_summary.json`` -- totals, per-class counts, image
  dimensions/formats, corrupted-file count, split sizes, fingerprint, versions.
* ``outputs/results/splits.csv``           -- the reproducible train/val/test assignment.
* ``outputs/results/image_metadata.csv``   -- per-file width/height/format/bytes.
* ``outputs/results/corrupted_files.csv``  -- unreadable files (empty if none).
* ``outputs/figures/class_distribution.png``
* ``outputs/figures/dataset_examples.png``

Everything printed or saved here is computed from the files themselves.

Usage::

    python -m src.inspect_dataset            # uses data/ and outputs/
    python -m src.inspect_dataset --data-dir /path/to/raw/color
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import DISPLAY_NAMES, ProjectConfig, environment_report, seed_everything, write_run_provenance
from . import data_loader as dl


def inspect_dataset(config: ProjectConfig, write: bool = True) -> dict:
    """Run the whole inspection and (optionally) persist its artefacts."""
    seed_everything(config.seed, config.deterministic_ops)
    config.ensure_directories()

    root = dl.find_dataset_root(config)
    mapping, discover_info = dl.build_class_mapping(root)
    print(f"[inspect] dataset root ......... {root}")
    print(f"[inspect] classes .............. {', '.join(mapping.class_names)}")

    frames = dl.collect_images(root, mapping, config)
    print(f"[inspect] image files found .... {len(frames)}")

    frames, report = dl.validate_images(frames, config)
    corrupt = report[report["error"].notna()] if "error" in report else report.iloc[0:0]
    print(f"[inspect] readable images ...... {len(frames)}")
    print(f"[inspect] corrupted/unreadable .. {len(corrupt)}")

    dist = dl.class_distribution(frames)
    splits = dl.stratified_split(frames, config)
    table = dl.split_table(splits)
    fingerprint = dl.dataset_fingerprint(frames)

    widths = report["width"].dropna()
    heights = report["height"].dropna()
    dimension_summary = {
        "count_measured": int(len(widths)),
        "width_min": int(widths.min()),
        "width_max": int(widths.max()),
        "height_min": int(heights.min()),
        "height_max": int(heights.max()),
        "unique_widths": sorted(int(w) for w in widths.unique().tolist())[:12],
        "unique_heights": sorted(int(h) for h in heights.unique().tolist())[:12],
        "formats": {str(k): int(v) for k, v in report["format"].value_counts().items()},
        "mean_file_bytes": float(pd.to_numeric(report["bytes"], errors="coerce").mean() or 0.0),
        "smallest_file_bytes": int(pd.to_numeric(report["bytes"], errors="coerce").min() or 0),
        "largest_file_bytes": int(pd.to_numeric(report["bytes"], errors="coerce").max() or 0),
        "total_dataset_mb": round(float(pd.to_numeric(report["bytes"], errors="coerce").sum() or 0) / 1e6, 2),
    }

    summary = {
        "dataset_root": str(root),
        "total_images_after_validation": int(len(frames)),
        "total_image_files_discovered": int(len(report)),
        "corrupted_files": int(len(corrupt)),
        "classes": {
            key: {
                "display_name": DISPLAY_NAMES[key],
                "folder": mapping.folder_names[key],
                "label_index": i,
                "images": int(dist.loc[dist["class_key"] == key, "count"].sum()),
            }
            for i, key in enumerate(mapping.class_keys)
        },
        "class_distribution": dist.to_dict(orient="records"),
        "imbalance_ratio_largest_over_smallest": round(dl.imbalance_ratio(dist), 4),
        "dimensions_and_formats": dimension_summary,
        "split_fractions": {
            "train": config.train_fraction,
            "validation": config.val_fraction,
            "test": config.test_fraction,
        },
        "split_counts": {k: int(v) for k, v in splits["split"].value_counts().items()},
        "split_by_class": table.to_dict(orient="index"),
        "group_aware_split": bool(config.group_aware_split),
        "seed": config.seed,
        "dataset_fingerprint": fingerprint,
        "unrecognised_dirs": discover_info["unrecognised_dirs"],
        "out_of_scope_dirs": discover_info["out_of_scope_dirs"],
        "class_weights_used": bool(config.class_weights),
        "class_weights": dl.compute_class_weights(splits[splits["split"] == "train"]),
        "environment": environment_report(config),
        "note": (
            "All numbers in this file were measured from the files on disk at run "
            "time; none are copied from the literature."
        ),
    }

    if write:
        config.result("dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        splits_out = splits.assign(path=splits["path"].astype(str))
        splits_out.to_csv(config.result("split_assignment.csv"), index=False)
        dl.save_splits(splits, config.result("splits.csv"), root)
        report.to_csv(config.result("image_metadata.csv"), index=False)
        corrupt.to_csv(config.result("corrupted_files.csv"), index=False)
        table.to_csv(config.result("split_table.csv"))
        mapping.save(config.class_map_path)
        config.save(config.result("run_config.json"))
        write_run_provenance(config, note="dataset inspection")
        _write_report_markdown(config, summary, dist, table)
        _make_figures(config, frames, splits, mapping, dist, table)

    print("\n=== dataset summary ===")
    print(f"total usable images : {summary['total_images_after_validation']}")
    print(dist.to_string(index=False))
    print(f"\nclass imbalance (largest/smallest): {summary['imbalance_ratio_largest_over_smallest']}")
    print("split sizes:", summary["split_counts"])
    print(f"corrupted files: {summary['corrupted_files']}")
    print(f"saved artefacts under: {config.artifacts_dir}")
    return summary


def _make_figures(config, frames, splits, mapping, dist, table) -> None:
    from . import visualize as viz

    viz.plot_class_distribution(dist, config.figure("class_distribution.png"), split_table=table)

    # Representative examples: the first *training-split* image of each class,
    # chosen deterministically from the sorted file list.
    picks: list[tuple[str, Path]] = []
    for key in mapping.class_keys:
        sub = splits[(splits["class_key"] == key) & (splits["split"] == "train")].sort_values("filename")
        if len(sub):
            picks.append((DISPLAY_NAMES[key], Path(sub["path"].iloc[0])))
    if picks:
        viz.plot_examples(
            picks,
            config.figure("dataset_examples.png"),
            ncols=len(picks),
            title="First training-split image of each class (real dataset files)",
        )


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional "tabulate" dependency."""
    try:
        return df.to_markdown()
    except (ImportError, ValueError):
        pass
    header = "| " + " | ".join([""] + [str(c) for c in df.columns]) + " |"
    rule = "| " + " | ".join(["---"] + ["---:" for _ in df.columns]) + " |"
    rows = [
        "| " + " | ".join([str(idx)] + [str(v) for v in record]) + " |"
        for idx, record in zip(df.index, df.itertuples(index=False))
    ]
    return "\n".join([header, rule] + rows)


def _write_report_markdown(config, summary: dict, dist: pd.DataFrame, table: pd.DataFrame) -> None:
    """A Chapter-3-ready write-up of the dataset, built from the measurements."""
    dims = summary["dimensions_and_formats"]
    lines = [
        "# Dataset report",
        "",
        "Generated by `python -m src.inspect_dataset`. Every value below was measured",
        "from the files on disk at run time.",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Total usable images: **{summary['total_images_after_validation']}**",
        f"- Unreadable/corrupted files: **{summary['corrupted_files']}**",
        f"- Image size range: {dims['width_min']}x{dims['height_min']} to "
        f"{dims['width_max']}x{dims['height_max']} px",
        f"- Formats: {dims['formats']}",
        f"- Total image bytes on disk: {dims['total_dataset_mb']} MB "
        f"(mean {int(dims['mean_file_bytes'])} B/file)",
        f"- Random seed: {summary['seed']}  |  dataset fingerprint: `{summary['dataset_fingerprint']}`",
        f"- Split fractions: {summary['split_fractions']} -> actual counts "
        f"{summary['split_counts']}",
        "",
        "## Images per class",
        "",
        "| Class | Folder name in dataset | Images | Share |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in dist.itertuples(index=False):
        folder = summary["classes"][row.class_key]["folder"]
        lines.append(f"| {row.class_name} | `{folder}` | {row.count} | {row.percent:.2f}% |")
    lines += [
        "",
        "## Split composition",
        "",
        _df_to_markdown(table),
        "",
        "## Scope and out-of-scope folders",
        "",
        f"- Unrecognised directories ignored: {summary['unrecognised_dirs'] or 'none'}",
        f"- Maize folders outside the four-class task, excluded: {summary['out_of_scope_dirs'] or 'none'}",
        "",
        "## Environment",
        "",
        "```",
        "\n".join(f"{k}: {v}" for k, v in summary["environment"].items()),
        "```",
        "",
        "> **Caveat that belongs in the thesis:** these are controlled single-leaf",
        "> photographs taken against a plain background, not field photographs.",
        "> Accuracy measured on this dataset therefore does not transfer directly to",
        "> in-field diagnosis (see README, Limitations).",
        "",
    ]
    path = config.report("dataset_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.inspect_dataset", description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Project data/ folder or the image root itself")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Smoke-test cap")
    parser.add_argument("--group-aware-split", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="Report only, write no files")
    args = parser.parse_args(argv)

    config = ProjectConfig().override(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        seed=args.seed,
        batch_size=args.batch_size,
        max_images_per_class=args.max_images_per_class,
        group_aware_split=args.group_aware_split or None,
    )
    try:
        inspect_dataset(config, write=not args.no_write)
    except dl.DatasetError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
