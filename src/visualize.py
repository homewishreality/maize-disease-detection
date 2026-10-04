"""Matplotlib helpers: every figure is drawn from measured data.

No number in any figure is typed in by hand -- each one is read from the
DataFrames/arrays passed in by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# Headless-safe (Colab, CI, Windows without a display server).  ``force=False``
# leaves a user's explicit backend choice alone.
matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Palette kept consistent across all figures of the report.
CLASS_COLORS = {
    "Healthy": "#2e7d32",
    "Common Rust": "#c62828",
    "Northern Leaf Blight": "#6d4c41",
    "Gray Leaf Spot": "#f9a825",
}
DEFAULT_COLOR = "#1565c0"


def _save(fig: plt.Figure, path: Path, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def plot_class_distribution(
    distribution: pd.DataFrame,
    path: Path,
    title: str = "Maize leaf images per class (measured from the dataset)",
    split_table: pd.DataFrame | None = None,
) -> Path:
    """Bar chart of images per class, annotated with the real counts.

    When *split_table* (class x split) is given, a grouped version is drawn so
    the reader can see that stratification kept the ratios equal.
    """
    if split_table is not None:
        table = split_table.drop(index="TOTAL", errors="ignore")
        classes = list(table.index)
        x = np.arange(len(classes))
        width = 0.26
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for i, (split, colour) in enumerate(
            [("train", "#1565c0"), ("validation", "#ef6c00"), ("test", "#2e7d32")]
        ):
            if split in table.columns:
                ax.bar(x + (i - 1) * width, table[split].to_numpy(), width, label=split, color=colour)
        for xi, cls in zip(x, classes):
            total = int(table.loc[cls, ["train", "validation", "test"]].sum())
            ax.annotate(f"n={total}", (xi, table.loc[cls, ["train", "validation", "test"]].max()),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
        ax.set_xticks(x, classes)
        ax.legend()
        ax.set_ylabel("Number of images")
        ax.set_title("Class distribution across train / validation / test splits")
        ax.grid(axis="y", alpha=0.3)
        return _save(fig, path)

    labels = list(distribution["class_name"])
    counts = distribution["count"].to_numpy(dtype=float)
    colours = [CLASS_COLORS.get(l, DEFAULT_COLOR) for l in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(labels, counts, color=colours)
    for bar, count in zip(bars, counts):
        ax.annotate(
            f"{int(count)}\n({count / counts.sum() * 100:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Number of images")
    ax.set_title(title)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_examples(
    image_paths: Sequence[tuple[str, Path]],
    path: Path,
    ncols: int = 4,
    title: str = "Representative training images per class (as stored in the dataset)",
) -> Path:
    """Grid of one or more real images per class, taken from the training split."""
    from PIL import Image

    n = len(image_paths)
    ncols = min(max(ncols, 1), n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.7 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, p) in zip(axes.ravel(), image_paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((224, 224))
            ax.imshow(np.asarray(im))
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.5, f"unreadable\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_training_history(
    history: dict[str, Sequence[float]] | pd.DataFrame,
    path: Path,
    title: str = "Training history",
) -> Path:
    """Accuracy and loss curves, from the Keras History dict / history.csv."""
    frame = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    if len(frame) == 0:
        raise ValueError("Empty training history; nothing to plot.")
    epochs = frame["epoch"].to_numpy() if "epoch" in frame.columns else np.arange(1, len(frame) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if "accuracy" in frame and "val_accuracy" in frame:
        axes[0].plot(epochs, frame["accuracy"], "o-", label="training accuracy", color=DEFAULT_COLOR)
        axes[0].plot(epochs, frame["val_accuracy"], "s-", label="validation accuracy", color="#ef6c00")
        axes[0].set_ylabel("accuracy (sparse categorical)")
        axes[0].set_title("Accuracy")
        axes[0].set_ylim(min(0.0, float(frame[["accuracy", "val_accuracy"]].min().min()) - 0.05), 1.02)
    if "loss" in frame and "val_loss" in frame:
        axes[1].plot(epochs, frame["loss"], "o-", label="training loss", color=DEFAULT_COLOR)
        axes[1].plot(epochs, frame["val_loss"], "s-", label="validation loss", color="#ef6c00")
        axes[1].set_ylabel("loss")
        axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str = "Confusion matrix (test set)",
    normalize: bool = False,
    subtitle: str | None = None,
) -> Path:
    """Confusion matrix with raw counts (and row percentages if *normalize*)."""
    matrix = np.asarray(matrix, dtype=float)
    k = len(class_names)
    if matrix.shape != (k, k):
        raise ValueError(f"Confusion matrix shape {matrix.shape} does not match {k} classes.")
    fig, ax = plt.subplots(figsize=(1.7 * k + 2.2, 1.6 * k + 1.9))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() if matrix.max() > 0 else 1)
    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
    for i in range(k):
        for j in range(k):
            text = f"{int(matrix[i, j])}"
            if normalize:
                row_total = matrix[i].sum()
                pct = matrix[i, j] / row_total * 100 if row_total else 0.0
                text = f"{int(matrix[i, j])}\n({pct:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > threshold else "black")
    ax.set_xticks(range(k), class_names, rotation=30, ha="right")
    ax.set_yticks(range(k), class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="images")
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(
    comparison: pd.DataFrame,
    path: Path,
    metrics: Sequence[str] = ("accuracy", "precision", "recall", "f1_score"),
    title: str = "Model comparison (test set)",
) -> Path:
    """Grouped bar chart of the metric columns for each candidate model."""
    available = [m for m in metrics if m in comparison.columns]
    if not available:
        raise ValueError(f"None of {list(metrics)} found in the comparison table.")
    models = list(comparison["model"].astype(str))
    x = np.arange(len(models))
    width = 0.8 / max(len(available), 1)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for i, metric in enumerate(available):
        values = comparison[metric].to_numpy(dtype=float)
        bars = ax.bar(x + (i - (len(available) - 1) / 2) * width, values, width, label=metric.replace("_", " "))
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(available))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def plot_prediction_probabilities(
    probabilities: dict[str, float],
    path: Path | None = None,
    title: str = "Predicted class probabilities",
) -> plt.Figure:
    """Horizontal probability bars; used by the CLI report and by the notebook."""
    names = list(probabilities.keys())
    values = np.array([probabilities[n] for n in names], dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    colours = [CLASS_COLORS.get(n, DEFAULT_COLOR) for n in names]
    ax.barh(names, values, color=colours)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f}%", (v, i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("model confidence (%)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig
