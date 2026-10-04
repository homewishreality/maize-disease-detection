"""Dataset location, class discovery, integrity checking and splitting.

The guiding rule of this module: *never assume* anything about the dataset.
Class folders are discovered, counted and integrity-checked at run time, and
the numbers printed in reports come straight out of those measurements.

Expected layout (one sub-directory per class, images inside it)::

    <dataset root>/
    |-- Corn_(maize)___Common_rust_/
    |   |-- RS_Rust 1563.JPG
    |   `-- ...
    |-- Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
    |-- Corn_(maize)___Northern_Leaf_Blight/
    `-- Corn_(maize)___healthy/
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    CLASS_ALIASES,
    DISPLAY_NAMES,
    FOLDER_KEYWORD_RULES,
    IMAGE_EXTENSIONS,
    KNOWN_OUT_OF_SCOPE,
    ProjectConfig,
)


class DatasetError(RuntimeError):
    """Raised for any problem the user can act on (missing data, bad layout)."""


@dataclass
class ClassMapping:
    """Mapping between canonical class keys, folder names and label indices.

    ``index`` -> canonical key, and the inverse, are what the model's output
    layer and the Streamlit app share, so the mapping is always saved to disk
    next to the weights (``models/class_names.json``).
    """

    #: ordered list of canonical keys, index == model output neuron
    class_keys: list[str]
    #: canonical key -> directory name found on disk
    folder_names: dict[str, str] = field(default_factory=dict)
    #: canonical key -> dataset-relative directory path
    relative_dirs: dict[str, str] = field(default_factory=dict)

    @property
    def class_names(self) -> list[str]:
        return [DISPLAY_NAMES[k] for k in self.class_keys]

    @property
    def num_classes(self) -> int:
        return len(self.class_keys)

    def to_dict(self) -> dict:
        return {
            "class_keys": self.class_keys,
            "class_names": self.class_names,
            "folder_names": self.folder_names,
            "dataset_relative_dirs": self.relative_dirs,
            "index_to_name": {str(i): n for i, n in enumerate(self.class_names)},
            "name_to_index": {n: i for i, n in enumerate(self.class_names)},
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Locating the dataset
# --------------------------------------------------------------------------
def _looks_like_image_dir(path: Path) -> bool:
    """True if *path* is a directory containing at least one image file."""
    if not path.is_dir():
        return False
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except (OSError, PermissionError):
        return False
    return False


def _normalise(name: str) -> str:
    """Folder-name normalisation: lower-case, non-alphanumerics -> ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _matches_class(folder_name: str) -> str | None:
    """Return the canonical class key for *folder_name*, or ``None``.

    Two passes: exact alias match first (cheap, unambiguous), then keyword
    rules for the many PlantVillage mirror spellings.
    """
    for key, aliases in CLASS_ALIASES.items():
        if folder_name in aliases:
            return key
    norm = _normalise(folder_name)
    # Only consider maize/corn folders so that a "Tomato___healthy" directory
    # in a full PlantVillage download is not mistaken for our healthy class.
    is_maize = any(tok in norm for tok in ("corn", "maize"))
    if not is_maize:
        return None
    hits = [key for key, kws in FOLDER_KEYWORD_RULES if any(kw in norm for kw in kws)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:  # ambiguous: trust the most specific rule (declaration order)
        order = [k for k, _ in FOLDER_KEYWORD_RULES]
        hits.sort(key=order.index)
        return hits[0]
    return None


def find_dataset_root(config: ProjectConfig) -> Path:
    """Locate the directory that holds the maize class folders.

    Searches ``config.data_dir`` and the candidate sub-paths in
    ``config.dataset_search_paths`` (recursively, depth-limited), so that a
    ``data/`` folder containing an extracted archive works without renaming.
    """
    base = Path(config.data_dir)
    if not base.exists():
        raise DatasetError(
            f"Dataset directory not found: {base}\n"
            "Please download the dataset and place it there, or run:\n"
            "    python -m src.dataset_setup --dest <project>/data\n"
            "See data/README.md for manual download instructions."
        )

    def score(candidate: Path) -> int:
        if not candidate.is_dir():
            return -1
        try:
            subdirs = [p for p in candidate.iterdir() if p.is_dir()]
        except (OSError, PermissionError):
            return -1
        return sum(1 for p in subdirs if _matches_class(p.name) and _looks_like_image_dir(p))

    checked: list[Path] = []
    candidates: list[Path] = [base]
    for rel in config.dataset_search_paths:
        candidates.append(base / rel)
    # Fall back to a shallow recursive search (covers "PlantVillage-master/...").
    candidates.extend(_shallow_dirs(base, max_depth=4))

    best: tuple[int, Path] | None = None
    for cand in candidates:
        checked.append(cand)
        n = score(cand)
        if n >= len(CLASS_ALIASES):
            return cand
        if n > 0 and (best is None or n > best[0]):
            best = (n, cand)
    if best is not None:
        raise DatasetError(
            f"Found a partial dataset at {best[1]} with {best[0]}/"
            f"{len(CLASS_ALIASES)} required maize classes.\n"
            "Expected these class folders (any of the accepted aliases):\n  "
            + "\n  ".join(f"{k}: {', '.join(v)}" for k, v in CLASS_ALIASES.items())
        )
    raise DatasetError(
        "No directory containing maize class folders was found.\n"
        f"Searched {len(checked)} candidate location(s) under: {base}\n"
        "Expected layout: <root>/<Corn_(maize)___ClassName>/*.jpg\n"
        "Run 'python -m src.dataset_setup --dest <project>/data' to fetch the "
        "maize subset, or read data/README.md."
    )


def _shallow_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    """Yield directories up to *max_depth* levels below *root* (bounded scan)."""
    root = Path(root)
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if child.name.startswith(".") or child.name in {"__pycache__"}:
                continue
            out.append(child)
            stack.append((child, depth + 1))
    return out


def discover_class_dirs(root: Path) -> tuple[dict[str, Path], list[Path]]:
    """Split first-level directories of *root* into matched / unmatched."""
    matched: dict[str, Path] = {}
    unmatched: list[Path] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        key = _matches_class(folder.name)
        if key is None:
            if _looks_like_image_dir(folder):
                unmatched.append(folder)
            continue
        if key in matched:
            raise DatasetError(
                f"Ambiguous dataset layout: class '{key}' matches two folders:\n"
                f"  {matched[key].name}\n  {folder.name}\n"
                "Keep exactly one folder per class, or extend CLASS_ALIASES in "
                "src/config.py."
            )
        if not _looks_like_image_dir(folder):
            raise DatasetError(
                f"Class folder '{folder.name}' exists but contains no readable "
                f"image files ({', '.join(IMAGE_EXTENSIONS)})."
            )
        matched[key] = folder
    return matched, unmatched


def build_class_mapping(root: Path) -> tuple[ClassMapping, dict]:
    """Discover class folders and produce the :class:`ClassMapping`.

    Returns the mapping plus an inspection dict with the raw per-folder file
    counts, the folders ignored as out-of-scope, and empty folders found.
    """
    root = Path(root)
    matched, unmatched = discover_class_dirs(root)

    missing = [k for k in CLASS_ALIASES if k not in matched]
    if missing:
        detail = "\n".join(
            f"  {k}: expected one of -> {', '.join(CLASS_ALIASES[k])}" for k in missing
        )
        raise DatasetError(
            f"Required class folder(s) not found in {root}:\n{detail}\n"
            "Folders that *were* found: "
            + (", ".join(p.name for p in sorted(root.iterdir()) if p.is_dir()) or "<none>")
        )

    # Stable ordering: healthy first, then the three diseases alphabetically,
    # which makes confusion matrices read naturally in the report.
    order = ["healthy", "common_rust", "gray_leaf_spot", "northern_leaf_blight"]
    class_keys = [k for k in order if k in matched] + sorted(set(matched) - set(order))

    mapping = ClassMapping(
        class_keys=class_keys,
        folder_names={k: matched[k].name for k in class_keys},
        relative_dirs={k: str(matched[k].relative_to(root)) for k in class_keys},
    )
    info = {
        "dataset_root": str(root),
        "matched_classes": {k: str(v) for k, v in matched.items()},
        "out_of_scope_dirs": [p.name for p in unmatched if p.name in KNOWN_OUT_OF_SCOPE],
        "unrecognised_dirs": [
            p.name for p in unmatched if p.name not in KNOWN_OUT_OF_SCOPE
        ],
    }
    return mapping, info


def _leaf_group(path: Path) -> str:
    """Heuristic identifier for the *source leaf* a photo came from.

    PlantVillage stores several photographs of the same leaf as
    ``<uuid>___... copy 2.jpg``.  Using the uuid/file stem as a group id lets
    the split keep siblings together (a leak the original paper explicitly
    avoided).  When no uuid prefix exists the stem itself is used, which makes
    grouping a no-op for e.g. ``RS_Rust 1563.JPG``.
    """
    stem = path.stem
    uuid_match = re.match(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", stem)
    return uuid_match.group(1) if uuid_match else stem


def collect_images(
    root: Path,
    mapping: ClassMapping,
    config: ProjectConfig | None = None,
    include_corrupt: bool = True,
) -> pd.DataFrame:
    """Build a DataFrame of every image file belonging to a mapped class.

    Columns: ``path``, ``class_key``, ``class_name``, ``label``, ``leaf_group``,
    ``filename``.  No pixel data is read here -- only file names -- so this is
    fast even for large trees.
    """
    config = config or ProjectConfig()
    rows: list[dict] = []
    for idx, key in enumerate(mapping.class_keys):
        folder = Path(root) / mapping.relative_dirs[key]
        files = sorted(
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if config.max_images_per_class:
            files = files[: config.max_images_per_class]
        for p in files:
            rows.append(
                {
                    "path": p,
                    "filename": p.name,
                    "class_key": key,
                    "class_name": DISPLAY_NAMES[key],
                    "label": idx,
                    "leaf_group": _leaf_group(p),
                }
            )
    if not rows:
        raise DatasetError(
            f"No image files ({', '.join(IMAGE_EXTENSIONS)}) found under {root}.\n"
            "Check that the archive was extracted (not merely downloaded)."
        )
    df = pd.DataFrame(rows)
    return df


def validate_images(df: pd.DataFrame, config: ProjectConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open every image with Pillow to detect corrupt or truncated files.

    Returns ``(valid, report)`` where *report* holds width/height/format bytes
    for each readable file and an ``error`` string for unreadable ones.  All
    dataset statistics printed elsewhere are derived from these measurements,
    never from values copied out of a paper.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # PlantVillage images are small; avoid the warning
    records = []
    limit = config.inspection_limit if config else None
    for i, row in enumerate(df.itertuples(index=False)):
        if limit is not None and i >= limit:
            break
        path = Path(row.path)
        rec = {"path": str(path), "width": None, "height": None, "format": None, "bytes": None, "error": None}
        try:
            rec["bytes"] = path.stat().st_size
            with Image.open(path) as im:
                im.verify()  # structural check: catches truncated/zero-byte files
            with Image.open(path) as im:
                rec["width"], rec["height"] = im.size
                rec["format"] = (im.format or "").lower()
        except Exception as exc:  # noqa: BLE001 - we report, not raise
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)
    report = pd.DataFrame(records)
    valid = report[report["error"].isna()]
    good_paths = set(valid["path"])
    df = df[df["path"].astype(str).isin(good_paths) | ~df["path"].astype(str).isin(set(report["path"]))]
    return df.reset_index(drop=True), report


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Add a ``split`` column of train/validation/test, stratified by class.

    Splitting happens on the **list of original image files**, before any
    augmentation exists, and before any pixel is touched -- so an augmented
    view of an image can never leak into the validation or test set.
    Fractions must sum to 1; ``config.seed`` makes it reproducible.
    """
    from sklearn.model_selection import train_test_split

    fractions = (config.train_fraction, config.val_fraction, config.test_fraction)
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions):.3f}")
    if min(fractions) <= 0:
        raise ValueError("All split fractions must be positive.")

    stratify_col = "leaf_group" if config.group_aware_split else "class_key"
    if config.group_aware_split:
        # Stratify on (class, group) so groups stay the size of 1 where possible.
        counts = df.groupby(["class_key", "leaf_group"]).size()
        if (counts > 1).any() and counts.max() > 20:
            print(
                "[split] warning: a few leaf groups contain many images; "
                "group-aware split may distort class balance."
            )

    train_val, test = train_test_split(
        df,
        test_size=config.test_fraction,
        stratify=df[stratify_col],
        random_state=config.seed,
        shuffle=True,
    )
    # ``train_val`` holds (1 - test_fraction) of the data, so the validation
    # share has to be expressed relative to that pool:
    #   val = 0.15/0.85 of train_val -> 15% of the whole dataset.
    # (Splitting the pool 50/50 would wrongly give 42.5%/42.5%/15%.)
    val_size_in_pool = config.val_fraction / (config.train_fraction + config.val_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_size_in_pool,
        stratify=train_val[stratify_col],
        random_state=config.seed,
        shuffle=True,
    )
    out = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    # Verify the guarantees instead of trusting the arithmetic above.
    _assert_disjoint(out)
    _assert_fractions(out, config)
    return out


def _assert_fractions(out: pd.DataFrame, config: ProjectConfig, tol: float = 0.01) -> None:
    """Check that each split really holds the requested share of the data."""
    actual = out["split"].value_counts(normalize=True).to_dict()
    target = {
        "train": config.train_fraction,
        "validation": config.val_fraction,
        "test": config.test_fraction,
    }
    bad = {
        k: (actual.get(k, 0.0), target[k])
        for k in target
        if abs(actual.get(k, 0.0) - target[k]) > tol
    }
    if bad:
        detail = ", ".join(f"{k}: {a:.1%} (wanted {t:.1%})" for k, (a, t) in bad.items())
        raise DatasetError(
            f"Split proportions are wrong -> {detail}. Fractions must be applied "
            "relative to the remaining pool; check stratified_split()."
        )


def _assert_disjoint(out: pd.DataFrame) -> None:
    by_split = {s: set(g["path"].astype(str)) for s, g in out.groupby("split")}
    if len(by_split) != 3:
        raise DatasetError(f"Expected 3 splits, found {sorted(by_split)}")
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for a, b in pairs:
        overlap = by_split[a] & by_split[b]
        if overlap:
            raise DatasetError(
                f"Data leakage: {len(overlap)} image(s) appear in both '{a}' and "
                f"'{b}' (e.g. {sorted(overlap)[0]})."
            )
    if sum(len(v) for v in by_split.values()) != len(out):
        raise DatasetError("Split sizes do not add up to the dataset size.")


def save_splits(df: pd.DataFrame, path: Path, root: Path | None = None) -> None:
    """Persist the split assignment so every model trains on exactly the same data.

    Paths are stored *relative* to the dataset root, so the same CSV works on a
    laptop and inside Colab.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = df[["filename", "class_key", "class_name", "label", "split", "leaf_group"]].copy()
    if root is not None:
        root = Path(root).resolve()
        keep["rel_path"] = [
            str(Path(p).resolve().relative_to(root)) for p in df["path"]
        ]
    else:
        keep["rel_path"] = [Path(p).name for p in df["path"]]
    keep["class_name"] = keep["class_name"].astype(str)
    keep.to_csv(path, index=False)


def load_splits(path: Path, root: Path) -> pd.DataFrame:
    """Load a saved split file, re-anchoring relative paths under *root*."""
    path, root = Path(path), Path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}\nRun the dataset inspection step first "
            "(python -m src.inspect_dataset ...)."
        )
    df = pd.read_csv(path)
    if "rel_path" not in df.columns:
        df["rel_path"] = df["filename"]
    # Pre-build "<class folder>/<file>" -> Path so re-anchoring is O(1).
    lookup: dict[str, Path] = {}
    if root.is_dir():
        for folder in (d for d in root.iterdir() if d.is_dir()):
            for child in folder.iterdir():
                if child.is_file():
                    lookup[f"{folder.name}/{child.name}"] = child

    paths: list[str] = []
    missing: list[str] = []
    for rel, name in zip(df["rel_path"].astype(str), df["filename"].astype(str)):
        candidate = root / rel
        if not candidate.exists():
            candidate = lookup.get(rel, candidate)
        if not Path(candidate).exists():
            found = list(root.rglob(name))
            candidate = found[0] if found else candidate
        if Path(candidate).exists():
            paths.append(str(candidate))
        else:
            missing.append(rel)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in {path.name} are no longer on disk "
            f"(e.g. {missing[0]}). Re-run the inspection step so the split file "
            "matches the current dataset."
        )
    df = df.copy()
    df["path"] = paths
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def split_table(df: pd.DataFrame) -> pd.DataFrame:
    """Contingency table of class x split, straight from the actual data."""
    table = pd.crosstab(df["class_name"], df["split"])
    for col in ("train", "validation", "test"):
        if col not in table.columns:
            table[col] = 0
    table = table[["train", "validation", "test"]]
    table["total"] = table.sum(axis=1)
    table.loc["TOTAL"] = table.sum(axis=0)
    return table


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and share, measured from the DataFrame."""
    counts = df["class_name"].value_counts()
    out = pd.DataFrame({"class_name": counts.index, "count": counts.values})
    out["percent"] = out["count"] / out["count"].sum() * 100.0
    out["class_key"] = out["class_name"].map(
        {v: k for k, v in DISPLAY_NAMES.items()}
    )
    return out.sort_values("class_key").reset_index(drop=True)


def imbalance_ratio(dist: pd.DataFrame) -> float:
    """Largest-class / smallest-class count (1.0 means perfectly balanced)."""
    counts = dist["count"].astype(float)
    return float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")


def compute_class_weights(df: pd.DataFrame) -> dict[int, float]:
    """Balanced class weights, as scikit-learn defines them.

    ``weight_c = n / (k * n_c)`` -- the strategy chosen for this project
    instead of oversampling, so that no image is duplicated into another split.
    """
    labels = df["label"].to_numpy()
    n, k = len(labels), len(set(labels))
    counts = pd.Series(labels).value_counts()
    return {int(c): float(n / (k * counts[c])) for c in sorted(counts.index)}


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable short hash of the exact image set (used to detect mismatches)."""
    h = hashlib.sha256()
    for name in sorted(df["filename"].astype(str)):
        h.update(name.encode("utf-8"))
    return h.hexdigest()[:12]
