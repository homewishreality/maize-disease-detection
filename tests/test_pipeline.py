"""Fast checks for the parts that are easy to get *silently* wrong.

Run either way::

    python tests/test_pipeline.py          # plain runner, no extra deps
    pytest tests/test_pipeline.py          # if you have pytest

Nothing here trains a model. The tests cover the guarantees the project claims:
class-folder mapping, corrupted-file detection, split proportions and
disjointness, augmentation being training-only, metric arithmetic, the
inference contract, the notebook's integrity, and two meta-checks that enforce
"no fabricated numbers" (no hard-coded dataset counts in the source, no
made-up accuracy claims in the README).

TensorFlow-based tests skip automatically when the dataset or TensorFlow is
unavailable, so the file stays runnable on a machine that has not been set up.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data_loader as dl  # noqa: E402
from src.config import CLASS_ALIASES, DISPLAY_NAMES, ProjectConfig  # noqa: E402
from src.disease_info import DISEASE_INFORMATION  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_fake_dataset(root: Path, counts: dict[str, int], size: int = 40) -> None:
    """Create a class-folder tree of real (tiny, valid) JPEG files."""
    for folder, n in counts.items():
        target = root / folder
        target.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = (np.random.default_rng(abs(hash((folder, i))) % 2**32)
                   .integers(0, 255, (size, size, 3))).astype("uint8")
            Image.fromarray(arr).save(target / f"img_{i:04d}.jpg")


def has_dataset() -> bool:
    try:
        dl.find_dataset_root(ProjectConfig())
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# dataset discovery / mapping
# --------------------------------------------------------------------------
def test_class_folder_mapping_accepts_mirror_spellings():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(
            root,
            {
                "Corn_(maize)___healthy": 3,
                "Corn_(maize)___Common_rust_": 3,
                # the spelling used by the canonical release:
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 3,
                "Corn_(maize)___Northern_Leaf_Blight": 3,
                # a mirror spelling, plus non-maize folders that must be ignored:
                "Tomato___Late_blight": 3,
                "Apple___Apple_scab": 3,
            },
        )
        mapping, info = dl.build_class_mapping(root)
        assert sorted(mapping.class_keys) == sorted(CLASS_ALIASES), mapping.class_keys
        assert mapping.folder_names["gray_leaf_spot"] == (
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
        )
        assert mapping.num_classes == 4
        assert mapping.class_names == [DISPLAY_NAMES[k] for k in mapping.class_keys]
        # non-maize folders are not mapped, and are reported as unrecognised
        assert "Tomato___Late_blight" in info["unrecognised_dirs"], info


def test_class_folder_mapping_alt_gray_leaf_spot_name():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(
            root,
            {
                "Corn_(maize)___healthy": 2,
                "Corn_(maize)___Common_rust_": 2,
                "Corn_(maize)___Gray_leaf_spot": 2,          # mirror naming
                "Corn_(maize)___Northern_Leaf_Blight": 2,
            },
        )
        mapping, _ = dl.build_class_mapping(root)
        assert mapping.folder_names["gray_leaf_spot"] == "Corn_(maize)___Gray_leaf_spot"


def test_missing_class_raises_useful_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(root, {"Corn_(maize)___healthy": 2, "Corn_(maize)___Common_rust_": 2})
        try:
            dl.build_class_mapping(root)
        except dl.DatasetError as exc:
            message = str(exc)
            assert "Northern_Leaf_Blight" in message and "Required class folder" in message, message
        else:  # pragma: no cover
            raise AssertionError("missing classes must raise DatasetError")


def test_empty_dataset_directory_raises_download_instructions():
    with tempfile.TemporaryDirectory() as tmp:
        config = ProjectConfig(data_dir=Path(tmp) / "absent")
        try:
            dl.find_dataset_root(config)
        except dl.DatasetError as exc:
            assert "not found" in str(exc) and "dataset_setup" in str(exc), exc
        else:  # pragma: no cover
            raise AssertionError("a missing dataset must raise DatasetError")


def test_corrupted_images_are_detected_and_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_dataset(
            root,
            {
                "Corn_(maize)___healthy": 4,
                "Corn_(maize)___Common_rust_": 4,
                "Corn_(maize)___Gray_leaf_spot": 4,
                "Corn_(maize)___Northern_Leaf_Blight": 4,
            },
        )
        # break two files: one truncated, one empty
        victims = sorted((root / "Corn_(maize)___healthy").iterdir())[:2]
        victims[0].write_bytes(b"\xff\xd8\xff\x00garbage-not-a-jpeg")
        victims[1].write_bytes(b"")
        config = ProjectConfig(data_dir=root)
        mapping, _ = dl.build_class_mapping(root)
        frames = dl.collect_images(root, mapping, config)
        assert len(frames) == 16, len(frames)
        good, report = dl.validate_images(frames, config)
        bad = report[report["error"].notna()]
        assert len(bad) == 2, f"expected 2 corrupt files, found {len(bad)}: {bad['error'].tolist()}"
        assert len(good) == 14, len(good)


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def _synthetic_frame(n_per_class: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for label, n in enumerate(n_per_class):
        for i in range(n):
            rows.append(
                {
                    "path": f"/fake/class{label}/img{i}.jpg",
                    "filename": f"img{i}.jpg",
                    "class_key": f"class{label}",
                    "class_name": f"Class {label}",
                    "label": label,
                    "leaf_group": f"g{label}_{i}",
                }
            )
    return pd.DataFrame(rows)


def test_split_proportions_are_70_15_15():
    df = _synthetic_frame((500, 400, 300, 200))
    config = ProjectConfig()
    out = dl.stratified_split(df, config)
    shares = out["split"].value_counts(normalize=True).to_dict()
    for name, want in (("train", 0.70), ("validation", 0.15), ("test", 0.15)):
        assert abs(shares[name] - want) < 0.01, (name, shares)
    # regression guard for the classic mistake: splitting the held-out pool in
    # half gives 42.5/42.5/15 instead of 70/15/15.
    assert shares["train"] > shares["validation"] * 3, shares


def test_split_is_disjoint_and_stratified():
    df = _synthetic_frame((500, 400, 300, 200))
    out = dl.stratified_split(df, ProjectConfig())
    by = {s: set(g["path"]) for s, g in out.groupby("split")}
    assert not (by["train"] & by["test"]) and not (by["train"] & by["validation"])
    assert not (by["validation"] & by["test"])
    assert sum(len(v) for v in by.values()) == len(df) == 1400

    table = dl.split_table(out)
    for cls in table.index.drop("TOTAL"):
        total = table.loc[cls, "total"]
        assert abs(table.loc[cls, "test"] / total - 0.15) < 0.02, (cls, table.loc[cls])


def test_split_is_reproducible_and_seed_sensitive():
    df = _synthetic_frame((120, 90, 60, 30))
    a = dl.stratified_split(df, ProjectConfig(seed=42))
    b = dl.stratified_split(df, ProjectConfig(seed=42))
    assert list(a["path"]) == list(b["path"]), "same seed must give the same split"
    c = dl.stratified_split(df, ProjectConfig(seed=7))
    assert set(a[a.split == "test"]["path"]) != set(c[c.split == "test"]["path"]), "seed must matter"


def test_bad_split_fractions_are_rejected():
    df = _synthetic_frame((50, 50, 50, 50))
    try:
        dl.stratified_split(df, ProjectConfig().override(train_fraction=0.8, val_fraction=0.1))
    except ValueError as exc:
        assert "sum to 1.0" in str(exc), exc
    else:  # pragma: no cover
        raise AssertionError("fractions that do not sum to 1 must be rejected")


def test_class_weight_formula():
    df = _synthetic_frame((600, 200, 200, 200))  # n=1200, k=4
    weights = dl.compute_class_weights(df)
    n, k = len(df), 4
    for label, count in enumerate((600, 200, 200, 200)):
        assert math.isclose(weights[label], n / (k * count), rel_tol=1e-9), (label, weights)
    assert weights[0] < weights[1], "majority class must be down-weighted"


def test_split_file_roundtrip():
    """The saved split file must re-resolve to existing files."""
    if not has_dataset():
        print("      skip: real dataset not present")
        return
    with tempfile.TemporaryDirectory() as tmp:
        config = ProjectConfig(artifacts_dir=Path(tmp))
        root = dl.find_dataset_root(config)
        mapping, _ = dl.build_class_mapping(root)
        frames = dl.collect_images(root, mapping, config)
        frames = dl.stratified_split(frames, config)
        path = Path(tmp) / "results" / "splits.csv"
        dl.save_splits(frames, path, root)
        loaded = dl.load_splits(path, root)
        assert len(loaded) == len(frames)
        assert set(loaded["split"]) == {"train", "validation", "test"}
        assert all(Path(p).exists() for p in loaded["path"].head(50))
        assert dl.dataset_fingerprint(loaded) == dl.dataset_fingerprint(frames)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_metrics_match_hand_computed_values():
    from src.evaluate import classification_report_frame, metrics_from_predictions

    names = ["Healthy", "Common Rust", "Gray Leaf Spot", "Northern Leaf Blight"]
    #  truth:      H H H H C C C G G N
    y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    #  predict:    H H H C C C G G G N
    y_pred = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    m = metrics_from_predictions(y_true, y_pred, names)

    assert math.isclose(m["accuracy"], 8 / 10, rel_tol=1e-9), m["accuracy"]
    # Healthy: 3 correct of 3 predicted -> precision 1, recall 3/4
    assert math.isclose(m["per_class"]["Healthy"]["precision"], 1.0, rel_tol=1e-9)
    assert math.isclose(m["per_class"]["Healthy"]["recall"], 0.75, rel_tol=1e-9)
    f1_healthy = 2 * 1.0 * 0.75 / (1.0 + 0.75)
    assert math.isclose(m["per_class"]["Healthy"]["f1"], f1_healthy, rel_tol=1e-9)
    # Gray Leaf Spot: 3 predictions of which 2 correct -> P=2/3; both true
    # Gray Leaf Spot images were found -> R=1.0
    assert math.isclose(m["per_class"]["Gray Leaf Spot"]["precision"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(m["per_class"]["Gray Leaf Spot"]["recall"], 1.0, rel_tol=1e-9)
    assert m["per_class"]["Gray Leaf Spot"]["support"] == 2
    # confusion matrix orientation: rows true, columns predicted
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert cm[0, 1] == 1 and cm[1, 2] == 1 and cm[2, 2] == 2, cm
    assert cm.diagonal().sum() == 8
    assert math.isclose(m["balanced_accuracy"], float(np.mean([c / cm[i].sum() for i, c in enumerate(cm.diagonal())])), rel_tol=1e-9)
    frame = classification_report_frame(m)
    assert set(frame["class"]) == {*names, "macro avg", "weighted avg", "accuracy"}


def test_metrics_handle_a_class_missing_from_the_test_set():
    from src.evaluate import metrics_from_predictions

    names = ["Healthy", "Common Rust", "Gray Leaf Spot", "Northern Leaf Blight"]
    y_true, y_pred = [0, 0, 1], [0, 1, 1]
    m = metrics_from_predictions(y_true, y_pred, names)
    assert len(m["confusion_matrix"]) == 4  # never shrinks below k classes
    assert m["per_class"]["Northern Leaf Blight"]["support"] == 0
    assert m["per_class"]["Northern Leaf Blight"]["precision"] == 0.0


# --------------------------------------------------------------------------
# disease information
# --------------------------------------------------------------------------
def test_disease_information_covers_every_class():
    assert set(DISEASE_INFORMATION) == set(CLASS_ALIASES), set(DISEASE_INFORMATION)
    for key, info in DISEASE_INFORMATION.items():
        assert info["summary"] and info["description"], key
        assert DISPLAY_NAMES[key] == info["display_name"]
    from src.disease_info import information_for

    for name in DISPLAY_NAMES.values():
        assert information_for(name)["display_name"] == name
    # no treatment prescriptions are allowed in a non-expert tool
    banned = ("spray", "dose", "kg/ha", "apply ", "insecticide", "fungicide product")
    for info in DISEASE_INFORMATION.values():
        blob = " ".join(info.values()).lower()
        assert not any(word in blob for word in banned), blob[:120]


def test_selection_weights_sum_to_one():
    from src.compare import SCORE_WEIGHTS

    assert math.isclose(sum(SCORE_WEIGHTS.values()), 1.0, rel_tol=1e-9), SCORE_WEIGHTS


# --------------------------------------------------------------------------
# TensorFlow-dependent tests
# --------------------------------------------------------------------------
def test_preprocessing_is_not_shared_between_architectures():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print("      skip: TensorFlow not installed")
        return
    from src import models as mdl
    from src.preprocessing import model_expects_internal_preprocessing

    config = ProjectConfig()
    results = {}
    for key in ("mobilenetv2", "efficientnetb0"):
        model, plan = mdl.build_model(key, 4, config, weights=None)
        results[key] = plan.internal_preprocessing
        assert model_expects_internal_preprocessing(model) == plan.internal_preprocessing
        tf.keras.backend.clear_session()
    # the two architectures must NOT be preprocessed identically
    assert results["mobilenetv2"] != results["efficientnetb0"], results
    # and the external preprocessing for MobileNetV2 must really rescale
    from src.preprocessing import PreprocessingPlan, apply_preprocessing

    import tensorflow as tf2

    external = PreprocessingPlan("mobilenetv2", "mobilenet_v2", False)
    out = apply_preprocessing(tf2.ones((1, 4, 4, 3)) * 200.0, external)
    assert abs(float(out[0, 0, 0, 0]) - (200.0 - 127.5) / 127.5) < 1e-4, float(out[0, 0, 0, 0])
    internal = PreprocessingPlan("efficientnetb0", "efficientnet", True)
    untouched = apply_preprocessing(tf2.ones((1, 4, 4, 3)) * 200.0, internal)
    assert abs(float(untouched[0, 0, 0, 0]) - 200.0) < 1e-6, "internal mode must pass 0-255 through"


def test_augmentation_applies_to_training_only():
    if not has_dataset():
        print("      skip: real dataset not present")
        return
    try:
        import tensorflow as tf
    except ImportError:
        print("      skip: TensorFlow not installed")
        return
    from src import preprocessing as prep
    from src.models import build_model

    config = ProjectConfig().override(max_images_per_class=6, batch_size=3, image_size=96)
    root = dl.find_dataset_root(config)
    mapping, _ = dl.build_class_mapping(root)
    frames = dl.collect_images(root, mapping, config)
    model, plan = build_model("mobilenetv2", mapping.num_classes, config, weights=None)

    plain = prep.build_dataset(frames, plan, config, augment=False, shuffle=False)
    a, _ = next(iter(plain.take(1)))
    b, _ = next(iter(prep.build_dataset(frames, plan, config, augment=False, shuffle=False).take(1)))
    assert bool(tf.reduce_all(tf.equal(a, b))), "validation pipeline must be deterministic"

    # shuffle is disabled on BOTH so the only difference is augmentation itself
    aug = prep.build_dataset(frames, plan, config, augment=True, shuffle=False)
    c, yc = next(iter(aug.take(1)))
    d, _ = next(iter(prep.build_dataset(frames, plan, config, augment=True, shuffle=False).take(1)))
    assert not bool(tf.reduce_all(tf.equal(a, c))), "training pipeline must differ from the raw image"
    assert c.shape == a.shape, (tuple(c.shape), tuple(a.shape))
    # Bounds depend on the plan: architectures that normalise internally are fed
    # [0,255]; MobileNetV2 gets the external (x-127.5)/127.5 rescale -> [-1,1].
    lo, hi = (float(tf.reduce_min(c)), float(tf.reduce_max(c)))
    if plan.internal_preprocessing:
        assert -0.01 <= lo and hi <= 255.01, (lo, hi)
    else:
        assert -1.01 <= lo and hi <= 1.01, (lo, hi)
    assert -1.02 <= lo <= hi <= 256.0, (lo, hi)
    # augmentation must be random per pass, otherwise it adds no data
    assert not bool(tf.reduce_all(tf.equal(c, d))), "two augmented passes should differ"
    # labels must survive augmentation unchanged
    _, yb = next(iter(plain.take(1)))
    assert list(yc.numpy()) == list(yb.numpy()), "augmentation must not reorder or alter labels"
    tf.keras.backend.clear_session()


def test_inference_contract_on_a_real_image():
    """Predictor output format: class, confidence %, all four probabilities."""
    if not has_dataset():
        print("      skip: real dataset not present")
        return
    from src.predict import MaizeDiseasePredictor, PredictionError

    config = ProjectConfig()
    if not config.final_model_path.exists():
        print("      skip: no trained model at models/best_maize_disease_model.keras")
        return
    predictor = MaizeDiseasePredictor(config=config)
    root = dl.find_dataset_root(config)
    mapping, _ = dl.build_class_mapping(root)
    sample = sorted((root / mapping.relative_dirs["healthy"]).iterdir())[0]

    result = predictor.predict(sample)
    assert set(result.probabilities) == set(predictor.class_names)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4
    assert result.class_name == predictor.class_names[result.class_index]
    assert result.confidence_percent.endswith("%")
    assert 0.0 <= result.confidence <= 1.0
    # the confidence shown must be the probability of the predicted class
    assert math.isclose(
        result.confidence, result.probabilities[result.class_name], rel_tol=1e-9
    )
    text = result.pretty_text()
    assert text.startswith("Prediction: ") and "Confidence: " in text
    assert "agricultural professional" in text, "disclaimer must be shown with every prediction"

    # bad input handling: an unsupported extension must be rejected with a clear
    # message (before Pillow is even asked to guess).
    with tempfile.TemporaryDirectory() as tmp:
        impostor = Path(tmp) / "leaf.tiff"
        shutil.copyfile(sample, impostor)
        try:
            predictor.predict(impostor)
        except PredictionError as exc:
            assert "Unsupported file type" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected PredictionError for a .tiff file")
        empty = Path(tmp) / "empty.jpg"
        empty.write_bytes(b"")
        try:
            predictor.predict(empty)
        except PredictionError as exc:
            assert "empty file" in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected PredictionError for a 0-byte file")

    try:
        MaizeDiseasePredictor(model_path=Path("/nonexistent/model.keras"))
    except PredictionError as exc:
        assert "train the model first" in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError("a missing model must raise a helpful PredictionError")


# --------------------------------------------------------------------------
# project-level meta checks (the anti-fabrication rules)
# --------------------------------------------------------------------------
def test_no_hardcoded_dataset_counts_in_source():
    """The dataset size must be measured, so it may not appear in the code."""
    # The statistics a student might be tempted to copy from a paper or from a
    # previous run. They must never be baked into the code.
    forbidden = ("3852", "3,852", "1192", "1162", "985", "513")
    import re as _re

    offenders = []
    for path in sorted((PROJECT_ROOT / "src").glob("*.py")) + sorted(
        (PROJECT_ROOT / "app").glob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if _re.search(rf"(?<![0-9.]){_re.escape(token)}(?![0-9])", text):
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"hard-coded dataset statistics found -> {offenders}"


def test_readme_makes_no_invented_performance_claims():
    import re

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    # no "model achieved NN% accuracy" style claims
    pattern = re.compile(
        r"(achieves?|reached|obtained|accuracy of)\D{0,25}\d{2}(\.\d+)?\s?%", re.IGNORECASE
    )
    hits = [m.group(0) for m in pattern.finditer(readme)]
    assert not hits, f"README appears to state measured results: {hits}"
    assert "After training, results will be reported here" in readme


def test_notebook_is_valid_complete_and_clean():
    import ast
    import nbformat

    path = PROJECT_ROOT / "notebooks" / "maize_disease_transfer_learning.ipynb"
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)

    sections = [
        line
        for cell in nb.cells
        if cell.cell_type == "markdown"
        for line in cell.source.split("\n")
        if line.startswith("## Section ")
    ]
    numbers = sorted(int(s.split("—")[0].replace("## Section", "").strip()) for s in sections)
    assert numbers == list(range(1, 23)), f"expected sections 1..22, got {numbers}"

    code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 20, len(code_cells)
    for i, src in enumerate(code_cells):
        probe = "\n".join("!" + l if l.startswith(("!", "%")) else l for l in src.split("\n"))
        try:
            ast.parse(probe)
        except SyntaxError as exc:  # pragma: no cover
            raise AssertionError(f"notebook code cell {i} does not parse: {exc}") from exc
    joined = "\n".join(code_cells)
    for token in ("__KEY__", "__NAME__"):
        assert token not in joined, f"generator artefact '{token}' leaked into the notebook"
    import re as _re

    assert not _re.search(r"(?<![A-Za-z0-9_])cfg\.", joined), (
        "generator-only name 'cfg' leaked into the notebook"
    )
    # the notebook must not fabricate anything either
    for forbidden in ("3852", "3,852"):
        assert forbidden not in joined, "notebook must measure, not assert, dataset size"


def test_requirements_cover_every_imported_module():
    imports: set[str] = set()
    for path in list((PROJECT_ROOT / "src").glob("*.py")) + [(PROJECT_ROOT / "app" / "streamlit_app.py")]:
        for line in path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line.startswith(("import ", "from ")):
                module = line.split()[1].split(".")[0]
                if module == "src" or module in sys.stdlib_module_names:
                    continue
                imports.add(module)
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    mapping = {
        "tensorflow": "tensorflow",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
        "streamlit": "streamlit",
        "keras": "tensorflow",
    }
    missing = {mapping.get(m, m) for m in imports if mapping.get(m, m) not in text}
    assert not missing, f"imported but not declared in requirements.txt: {missing}"


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures: list[tuple[str, str]] = []
    print(f"running {len(tests)} checks\n")
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            failures.append((name, tb))
            print(f"FAIL  {name}")
    if failures:
        print("\n" + "=" * 70)
        for name, tb in failures:
            print(f"--- {name} ---\n{tb}")
        print(f"\n{len(failures)} of {len(tests)} checks FAILED")
        return 1
    print(f"\nall {len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
