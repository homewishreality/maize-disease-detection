"""Central configuration for the maize disease detection project.

Every hyper-parameter, path and reproducibility setting lives here so that the
experiment can be re-configured without editing the training code.  Nothing in
this module hard-codes an absolute machine path: all paths are resolved
relative to the project root, and can be overridden from the command line or
through environment variables (which is what Google Colab uses).

Author: undergraduate final-year project.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
ENV_DATA_DIR = "MAIZE_DATA_DIR"
ENV_ARTIFACTS_DIR = "MAIZE_ARTIFACTS_DIR"
ENV_MODELS_DIR = "MAIZE_MODELS_DIR"
DEFAULT_DATA_DIR: Path = Path(os.environ.get(ENV_DATA_DIR, PROJECT_ROOT / "data"))
# Environment variables are read *here*, when defaults are built, so that an
# explicit --data-dir / --artifacts-dir always wins over the environment.
DEFAULT_ARTIFACTS_DIR: Path = Path(os.environ.get(ENV_ARTIFACTS_DIR, PROJECT_ROOT / "outputs"))
DEFAULT_MODELS_DIR: Path = Path(os.environ.get(ENV_MODELS_DIR, PROJECT_ROOT / "models"))
#: Narrative reports (dataset report, comparison write-up) for thesis Chapters 3/4.
DEFAULT_REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Environment variables take precedence over the defaults.  This keeps the
# same code working on a laptop, a Linux server and a Colab runtime.
ENV_DATA_DIR = "MAIZE_DATA_DIR"  # noqa: E305
ENV_ARTIFACTS_DIR = "MAIZE_ARTIFACTS_DIR"
ENV_MODELS_DIR = "MAIZE_MODELS_DIR"


# --------------------------------------------------------------------------
# Label space
# --------------------------------------------------------------------------
#: Canonical class identifiers used throughout the project.  Directory names in
#: the various public PlantVillage mirrors are *mapped onto* these names (see
#: :data:`CLASS_ALIASES`), so the code never depends on an exact folder name.
CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "healthy": ("Corn_(maize)___healthy",),
    "common_rust": ("Corn_(maize)___Common_rust_",),
    "northern_leaf_blight": ("Corn_(maize)___Northern_Leaf_Blight",),
    # In the canonical PlantVillage release the gray-leaf-spot folder is named
    # "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"; some mirrors call it
    # "Corn_(maize)___Gray_leaf_spot".  Both are accepted.
    "gray_leaf_spot": (
        "Corn_(maize)___Gray_leaf_spot",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    ),
}

#: Human-readable labels used in reports, figures and the Streamlit UI.
DISPLAY_NAMES: dict[str, str] = {
    "healthy": "Healthy",
    "common_rust": "Common Rust",
    "northern_leaf_blight": "Northern Leaf Blight",
    "gray_leaf_spot": "Gray Leaf Spot",
}

#: Substrings used to recognise a maize class folder when the folder name does
#: not match any alias exactly.  Ordered, and deliberately mutually exclusive:
#: "blight" must be tested before "leaf_spot" so that Northern Leaf Blight is
#: never confused with Gray Leaf Spot.
FOLDER_KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("healthy", ("healthy",)),
    ("northern_leaf_blight", ("northern_leaf_blight", "northern", "blight")),
    ("common_rust", ("common_rust", "rust")),
    ("gray_leaf_spot", ("gray_leaf_spot", "grey_leaf_spot", "cercospora")),
)

#: Directory names that belong to maize but are outside the four-class scope of
#: this project.  They are reported to the user and then excluded, never
#: silently folded into another class.
KNOWN_OUT_OF_SCOPE: tuple[str, ...] = (
    "Corn_(maize)___Maydis_eye_spot",
    "Corn_(maize)___Tar_spot",
    "Corn_(maize)___Grey_leaf_spot_unverified",
)

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------
# Configuration object
# --------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    """All tunable experiment settings in one place.

    The defaults target the Google Colab (GPU) run described in the README.
    :meth:`override` is used by the CLIs to change single fields.
    """

    # ---- reproducibility -------------------------------------------------
    seed: int = 42
    deterministic_ops: bool = True

    # ---- dataset ---------------------------------------------------------
    data_dir: Path = DEFAULT_DATA_DIR
    #: Where figures/CSV/JSON artefacts are written.
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    #: Where trained weights and the class mapping are written.
    models_dir: Path = DEFAULT_MODELS_DIR
    #: Where the markdown narrative reports are written.
    reports_dir: Path = DEFAULT_REPORTS_DIR
    #: Sub-directories searched, in order, when ``data_dir`` is not itself the
    #: image folder root (e.g. it is the project ``data/`` folder).
    dataset_search_paths: tuple[str, ...] = (
        ".",
        "pv/raw/color",
        "PlantVillage/raw/color",
        "raw/color",
        "raw/color",
        "color",
        "dataset/raw/color",
        "PlantVillage/color",
        "maize",
    )
    #: Number of images whose header is opened to collect dimension/format
    #: statistics.  ``None`` means "inspect every image".
    inspection_limit: int | None = None

    # ---- splitting -------------------------------------------------------
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    #: Group-aware splitting keeps all photos of the same source leaf in one
    #: subset (see README, "Data leakage from repeated photographs").
    group_aware_split: bool = False

    # ---- images ----------------------------------------------------------
    image_size: int = 224
    batch_size: int = 32
    shuffle_buffer: int = 2000

    # ---- head architecture ----------------------------------------------
    dense_units: int = 128
    dropout_rate: float = 0.30
    l2_weight: float = 1e-4

    # ---- stage 1: feature extraction ------------------------------------
    feature_extraction_epochs: int = 15
    head_learning_rate: float = 1e-3

    # ---- stage 2: fine-tuning -------------------------------------------
    finetune_epochs: int = 15
    finetune_learning_rate: float = 1e-4
    #: Fraction of the *pretrained base* unfrozen in stage 2, counted from the
    #: top of the network.  0.25 for both models; documented in the README.
    finetune_unfreeze_fraction: float = 0.25

    # ---- callbacks -------------------------------------------------------
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    lr_reduce_min_lr: float = 1e-6

    # ---- tf.data memory behaviour ---------------------------------------
    #: ``None`` lets TensorFlow autotune (best on a Colab GPU box).  On a
    #: machine with <4 GB RAM, autotuned prefetch buffers can exceed the
    #: available memory, which shows up as the process simply being "Killed".
    #: ``--low-memory`` on the CLI sets these to small fixed values.
    tf_data_num_parallel_calls: int | None = None
    tf_data_prefetch: int | None = None

    # ---- misc ------------------------------------------------------------
    class_weights: bool = True
    low_confidence_threshold: float = 0.60
    #: Optional cap on images per class (smoke tests on small machines).
    max_images_per_class: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- utils
    def override(self, **kwargs: Any) -> "ProjectConfig":
        """Return a copy with the given fields replaced (ignoring ``None``)."""
        updates = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **updates)

    def model_artifact_path(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}_maize.keras"

    @property
    def final_model_path(self) -> Path:
        return self.models_dir / "best_maize_disease_model.keras"

    @property
    def class_map_path(self) -> Path:
        return self.models_dir / "class_names.json"

    @property
    def inference_config_path(self) -> Path:
        return self.models_dir / "inference_config.json"

    # ---- artefact locations (outputs/) ----------------------------------
    def figure(self, name: str) -> Path:
        return self.artifacts_dir / "figures" / name

    def result(self, name: str) -> Path:
        return self.artifacts_dir / "results" / name

    def prediction(self, name: str) -> Path:
        return self.artifacts_dir / "predictions" / name

    def report(self, name: str) -> Path:
        return self.reports_dir / name

    def ensure_directories(self) -> None:
        for sub in ("figures", "results", "predictions"):
            (self.artifacts_dir / sub).mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["data_dir"] = str(self.data_dir)
        out["artifacts_dir"] = str(self.artifacts_dir)
        out["models_dir"] = str(self.models_dir)
        out["reports_dir"] = str(self.reports_dir)
        out["split_fractions"] = {
            "train": self.train_fraction,
            "validation": self.val_fraction,
            "test": self.test_fraction,
        }
        out["classes"] = list(CLASS_ALIASES.keys())
        out["display_names"] = DISPLAY_NAMES
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser())
        # An explicit environment variable wins over the default, which lets
        # Colab redirect artefacts to Google Drive without CLI plumbing.
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir).expanduser())
        object.__setattr__(self, "models_dir", Path(self.models_dir).expanduser())
        object.__setattr__(self, "reports_dir", Path(self.reports_dir).expanduser())
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser())


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def seed_everything(seed: int, deterministic_ops: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow; enable deterministic ops if asked.

    Full bit-level determinism on a GPU is *not* guaranteed by TensorFlow
    (some cuDNN kernels are non-deterministic by design).  This is stated in
    the README rather than glossed over.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError as exc:  # pragma: no cover
        raise ImportError("NumPy is required: pip install numpy") from exc

    try:
        import tensorflow as tf
    except ImportError:
        return

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover - depends on TF build
            print(f"[config] op determinism unavailable: {exc}")


def configure_gpu_memory_growth() -> dict[str, Any]:
    """Enable GPU memory growth so a Colab restart is rarely needed.

    Returns a small report dict describing what was found.
    """
    report: dict[str, Any] = {"gpu_detected": False, "gpus": []}
    try:
        import tensorflow as tf
    except ImportError:
        report["error"] = "TensorFlow is not installed"
        return report

    gpus = tf.config.list_physical_devices("GPU")
    report["gpu_detected"] = bool(gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            report["gpus"].append({"name": gpu.name, "memory_growth": "enabled"})
        except RuntimeError as exc:  # already initialised
            report["gpus"].append({"name": gpu.name, "memory_growth": f"skipped: {exc}"})
    return report


def available_memory_mb() -> dict[str, Any]:
    """Best-effort free/total memory reading, used to warn before an OOM kill.

    Deliberately dependency-free: ``/proc/meminfo`` on Linux (including Colab),
    ``GlobalMemoryStatusEx`` on Windows, ``vm_stat`` on macOS.  Returns
    ``{"available_mb": None, ...}`` when nothing can be read, so callers never
    fail because the platform was unusual.
    """
    info: dict[str, Any] = {"available_mb": None, "total_mb": None, "source": None}
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            values: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                key, _, rest = line.partition(":")
                kb = rest.strip().split(" ")[0]
                if kb.isdigit():
                    values[key.strip()] = int(kb)
            for key in ("MemAvailable", "MemFree"):
                if key in values:
                    info["available_mb"] = round(values[key] / 1024.0, 1)
                    break
            if "MemTotal" in values:
                info["total_mb"] = round(values["MemTotal"] / 1024.0, 1)
            info["source"] = "/proc/meminfo"
            return info
        if sys.platform.startswith("win"):  # pragma: no cover - platform specific
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                info["available_mb"] = round(stat.ullAvailPhys / 1024**2, 1)
                info["total_mb"] = round(stat.ullTotalPhys / 1024**2, 1)
                info["source"] = "GlobalMemoryStatusEx"
                return info
    except Exception as exc:  # noqa: BLE001 - diagnostics only, never fatal
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def environment_report(config: ProjectConfig | None = None) -> dict[str, Any]:
    """Collect the run metadata that must appear in the thesis appendix.

    Only values that can actually be read from the machine are recorded; if a
    library is missing the value is ``None`` and a warning key is added.
    """
    report: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "tensorflow_version": None,
        "keras_version": None,
        "numpy_version": None,
        "pandas_version": None,
        "scikit_learn_version": None,
        "gpu_available": False,
        "gpu_names": [],
        "cpu_count": os.cpu_count(),
        "seed": config.seed if config else None,
        "image_size": config.image_size if config else None,
        "batch_size": config.batch_size if config else None,
        "timestamp_utc": None,
    }
    try:
        from datetime import datetime, timezone

        report["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    except Exception:  # pragma: no cover
        pass

    for key, module in (
        ("tensorflow_version", "tensorflow"),
        ("keras_version", "keras"),
        ("numpy_version", "numpy"),
        ("pandas_version", "pandas"),
        ("scikit_learn_version", "sklearn"),
    ):
        try:
            mod = __import__(module)
            report[key] = mod.__version__
        except ImportError:
            report[key] = None

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        report["gpu_available"] = bool(gpus)
        report["gpu_names"] = [g.name for g in gpus]
    except ImportError:
        report["warnings"] = report.get("warnings", []) + ["TensorFlow not importable"]
    return report


def write_run_provenance(config: ProjectConfig, note: str | None = None) -> Path:
    """Persist config + environment so a run can be identified later.

    This is what prevents a student from accidentally reporting a smoke-test
    run as if it were the real experiment: the file records epochs and device.
    """
    payload = {"config": config.to_dict(), "environment": environment_report(config)}
    if note:
        payload["run_note"] = note
    path = config.result("run_provenance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON file with a friendly error message."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            "Run the training pipeline first (python -m src.train ...) or the "
            "Google Colab notebook; these metadata files are produced by it."
        )
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"File {path} is not valid JSON: {exc}") from exc
