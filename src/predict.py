"""Single-image inference pipeline (command line) and the API used by Streamlit.

The pipeline is the documented nine steps:

1. load image 2. validate image 3. resize 4. apply the model's own
preprocessing 5. run the prediction 6. read class probabilities 7. identify the
predicted class 8. report confidence 9. return all probabilities.

Steps 3 and 4 reuse the *same* functions as training (:mod:`src.preprocessing`),
so training and deployment cannot drift apart.

Usage::

    python -m src.predict --image path/to/leaf.jpg
    python -m src.predict --image a.jpg --image b.png --json
    python -m src.predict --dir some/folder --limit 20 --csv outputs/predictions/sample_predictions.csv
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ProjectConfig, load_json
from .disease_info import DISCLAIMER, LOW_CONFIDENCE_MESSAGE, information_for

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_USEFUL_EDGE = 48  # below this the resize step is pure guesswork
LOW_RESOLUTION_EDGE = 112  # half the model input: warn, but still try


class PredictionError(RuntimeError):
    """User-facing inference failure with an actionable message."""


@dataclass
class PredictionResult:
    """One prediction, with everything the UI and the CSV export need."""

    class_name: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]
    image_source: str = ""
    low_confidence: bool = False
    warnings: list[str] = field(default_factory=list)
    inference_ms: float | None = None
    model_name: str = ""
    image_size: int | None = None
    original_size: tuple[int, int] | None = None

    @property
    def confidence_percent(self) -> str:
        """Confidence as a percentage string, e.g. ``'96.42%'``."""
        return f"{self.confidence * 100.0:.2f}%"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image_source,
            "predicted_class": self.class_name,
            "predicted_index": self.class_index,
            "confidence_percent": round(self.confidence * 100.0, 2),
            "low_confidence": self.low_confidence,
            "warnings": "; ".join(self.warnings),
            "inference_ms": None if self.inference_ms is None else round(self.inference_ms, 2),
            "model": self.model_name,
            **{f"prob_{name}": round(prob * 100.0, 2) for name, prob in self.probabilities.items()},
        }

    def pretty_text(self) -> str:
        lines = [
            f"Prediction: {self.class_name}",
            f"Confidence: {self.confidence_percent}",
            "",
            "Class probabilities:",
        ]
        width = max(len(n) for n in self.probabilities)
        for name, prob in sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {name:<{width}}  {prob * 100.0:6.2f}%")
        if self.inference_ms is not None:
            lines.append(f"\nInference time: {self.inference_ms:.1f} ms")
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        lines.append("")
        lines.append(f"About '{self.class_name}': {information_for(self.class_name)['summary']}")
        lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)


class MaizeDiseasePredictor:
    """Load the exported model once, then classify images.

    Parameters
    ----------
    model_path, inference_config_path, class_map_path:
        Default to ``models/best_maize_disease_model.keras``,
        ``models/inference_config.json`` and ``models/class_names.json``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        inference_config_path: str | Path | None = None,
        class_map_path: str | Path | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.model_path = Path(model_path) if model_path else self.config.final_model_path
        self.inference_config_path = (
            Path(inference_config_path) if inference_config_path else self.config.inference_config_path
        )
        self.class_map_path = Path(class_map_path) if class_map_path else self.config.class_map_path

        if not self.model_path.exists():
            raise PredictionError(
                f"No trained model found at {self.model_path}.\n"
                "Please train the model first: run the Google Colab notebook, or\n"
                "    python -m src.train --model mobilenetv2\n"
                "    python -m src.compare\n"
                "The second command writes models/best_maize_disease_model.keras."
            )

        self.inference_config: dict[str, Any] = {}
        if self.inference_config_path.exists():
            self.inference_config = load_json(self.inference_config_path)
        else:
            print(
                f"[predict] {self.inference_config_path} not found; reading class names from "
                f"{self.class_map_path.name} and detecting preprocessing at runtime."
            )
        if self.class_map_path.exists():
            self.class_names = list(load_json(self.class_map_path).get("class_names") or [])
        else:
            self.class_names = []
        if not self.class_names:
            self.class_names = list(self.inference_config.get("classes") or [])
        if not self.class_names:
            raise PredictionError(
                f"Could not determine the class order: neither {self.class_map_path} nor "
                f"{self.inference_config_path} lists class names. Re-run 'python -m src.compare'."
            )

        self.image_size = int(self.inference_config.get("image_size") or self.config.image_size)
        self.low_confidence_threshold = float(
            self.inference_config.get("low_confidence_threshold") or self.config.low_confidence_threshold
        )
        self.model_name = str(self.inference_config.get("display_name") or self.model_path.stem)
        self._model, self._plan = self._load_model()

    # ---------------------------------------------------------------- setup
    def _load_model(self):
        from .evaluate import load_trained_model

        model, plan = load_trained_model(
            self.model_path,
            self.inference_config_path if self.inference_config_path.exists() else None,
            self.config,
        )
        if plan.model_name in ("", "unknown"):
            plan = type(plan)(plan.model_name, plan.module_name, plan.internal_preprocessing)
        return model, plan

    # ------------------------------------------------------------ decoding
    def _preprocess_pil(self, image):
        """Resize + model-specific preprocessing for a PIL image -> (1, S, S, 3)."""
        import numpy as np
        import tensorflow as tf

        from .preprocessing import apply_preprocessing

        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype="float32")  # [0, 255]
        tensor = tf.convert_to_tensor(array[tf.newaxis, ...])
        return apply_preprocessing(tensor, self._plan)

    def _validate_pil(self, image, source: str):
        """Structural checks + quality warnings (returns the image, may raise)."""
        from PIL import ImageStat

        if image is None:
            raise PredictionError(
                f"{source}: the file could not be opened as an image by Pillow. "
                "Supported formats are JPG, JPEG and PNG."
            )
        if getattr(image, "format", None) and image.format.upper() not in {"JPEG", "PNG"}:
            print(
                f"[predict] note: '{image.format}' input converted to RGB; "
                "JPG/PNG are the documented formats."
            )
        width, height = image.size
        if width < MIN_USEFUL_EDGE or height < MIN_USEFUL_EDGE:
            raise PredictionError(
                f"{source}: image is only {width}x{height} px. Provide a photo where the leaf "
                f"fills the frame (at least {MIN_USEFUL_EDGE}px on each side; 256px or more is better)."
            )
        warnings: list[str] = []
        if min(width, height) < LOW_RESOLUTION_EDGE:
            warnings.append(
                f"Low resolution ({width}x{height} px): the image has to be enlarged to "
                f"{self.image_size}x{self.image_size} for the model, which can hide fine symptoms "
                "such as young rust pustules."
            )
        if min(width, height) > 4 * self.image_size:
            warnings.append(
                f"Very large image ({width}x{height} px): it is downscaled to "
                f"{self.image_size}x{self.image_size}, so small lesions may disappear."
            )
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        if max(stat.stddev) < 12.0:
            warnings.append(
                "The image has almost no colour variation; it may be blank, silhouetted or out of focus."
            )
        return rgb, warnings

    # ------------------------------------------------------------- predict
    def predict(self, source: str | Path | bytes | bytearray | "Any") -> PredictionResult:  # noqa: F821
        """Classify one image given as a path, raw bytes or a PIL image."""
        import numpy as np
        from PIL import Image, UnidentifiedImageError

        warnings: list[str] = []
        label = ""
        if isinstance(source, (str, Path)):
            path = Path(source)
            label = str(path)
            if not path.exists():
                raise PredictionError(f"Image file not found: {path}")
            if path.is_dir():
                raise PredictionError(f"{path} is a directory; pass an image file (or use --dir).")
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                raise PredictionError(
                    f"Unsupported file type '{path.suffix}' for {path.name}.\n"
                    f"Upload one of: {', '.join(sorted(ALLOWED_SUFFIXES))}."
                )
            size_bytes = path.stat().st_size
            if size_bytes == 0:
                raise PredictionError(f"{path} is an empty file (0 bytes); the download or copy failed.")
            try:
                with Image.open(path) as im:
                    im.load()
                    image = im.copy()
            except (UnidentifiedImageError, OSError) as exc:
                raise PredictionError(
                    f"{path} could not be decoded as an image ({type(exc).__name__}: {exc}). "
                    "The file may be corrupted or truncated."
                ) from exc
        elif isinstance(source, (bytes, bytearray)):
            label = "<bytes>"
            if len(source) == 0:
                raise PredictionError("Received 0 bytes of image data.")
            try:
                with Image.open(io.BytesIO(source)) as im:
                    im.load()
                    image = im.copy()
            except (UnidentifiedImageError, OSError) as exc:
                raise PredictionError(
                    f"The uploaded data is not a readable image ({type(exc).__name__}: {exc})."
                ) from exc
        elif hasattr(source, "size") and hasattr(source, "convert"):  # PIL.Image.Image
            image = source
            label = "<PIL.Image>"
        else:
            raise PredictionError(f"Cannot predict from object of type {type(source).__name__}.")

        original_size = tuple(int(v) for v in image.size)
        image, quality_warnings = self._validate_pil(image, label)
        warnings.extend(quality_warnings)

        prepared = self._preprocess_pil(image)
        t0 = time.perf_counter()
        raw = self._model.predict(prepared, verbose=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        probabilities = np.asarray(raw).ravel()
        if probabilities.size != len(self.class_names):
            raise PredictionError(
                f"The model returned {probabilities.size} scores but {len(self.class_names)} class "
                "names are configured; models/class_names.json does not match the weights."
            )
        # Softmax output should already sum to 1; normalise defensively so the
        # percentages shown to a user are internally consistent.
        total = float(probabilities.sum())
        if total > 0 and abs(total - 1.0) > 1e-3:
            probabilities = probabilities / total
            warnings.append("Probabilities were re-normalised to sum to 100%.")

        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        class_name = self.class_names[index]
        low = confidence < self.low_confidence_threshold
        if low:
            warnings.append(LOW_CONFIDENCE_MESSAGE)
        margin = float(np.sort(probabilities)[-1] - np.sort(probabilities)[-2])
        if margin < 0.05:
            warnings.append(
                f"The top two classes are only {margin * 100:.1f} percentage points apart; the visual "
                "evidence is ambiguous between them."
            )
        return PredictionResult(
            class_name=class_name,
            class_index=index,
            confidence=confidence,
            probabilities={n: float(p) for n, p in zip(self.class_names, probabilities)},
            image_source=label,
            low_confidence=low,
            warnings=warnings,
            inference_ms=elapsed_ms,
            model_name=self.model_name,
            image_size=self.image_size,
            original_size=original_size,
        )

    # convenience alias
    predict_path = predict

    def batch_report(self, paths: list[Path], limit: int | None = None) -> "Any":
        """Predict several images and return a pandas DataFrame (for the report)."""
        import pandas as pd

        rows = []
        for path in (paths if limit is None else paths[:limit]):
            try:
                rows.append(self.predict(path).to_dict())
            except PredictionError as exc:
                rows.append({"image": str(path), "error": str(exc)})
        return pd.DataFrame(rows)


def _collect_image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        raise PredictionError(f"Directory not found: {directory}")
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in ALLOWED_SUFFIXES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.predict", description=__doc__)
    parser.add_argument("--image", action="append", default=[], help="One or more image files")
    parser.add_argument("--dir", default=None, help="Classify every JPG/PNG under this folder")
    parser.add_argument("--limit", type=int, default=10, help="Max images when using --dir")
    parser.add_argument("--csv", default=None, help="Write a prediction CSV (e.g. outputs/predictions/sample_predictions.csv)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--model", default=None, help="Override the .keras path")
    args = parser.parse_args(argv)

    paths: list[Path] = [Path(p) for p in args.image]
    if args.dir:
        paths += _collect_image_paths(Path(args.dir))
    if not paths:
        parser.error("Provide --image <file> or --dir <folder>.")

    try:
        predictor = MaizeDiseasePredictor(model_path=args.model)
    except PredictionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results = []
    for path in paths:
        try:
            result = predictor.predict(path)
        except PredictionError as exc:
            print(f"ERROR: {exc}\n", file=sys.stderr)
            continue
        results.append(result)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.pretty_text())
            print("-" * 60)

    if args.csv and results:
        import pandas as pd

        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([r.to_dict() for r in results]).to_csv(out, index=False)
        print(f"wrote {len(results)} prediction rows -> {out}")
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
