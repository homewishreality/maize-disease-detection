"""Maize Disease Detection System -- Streamlit front end.

Run from the project root::

    streamlit run app/streamlit_app.py

The page does exactly four things: take an image, show the model's prediction
with all four class probabilities, explain what the predicted class means, and
state the limits of the result.  All model logic lives in :mod:`src.predict`,
so this file contains no ML code of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project's ``src`` package importable no matter where Streamlit is
# started from (project root, app/ folder, or a Colab mount).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig  # noqa: E402
from src.disease_info import (  # noqa: E402
    DISEASE_INFORMATION,
    DISCLAIMER,
    SCOPE_WARNING,
    information_for,
)
from src.predict import MaizeDiseasePredictor, PredictionError  # noqa: E402

st.set_page_config(
    page_title="Maize Disease Detection System",
    page_icon="🌽",
    layout="centered",
)

#: Models live in <project>/models; override with MAIZE_MODELS_DIR (see src/config.py).
MODEL_OVERRIDE = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource(show_spinner="Loading the trained model ...")
def get_predictor():
    """Load the exported model once per session (cached by Streamlit)."""
    config = ProjectConfig()
    models_dir = MODEL_OVERRIDE if (MODEL_OVERRIDE / "best_maize_disease_model.keras").exists() else config.models_dir
    return MaizeDiseasePredictor(
        model_path=models_dir / "best_maize_disease_model.keras",
        inference_config_path=models_dir / "inference_config.json",
        class_map_path=models_dir / "class_names.json",
        config=config,
    )


# ---------------------------------------------------------------- header
st.title("🌽 Maize Disease Detection System")
st.caption(
    "Deep-learning image classification for four maize leaf conditions, using "
    "transfer learning (MobileNetV2 / EfficientNetB0) on the PlantVillage "
    "maize subset."
)

# ------------------------------------------------------------- model status
predictor = None
load_error: str | None = None
try:
    predictor = get_predictor()
except Exception as exc:  # noqa: BLE001  - never crash the page on load
    load_error = f"{type(exc).__name__}: {exc}"

if predictor is None:
    st.error(
        "**No usable model found.**\n\n"
        f"{load_error}\n\n"
        "Train the model (see the README, section *Training*) or copy the "
        "`models/` folder produced by the Google Colab notebook next to this app."
    )
    st.stop()

# ------------------------------------------------------------------ upload
st.subheader("1. Upload a maize leaf image")
uploaded = st.file_uploader(
    "Accepted formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    key="leaf_image",
    help="For best results: one leaf, in focus, filling most of the frame, "
    "daylight, plain or blurred background.",
)

if uploaded is None:
    st.info("Choose an image to begin. Nothing is uploaded to a server; inference runs locally.")
    with st.expander("What the four classes are", expanded=False):
        for key, info in DISEASE_INFORMATION.items():
            st.markdown(f"**{info['display_name']}** — {info['summary']}")
    with st.expander("Model information"):
        st.json(
            {
                "model": predictor.model_name,
                "image_size": predictor.image_size,
                "classes": predictor.class_names,
                "low_confidence_threshold": predictor.low_confidence_threshold,
                "weights_file": str(predictor.model_path),
            }
        )
    st.caption(DISCLAIMER)
    st.stop()

image_bytes = uploaded.getvalue()

left, right = st.columns([1, 1])
with left:
    st.image(image_bytes, caption=f"Uploaded: {uploaded.name}")
    st.caption(f"Decoded size: {len(image_bytes) / 1024:.0f} KB")
with right:
    st.subheader("2. Prediction")
    with st.spinner("Running the model ..."):
        try:
            result = predictor.predict(image_bytes)
        except PredictionError as exc:
            st.error(f"Could not process the image.\n\n{exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.stop()

    st.markdown(f"### {result.class_name}")
    st.metric(
        "Model confidence",
        result.confidence_percent,
        help="Softmax probability of the predicted class - a model estimate, not certainty.",
    )
    if result.low_confidence:
        st.warning(
            "Low confidence prediction. Consider providing a clearer image or "
            "consulting an agricultural expert.",
            icon="⚠️",
        )
    for warning in result.warnings:
        if not warning.startswith("Low-confidence"):
            st.caption(f"ℹ️ {warning}")

with right:
    st.subheader("3. Class probabilities")
    st.caption("Model estimates for all four classes (they sum to 100%).")
    for name in sorted(result.probabilities, key=lambda c: -result.probabilities[c]):
        value = result.probabilities[name] * 100.0
        st.progress(min(max(value / 100.0, 0.0), 1.0), text=f"{name}: {value:.2f}%")

st.subheader("4. About the predicted class")
info = information_for(result.class_name)
st.markdown(f"**{info['display_name']}** — *{info['summary']}*")
st.write(info["description"])
if info.get("signs"):
    st.caption(f"Typical signs: {info['signs']}")
st.caption(
    "Management decisions (resistant hybrids, crop protection, thresholds) must "
    "come from local extension guidance; this tool does not advise on treatment."
)

# ----------------------------------------------------------- diagnostics
with st.expander("Diagnostics and model metadata", expanded=False):
    st.write(
        {
            "predicted index": result.class_index,
            "inference time (ms)": None if result.inference_ms is None else round(result.inference_ms, 1),
            "uploaded size (px)": result.original_size,
            "model input size (px)": result.image_size,
            "weights": predictor.model_path.name,
        }
    )
    meta = predictor.inference_config or {}
    if meta:
        st.markdown("**Training / evaluation summary**")
        st.json(
            {
                "model": meta.get("display_name"),
                "test_metrics": meta.get("test_metrics"),
                "trained_on": meta.get("trained_on"),
                "preprocessing": (meta.get("preprocessing") or {}).get("description"),
                "environment": {
                    k: (meta.get("environment") or {}).get(k)
                    for k in ("tensorflow_version", "python_version", "gpu_available", "timestamp_utc")
                },
            }
        )
    st.markdown("**Scope of the result**")
    st.caption(SCOPE_WARNING)

st.divider()
st.caption(f"⚠️ {DISCLAIMER}")
