# `models/` — trained weights (generated, not committed)

Written by `python -m src.train` and `python -m src.compare`:

| File | Written by | Contents |
| --- | --- | --- |
| `mobilenetv2_maize.keras` | `src.train` | Keras 3 archive: MobileNetV2 transfer-learning model |
| `efficientnetb0_maize.keras` | `src.train` | Keras 3 archive: EfficientNetB0 transfer-learning model |
| `best_maize_disease_model.keras` | `src.compare` | copy of the **selected** model — the one the Streamlit app loads |
| `class_names.json` | `src.compare` (also `src.inspect_dataset`) | class index ↔ name mapping, i.e. the model's output order |
| `inference_config.json` | `src.compare` | image size, preprocessing decision, low-confidence threshold, metrics, versions |

The two JSON files are what make the weights usable on their own: no prediction code depends on
leftover notebook state. If you retrain, all of them are rewritten.

Kept out of git because they are tens of megabytes; in Colab, download them from the zip produced by the
last notebook section, or copy `models/` to Drive (`MOUNT_DRIVE = True`).

To regenerate just the class mapping without training, run `python -m src.inspect_dataset`.
