# Development of a Maize Disease Detection System for Food Security Using Deep Learning Techniques

An undergraduate final-year project: an image-classification system that takes a photograph of a
maize (corn) leaf and predicts one of four conditions using **transfer learning**, packaged with a
reproducible Colab notebook, a command-line pipeline and a Streamlit web application.

> **Read this first — what is and is not in this repository.**
> Every line of code here is real and executable, and the dataset inspection stage has been run
> against the actual PlantVillage maize images. The *model results* (accuracy, precision, recall,
> F1, confusion matrices, training times) are **not** pre-filled anywhere in this README: they are
> produced by running the training pipeline, and they are written automatically into
> `outputs/results/` and `reports/`. Placeholders are used instead of invented numbers.

---

## 1. Project description

Maize is a staple crop for hundreds of millions of people, and yield losses from foliar diseases are
one of the main threats to food security for smallholder growers, who often have limited access to a
plant clinic or extension officer. This project investigates whether a convolutional neural network
built with **transfer learning** can classify maize leaf photographs into four categories well enough
to act as a *screening aid* in front of an extension service — not as a replacement for one.

Instead of training a CNN from scratch (which needs far more data than an undergraduate project can
collect), the system starts from ImageNet-pretrained weights and trains only a small classification
head, then fine-tunes the top of the network. Two backbones are compared under identical conditions so
that the choice of architecture is an *empirical* result rather than an assumption:

* **MobileNetV2** — small, fast, suitable for low-power devices.
* **EfficientNetB0** — larger, more accurate on many transfer-learning tasks.

## 2. Research objectives

1. To build a reproducible deep-learning pipeline that classifies maize leaf images into
   *Healthy*, *Common Rust*, *Northern Leaf Blight* and *Gray Leaf Spot*.
2. To apply and compare two transfer-learning architectures (MobileNetV2, EfficientNetB0) under an
   identical data split, augmentation and evaluation protocol.
3. To measure performance with metrics appropriate to an imbalanced dataset — per-class precision,
   recall, F1, macro/weighted averages and the confusion matrix — rather than accuracy alone.
4. To select the final model with a documented criterion that includes generalisation, model size and
   inference cost, not just test accuracy.
5. To deliver a usable inference interface (CLI + Streamlit) that reports confidence as an estimate
   and points the user to professional advice.

A deliberately **non**-objective: claiming field-ready diagnostic accuracy. See §14 Limitations.

## 3. Features

- Robust dataset discovery: class folders are *found and counted*, never assumed; folder-name
  variations across PlantVillage mirrors are handled by an alias/keyword mapping.
- Integrity checking: every image is opened with Pillow; corrupted/truncated files are listed in a CSV
  and excluded from training (with the count reported).
- Stratified 70 / 15 / 15 train-validation-test split performed on the **original files, before any
  augmentation**, so an augmented copy can never leak between subsets; disjointness and proportions
  are asserted in code, and the split is saved to CSV so both models train on exactly the same images.
- Architecture-specific preprocessing, determined by inspecting the built model (see §8).
- Two-stage transfer learning with `EarlyStopping`, `ModelCheckpoint` and `ReduceLROnPlateau`;
  the best epoch is chosen on **validation** loss only.
- Balanced class weights (the dataset is imbalanced ~2.3:1) instead of oversampling.
- Full artefact export for thesis Chapters 3–4: figures, CSVs, JSON metadata, markdown reports.
- Model selection with a documented composite score and a significance sanity-check.
- Single-image inference with validation, confidence, all four probabilities, and low-confidence flags.
- Streamlit app with image upload, preview, prediction, probability bars, disease information,
  disclaimer, and a diagnostics panel.

## 4. Dataset

**PlantVillage maize (corn) subset** — expert-verified leaf photographs released with

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant
> Disease Detection.* Frontiers in Plant Science, 7:1419. <https://doi.org/10.3389/fpls.2016.01419>

The images were taken at US research stations (Penn State, Florida, Cornell): a single leaf was
detached, placed on a uniform grey/black background and photographed outdoors with a compact camera in
automatic mode. Only leaves with a confirmed diagnosis were included.

**Why it is suitable here:** it is public, licence-clear for academic use, label-verified, large enough
for transfer learning, and it is the standard benchmark in this research area, which makes results
comparable to published work. **Why it must be qualified:** the images are controlled single-leaf
photographs. They do **not** represent whole-field appearance, early-stage symptoms, mixed diseases,
nutrient disorders, pest damage, variable backgrounds or camera quality. See §14.

### Dataset classes

| Canonical key | PlantVillage folder name | Meaning |
| --- | --- | --- |
| `healthy` | `Corn_(maize)___healthy` | no visible disease |
| `common_rust` | `Corn_(maize)___Common_rust_` | *Puccinia sorghi* rust pustules |
| `northern_leaf_blight` | `Corn_(maize)___Northern_Leaf_Blight` | *Exserohilum turcicum* cigar-shaped lesions |
| `gray_leaf_spot` | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` (some mirrors: `Corn_(maize)___Gray_leaf_spot`) | *Cercospora zeae-maydis* rectangular, vein-bounded lesions |

The exact number of images **per class is measured by the code**, not quoted from a paper. Run:

```bash
python -m src.inspect_dataset
```

It prints and saves the totals, per-class counts, image dimensions, formats, corrupted-file count and
the resulting split sizes (`outputs/results/dataset_summary.json`, `outputs/figures/class_distribution.png`).
If your copy contains additional maize folders outside these four classes (newer PlantVillage releases
added e.g. *Maydis eye spot* and *Tar spot*), they are reported as *out of scope and excluded* — never
silently merged into another class.

## 5. Project structure

```
maize-disease-detection/
├── README.md                     ← this file
├── requirements.txt
├── .gitignore
├── LICENSE                       ← code licence (MIT); the dataset licence is separate, see §23
├── data/                         ← dataset (downloaded; not in git)
│   ├── README.md                    where to put it, licence, alternatives
│   └── dataset_provenance.json      written by the setup step: repo, commit, counts, verification
├── notebooks/
│   ├── maize_disease_transfer_learning.ipynb   ← the Colab notebook (22 sections)
│   └── build_notebook.py           ← generator that rebuilds that .ipynb (see §19)
├── src/
│   ├── __init__.py
│   ├── config.py                 ← ALL hyper-parameters, paths, seed, environment report
│   ├── dataset_setup.py          ← download + verify the maize subset (git sparse clone)
│   ├── data_loader.py            ← discovery, class mapping, validation, stratified split
│   ├── preprocessing.py          ← decode/resize, per-model preprocessing, augmentation
│   ├── models.py                 ← MobileNetV2 / EfficientNetB0 heads, freezing, callbacks
│   ├── train.py                  ← two-stage training driver (CLI)
│   ├── evaluate.py               ← test-set metrics, reports, model loading (CLI)
│   ├── compare.py                ← comparison table, selection, final export (CLI)
│   ├── predict.py                ← inference pipeline + MaizeDiseasePredictor (CLI)
│   ├── inspect_dataset.py        ← dataset inspection stage (CLI)
│   ├── visualize.py              ← all matplotlib figures
│   └── disease_info.py           ← static, non-prescriptive disease descriptions
├── app/
│   └── streamlit_app.py          ← web UI
├── models/                       ← saved weights + class mapping (generated; not in git)
│   └── README.md
├── outputs/                      ← figures, CSVs, JSON metrics (generated; not in git)
│   └── README.md                 ← what every generated file contains
├── reports/                      ← generated markdown write-ups for Chapters 3–4
│   └── README.md
└── tests/
    └── test_pipeline.py          ← fast checks: mapping, split integrity, metric maths, inference
```

Design note: the notebook imports the modules in `src/` instead of duplicating them, so the Colab
experiment and the local command-line experiment cannot drift apart.

## 6. Technologies

Python 3 · TensorFlow / Keras · NumPy · pandas · Matplotlib · scikit-learn · Pillow · Streamlit ·
Google Colab (T4 GPU optional) · git (only for the dataset download).

Deliberately absent: Flask/FastAPI, React/Node, Docker, databases, YOLO, object detection, custom CUDA.
Classification of a single leaf is the task; the simpler the stack, the more of it you can defend.

## 7. Installation (local machine)

```bash
# 1. get the code
git clone <your-repository-url> maize-disease-detection
cd maize-disease-detection

# 2. create an environment (Python 3.10 - 3.13)
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install (no dataset needed for this step):

```bash
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

On a machine without an NVIDIA GPU, install the CPU build (`pip install tensorflow-cpu`); it works,
just far more slowly (§10 note on hardware).

## 8. About preprocessing (a detail that matters)

`tensorflow.keras.applications.<model>.preprocess_input` is **not** interchangeable between
architectures, and in Keras 3 it is not even the same kind of thing:

* `mobilenet_v2.preprocess_input` really rescales (`(x-127.5)/127.5`), and the `MobileNetV2` body has
  no preprocessing layers → the caller **must** apply it.
* `efficientnet.preprocess_input` is a documented **no-op** in Keras 3, because `EfficientNetB0`
  already contains `Rescaling → Normalization → Rescaling` internally → the pipeline must feed
  `[0, 255]` images and let the model normalise them. Applying an extra `(x-127.5)/127.5` here would
  double-normalise and silently degrade training.

`src/preprocessing.py` therefore *inspects the built model* and adapts
(`model_expects_internal_preprocessing`). The decision actually taken is printed during training and
recorded in `outputs/results/preprocessing_<model>.json`, `train_meta_<model>.json` and
`models/inference_config.json`, so the thesis can state exactly what each model was fed.
Inference reuses the same function, so training and deployment cannot disagree.

## 9. Dataset setup

Option A — automatic (recommended, and what the notebook does):

```bash
python -m src.dataset_setup --dest data
```

It performs a **blobless sparse git clone** of the dataset authors' own repository
(`https://github.com/spMohanty/PlantVillage-Dataset`), checking out only
`raw/color/Corn_(maize)___*`, then counts what arrived and writes
`data/dataset_provenance.json` (repository, commit hash, commit date). No scraping, no Kaggle
credentials. See `data/README.md` for manual alternatives and licence notes.

Option B — you already have the images: no renaming is required. Point the pipeline at the folder that
contains the four class directories:

```bash
python -m src.inspect_dataset --data-dir /path/to/PlantVillage/raw/color
```

Then inspect (mandatory step — it is also what produces the split file used by training):

```bash
python -m src.inspect_dataset
```

## 10. Training

### In Google Colab (recommended)

1. Upload/push this project to GitHub or Google Drive, open `notebooks/maize_disease_transfer_learning.ipynb`
   in Colab (`File > Open notebook > GitHub/Drive`).
2. `Runtime > Change runtime type > T4 GPU` (Free tier is enough).
3. Run all cells top to bottom. Sections 1–22 are labelled in the notebook; the dataset is downloaded
   automatically, inspected, split, both models trained (feature extraction then fine-tuning),
   evaluated, compared, and the winner exported to `models/best_maize_disease_model.keras`.
4. The last cell zips `models/`, `outputs/` and `reports/` to
   `/content/maize_results_models.zip`, `..._outputs.zip` and `..._reports.zip` (outside Colab, next to the
   project folder). Download those — or any single file from the Files sidebar — and keep them with your
   thesis: they *are* your experimental record.

Typical GPU cost with the default configuration (15 + 15 epochs, batch 32, ~2.7k training images):
tens of minutes per model. The notebook prints the measured wall-clock time per stage; do not reuse
someone else's figure.

> **Measured footprint (this project's own runs, CPU, batch 16, 224x224):** importing
> TensorFlow alone is ~0.66 GB RSS, and one model's full two-stage training peaks at ~1.27 GB
> (MobileNetV2) / ~1.36 GB (EfficientNetB0) even on a 160-image demo. Budget **~4 GB free RAM per
> model in one process** (more at batch 32/64, or when several models are trained inside one long-lived
> kernel such as a notebook). The trainer records `peak_memory_mb` in `train_meta_<model>.json`, so you
> can quote your own number instead of this one.
>
> Hardware reality check: a CPU-only machine with ~2 GB RAM can *run* the pipeline but not train a
> 224×224 network over thousands of images in reasonable time — we hit exactly that while building
> this project (the process is killed with exit code 137 when memory runs out). The trainer detects low
> available memory, shrinks `tf.data` prefetch/parallelism and batch size, and prints advice;
> `--low-memory` forces that mode, and `--max-images-per-class 400` makes a smoke test fast.

### Locally (CLI, same code as the notebook)

```bash
# both models, one after the other, then comparison + export
python -m src.train --model both

# or individually, with explicit control
python -m src.train --model mobilenetv2    --epochs 15 --finetune-epochs 15
python -m src.train --model efficientnetb0 --epochs 15 --finetune-epochs 15
python -m src.compare
```

Useful flags (all also available through `src/config.py`):

| Flag | Meaning |
| --- | --- |
| `--epochs N` / `--finetune-epochs N` | stage 1 / stage 2 epochs |
| `--batch-size N`, `--image-size N` | throughput and input resolution |
| `--head-lr`, `--finetune-lr` | learning rates (stage 1 / stage 2) |
| `--patience N` | `EarlyStopping` patience on validation loss |
| `--seed N` | reproducibility seed (default 42) |
| `--no-class-weights` | ablation: train without re-weighting |
| `--weights None` | ablation: no ImageNet initialisation (trains from scratch) |
| `--max-images-per-class N` | fast smoke test on a small machine |
| `--group-aware-split` | keep all photos of the same source leaf in one split (see §14) |
| `--low-memory` | cap `tf.data` prefetch/parallelism |
| `--data-dir DIR` | look for the dataset somewhere else |
| `--artifacts-dir DIR` | write `results/`, `figures/`, `checkpoints/`, `predictions/` under `DIR` |
| `--note TEXT`, `--quiet` | annotate `run_provenance.json`, suppress per-epoch logs |

> **Where a run writes.** `--artifacts-dir` moves the *metrics, figures, checkpoints and histories only*.
> Saved weights are not part of it: they go to `models/`, or to the directory named by the
> `MAIZE_MODELS_DIR` environment variable. The other two roots have the same form — `MAIZE_DATA_DIR`,
> `MAIZE_ARTIFACTS_DIR` — and an explicit `--data-dir` / `--artifacts-dir` flag always wins over them. No
> absolute path is baked into the code: outside Colab everything resolves relative to the project folder. There is deliberately no `--models-dir`
> flag, so if you point `--artifacts-dir` at a scratch folder and then run `src.compare`, the exported
> `best_maize_disease_model.keras` still lands in `models/` (or in `MAIZE_MODELS_DIR`, if you set it) —
> which is the one thing about this CLI that is easy to misread.

Everything about a run is recorded in `outputs/results/run_config.json` and
`outputs/results/run_provenance.json` (versions, GPU, seed, image size, batch size, epochs, learning
rates, peak memory), so a result can always be traced back to the settings that produced it.

## 11. Training strategy (what the code does)

| | Stage 1 — feature extraction | Stage 2 — fine-tuning |
| --- | --- | --- |
| pretrained base | **fully frozen** (all base layers `trainable=False`) | top 25% of base layers unfrozen, bottom 75% frozen |
| trainable part | GAP + Dropout + Dense(128) + Dense(4) head | head **+** those unfrozen upper base layers |
| optimizer | Adam, `lr = 1e-3` | Adam, `lr = 1e-4` (10× smaller) |
| BatchNorm in base | not updated (`training=False` for the base) | deliberately kept frozen (weights *and* moving statistics) |
| loss | `SparseCategoricalCrossentropy` | same |
| stopping | `EarlyStopping` on `val_loss`, patience 5 | same, restarted for stage 2 |

`python -m src.train` prints, and `train_meta_<model>.json` stores, exactly which base layers were
unfrozen (e.g. MobileNetV2: `block_13_expand … out_relu`, 25 of 154 layers) together with the trainable
parameter count for each stage — that is the "document which layers are trainable" requirement, produced
automatically rather than by hand.

## 12. Class imbalance

The class counts are uneven (measured by the inspection step — for the maize subset used here the
largest/smallest ratio is ≈ 2.3). The chosen strategy is **balanced class weights**
(`weight_c = n / (k · n_c)`, computed from the **training** split only), not oversampling:

* oversampling duplicates images, and with a benchmark dataset whose photos of the same leaf recur it
  increases the chance of near-duplicates across splits;
* re-weighting keeps the dataset exactly as it is, so the split, the counts and the figures remain
  honest and auditable;
* Gray Leaf Spot is the smallest class, so recall for it is reported per class — the project judges
  success by macro-F1, not by overall accuracy that a majority class can inflate.

Ablation if you want evidence that the choice matters: `--no-class-weights` and compare
`outputs/results/classification_report_*.csv` (per-class rows) for the two runs.

## 13. Evaluation, comparison and model selection

```bash
python -m src.evaluate --model mobilenetv2       # re-evaluate a saved model on the test split
python -m src.compare                            # comparison table + selection + export
```

`src.evaluate` is also usable on its own, which is how you re-run the numbers for the thesis without
retraining: `--weights PATH` points at any `.keras` file (default `models/<model>_maize.keras`), `--data-dir`
and `--artifacts-dir` relocate the inputs and outputs, and `--no-write` prints the report without touching the
saved artefacts. It reuses the `outputs/results/splits.csv` written during training, so the split cannot
accidentally change between training and evaluation, and it compares the current dataset fingerprint with the
one recorded in `train_meta_<model>.json`, warning if the folder contents have changed underneath you.
(`--weights` means something different here from `--weights None` in `src.train`, where it selects random
initialisation.)

Reported per model, on the untouched test set: accuracy, per-class precision/recall/F1/support,
macro **and** weighted averages, balanced accuracy (mean per-class recall), the confusion matrix (raw +
row-normalised), mean confidence, number of low-confidence predictions, parameter counts (total and
trainable), model file size, training wall-clock time per stage, and measured single-image inference
latency (mean/median/std over 16 test images).

`python -m src.compare` writes `outputs/results/model_comparison.csv`,
`reports/model_comparison_and_selection.md`, `outputs/figures/model_comparison.png`, and selects the
final model with this documented score:

```
score = 0.40·macro-F1 + 0.20·balanced-accuracy + 0.20·generalisation + 0.20·efficiency
```

(generalisation = 1 − |train − validation| accuracy gap from the final fine-tuning epoch, so the test
set never influences selection; efficiency = relative model size and latency). Accuracy differences
smaller than the test set's 95% confidence half-width are flagged as *not* meaningful evidence — with
~580 test images that half-width is a few percentage points — in which case the cheaper model wins.
**MobileNetV2 is not assumed to be better**; the winner is whichever the numbers support.

## 14. Expected results section

Do not copy any numbers into your thesis until you have run the experiment. After training, the
generated files hold everything:

| What you need | Where it comes from |
| --- | --- |
| Total images, per-class counts, dimensions, corrupted count | `outputs/results/dataset_summary.json` |
| Split sizes (train/val/test per class) | `outputs/results/split_table.csv` |
| Class distribution figure | `outputs/figures/class_distribution.png` |
| Per-class precision/recall/F1/support | `outputs/results/classification_report_<model>.csv` |
| Confusion matrices | `outputs/figures/confusion_matrix_<model>.png` (+ `.csv`) |
| Training/validation curves | `outputs/figures/training_history_<model>.png`, `outputs/results/training_history_<model>.csv` |
| Model comparison + training time + sizes + latency | `outputs/results/model_comparison.csv` |
| Chosen model and the written rationale | `outputs/results/model_selection.json` |
| Ready-to-paste narrative | `reports/dataset_report.md`, `reports/model_comparison_and_selection.md` |
| Per-image test predictions (for error analysis) | `outputs/predictions/test_predictions_<model>.csv` |

After training, results will be reported here — e.g. *MobileNetV2 reached X% accuracy and Y macro-F1;
EfficientNetB0 reached …; Z was selected because …* — generated from those files.

## 15. Saving the model

`python -m src.compare` writes the selected model as a modern Keras archive plus explicit metadata:

```
models/best_maize_disease_model.keras    ← the winning model (Keras 3 zip archive)
models/class_names.json                  ← class order used by the output layer
models/inference_config.json             ← image size, preprocessing, thresholds, metrics, versions
```

Nothing needed for inference is left as undocumented notebook state: `inference_config.json` records
`image_size`, the class order, the preprocessing decision, the low-confidence threshold, the dataset
fingerprint and the library versions, and the app reads it.

## 16. Running the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501>. Requires `models/best_maize_disease_model.keras` (i.e. training must
have been run, or the Colab artefacts copied into `models/`). The page shows: upload → image preview →
predicted class → confidence → all four class probabilities → information about the predicted class →
disclaimer, plus a *Diagnostics* expander with model metadata, measured test metrics and the
scope-of-the-result warning.

### Inference from the command line

```bash
python -m src.predict --image path/to/leaf.jpg
python -m src.predict --image a.jpg --json
python -m src.predict --dir some/folder --limit 20 --csv outputs/predictions/sample_predictions.csv
```

Example output (format shown; the numbers come from your trained model):

```
Prediction: Common Rust
Confidence: 96.42%

Class probabilities:
  Common Rust            96.42%
  Gray Leaf Spot          1.35%
  Healthy                 1.18%
  Northern Leaf Blight    1.05%

Inference time: 12.4 ms
...
This system provides an AI-based image classification result and should not replace
diagnosis or advice from a qualified agricultural professional.
```

### Python API

```python
from src.predict import MaizeDiseasePredictor

predictor = MaizeDiseasePredictor()                 # loads models/best_maize_disease_model.keras
result = predictor.predict("field_photo.jpg")        # path, bytes or PIL.Image all work
print(result.class_name, result.confidence_percent) # 'Common Rust', '96.42%'
print(result.probabilities)                          # {'Healthy': ..., ...}
print(result.low_confidence, result.warnings)        # threshold default 0.60
```

## 17. Confidence handling

Confidence is the softmax probability, i.e. a model estimate, and it is **not** calibrated
probability of correctness. The project therefore (a) never prints "certain", (b) flags predictions
below `low_confidence_threshold` (default 0.60) with *"Low confidence prediction. Consider providing a
clearer image or consulting an agricultural expert."*, (c) warns when the top two classes are within 5
percentage points of each other, (d) warns about inputs that are too small, huge (heavily downscaled)
or nearly blank, and (e) reports how many test predictions were low-confidence, so over-confidence in
the benchmark setting is visible rather than hidden.

## 18. Reproducibility

* seed `42` (change with `--seed`) for Python, NumPy and TensorFlow; `tf.config.experimental.enable_op_determinism()`
  is enabled when the build supports it.
* The split is written to `outputs/results/splits.csv` and reused, so both models — and any later
  re-run — train on identical images. `dataset_summary.json` also stores a SHA-256-based *fingerprint*
  of the file set; a mismatch forces re-inspection rather than silently training on a different split.
* Versions, device, image size, batch size, epochs and learning rates are recorded per run.
* Honest caveat: on a GPU, some cuDNN kernels are non-deterministic, so re-running can shift results in
  the last decimal places. Bit-identical metrics are *not* guaranteed; the protocol (data → split →
  preprocessing → config) is.

## 19. Tests

```bash
python tests/test_pipeline.py          # no training needed
```

22 checks, all in plain `assert`s so no test framework is required. They cover the parts that are easy to
get silently wrong: class-folder mapping (including the `Cercospora_leaf_spot Gray_leaf_spot` spelling),
split proportions and disjointness, the augmentation-is-training-only property, metric arithmetic against
hand-computed examples, the confusion matrix shape, the inference output contract, that the README and the
source contain no invented dataset counts, and that error messages are actionable when the model or dataset
is missing.

Two notes on running them. They import TensorFlow, so they need the `requirements.txt` environment (that
import is most of the ~20 s runtime), and `test_inference_contract_on_a_real_image` **skips itself** unless a
trained `models/best_maize_disease_model.keras` exists — a green run on a fresh clone (dataset present, no weights) therefore reports
`all 22 checks passed` with that one check having had nothing to load. Run it after training, or point
`MAIZE_MODELS_DIR` at a directory that holds weights, to exercise the full contract.

The notebook is generated, not hand-edited: `python notebooks/build_notebook.py` rebuilds
`maize_disease_transfer_learning.ipynb` from the same cell templates used for `src/`, and the generator
refuses to write the file if any code cell fails to parse, contains an unsubstituted token, or refers to a
config attribute through a name that does not exist in the notebook. Edit that generator if you change the
pipeline, then re-run it; the notebook check in the suite above validates the result.

## 20. Limitations

1. **Benchmark setting, not field setting.** PlantVillage images are single detached leaves on a plain
   background. Accuracy on them is an upper bound; a field photo with two lesions, dust, shadow,
   another crop, or another disease is out of distribution. Do not report the benchmark figure as
   "field accuracy".
2. **Four classes only.** The model *must* answer one of them; a nitrogen deficiency, a herbicide
   injury, an armyworm-damaged leaf or a Southern rust infection is forced into the nearest of four
   labels. It cannot say "unknown" (no rejection class was trained).
3. **Class imbalance and a small minority class.** The smallest class is roughly 13% of the data
   (~500 images), so its per-class recall is the number to watch, and a handful of images can move it.
4. **Possible repeated-leaf leakage.** PlantVillage contains several photographs of the same physical
   leaf. Splitting by image can therefore put near-duplicates in train and test, inflating scores. The
   original dataset paper explicitly guarded against this; `--group-aware-split` does the same here via
   a filename heuristic. Report which you used.
5. **No localisation.** The model cannot say *where* on the leaf it looked, nor count lesions;
   Grad-CAM-style explanation is left as future work.
6. **Single image, single plant.** Real diagnosis uses agronomic context: growth stage, variety,
   weather, field history. None of that is available here.
7. **Softmax confidence is uncalibrated**, and preprocessing/resize choices bound what a phone photo
   can express (224×224).
8. **Generalisation to other regions/seasons is untested** — no independent geographic hold-out was
   available.
9. This project does not, and cannot, guarantee food security. It is a screening-aid prototype whose
   value is the reproducible method, not a deployed product.

## 21. Ethical considerations

* **Do not act on the model alone.** Management decisions (planting, resistant hybrids, crop
  protection) have cost, health and environmental consequences; the UI always shows the disclaimer and
  never recommends a treatment. The disease text is descriptive on purpose.
* **Failure consequences are asymmetric.** A false "Healthy" can let an outbreak spread; a false
  "diseased" can trigger needless pesticide use. That is why low-confidence outputs are highlighted and
  why recall of the disease classes is reported per class.
* **Smallholder access and language.** A phone-based tool assumes a smartphone, data credit and English
  UI; that bias is worth stating in your thesis.
* **Farmer consent and image ownership.** Field data collection needs informed consent; images of a
  farm can reveal practices a farmer may not want published. The dataset used here is public research
  data, but this matters for any extension of the work.
* **Attribution and licence** of the dataset are recorded (`data/dataset_provenance.json`).
* **No fabricated evidence.** Reported metrics must come from the generated artefacts.

## 22. Future improvements

1. Field-image fine-tuning (and a proper geographic hold-out) to measure the domain gap directly.
2. An "other/unknown" rejection class, or thresholded out-of-distribution detection.
3. Multi-label support (two diseases on one leaf is realistic).
4. Calibration (temperature scaling / Platt scaling) so confidence means something, and reported ECE.
5. Grad-CAM or similar saliency to show *what* the model looked at, for trust and error analysis.
6. More architectures under the same protocol (e.g. EfficientNetB2, a small ViT) and a from-scratch
   baseline to demonstrate transfer learning's benefit quantitatively (`--weights None` already
   supports that ablation).
7. Mobile/edge deployment (TFLite) with a latency and accuracy trade-off measurement.
8. Wider temporal coverage: same plant photographed over days to track progression.
9. Multi-crop extension with a shared preprocessing/backbone pipeline.

## 23. Dataset attribution

Sharada P. Mohanty, David P. Hughes and Marcel Salathé, *"Using Deep Learning for Image-Based Plant
Disease Detection"*, Frontiers in Plant Science, 2016. DOI: [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)

The image files were obtained from the authors' public repository
<https://github.com/spMohanty/PlantVillage-Dataset> (`raw/color/`), which is the distribution the
PlantVillage project itself points people to now that `plantvillage.org` no longer serves the archive.
Record the commit you used — `python -m src.dataset_setup` writes it to `data/dataset_provenance.json`.
Please verify the licence terms of the copy you download (the release has been described as
Creative-Commons-style open research data) and cite the paper.

## 24. License

The **code in this repository** is provided for academic use under the MIT License (see `LICENSE`);
add your own name and institution before submitting. The **dataset** is *not* covered by that license —
it belongs to the PlantVillage authors and is used here for non-commercial research with attribution.

```
MIT License (code)

Copyright (c) 2026 <your name>, <your department>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## Appendix A — one-page command list

```bash
python -m src.dataset_setup --dest data     # 1. get the data (sparse clone)
python -m src.inspect_dataset                # 2. measure it, make the split, figures
python -m src.train --model mobilenetv2      # 3. train + fine-tune model 1
python -m src.train --model efficientnetb0   # 4. train + fine-tune model 2
python -m src.compare                        # 5. compare, select, export models/
python -m src.evaluate --model mobilenetv2   #    (re-evaluate a saved model any time)
python -m src.predict --image my.jpg         # 6. classify an image
streamlit run app/streamlit_app.py           # 7. open the web app
python tests/test_pipeline.py                # 8. sanity checks
```

## Appendix B — sample output formats

`outputs/results/classification_report_mobilenetv2.csv` — columns and one row per class, plus the two
averages and the accuracy row (the file is written by `src/evaluate.py`; **no value in it is typed in**):

```csv
class,precision,recall,f1-score,support,correct,misclassified
Healthy,<measured>,<measured>,<measured>,<count>,<count>,<count>
Common Rust,<measured>,<measured>,<measured>,<count>,<count>,<count>
Gray Leaf Spot,<measured>,<measured>,<measured>,<count>,<count>,<count>
Northern Leaf Blight,<measured>,<measured>,<measured>,<count>,<count>,<count>
macro avg,<measured>,<measured>,<measured>,<total>,,
weighted avg,<measured>,<measured>,<measured>,<total>,,
accuracy,,,,<total>,<count>,<count>
```

`outputs/figures/confusion_matrix_mobilenetv2.png` is a 4x4 grid of counts, rows = true class,
columns = predicted class, titled with the measured accuracy and macro-F1 — the exact layout can be
seen in the figure produced by the inspection stage of this repository.

`outputs/results/model_comparison.csv` columns:

```
model, display_name, accuracy, precision, recall, f1_score, macro_f1, macro_precision, macro_recall,
balanced_accuracy, mean_confidence, low_confidence_predictions, test_images, accuracy_95ci_halfwidth,
parameters_total, parameters_trainable, model_size_mb, training_time_seconds, inference_mean_ms,
inference_median_ms, best_val_loss, train_val_accuracy_gap, val_accuracy,
component_macro_f1, component_balanced_accuracy, component_generalisation, component_efficiency,
selection_score
```

`outputs/predictions/sample_predictions.csv` columns:

```
image, predicted_class, predicted_index, confidence_percent, low_confidence, warnings, inference_ms,
model, prob_Healthy, prob_Common_Rust, prob_Gray_Leaf_Spot, prob_Northern_Leaf_Blight
```
