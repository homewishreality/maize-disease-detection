"""Generate ``notebooks/maize_disease_transfer_learning.ipynb``.

The notebook is generated from one source description so its 22 sections stay in
sync with ``src/``. It contains no analysis logic of its own: every substantive
step calls the same functions the command-line tools call, so the Colab run and
the local run are the same experiment.

Regenerate after editing::

    python notebooks/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

# Pull the *real* defaults into the notebook text so the prose can never drift
# out of sync with src/config.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ProjectConfig  # noqa: E402

cfg = ProjectConfig()

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    """Markdown cell; dedented (not line-stripped) so fenced code keeps indent."""
    import textwrap

    body = textwrap.dedent(text.strip("\\n"))
    cells.append(nbf.v4.new_markdown_cell(body))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\\n")))


def cell(template: str, **tokens: object) -> None:
    """Append a code cell from a template.

    Notebook code is written with ``__TOKENS__`` instead of f-string
    placeholders, because f-strings would try to evaluate the *notebook's* local
    variables (``stage``, ``info``, ...) at generation time.
    """
    text = template
    for name, value in tokens.items():
        text = text.replace(f"__{name}__", str(value))
    cells.append(nbf.v4.new_code_cell(text.strip("\\n")))


# ========================================================== front matter
md(
    """
# Maize Disease Detection Using Transfer Learning

**Project:** Development of a Maize Disease Detection System for Food Security Using Deep Learning
Techniques.

This notebook is the complete experiment: dataset acquisition → inspection → stratified split →
preprocessing → augmentation → **MobileNetV2** (feature extraction, fine-tuning, evaluation) →
**EfficientNetB0** (same) → comparison → model selection → save → single-image prediction → export for
the Streamlit app.

Run it **top to bottom** on a GPU runtime (`Runtime ▸ Change runtime type ▸ T4 GPU`).

* Every number printed here is measured during *your* run; nothing is hard-coded, so the outputs you
  generate are exactly the results you report in the thesis.
* All logic lives in the project's `src/` package; this notebook only calls it, which is why the
  Colab run and the command-line run cannot disagree.

> **Demo mode.** `DEMO_MODE = True` in Section 3 shrinks the experiment to 40 images per class and
> 2 + 1 epochs so you can verify the whole pipeline on a weak machine in a few minutes. A demo run is a
> pipeline check, **not** results: never report its numbers as the experiment.
"""
)

# ========================================================== Section 1
md(
    """
## Section 1 — Environment setup

Optionally mount Google Drive (recommended: the trained model and artefacts then survive the session),
confirm a GPU is attached, and install `streamlit` so the last section can package an app-ready folder.
"""
)
code(
    """
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image as IPImage, display
from PIL import Image as PILImage

MOUNT_DRIVE = False   # True -> keep data/models/outputs under /content/drive/MyDrive/maize
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Drive mounted at /content/drive")

try:
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                        capture_output=True, text=True)
    print(out.stdout.strip() or "no GPU reported")
except FileNotFoundError:
    print("nvidia-smi not found -> running without a GPU (fine for demo mode, slow for real training).")

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'streamlit'], check=False)

# A RAM check now beats a dead kernel later. Importing TensorFlow costs ~0.7 GB
# on its own, and one trained model peaks around 1.3-2.5 GB, so a 2 GB runtime
# cannot finish this notebook even though every individual step is correct.
sys.path.insert(0, str(Path.cwd()) if Path.cwd().name != 'notebooks' else str(Path.cwd().parent))
try:
    from src.config import available_memory_mb
    _mem = available_memory_mb()
except Exception:
    _mem = {'available_mb': None, 'total_mb': None}
if _mem.get('available_mb') is not None:
    print(f"RAM available to this runtime: {_mem['available_mb']:.0f} MB "
          f"of {_mem.get('total_mb', '?')} MB total")
    if _mem['available_mb'] < 4000:
        print("  --> under ~4 GB, run the notebook in DEMO_MODE (below) or train ONE model per "
              "session with the command line: python -m src.train --model mobilenetv2. "
              "A Colab GPU runtime has 12+ GB and needs no changes.")
print("environment check complete")
"""
)

# ========================================================== Section 2
md(
    """
## Section 2 — Locate the project code, then import it

The notebook needs the project folder (containing `src/`). Either upload the repository as a **zip** to
the Colab file browser — it is found and unpacked automatically below — or clone it:

```python
!git clone https://github.com/<your-username>/maize-disease-detection.git
```
"""
)
code(
    """
CANDIDATES = [
    Path('/content/maize-disease-detection'),
    Path('/content/drive/MyDrive/maize/maize-disease-detection'),
    Path('/content/drive/MyDrive/maize-disease-detection'),
    Path.cwd(),
    Path.cwd().parent,
    Path('/content'),
]


def find_project_root():
    # Return the folder that contains src/config.py, unpacking a zip if needed.
    for base in CANDIDATES:
        if not base.exists():
            continue
        if (base / 'src' / 'config.py').exists():
            return base
        for candidate in sorted(p for p in base.iterdir() if p.is_file() and p.suffix == '.zip'):
            with zipfile.ZipFile(candidate) as zf:
                if any(n.endswith('src/config.py') for n in zf.namelist()):
                    zf.extractall(candidate.parent)
                    top = next(n for n in zf.namelist() if n.endswith('src/config.py')).split('/')[0]
                    return candidate.parent / top
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if (sub / 'src' / 'config.py').exists():
                return sub
    return None


PROJECT_ROOT = find_project_root()
if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not find the project (src/config.py). Upload the repository zip into /content, "
        "or clone it: !git clone https://github.com/<you>/maize-disease-detection.git"
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print("project root:", PROJECT_ROOT)

import tensorflow as tf                      # imported before the rest so CUDA discovery happens once

from src.config import (
    ProjectConfig,
    DISPLAY_NAMES,
    environment_report,
    seed_everything,
    write_run_provenance,
)
from src import data_loader as dl
from src import evaluate as ev
from src import models as mdl
from src import preprocessing as prep
from src import visualize as viz
from src import train as trn
from src.train import load_split_frames
from src.compare import compare_and_select
from src.inspect_dataset import inspect_dataset
from src.dataset_setup import prepare_dataset
from src.predict import MaizeDiseasePredictor

print("tensorflow", tf.__version__, "| GPUs:", [g.name for g in tf.config.list_physical_devices('GPU')])
"""
)

# ========================================================== Section 3
md(
    """
## Section 3 — Configuration

One object holds **every** hyper-parameter: image size, batch size, split fractions, head width,
dropout, both learning rates, both epoch budgets, the unfreeze fraction, patience, the low-confidence
threshold and the seed. Nothing else in the project needs editing, and the configuration is saved with
the results so every figure can be traced back to the settings that produced it.

`seed_everything` seeds Python, NumPy and TensorFlow; the same seed drives the stratified split, so the
data partition is reproducible.
"""
)
code(
    """
DEMO_MODE = False      # True = 40 images/class and 2+1 epochs: pipeline check, NOT results

config = ProjectConfig(
    data_dir=PROJECT_ROOT / 'data',
    artifacts_dir=PROJECT_ROOT / 'outputs',
    models_dir=PROJECT_ROOT / 'models',
    reports_dir=PROJECT_ROOT / 'reports',
    seed=42,
    image_size=224,
    train_fraction=0.70,
    val_fraction=0.15,
    test_fraction=0.15,
    dense_units=128,
    dropout_rate=0.30,
    l2_weight=1e-4,
    feature_extraction_epochs=2 if DEMO_MODE else 15,
    head_learning_rate=1e-3,
    finetune_epochs=1 if DEMO_MODE else 15,
    finetune_learning_rate=1e-4,
    finetune_unfreeze_fraction=0.25,
    early_stopping_patience=2 if DEMO_MODE else 5,
    class_weights=True,
    low_confidence_threshold=0.60,
    batch_size=16 if DEMO_MODE else 32,
    max_images_per_class=40 if DEMO_MODE else None,
    group_aware_split=False,      # True keeps all photos of one source leaf in the same split
    tf_data_num_parallel_calls=2 if DEMO_MODE else None,
    tf_data_prefetch=1 if DEMO_MODE else None,
)
seed_everything(config.seed, config.deterministic_ops)
config.ensure_directories()

display(pd.DataFrame([config.to_dict()]).T.rename(columns={0: 'value'}))
pd.DataFrame([config.to_dict()]).T.to_csv(config.result('notebook_configuration.csv'), header=False)
print("DEMO_MODE =", DEMO_MODE)
"""
)

# ========================================================== Section 4
md(
    """
## Section 4 — Dataset acquisition

The maize images come from the dataset **authors' own public repository**
(`spMohanty/PlantVillage-Dataset`), since `plantvillage.org` no longer serves the archive. The download
is a *blobless sparse* git clone that checks out **only** the four `raw/color/Corn_(maize)___*`
folders — tens of MB, no Kaggle account, no scraping of random mirrors.

If a usable dataset is already present nothing is downloaded. The exact commit used is written to
`data/dataset_provenance.json`, which is the provenance record your examiner may ask for.
"""
)
code(
    """
report = prepare_dataset(config.data_dir)
DATA_ROOT = Path(report['root'])
print("dataset root :", DATA_ROOT)
print("source       :", report.get('source'))
provenance = report.get('provenance') or {}
if provenance:
    print("repository   :", provenance.get('repository'))
    print("commit       :", provenance.get('commit'), "(", provenance.get('commit_date'), ")")
print("\\nimages per class, counted from the download:")
display(pd.DataFrame([report['verification']['images_per_class']]).T.rename(columns={0: 'images'}))
"""
)

# ========================================================== Section 5
md(
    """
## Section 5 — Dataset inspection (measure, never assume)

This stage discovers the class directories, verifies that all four required classes exist, counts the
images, opens every file with Pillow to detect corrupt/truncated images, measures dimensions and
formats, builds and validates the stratified split, and writes everything to `outputs/`. If your copy
has extra maize folders outside the four-class scope (newer PlantVillage releases added *Maydis eye
spot* and *Tar spot*), they are reported as excluded rather than merged into a class.
"""
)
code(
    """
summary = inspect_dataset(config, write=True)

print("\\n=== measured dataset summary ===")
print("total usable images :", summary['total_images_after_validation'])
print("corrupted files     :", summary['corrupted_files'])
print("imbalance ratio     :", summary['imbalance_ratio_largest_over_smallest'])
display(pd.DataFrame(summary['class_distribution']))
print("dimensions / formats:", json.dumps(summary['dimensions_and_formats'], indent=2))
print("out-of-scope maize folders excluded:", summary['out_of_scope_dirs'] or 'none')
print("unrecognised directories ignored   :", summary['unrecognised_dirs'] or 'none')
"""
)

# ========================================================== Section 6
md("## Section 6 — Visualisation of the dataset")
code(
    """
display(IPImage(filename=str(config.figure('class_distribution.png')), width=780))
display(IPImage(filename=str(config.figure('dataset_examples.png')), width=900))
print("Representative images above are real files from the training split: note the single detached "
      "leaf and the uniform background -- this is a controlled benchmark, not field photography.")
"""
)

# ========================================================== Section 7
md(
    """
## Section 7 — Train / validation / test split (70 / 15 / 15, stratified)

The split is computed on the **list of original image files, before any augmentation exists** — that is
the mechanism that stops an augmented view of a training image from appearing in validation or test.
`src.data_loader` then *asserts* that the three subsets are disjoint and that each holds the requested
share, so the property is verified rather than claimed. The assignment is written to
`outputs/results/splits.csv` and reused by both models, which makes the comparison fair.
"""
)
code(
    """
mapping, _ = dl.build_class_mapping(DATA_ROOT)
frames, mapping = load_split_frames(config, DATA_ROOT, mapping)

counts = dl.split_table(frames)
counts.to_csv(config.result('split_table_notebook.csv'))
display(counts)

train_df, val_df, test_df = (frames[frames['split'] == s] for s in ('train', 'validation', 'test'))
print(f"shares: train {len(train_df)/len(frames):.1%}  val {len(val_df)/len(frames):.1%}  "
      f"test {len(test_df)/len(frames):.1%}")
print("\\nclass weights (from the TRAIN split only, so the test set never influences training):")
weights = dl.compute_class_weights(train_df)
display(pd.DataFrame([{'class': mapping.class_names[k], 'weight': round(v, 4),
                       'train_images': int((train_df['label'] == k).sum())} for k, v in weights.items()]))
print("dataset fingerprint:", dl.dataset_fingerprint(frames), "-- stored in dataset_summary.json")
"""
)

# ========================================================== Section 8
md(
    """
## Section 8 — Preprocessing (architecture-specific)

`keras.applications.<model>.preprocess_input` is **not** interchangeable between architectures, and in
Keras 3 it is not even the same kind of object:

| model | `preprocess_input` | preprocessing layers inside the model | what the pipeline must feed |
| --- | --- | --- | --- |
| MobileNetV2 | applies `(x-127.5)/127.5` | none | rescaled tensor |
| EfficientNetB0 | documented **no-op** | `Rescaling → Normalization → Rescaling` | float `[0,255]` |

Applying an extra `(x-127.5)/127.5` to EfficientNetB0 would double-normalise and quietly waste the
pretrained features. So `src.preprocessing` **inspects the built model** and adapts
(`model_expects_internal_preprocessing`). The cell below prints the decision for both architectures and
checks the numeric range actually fed to each one.
"""
)
code(
    """
sample = train_df.head(8)

plans = {}
for key in ('mobilenetv2', 'efficientnetb0'):
    # Build only the pretrained *base* (no weights) -- that is where the
    # preprocessing layers live, so it is enough to detect the strategy.
    base = mdl.get_backbone(key).build_base(weights=None,
                                            input_shape=(config.image_size, config.image_size, 3))
    plan = prep.build_plan(key, mdl.get_backbone(key).module_name, base)
    plans[key] = plan
    ds = prep.build_dataset(sample, plan, config, augment=False, shuffle=False)
    x, y = next(iter(ds.take(1)))
    print(f"{key:15s} internal_preprocessing={plan.internal_preprocessing}")
    print(f"   -> {plan.description}")
    print(f"   fed to the network: shape {tuple(x.shape)}, "
          f"range [{float(tf.reduce_min(x)):.2f}, {float(tf.reduce_max(x)):.2f}]")
    print(f"   first layers of the base: {[l.__class__.__name__ for l in base.layers[:4]]}")
    del base, ds
    tf.keras.backend.clear_session()

expected = {'mobilenetv2': False, 'efficientnetb0': True}
for key, want in expected.items():
    got = plans[key].internal_preprocessing
    print(f"   check {key}: internal_preprocessing={got} (expected on this Keras version: {want})"
          + ("  OK" if got == want else "  <-- Keras changed; src.preprocessing adapted automatically"))

import gc; gc.collect()
print("\\nlabels are integers (SparseCategoricalCrossentropy), class names:", mapping.class_names)
"""
)

# ========================================================== Section 9
md(
    """
## Section 9 — Data augmentation (training split only)

Transformations and why: horizontal flip (leaf orientation in a photo is arbitrary; *no* vertical flip,
which is not physically symmetric for a dorsiventral leaf), ±3.6° rotation, 10% zoom, 10% translation
(hand-held framing), brightness/contrast ±10% (sun vs cloud). **No hue or saturation shifts**: rust
pustules versus blight lesions versus grey rectangular spots are partly defined by colour, so altering
colour statistics could destroy the signal the model is supposed to learn. Validation and test images are
never augmented, and they are never shuffled.
"""
)
code(
    """
aug_layer = prep.build_augmentation(config.image_size)
aug_layer.summary()

row = train_df.iloc[0]
base = prep.decode_and_resize(tf.constant(str(row['path'])), config.image_size)[tf.newaxis, ...]
tiles = [np.clip(base.numpy(), 0, 255)[0].astype('uint8')]
for _ in range(5):
    tiles.append(np.clip(aug_layer(base, training=True).numpy(), 0, 255)[0].astype('uint8'))

fig, axes = plt.subplots(1, len(tiles), figsize=(3.1 * len(tiles), 3.3))
for ax, img in zip(axes, tiles):
    ax.imshow(img); ax.axis('off')
axes[0].set_title('original', fontsize=9)
for i, ax in enumerate(axes[1:], 1):
    ax.set_title(f'aug {i}', fontsize=9)
plt.suptitle(f"training image of class '{row['class_name']}' (resized to "
             f"{config.image_size}x{config.image_size}, float32 in [0,255])")
plt.show()

# Explicit check that the non-augmented path leaves pixels untouched.
plan = plans['mobilenetv2']
ds_plain = prep.build_dataset(val_df.head(1), plan, config, augment=False, shuffle=False)
x_a, _ = next(iter(ds_plain.take(1)))
x_b, _ = next(iter(ds_plain.take(1)))
print("validation batch identical across calls (no augmentation, no shuffle):",
      bool(tf.reduce_all(tf.equal(x_a, x_b)).numpy()))
"""
)

# ====================================================== Sections 10-13, 14-17
for first, key, name in ((10, 'mobilenetv2', 'MobileNetV2'), (14, 'efficientnetb0', 'EfficientNetB0')):
    md(
        f"""
## Section {first} — {name}: architecture, data and input pipelines

Required skeleton, built by `src.models.build_model` (defaults shown; edit Section 3 to change them):

```
Input({cfg.image_size},{cfg.image_size},3)
  -> {name} base, weights='imagenet', include_top=False     <- fetched automatically by Keras
  -> GlobalAveragePooling2D
  -> Dropout({cfg.dropout_rate})
  -> Dense({cfg.dense_units}, relu, L2={cfg.l2_weight:g})
  -> Dense(4, softmax)
```

Three notes on how it is wired:

* the base is called with `training=False` in **both** stages, so BatchNorm inside the pretrained
  network never updates its moving statistics while it is being adapted;
* `src.preprocessing` decides this architecture's input normalisation automatically (Section 8);
* `tf.data` is lazy (decode -> resize -> *optional* augment -> preprocess -> prefetch) with **no**
  `.cache()`, because caching ~2.7k images at {cfg.image_size}x{cfg.image_size}x3 float32 would need
  ~1.6 GB of RAM, which a Colab CPU runtime does not have.
"""
    )
    cell(
        """
# Colab gives one kernel process for the whole notebook, so the previous
# candidate's graph must be released before building the next one -- otherwise
# both models plus their input pipelines sit in RAM at once (that is exactly how
# a "kernel died" / exit-137 failure shows up on a 12 GB runtime).
for _name in ("ctx", "pipelines", "h1", "h2", "m1", "m2", "meta"):
    globals().pop(_name, None)
import gc
gc.collect()
tf.keras.backend.clear_session()

ctx = trn.prepare_experiment('__KEY__', config, weights='imagenet')
ctx['model'].summary()
pipelines = trn.build_pipelines(ctx)
for split, ds in pipelines.items():
    print(f"  {split:11s} batches={int(ds.cardinality().numpy()):4d}  "
          f"augmentation={'YES' if split == 'train' else 'none'}")
print("\\npreprocessing decision:", ctx['plan'].description)
config.ensure_directories()
Path(config.result('preprocessing___KEY__.json')).write_text(
    json.dumps({'model': '__KEY__', 'plan': ctx['plan'].__dict__}, indent=2), encoding='utf-8')
""",
        KEY=key,
    )
    md(
        f"""
## Section {first + 1} — {name}: stage 1, feature extraction

The pretrained base is **entirely frozen**; only the new head trains, with Adam at
`lr={cfg.head_learning_rate:g}` for up to {cfg.feature_extraction_epochs} epochs. Balanced class weights are
computed from the **training** split only. `EarlyStopping`, `ModelCheckpoint` and `ReduceLROnPlateau`
all monitor `val_loss`, so the best epoch is chosen on validation performance -- never on the test set.
"""
    )
    cell(
        """
h1, m1 = trn.stage_one(ctx, pipelines['train'], pipelines['validation'], quiet=False)
print("\\nstage 1 record:", json.dumps({k: v for k, v in m1.items() if k != 'trainable_layer_names_in_base'},
                                     indent=2))
""",
        KEY=key,
    )
    md(
        f"""
## Section {first + 2} — {name}: stage 2, fine-tuning

Only the top {cfg.finetune_unfreeze_fraction:.0%} of the base is unfrozen, at `lr={cfg.finetune_learning_rate:g}` (10x smaller),
with BatchNorm deliberately kept frozen. Not unfreezing everything is the point: with ~2.7k images the
network would otherwise overfit and forget its ImageNet features. The trainable layer names are recorded
below and in `train_meta___KEY__.json`, which is the "document exactly which layers are trainable"
requirement produced automatically.
"""
    )
    cell(
        """
h2, m2 = trn.stage_two(ctx, pipelines['train'], pipelines['validation'], quiet=False)
unfrozen = m2['trainable_layer_names_in_base']
print(f"base layers: {m2['backbone_layers']} total, {m2['base_layers_frozen']} frozen, "
      f"{m2['base_layers_trainable']} trainable")
print(f"first trainable: {m2['first_trainable_base_layer']} | last trainable: {m2['last_trainable_base_layer']}")
print("trainable parameters after unfreezing:", f"{m2['trainable_parameters']:,}",
      "of", f"{m2['parameters']:,}")
print("\\nfirst 12 trainable base layers:", unfrozen[:12])
""",
        KEY=key,
    )
    md(
        f"""
## Section {first + 3} — {name}: evaluation, saving and reporting

`finalize` saves the weights, then evaluates **once** on the untouched test split: accuracy, per-class
precision/recall/F1/support with the real class names, macro and weighted averages, balanced accuracy,
confusion matrix, mean confidence and the low-confidence count. It also measures model size, peak memory
and single-image inference latency.
"""
    )
    cell(
        """
meta = trn.finalize(ctx, [h1, h2], {'feature_extraction': m1, 'fine_tuning': m2},
                    pipelines['test'],
                    run_note='DEMO run - pipeline check only' if DEMO_MODE else None)
display(pd.read_csv(config.result('classification_report___KEY__.csv')))
print("confusion matrix (rows = true label, columns = predicted label):")
display(pd.read_csv(config.result('confusion_matrix___KEY__.csv'), index_col=0))
display(IPImage(filename=str(config.figure('confusion_matrix___KEY__.png')), width=620))
display(IPImage(filename=str(config.figure('training_history___KEY__.png')), width=900))
print("sizes/timings:", json.dumps({'parameters': meta['parameter_counts'],
                                    'model_size_mb': meta['model_size_mb'],
                                    'training_seconds_total': meta['training_seconds_total'],
                                    'peak_memory_mb': meta['peak_memory_mb'],
                                    'inference': meta['inference_time']}, indent=2))

# keep a compact copy of what matters, then release the graph
__KEY__summary = {
    'test_metrics': {k: meta['test_metrics'][k] for k in
                     ('accuracy', 'f1_macro', 'balanced_accuracy', 'mean_confidence')},
    'parameter_counts': meta['parameter_counts'],
    'model_size_mb': meta['model_size_mb'],
    'training_seconds_total': meta['training_seconds_total'],
    'peak_memory_mb': meta['peak_memory_mb'],
}
del ctx, pipelines, h1, h2
import gc as _gc; _gc.collect(); tf.keras.backend.clear_session()
print("released this model's graph:", __KEY__summary['model_size_mb'], "MB saved")
""",
        KEY=key,
    )


# ========================================================== Section 18
md(
    """
## Section 18 — Model comparison

The table is assembled from the two runs' metadata: accuracy, precision, recall, F1, macro-F1, balanced
accuracy, total and trainable parameters, model file size, measured training wall-clock time, and
measured single-image inference latency. Values come from `outputs/results/train_meta_*.json`, so they
are the numbers you produced, not values copied from a paper.
"""
)
code(
    """
comparison = pd.read_csv(config.result('model_comparison_raw.csv'))
show = [c for c in ('display_name', 'accuracy', 'precision', 'recall', 'f1_score', 'macro_f1',
                    'balanced_accuracy', 'parameters_total', 'parameters_trainable', 'model_size_mb',
                    'training_time_seconds', 'inference_mean_ms', 'val_accuracy',
                    'train_val_accuracy_gap', 'accuracy_95ci_halfwidth', 'test_images')
        if c in comparison.columns]
display(comparison[show])
comparison[show].to_csv(config.result('model_comparison_notebook.csv'), index=False)
display(IPImage(filename=str(config.figure('model_comparison.png')), width=760))
"""
)

# ========================================================== Section 19
md(
    """
## Section 19 — Final model selection

`src.compare` scores each candidate with a documented composite criterion, so accuracy is not the only
consideration and the test set never influences the choice:

```
score = 0.40·macro-F1 + 0.20·balanced-accuracy + 0.20·generalisation + 0.20·efficiency
```

*generalisation* = 1 − |train − validation| accuracy gap (from the final fine-tuning epoch, i.e. **not**
from the test set); *efficiency* = relative model size and latency. If the accuracy gap between two
models is smaller than the test set's 95% confidence half-width, the code states that the difference is
not meaningful evidence and prefers the cheaper model. `MobileNetV2` is **not** assumed to win — the
numbers decide, and the written rationale is saved to `outputs/results/model_selection.json`.
"""
)
code(
    """
selection = compare_and_select(config, export=True)
print("\\n" + selection['rationale'])
display(pd.DataFrame(selection['ranking']))
print("complete comparison (both models trained)?", selection['complete_comparison'])
Path(config.report('model_selection_note.md')).write_text(
    "# Model selection\\n\\n" + selection['rationale'] + "\\n", encoding='utf-8')
"""
)

# ========================================================== Section 20
md(
    """
## Section 20 — Save the selected model

`models/best_maize_disease_model.keras` (modern Keras archive) plus `class_names.json` and
`inference_config.json`, which carry the class order, image size, preprocessing decision, thresholds,
dataset fingerprint and library versions. That is what makes the weights usable on their own — no
undocumented notebook state is required to run inference later.
"""
)
code(
    """
for p in sorted((PROJECT_ROOT / 'models').iterdir()):
    print(f"{p.name:40s} {p.stat().st_size / 1e6:8.2f} MB")

print("\\nmodels/inference_config.json:")
inference_cfg = json.loads((PROJECT_ROOT / 'models' / 'inference_config.json').read_text(encoding='utf-8'))
display(pd.DataFrame([
    {'key': k, 'value': json.dumps(v) if isinstance(v, (dict, list)) else v}
    for k, v in inference_cfg.items()
]).set_index('key').T)

history_csvs = sorted(config.artifacts_dir.rglob('training_history_*.csv'))
print("\\ntraining histories:", [str(h.relative_to(PROJECT_ROOT)) for h in history_csvs])
"""
)

# ========================================================== Section 21
md(
    """
## Section 21 — Single-image prediction

The documented inference contract: load → validate → resize → model-specific preprocessing → predict →
probabilities → predicted class → confidence → all class probabilities. It reuses
`src.preprocessing`, the same code path used in training, so training and deployment cannot drift.

A **test-split** image is used here, and the confidence shown is whatever your model actually outputs.
"""
)
code(
    """
predictor = MaizeDiseasePredictor(
    model_path=PROJECT_ROOT / 'models' / 'best_maize_disease_model.keras',
    inference_config_path=PROJECT_ROOT / 'models' / 'inference_config.json',
    class_map_path=PROJECT_ROOT / 'models' / 'class_names.json',
    config=config,
)

test_paths = [str(p) for p in test_df['path'].head(10)]
result = predictor.predict(test_paths[0])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].imshow(PILImage.open(result.image_source).convert('RGB').resize((224, 224)))
axes[0].set_title(f"TRUE (test label): {test_df.iloc[0]['class_name']}")
axes[0].axis('off')
probs = result.probabilities
order = sorted(probs, key=lambda k: -probs[k])
axes[1].barh(order[::-1], [probs[k] * 100 for k in order[::-1]], color='#1565c0')
for i, k in enumerate(order[::-1]):
    axes[1].text(probs[k] * 100 + 1, i, f"{probs[k] * 100:.2f}%", va='center', fontsize=9)
axes[1].set_xlim(0, 112)
axes[1].set_xlabel('model confidence (%)')
axes[1].set_title(f"PREDICTED: {result.class_name}  ({result.confidence_percent})")
plt.tight_layout(); plt.show()

print(result.pretty_text())

rows = [predictor.predict(p).to_dict() for p in test_paths]
sample = pd.DataFrame(rows)
sample.to_csv(config.prediction('sample_predictions.csv'), index=False)
display(sample[['image', 'predicted_class', 'confidence_percent', 'low_confidence', 'inference_ms']])
"""
)

# ========================================================== Section 22
md(
    """
## Section 22 — Export artefacts (Streamlit app + download)

`models/` is all the Streamlit app needs: `streamlit run app/streamlit_app.py` from the project root.
The cell below also records the environment (Python/TF/GPU versions) and zips `models/`, `outputs/` and
`reports/` so your complete experimental record can be downloaded in one go.
"""
)
code(
    """
env = environment_report(config)
write_run_provenance(config, note='notebook run' + (' (DEMO MODE)' if DEMO_MODE else ''))
display(pd.DataFrame([{'key': k, 'value': v} for k, v in env.items()]).set_index('key').T)

zip_base = Path('/content/maize_results') if Path('/content').exists() else PROJECT_ROOT / 'maize_results'
for folder in ('models', 'outputs', 'reports'):
    source = PROJECT_ROOT / folder
    if source.exists():
        archive = shutil.make_archive(f"{zip_base}_{folder}", 'zip', source)
        print("zipped", folder, "->", archive)

print("\\nFiles to keep with your thesis:")
for pattern in ('dataset_summary.json', 'split_table.csv', 'classification_report_*.csv',
                'confusion_matrix_*.csv', 'model_comparison.csv', 'model_selection.json',
                'run_provenance.json'):
    for f in sorted(config.artifacts_dir.rglob(pattern)):
        print("  ", f.relative_to(PROJECT_ROOT))
print("\\nRun the app locally with:  streamlit run app/streamlit_app.py")
"""
)

# ---------------------------------------------------------------- self-checks
for i, c in enumerate(cells):
    if c.cell_type != "code":
        continue
    src = c.source
    problems = []
    import re as _re

    try:
        import ast as _ast

        probe = chr(10).join(
            "!" + line if line.startswith(("!", "%")) else line
            for line in src.split(chr(10))
        )
        _ast.parse(probe)
    except SyntaxError as exc:
        problems.append(f"cell does not parse ({exc.msg} at line {exc.lineno})")
    if _re.search(r"(?<![A-Za-z0-9_])cfg\.", src):
        problems.append("references generator-only name 'cfg'")
    for token in ("__KEY__", "__NAME__"):
        if token in src:
            problems.append(f"unsubstituted template token {token}")
    if problems:
        raise SystemExit(f"generator bug, code cell {i}: " + "; ".join(problems))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "file_extension": ".py", "pygments_lexer": "ipython3"},
    "accelerator": "GPU",
    "colab": {"collapsed_sections": [], "toc_visible": True},
}

out = Path(__file__).resolve().parent / "maize_disease_transfer_learning.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(cells)} cells: "
      f"{sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
