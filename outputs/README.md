# `outputs/` — generated experiment artefacts

Everything in this folder (except this `README.md`) is **produced by running the pipeline**; nothing is
typed in by hand, and the folder is ignored by git so a fresh clone regenerates its own numbers.

```
outputs/
├── figures/
│   ├── class_distribution.png                    # measured images per class, + split
│   ├── dataset_examples.png                      # real training images, one per class
│   ├── training_history_mobilenetv2.png          # accuracy & loss, both stages
│   ├── training_history_efficientnetb0.png
│   ├── confusion_matrix_mobilenetv2.png          # test set, raw counts
│   ├── confusion_matrix_mobilenetv2_normalised.png
│   ├── confusion_matrix_efficientnetb0.png
│   ├── confusion_matrix_efficientnetb0_normalised.png
│   └── model_comparison.png                      # side-by-side scores
├── results/
│   ├── dataset_summary.json                      # totals, dimensions, formats, corrupted count
│   ├── image_metadata.csv                        # per-file width/height/format/bytes
│   ├── corrupted_files.csv                       # unreadable files (header only if none)
│   ├── split_table.csv, splits.csv, split_assignment.csv
│   ├── training_history_<model>.csv              # per-epoch metrics (re-plottable)
│   ├── classification_report_<model>.csv         # per-class precision/recall/F1/support
│   ├── confusion_matrix_<model>.csv              # raw counts
│   ├── test_metrics_<model>.json                 # all test metrics + mean confidence
│   ├── preprocessing_<model>.json                # which preprocessing that model needs
│   ├── train_meta_<model>.json                   # params, sizes, timings, stages, config
│   ├── model_comparison.csv, model_comparison_raw.csv
│   ├── model_selection.json                      # chosen model + written rationale
│   ├── run_config.json, run_provenance.json      # exact settings + versions of this run
│   └── checkpoints/                              # best-epoch weights from each stage
└── predictions/
    ├── test_predictions_<model>.csv              # per-image test predictions
    └── sample_predictions.csv                     # from `python -m src.predict --csv ...`
```

**`<model>`** is `mobilenetv2` or `efficientnetb0`.

Which file answers which thesis question:

| Question | File |
| --- | --- |
| How much data, and what shape? | `results/dataset_summary.json`, `figures/class_distribution.png` |
| How was it split, and how do we know nothing leaked? | `results/split_table.csv`, `results/splits.csv` |
| What did training look like? | `results/training_history_<model>.csv`, `figures/training_history_<model>.png` |
| How good is it, per class? | `results/classification_report_<model>.csv`, `figures/confusion_matrix_<model>.png` |
| Which model won, and why? | `results/model_comparison.csv`, `results/model_selection.json` |
| Can this exact result be re-created? | `results/run_provenance.json` (versions, seed, device, all hyper-parameters) |
| Where does it fail? | `predictions/test_predictions_<model>.csv` (filter `correct == False`) |

If a figure or CSV is missing, the step that writes it has not been run yet:
`python -m src.inspect_dataset` → `python -m src.train --model both` → `python -m src.compare`.
