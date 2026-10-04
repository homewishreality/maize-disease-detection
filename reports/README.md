# `reports/` — narrative summaries generated from the run

Markdown write-ups assembled from the measured artefacts (no typed-in numbers):

- **`dataset_report.md`** — written by `python -m src.inspect_dataset`: dataset root, real per-class
  counts, measured image dimensions/formats, corrupted-file count, split composition table,
  out-of-scope folders, environment/versions, and the controlled-environment caveat that belongs in a
  thesis limitations section.
- **`model_comparison_and_selection.md`** — written by `python -m src.compare`: the comparison table
  (accuracy / precision / recall / F1, macro-F1, balanced accuracy, parameters, model size, training
  time, inference latency), each accuracy with its 95% confidence half-width, the composite selection
  score with its weights, and per-class test results for each model.
- **`model_selection_note.md`** — the one-paragraph decision note written by the notebook.

Both are regenerated on every run and ignored by git, so they always describe the experiment you
actually performed. Nothing in them is filled in manually: if a value is missing from the metadata, the
cell renders empty rather than inventing one.
