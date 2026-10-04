# `data/` — where the dataset goes

This directory is **not** committed to git (see `.gitignore`); it holds the image dataset only.

## Automatic download (recommended)

From the project root:

```bash
python -m src.dataset_setup --dest data
```

This performs a *blobless sparse* `git clone` of the dataset authors' own repository,
`https://github.com/spMohanty/PlantVillage-Dataset`, checking out **only** the four maize folders
(`raw/color/Corn_(maize)___*`), and then verifies what arrived by counting the images it actually
found. Resulting layout:

```
data/
├── README.md                     ← this file (kept in git)
├── dataset_provenance.json       ← written by the download: repo, commit, date, counts
└── pv/
    └── raw/
        └── color/
            ├── Corn_(maize)___Common_rust_/
            ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
            ├── Corn_(maize)___Northern_Leaf_Blight/
            └── Corn_(maize)___healthy/
```

`src/data_loader.py` *discovers* `data/pv/raw/color` by itself (it searches `data/` and common
extraction layouts for directories containing maize class folders), so an archive extracted elsewhere
under `data/` also works. To point at the images directly:

```bash
python -m src.inspect_dataset --data-dir /path/to/raw/color
```

Download size is a few tens of MB (the maize subset alone, ~52 MB of JPEGs), not the multi-GB full
repository, because the sparse checkout requests only these four folders.

## Manual alternatives

1. **Clone the whole repository** (large — every crop, every variant):
   `git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git data/pv`,
   then use `data/pv/raw/color`.
2. **Mendeley Data / Kaggle mirrors** of PlantVillage exist. If you use one, drop its four maize class
   folders anywhere under `data/` and re-run the inspection step. Mirror folder names differ slightly
   (for example `Corn_(maize)___Gray_leaf_spot` instead of
   `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`); the alias/keyword mapping in
   `src/config.py` (`CLASS_ALIASES`, `FOLDER_KEYWORD_RULES`) accepts them, so **never** rename
   thousands of files by hand.
3. The original `plantvillage.org` image download is **no longer served**, which is why the authors'
   public repository is used here.

## Attribution / licence

Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease
Detection.* Frontiers in Plant Science, 7, 1419. <https://doi.org/10.3389/fpls.2016.01419>

The images were released as open research data by the PlantVillage project (described by the authors as
Creative-Commons-style licensing). **Verify the licence text attached to the copy you download** and
cite the paper in your thesis; the dataset is not covered by this repository's MIT licence.

## Expect counts to be measured, not asserted

`python -m src.inspect_dataset` reports images per class, image dimensions, formats and any corrupted
files **from your copy of the data** (`outputs/results/dataset_summary.json`). The code never assumes a
total; if you see a number in a generated report, it came from counting files.
