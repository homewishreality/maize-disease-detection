"""Dataset acquisition for the maize (PlantVillage) subset.

Provenance, stated plainly:

* The images are the **PlantVillage** maize classes released by Mohanty, Hughes
  and Salathe (2016), *Frontiers in Plant Science* 7:1419.
* ``plantvillage.org`` no longer serves the image archive, so the script clones
  the dataset authors' own public repository,
  ``https://github.com/spMohanty/PlantVillage-Dataset``, and checks out **only**
  the four maize folders (``raw/color/...``) with a blobless sparse clone.  No
  scraping, no random mirrors, no Kaggle credentials required.
* The original release is documented by the authors as CC BY / CC0-licensed
  open-access research data; keep the citation in the thesis even though the
  hosting has moved.  Verify the licence text of whatever copy you download.

Run it with::

    python -m src.dataset_setup --dest data
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .config import CLASS_ALIASES, DISPLAY_NAMES, ProjectConfig

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset"
BRANCH = "master"
#: Directory (inside the repo) holding the full-colour images per class.
COLOR_SUBDIR = "raw/color"
#: Repo paths to check out: the four maize/corn classes of this project.
MAIZE_PATHS: tuple[str, ...] = tuple(
    f"{COLOR_SUBDIR}/{folder}" for folders in CLASS_ALIASES.values() for folder in folders
)


class DatasetSetupError(RuntimeError):
    pass


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> str:
    """Run a command, returning stdout; raise with a readable message on error."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise DatasetSetupError(
            f"Command not found: {cmd[0]}. Install it first "
            f"(on Ubuntu: 'sudo apt-get install {cmd[0]}'; on Windows install "
            f"{cmd[0]} and make sure it is on PATH)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DatasetSetupError(f"'{' '.join(cmd)}' timed out after {timeout}s.") from exc
    if proc.returncode != 0:
        raise DatasetSetupError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout.strip()[-800:]}\nstderr: {proc.stderr.strip()[-800:]}"
        )
    return proc.stdout


def git_available() -> bool:
    return shutil.which("git") is not None


def sparse_clone_maize(dest: Path, verbose: bool = True) -> dict:
    """Blobless sparse clone of only the four maize class folders.

    Only the small ``raw/color/Corn_*`` folders are downloaded, which is a few
    tens of megabytes rather than the several gigabytes of the full repository.
    """
    if not git_available():
        raise DatasetSetupError(
            "git is required for the automatic download but was not found on PATH.\n"
            "Install git, or download the four maize class folders manually and "
            "place them under: " + str(dest) + " -- see data/README.md."
        )
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    repo_dir = dest / "pv"
    if repo_dir.exists() and any(repo_dir.iterdir()):
        print(f"[dataset] reusing existing checkout at {repo_dir}")
    else:
        print(f"[dataset] cloning {REPO_URL} (blobless, maize folders only) ...")
        _run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                "--branch", BRANCH, REPO_URL, str(repo_dir),
            ],
            cwd=dest,
            timeout=3600,
        )
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=repo_dir, timeout=600)
    _run(["git", "sparse-checkout", "set", *MAIZE_PATHS], cwd=repo_dir, timeout=900)
    _run(["git", "checkout", BRANCH], cwd=repo_dir, timeout=3600)
    if verbose:
        print(f"[dataset] extracted to {repo_dir / COLOR_SUBDIR}")
    return {
        "repository": REPO_URL,
        "branch": BRANCH,
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo_dir).strip(),
        "commit_date": _run(["git", "log", "-1", "--format=%ci"], cwd=repo_dir).strip(),
        "sparse_paths": list(MAIZE_PATHS),
        "local_root": str(repo_dir / COLOR_SUBDIR),
    }


def verify_dataset(root: Path, config: ProjectConfig | None = None) -> dict:
    """Count real images per class right after downloading (no assumptions)."""
    from . import data_loader as dl

    config = config or ProjectConfig(data_dir=root)
    mapping, info = dl.build_class_mapping(Path(root))
    counts = {}
    for key in mapping.class_keys:
        folder = Path(root) / mapping.relative_dirs[key]
        counts[DISPLAY_NAMES[key]] = sum(
            1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    return {
        "root": str(root),
        "classes_found": mapping.class_keys,
        "images_per_class": counts,
        "total_images": sum(counts.values()),
        "unrecognised_dirs": info["unrecognised_dirs"],
        "out_of_scope_dirs": info["out_of_scope_dirs"],
    }


def prepare_dataset(dest: Path, config: ProjectConfig | None = None) -> dict:
    """Full acquisition routine: clone if needed, then verify and record provenance.

    If a usable dataset already exists under *dest*, nothing is downloaded --
    the existing copy is verified and reported instead.  A half-finished
    ``data/pv`` checkout is repaired by re-running the sparse checkout, which is
    safe: it only writes files git tracks.
    """
    from . import data_loader as dl

    dest = Path(dest)
    config = (config or ProjectConfig()).override(data_dir=dest)
    record: dict = {"dest": str(dest)}
    try:
        existing = dl.find_dataset_root(config)
        print(f"[dataset] usable dataset already present at {existing}; skipping download")
        record["source"] = "pre-existing"
        record["root"] = str(existing)
    except dl.DatasetError:
        if (dest / "pv").exists():
            print("[dataset] incomplete checkout found; completing the sparse checkout")
        record["source"] = "sparse git clone"
        prov = sparse_clone_maize(dest)
        record.update({"provenance": prov, "root": prov["local_root"]})

    root = Path(record["root"])
    record["verification"] = verify_dataset(root, config)
    if record["verification"]["total_images"] == 0:
        raise DatasetSetupError(
            f"The dataset at {root} contains 0 usable images. Delete {root} and "
            "re-run, or extract the archive manually (data/README.md)."
        )
    prov_path = dest / "dataset_provenance.json"
    prov_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["provenance_file"] = str(prov_path)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset_setup",
        description="Download/verify the PlantVillage maize subset used by this project.",
    )
    parser.add_argument("--dest", default=str(ProjectConfig().data_dir), help="Where to place the data/")
    parser.add_argument("--verify-only", action="store_true", help="Do not download, only inspect.")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    if args.verify_only:
        from . import data_loader as dl

        root = dl.find_dataset_root(ProjectConfig(data_dir=dest))
        report = verify_dataset(root)
        print(json.dumps(report, indent=2))
        return 0
    record = prepare_dataset(dest)
    print("\n=== dataset ready ===")
    print(f"root: {record['root']}")
    for name, count in record["verification"]["images_per_class"].items():
        print(f"  {name:<22} {count:>5} images")
    print(f"  {'TOTAL':<22} {record['verification']['total_images']:>5} images")
    if record["verification"]["unrecognised_dirs"]:
        print(f"  ignored (unrecognised) folders: {record['verification']['unrecognised_dirs']}")
    if record["verification"]["out_of_scope_dirs"]:
        print(
            "  maize folders outside the 4-class scope, excluded: "
            f"{record['verification']['out_of_scope_dirs']}"
        )
    print(f"provenance recorded in {record.get('provenance_file')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
