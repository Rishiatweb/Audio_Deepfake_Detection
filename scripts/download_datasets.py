"""Download FoR + ITW datasets and reorganize into expected directory structure.

FoR: Kaggle (requires ~/.kaggle/kaggle.json)
ITW: HuggingFace (public, no auth)

Usage:
    python scripts/download_datasets.py --datasets for itw
    python scripts/download_datasets.py --datasets for   # FoR only
    python scripts/download_datasets.py --datasets itw   # ITW only
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


# ─── FoR ─────────────────────────────────────────────────────────────────────

def download_for(data_dir: Path) -> None:
    """Download FoR dataset via Kaggle API."""
    try:
        import kaggle  # noqa: F401 — triggers credential check
    except ImportError:
        print("ERROR: kaggle not installed. Run: uv pip install kaggle")
        return

    credentials = Path.home() / ".kaggle" / "kaggle.json"
    if not credentials.exists():
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("  1. Go to kaggle.com -> Account -> API -> Create New Token")
        print("  2. Save kaggle.json to ~/.kaggle/kaggle.json")
        return

    out = data_dir / "for_download"
    out.mkdir(parents=True, exist_ok=True)

    zip_path = out / "the-fake-or-real-dataset.zip"
    if zip_path.exists():
        print(f"FoR zip already exists at {zip_path}, skipping download.")
    else:
        print("Downloading FoR dataset from Kaggle (~17 GB)...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", "mohammedabdeldayem/the-fake-or-real-dataset",
             "-p", str(out)],
            check=True,
        )
        print(f"Downloaded: {zip_path}")

    _extract_for(zip_path, data_dir / "for-dataset")


def _extract_for(zip_path: Path, dest: Path) -> None:
    """Extract FoR zip to expected directory structure."""
    if dest.exists() and any(dest.iterdir()):
        print(f"FoR already extracted at {dest}, skipping.")
        return

    print(f"Extracting FoR to {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("FoR extraction complete.")
    _verify_for(dest)


def _verify_for(dest: Path) -> None:
    subsets = ["for-original", "for-norm", "for-2seconds", "for-rerecorded"]
    found = []
    for sub in subsets:
        # Check both flat and nested structures
        candidates = [
            dest / sub / sub,
            dest / sub,
            dest / sub.replace("-", "_"),
        ]
        for c in candidates:
            if c.exists():
                found.append(sub)
                break
    print(f"FoR subsets found: {found}")
    if len(found) < 4:
        missing = [s for s in subsets if s not in found]
        print(f"WARNING: Missing subsets: {missing}")
        print(f"  Check structure under: {dest}")
        _show_tree(dest, max_depth=3)


def _show_tree(path: Path, max_depth: int = 2, _depth: int = 0) -> None:
    if _depth > max_depth:
        return
    indent = "  " * _depth
    for child in sorted(path.iterdir())[:8]:
        print(f"{indent}{child.name}/")
        if child.is_dir() and _depth < max_depth:
            _show_tree(child, max_depth, _depth + 1)


# ─── ITW ─────────────────────────────────────────────────────────────────────

def download_itw(data_dir: Path) -> None:
    """Download ITW dataset from HuggingFace (public)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: uv pip install huggingface_hub")
        return

    out = data_dir / "itw_download"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "release_in_the_wild.zip"

    if zip_path.exists():
        print(f"ITW zip already exists at {zip_path}, skipping download.")
    else:
        print("Downloading ITW dataset from HuggingFace (~8 GB)...")
        downloaded = hf_hub_download(
            repo_id="mueller91/In-The-Wild",
            filename="release_in_the_wild.zip",
            repo_type="dataset",
            local_dir=str(out),
        )
        print(f"Downloaded: {downloaded}")
        zip_path = Path(downloaded)

    _extract_itw(zip_path, data_dir / "in-the-wild")


def _extract_itw(zip_path: Path, dest: Path) -> None:
    """Extract and reorganize ITW into real/ and fake/ subdirs."""
    if dest.exists() and any(dest.iterdir()):
        print(f"ITW already extracted at {dest}, skipping.")
        return

    print(f"Extracting ITW to {dest} ...")
    dest.mkdir(parents=True, exist_ok=True)

    # Inspect zip structure first
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"  Zip contains {len(names)} files.")
        if names:
            print(f"  Sample paths: {names[:5]}")

    # Extract to temp
    tmp = dest.parent / "itw_tmp"
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    # Reorganize into real/ fake/ based on label
    # ITW structure: flat dir with meta.csv OR subdirs real/fake
    _reorganize_itw(tmp, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ITW extraction and reorganization complete.")


def _reorganize_itw(src: Path, dest: Path) -> None:
    """Map ITW extracted files into dest/real/ and dest/fake/."""
    (dest / "real").mkdir(parents=True, exist_ok=True)
    (dest / "fake").mkdir(parents=True, exist_ok=True)

    # Case 1: already has real/fake subdirs
    if (src / "real").exists() or (src / "fake").exists():
        for lbl in ["real", "fake"]:
            for wav in (src / lbl).rglob("*.wav"):
                shutil.copy2(wav, dest / lbl / wav.name)
        return

    # Case 2: single flat dir with meta.csv
    # Find meta CSV
    metas = list(src.rglob("meta.csv")) + list(src.rglob("*.csv"))
    if metas:
        import pandas as pd
        meta = pd.read_csv(metas[0])
        print(f"  Meta CSV columns: {list(meta.columns)}")
        # Common column names: file, label / filename, label / path, label
        file_col = next((c for c in meta.columns if "file" in c.lower() or "path" in c.lower()), None)
        label_col = next((c for c in meta.columns if "label" in c.lower() or "class" in c.lower() or "fake" in c.lower()), None)
        if file_col and label_col:
            for _, row in meta.iterrows():
                wav = src / row[file_col]
                if not wav.exists():
                    # Search recursively
                    matches = list(src.rglob(Path(row[file_col]).name))
                    wav = matches[0] if matches else None
                if wav and wav.exists():
                    lbl = "fake" if str(row[label_col]).lower() in ("1", "fake", "spoof", "synthesized") else "real"
                    shutil.copy2(wav, dest / lbl / wav.name)
            return

    # Case 3: subdirs named by speaker/system — no meta; use dir name heuristic
    # Fallback: walk all wavs, use parent dir name
    real_keywords = {"real", "genuine", "original", "bona_fide", "bonafide"}
    fake_keywords = {"fake", "spoof", "synth", "tts", "vc", "clone", "deepfake"}
    n_real = n_fake = 0
    for wav in src.rglob("*.wav"):
        parts_lower = {p.lower() for p in wav.parts}
        if parts_lower & fake_keywords:
            shutil.copy2(wav, dest / "fake" / wav.name)
            n_fake += 1
        elif parts_lower & real_keywords:
            shutil.copy2(wav, dest / "real" / wav.name)
            n_real += 1
        else:
            # Unknown — put in real as default (won't bias much)
            shutil.copy2(wav, dest / "real" / wav.name)
            n_real += 1

    print(f"  Reorganized: {n_real} real, {n_fake} fake")
    if n_fake == 0:
        print("  WARNING: No fake files found. Check zip structure.")
        _show_tree(src, max_depth=3)


# ─── Config updater ───────────────────────────────────────────────────────────

def update_config_paths(data_dir: Path) -> None:
    """Update configs/default.yaml with absolute paths to downloaded data."""
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        return
    text = config_path.read_text(encoding="utf-8")
    for_path = (data_dir / "for-dataset").resolve()
    itw_path = (data_dir / "in-the-wild").resolve()
    # Replace path lines
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if "for_base:" in line:
            new_lines.append(f"  for_base: {for_path}")
        elif "itw_root:" in line:
            new_lines.append(f"  itw_root: {itw_path}")
        else:
            new_lines.append(line)
    config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Updated configs/default.yaml:")
    print(f"  for_base: {for_path}")
    print(f"  itw_root: {itw_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download FoR + ITW datasets")
    p.add_argument("--datasets", nargs="+", default=["for", "itw"],
                   choices=["for", "itw"], help="Which datasets to download")
    p.add_argument("--data-dir", default="data", help="Root data directory")
    p.add_argument("--update-config", action="store_true", default=True,
                   help="Update configs/default.yaml with paths")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if "itw" in args.datasets:
        download_itw(data_dir)
    if "for" in args.datasets:
        download_for(data_dir)

    if args.update_config:
        update_config_paths(data_dir)

    print("\nDataset download complete.")
    print("Next: python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
