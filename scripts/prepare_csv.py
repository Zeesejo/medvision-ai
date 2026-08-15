"""
Prepare deterministic NIH ChestX-ray14 train/validation/test manifests.

The official NIH train/test split is preserved. The train pool is further split
into train/validation sets at the patient level when a patient identifier is
available, preventing the same patient from appearing in both partitions.

Example:
    python scripts/prepare_csv.py \
        --archive_dir /path/to/nih-chestxray14 \
        --output_dir data/splits \
        --val_frac 0.10 \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.constants import CLASS_NAMES

IMAGE_SUBDIRS = [f"images_{i:03d}/images" for i in range(1, 13)]


def find_image(archive_dir: str, filename: str) -> str | None:
    for sub in IMAGE_SUBDIRS:
        candidate = os.path.join(archive_dir, sub, filename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    candidate = os.path.join(archive_dir, "images", filename)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    return None


def load_metadata(archive_dir: str) -> pd.DataFrame:
    meta_path = os.path.join(archive_dir, "Data_Entry_2017.csv")
    meta = pd.read_csv(meta_path)
    required = {"Image Index", "Finding Labels"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    return meta


def build_dataframe(meta: pd.DataFrame, archive_dir: str, filenames: list[str]) -> pd.DataFrame:
    frame = meta[meta["Image Index"].isin(set(filenames))].copy()

    for cls in CLASS_NAMES:
        frame[cls] = frame["Finding Labels"].apply(
            lambda value: int(cls in str(value).split("|"))
        )

    print(f"Resolving {len(frame):,} image paths ...", end="", flush=True)
    frame["image_path"] = frame["Image Index"].apply(
        lambda filename: find_image(archive_dir, filename)
    )
    missing = int(frame["image_path"].isna().sum())
    frame = frame.dropna(subset=["image_path"]).reset_index(drop=True)
    print(f" done ({missing} missing files skipped)")

    keep = ["Image Index", "image_path"]
    if "Patient ID" in frame.columns:
        keep.append("Patient ID")
    keep.extend(CLASS_NAMES)
    return frame[keep]


def split_train_validation(
    pool: pd.DataFrame,
    val_frac: float,
    seed: int,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if strategy == "patient" and "Patient ID" in pool.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(splitter.split(pool, groups=pool["Patient ID"]))
        train_df = pool.iloc[train_idx].copy()
        val_df = pool.iloc[val_idx].copy()
        effective_strategy = "patient"
    else:
        train_df, val_df = train_test_split(
            pool,
            test_size=val_frac,
            random_state=seed,
            shuffle=True,
        )
        effective_strategy = "image"

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        effective_strategy,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reproducible ChestX-ray14 split CSVs")
    parser.add_argument("--archive_dir", required=True, help="Root of the NIH ChestX-ray14 archive")
    parser.add_argument("--output_dir", default="data/splits", help="Where split manifests are written")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["patient", "image"],
        default="patient",
        help="Validation split strategy. Patient-level is preferred for publication runs.",
    )
    args = parser.parse_args()

    archive_dir = os.path.abspath(args.archive_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(archive_dir)

    with open(os.path.join(archive_dir, "train_val_list.txt"), encoding="utf-8") as handle:
        train_val_files = [line.strip() for line in handle if line.strip()]
    with open(os.path.join(archive_dir, "test_list.txt"), encoding="utf-8") as handle:
        test_files = [line.strip() for line in handle if line.strip()]

    print(f"Official train/val pool: {len(train_val_files):,} images")
    print(f"Official test pool     : {len(test_files):,} images")

    train_val_pool = build_dataframe(meta, archive_dir, train_val_files)
    test_df = build_dataframe(meta, archive_dir, test_files)
    train_df, val_df, effective_strategy = split_train_validation(
        train_val_pool,
        val_frac=args.val_frac,
        seed=args.seed,
        strategy=args.strategy,
    )

    if effective_strategy == "patient":
        train_patients = set(train_df["Patient ID"].tolist())
        val_patients = set(val_df["Patient ID"].tolist())
        overlap = train_patients.intersection(val_patients)
        if overlap:
            raise RuntimeError(f"Patient leakage detected between train and validation: {len(overlap)} patients")

    paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    manifest = {
        "dataset": "NIH ChestX-ray14",
        "archive_dir": archive_dir,
        "seed": args.seed,
        "requested_strategy": args.strategy,
        "effective_strategy": effective_strategy,
        "validation_fraction": args.val_frac,
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "split_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nPrepared split manifests")
    print(f"Strategy: {effective_strategy}")
    print(f"Train   : {len(train_df):,}")
    print(f"Val     : {len(val_df):,}")
    print(f"Test    : {len(test_df):,}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
