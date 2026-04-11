"""
prepare_csv.py — Generate train_val.csv, val.csv, test.csv for MedVision-AI
============================================================================

Reads the official NIH ChestX-ray14 metadata and split lists, one-hot
encodes the 14 pathology labels, resolves each image to its actual path
(images live in images_001 … images_012 sub-folders), and writes:

    <archive_dir>/train_val.csv   — training + validation pool
    <archive_dir>/val.csv         — 10 % random hold-out from train_val
    <archive_dir>/test.csv        — official test split

Usage (run once before training):
    python scripts/prepare_csv.py

Or with explicit paths:
    python scripts/prepare_csv.py \\
        --archive_dir "F:/NIH Chest X-Ray dataset/archive" \\
        --val_frac 0.10 \\
        --seed 42
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

IMAGE_SUBDIRS = [f"images_{i:03d}/images" for i in range(1, 13)]


def find_image(archive_dir: str, filename: str) -> str | None:
    """Return the full path to filename searching all images_00x sub-folders."""
    for sub in IMAGE_SUBDIRS:
        p = os.path.join(archive_dir, sub, filename)
        if os.path.isfile(p):
            return p
    # Flat fallback (some Kaggle archives extract without sub-folders)
    p = os.path.join(archive_dir, "images", filename)
    if os.path.isfile(p):
        return p
    return None


def build_dataframe(archive_dir: str, filenames: list[str]) -> pd.DataFrame:
    """
    Load Data_Entry_2017.csv, filter to filenames, resolve image paths,
    one-hot encode labels, and drop rows whose image is missing on disk.
    """
    meta_path = os.path.join(archive_dir, "Data_Entry_2017.csv")
    meta = pd.read_csv(meta_path)

    # Keep only the rows in our split
    meta = meta[meta["Image Index"].isin(set(filenames))].copy()

    # One-hot encode the pipe-separated Finding Labels column
    for cls in CLASSES:
        meta[cls] = meta["Finding Labels"].apply(
            lambda s: 1 if cls in str(s).split("|") else 0
        )

    # Resolve each image to its actual path on disk
    print(f"  Resolving {len(meta):,} image paths …", end="", flush=True)
    meta["image_path"] = meta["Image Index"].apply(
        lambda fn: find_image(archive_dir, fn)
    )
    missing = meta["image_path"].isna().sum()
    meta = meta.dropna(subset=["image_path"])
    print(f" done  ({missing} missing files skipped)")

    # Keep only columns the trainer needs
    keep = ["Image Index", "image_path"] + CLASSES
    return meta[keep].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare NIH ChestX-ray14 CSVs")
    parser.add_argument(
        "--archive_dir",
        default="F:/NIH Chest X-Ray dataset/archive",
        help="Root of the NIH archive (contains Data_Entry_2017.csv)",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.10,
        help="Fraction of train_val to hold out as val.csv (default 0.10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arch = args.archive_dir

    # ── Load official split lists ─────────────────────────────────────────────
    tv_list_path = os.path.join(arch, "train_val_list.txt")
    te_list_path = os.path.join(arch, "test_list.txt")

    with open(tv_list_path) as f:
        train_val_files = [l.strip() for l in f if l.strip()]
    with open(te_list_path) as f:
        test_files = [l.strip() for l in f if l.strip()]

    print(f"train_val pool : {len(train_val_files):,} images")
    print(f"test pool      : {len(test_files):,} images")

    # ── Build DataFrames ──────────────────────────────────────────────────────
    print("\nBuilding train_val DataFrame …")
    tv_df = build_dataframe(arch, train_val_files)

    print("Building test DataFrame …")
    te_df = build_dataframe(arch, test_files)

    # ── Split train_val → train + val ─────────────────────────────────────────
    train_df, val_df = train_test_split(
        tv_df, test_size=args.val_frac, random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    tv_csv  = os.path.join(arch, "train_val.csv")
    val_csv = os.path.join(arch, "val.csv")
    te_csv  = os.path.join(arch, "test.csv")

    train_df.to_csv(tv_csv,  index=False)
    val_df.to_csv(val_csv,   index=False)
    te_df.to_csv(te_csv,     index=False)

    print(f"\n✓ train_val.csv  → {tv_csv}   ({len(train_df):,} rows)")
    print(f"✓ val.csv        → {val_csv}  ({len(val_df):,} rows)")
    print(f"✓ test.csv       → {te_csv}   ({len(te_df):,} rows)")
    print("\nDone — now run:")
    print(f"  python src/train_v3.py "
          f"--train_csv \"{tv_csv}\" "
          f"--val_csv \"{val_csv}\" "
          f"--img_dir \"{arch}\" "
          f"--backbone densenet121 "
          f"--wb_entity zeemaokik-university-of-bremen "
          f"--num_workers 0")


if __name__ == "__main__":
    main()
