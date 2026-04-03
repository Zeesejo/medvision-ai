"""
Download MedMNIST datasets via the medmnist package.
All datasets are standardized 28x28 or 64x64 images.

Usage:
    python data/scripts/download_medmnist.py --dataset chestmnist --output_dir data/raw/medmnist

Available datasets: pathmnist, chestmnist, dermamnist, octmnist, pneumoniamnist,
                    retinamnist, breastmnist, organamnist, organcmnist, organSmnist
"""
import argparse
from pathlib import Path


def download(dataset: str, output_dir: str, size: int = 64) -> None:
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise ImportError("Run: pip install medmnist")

    info = INFO[dataset]
    DataClass = getattr(medmnist, info["python_class"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        DataClass(split=split, download=True, size=size, root=str(out))
        print(f"Downloaded {dataset} [{split}] to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chestmnist")
    parser.add_argument("--output_dir", type=str, default="data/raw/medmnist")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    download(args.dataset, args.output_dir, args.size)
