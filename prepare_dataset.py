"""Create the working dataset for the supported Indian currency notes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUPPORTED_DENOMINATIONS = ("10", "20", "50", "100", "200", "500")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def prepare_dataset(source: Path, destination: Path, clean: bool = False) -> dict[str, int]:
    """Copy only supported denomination folders into a separate dataset."""
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")

    if destination.exists() and clean:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for denomination in SUPPORTED_DENOMINATIONS:
        source_class = source / denomination
        if not source_class.is_dir():
            raise FileNotFoundError(f"Missing denomination folder: {source_class}")

        destination_class = destination / denomination
        destination_class.mkdir(exist_ok=True)
        image_paths = sorted(
            path
            for path in source_class.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            shutil.copy2(image_path, destination_class / image_path.name)
        counts[denomination] = len(image_paths)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the supported INR denomination folders."
    )
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data_filtered"))
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace an existing output directory before copying.",
    )
    args = parser.parse_args()

    counts = prepare_dataset(args.source, args.output, args.clean)
    print(f"Prepared dataset: {args.output}")
    for denomination, count in counts.items():
        print(f"  {denomination}: {count} images")


if __name__ == "__main__":
    main()
