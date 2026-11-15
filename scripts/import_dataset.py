"""Rebuild an Olivetti NPZ archive from exported PNGs and labels.csv."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from face_recognition import IMAGE_SIZE, ROOT, validate_dataset


def import_dataset(source, output):
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new file with --output.")
    with (source / "labels.csv").open(newline="", encoding="utf-8-sig") as manifest:
        rows = sorted(csv.DictReader(manifest), key=lambda row: int(row["sample_index"]))
    if [int(row["sample_index"]) for row in rows] != list(range(400)):
        raise ValueError("labels.csv must list every sample index from 0 to 399 once.")

    images = []
    labels = []
    filenames = set()
    for row in rows:
        filename = row["filename"]
        path = Path(filename)
        if path.name != filename or path.suffix.lower() != ".png":
            raise ValueError(f"Expected a PNG filename within the dataset folder: {filename!r}")
        if filename.casefold() in filenames:
            raise ValueError(f"Duplicate filename in labels.csv: {filename}")
        filenames.add(filename.casefold())
        with Image.open(source / path) as photo:
            if photo.format != "PNG" or photo.mode != "L" or photo.size != IMAGE_SIZE:
                raise ValueError(f"{filename} must be an 8-bit grayscale 64 × 64 PNG.")
            images.append(np.asarray(photo, dtype=np.float32) / 255)
        labels.append(int(row["subject_id"]))

    images = np.stack(images)
    labels = np.asarray(labels)
    validate_dataset(images, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dataset-import-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with staged.open("wb") as file:
            np.savez_compressed(file, images=images, labels=labels.astype(np.int16))
        staged.rename(output)
    print(f"Packed {len(images)} PNGs and their subject IDs into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dataset_png",
                        help="Folder with the PNGs and labels.csv (default: dataset_png).")
    parser.add_argument("--output", type=Path, default=ROOT / "olivetti_faces_rebuilt.npz",
                        help="New NPZ file (default: olivetti_faces_rebuilt.npz).")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("--output requires a filename ending in .npz")
    try:
        import_dataset(args.input.resolve(), args.output.resolve())
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Dataset import failed: {error}\n")


if __name__ == "__main__":
    main()
