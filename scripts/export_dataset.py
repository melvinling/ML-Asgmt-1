"""Export the Olivetti dataset as individual PNGs with a CSV index."""

import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from face_recognition import ROOT, load_dataset


def export_dataset(output):
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new folder with --output.")
    images, labels = load_dataset()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dataset-export-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        with (stage / "labels.csv").open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(["sample_index", "subject_id", "filename"])
            for index, (photo, subject) in enumerate(zip(images, labels, strict=True)):
                filename = f"sample_{index:03d}_subject_{subject:02d}.png"
                pixels = np.rint(photo * 255).astype(np.uint8)
                Image.fromarray(pixels).save(stage / filename)
                writer.writerow([index, subject, filename])
        stage.rename(output)
    print(f"Exported {len(images)} PNGs and labels.csv to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_png",
                        help="New folder for the PNGs and CSV (default: dataset_png).")
    args = parser.parse_args()
    try:
        export_dataset(args.output.resolve())
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Dataset export failed: {error}\n")


if __name__ == "__main__":
    main()
