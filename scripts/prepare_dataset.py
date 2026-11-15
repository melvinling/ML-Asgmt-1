"""Rebuild the local dataset without fitting a model."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from face_recognition import DATASET, ROOT, validate_dataset


MAT_SHA256 = "b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794"


def from_mat(path):
    from scipy.io import loadmat

    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != MAT_SHA256:
        raise ValueError("The MAT file does not match the checksum in scikit-learn's loader.")
    # Match scikit-learn's pixel ordering and scaling for the same source file.
    faces = np.float32(loadmat(path)["faces"].T)
    faces -= faces.min()
    faces /= faces.max()
    images = faces.reshape(400, 64, 64).transpose(0, 2, 1)
    return images, np.arange(400) // 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, help="Use a verified local copy of olivettifaces.mat.")
    args = parser.parse_args()
    try:
        with TemporaryDirectory(prefix="olivetti-", dir=ROOT) as temporary:
            if args.mat is None:
                from sklearn.datasets import fetch_olivetti_faces

                dataset = fetch_olivetti_faces(data_home=temporary, shuffle=False)
                images, labels = dataset.images, dataset.target
            else:
                images, labels = from_mat(args.mat)
            validate_dataset(images, labels)
            staged = Path(temporary) / DATASET.name
            np.savez_compressed(staged, images=images, labels=labels.astype(np.int16))
            staged.replace(DATASET)
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    with DATASET.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    print(f"Prepared {DATASET}\nSHA-256 {sha256}")


if __name__ == "__main__":
    main()
