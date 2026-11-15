"""Train and demonstrate an eigenface recognizer for KNR4633 Assignment 1."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "olivetti_faces.npz"
RESULTS = ROOT / "results"
SEED = 4633
IMAGE_SIZE = (64, 64)
COMPONENT_COUNTS = (10, 25, 50, 100)


def load_dataset():
    with np.load(DATASET, allow_pickle=False) as archive:
        images = archive["images"]
        labels = archive["labels"]
    validate_dataset(images, labels)
    return images, labels


def validate_dataset(images, labels):
    if images.shape != (400, *IMAGE_SIZE) or labels.shape != (400,):
        raise ValueError("Expected 400 grayscale 64 × 64 images and 400 labels.")
    if not np.isfinite(images).all() or images.min() != 0 or images.max() != 1:
        raise ValueError("Expected finite pixel values spanning [0, 1].")
    subjects, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(subjects, np.arange(40)) or not np.all(counts == 10):
        raise ValueError("Expected ten images for each subject ID from 0 to 39.")


def split_dataset(labels):
    rng = np.random.default_rng(SEED)
    subjects = rng.permutation(np.unique(labels))
    enrolled = subjects[:30]
    validation_unknown = subjects[30:35]
    test_unknown = subjects[35:]
    grouped = np.stack([rng.permutation(np.flatnonzero(labels == s)) for s in enrolled])
    return {
        "train": grouped[:, :6].ravel(),
        "validation_known": grouped[:, 6:8].ravel(),
        "test_known": grouped[:, 8:].ravel(),
        "validation_unknown": np.flatnonzero(np.isin(labels, validation_unknown)),
        "test_unknown": np.flatnonzero(np.isin(labels, test_unknown)),
    }


def fit_eigenfaces(samples):
    """Return the training mean, positive covariance eigenvalues and eigenfaces."""
    n_samples, n_features = samples.shape
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / (n_samples - 1)
    gram = centered @ centered.T / (n_samples - 1)

    eigenvalues, gram_vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    tolerance = np.finfo(gram.dtype).eps * max(n_samples, n_features) * eigenvalues[order[0]]
    keep = order[eigenvalues[order] > tolerance]
    if keep.size == 0:
        raise ValueError("Training images must contain some variation.")
    eigenvalues = eigenvalues[keep]
    bases = centered.T @ gram_vectors[:, keep]
    bases /= np.sqrt((n_samples - 1) * eigenvalues)

    # Check that the recovered directions diagonalize the pixel covariance.
    residual = covariance @ bases - bases * eigenvalues
    if np.linalg.norm(residual) > 1e-8 * np.linalg.norm(eigenvalues):
        raise ValueError("Recovered eigenfaces failed the pixel covariance check.")
    return mean, eigenvalues, bases, float(np.trace(covariance))


def project(samples, mean, basis):
    # Use the same reduction for single-image inference and batched evaluation.
    return np.einsum("ij,jk->ik", samples - mean, basis)


def nearest(probes, gallery, labels):
    """Compute Euclidean distance to the closest gallery image for each probe."""
    matches = np.empty(len(probes), dtype=labels.dtype)
    distances = np.empty(len(probes), dtype=np.float64)
    for i, probe in enumerate(probes):
        differences = gallery - probe
        squared_distances = np.einsum("ij,ij->i", differences, differences)
        index = squared_distances.argmin()
        matches[i] = labels[index]
        distances[i] = np.sqrt(squared_distances[index])
    return matches, distances


def calibrate_threshold(known_true, known_matches, known_distances, unknown_distances):
    scores = np.concatenate([known_distances, unknown_distances])
    # Include the operating point which rejects every validation image.
    candidates = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores)]
    correct = known_matches == known_true

    def objective(threshold):
        correct_known = np.count_nonzero(correct & (known_distances <= threshold))
        rejected_unknown = np.count_nonzero(unknown_distances > threshold)
        # Multiply the two rates by their common denominator for exact ties.
        return correct_known * len(unknown_distances) + rejected_unknown * len(known_distances)

    return float(max(candidates, key=objective))


def evaluate(labels, known_indices, unknown_indices, matches, distances, threshold):
    indices = np.r_[known_indices, unknown_indices]
    is_enrolled = np.arange(len(indices)) < len(known_indices)
    accepted = distances <= threshold
    correct = matches == labels[indices]
    known_total = len(known_indices)
    unknown_total = len(unknown_indices)
    rank1_correct = int(np.sum(is_enrolled & correct))
    correctly_identified = int(np.sum(is_enrolled & accepted & correct))
    false_accepts = int(np.sum(~is_enrolled & accepted))
    metrics = {
        "threshold": threshold,
        "known_total": known_total,
        "unknown_total": unknown_total,
        "rank1_correct": rank1_correct,
        "rank1_accuracy": rank1_correct / known_total,
        "enrolled_identified": correctly_identified,
        "enrolled_rejected": int(np.sum(is_enrolled & ~accepted)),
        "enrolled_misidentified": int(np.sum(is_enrolled & accepted & ~correct)),
        "unknown_false_accepts": false_accepts,
        "unknown_rejected": unknown_total - false_accepts,
        "unknown_false_accept_rate": false_accepts / unknown_total,
        "correct_identification_rate": correctly_identified / known_total,
    }
    rows = [
        {
            "sample_index": int(index),
            "actual_subject_id": int(labels[index]),
            "enrolled": bool(enrolled),
            "nearest_subject_id": int(match),
            "distance": float(distance),
            "prediction": int(match) if accept else "Unknown",
        }
        for index, enrolled, match, distance, accept in
        zip(indices, is_enrolled, matches, distances, accepted, strict=True)
    ]
    return metrics, rows


def evaluate_model(features, labels, split, gallery):
    gallery_labels = labels[split["train"]]
    validation_known = split["validation_known"]
    validation_unknown = split["validation_unknown"]
    validation_indices = np.r_[validation_known, validation_unknown]
    matches, distances = nearest(features[validation_indices], gallery, gallery_labels)
    n_known = len(validation_known)
    threshold = calibrate_threshold(
        labels[validation_known], matches[:n_known], distances[:n_known], distances[n_known:]
    )
    validation, _ = evaluate(
        labels, validation_known, validation_unknown, matches, distances, threshold
    )

    test_known = split["test_known"]
    test_unknown = split["test_unknown"]
    matches, distances = nearest(
        features[np.r_[test_known, test_unknown]], gallery, gallery_labels
    )
    test, rows = evaluate(labels, test_known, test_unknown, matches, distances, threshold)
    return {"validation": validation, "test": test}, rows


def save_figures(output, mean, eigenvalues, basis, total_variance, sweep, best):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), layout="constrained")
    for i, axis in enumerate(axes.flat):
        if i == 0:
            axis.imshow(mean.reshape(IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
            axis.set_title("Training mean")
        else:
            eigenface = basis[:, i - 1].reshape(IMAGE_SIZE)
            limit = np.abs(eigenface).max()
            axis.imshow(eigenface, cmap="gray", vmin=-limit, vmax=limit)
            axis.set_title(f"Eigenface {i}")
        axis.set_axis_off()
    figure.savefig(output / "eigenfaces.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
    axes[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues.cumsum() / total_variance)
    axes[0].axvline(best, color="#9f4c24", linestyle="--", label=f"Selected m = {best}")
    axes[0].set(xlabel="Components retained", ylabel="Cumulative explained variance", ylim=(0, 1.03))
    axes[0].legend(frameon=False, loc="lower right")
    counts = [row["components"] for row in sweep]
    accuracy = [row["rank1_accuracy"] for row in sweep]
    axes[1].plot(counts, accuracy, marker="o")
    axes[1].scatter(best, accuracy[counts.index(best)], color="#9f4c24", zorder=3)
    axes[1].set(xlabel="Components retained", ylabel="Validation rank-1 accuracy", xticks=counts)
    for axis in axes:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "component_sweep.png", dpi=220)
    plt.close(figure)


def train(output):
    if output.exists():
        raise ValueError(f"Results already exist at {output}. Use --run-dir to start a new run.")
    images, labels = load_dataset()
    samples = images.reshape(len(images), -1).astype(np.float64)
    split = split_dataset(labels)
    train_indices = split["train"]
    print(f"Fitting PCA on {len(train_indices)} training images...", flush=True)
    mean, eigenvalues, basis, total_variance = fit_eigenfaces(samples[train_indices])
    if basis.shape[1] < max(COMPONENT_COUNTS):
        raise ValueError("The training split has too few positive eigenvalues for the sweep.")

    validation_known = split["validation_known"]
    sweep = []
    projections = project(samples, mean, basis[:, :max(COMPONENT_COUNTS)])
    for count in COMPONENT_COUNTS:
        matches, _ = nearest(
            projections[validation_known, :count], projections[train_indices, :count],
            labels[train_indices]
        )
        correct = int(np.sum(matches == labels[validation_known]))
        sweep.append({
            "components": count,
            "rank1_correct": correct,
            "rank1_accuracy": correct / len(validation_known),
            "explained_variance": float(eigenvalues[:count].sum() / total_variance),
        })
    best = max(sweep, key=lambda row: (row["rank1_correct"], -row["components"]))["components"]
    selected_basis = np.ascontiguousarray(basis[:, :best])
    features = project(samples, mean, selected_basis)
    gallery = features[train_indices]
    pca_metrics, pca_rows = evaluate_model(features, labels, split, gallery)
    raw_metrics, raw_rows = evaluate_model(samples, labels, split, samples[train_indices])
    with DATASET.open("rb") as file:
        dataset_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    metrics = {
        "seed": SEED,
        "dataset_sha256": dataset_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "split_sizes": {name: len(indices) for name, indices in split.items()},
        "selected_components": best,
        "component_sweep": sweep,
        "pca_1nn": pca_metrics,
        "raw_pixel_1nn": raw_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-", dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        np.savez_compressed(
            stage / "model.npz", mean=mean, basis=selected_basis, gallery=gallery,
            labels=labels[train_indices], threshold=pca_metrics["test"]["threshold"]
        )
        np.savez_compressed(stage / "split_indices.npz", **split)
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        with (stage / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["model", *pca_rows[0]])
            writer.writeheader()
            for name, rows in (("pca_1nn", pca_rows), ("raw_pixel_1nn", raw_rows)):
                writer.writerows({"model": name, **row} for row in rows)
        save_figures(stage, mean, eigenvalues, basis, total_variance, sweep, best)
        stage.rename(output)
    print(f"Selected {best} components. Model, metrics and figures saved to {output}")


def predict(model_path, sample_index, image_path, preview_path=None):
    if not model_path.is_file():
        raise ValueError(f"Model file is missing at {model_path}. Run the train command first.")
    if sample_index is not None:
        images, subject_ids = load_dataset()
        sample = images[sample_index]
        actual_subject = int(subject_ids[sample_index])
    else:
        with Image.open(image_path) as image:
            gray = ImageOps.exif_transpose(image).convert("L")
            resized = gray.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            sample = np.asarray(resized, dtype=np.float64) / 255
    probe = sample.astype(np.float64).reshape(1, -1)
    with np.load(model_path, allow_pickle=False) as model:
        enrolled_subjects = model["labels"]
        features = project(probe, model["mean"], model["basis"])
        matches, distances = nearest(features, model["gallery"], enrolled_subjects)
        threshold = float(model["threshold"].item())
    distance = float(distances[0])
    prediction = f"Subject {matches[0]}" if distance <= threshold else "Unknown"
    print(f"{prediction} (distance {distance:.6g}, threshold {threshold:.6g})")
    if sample_index is not None:
        enrolled = actual_subject in enrolled_subjects
        expected = f"Subject {actual_subject}" if enrolled else "Unknown"
        status = "enrolled" if enrolled else "unenrolled"
        verdict = "Correct" if prediction == expected else "Incorrect"
        print(f"Dataset sample {sample_index}: Subject {actual_subject} ({status})")
        print(f"Expected: {expected}. {verdict}.")
    if preview_path is not None:
        pixels = np.rint(sample * 255).astype(np.uint8)
        preview = Image.fromarray(pixels).resize((256, 256), Image.Resampling.NEAREST)
        with preview_path.open("xb") as file:
            preview.save(file, format="PNG")
        print(f"Photo preview saved to {preview_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-data", help="Validate the bundled dataset and show split sizes.")
    training = commands.add_parser("train", help="Fit, evaluate and save the eigenface model.")
    training.add_argument("--run-dir", type=Path, default=RESULTS, help="New output directory.")
    prediction = commands.add_parser("predict", help="Classify a dataset sample or a cropped face.")
    prediction.add_argument("--model", type=Path, default=RESULTS / "model.npz")
    prediction.add_argument("--preview", type=Path, metavar="PNG",
                            help="Save the processed input photo to a new PNG file.")
    inputs = prediction.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sample-index", type=int, metavar="0–399")
    inputs.add_argument("--image", type=Path, help="Path to a cropped frontal face image.")
    args = parser.parse_args()
    if args.command == "predict" and args.sample_index is not None:
        if not 0 <= args.sample_index < 400:
            parser.error("--sample-index must be between 0 and 399")
    if args.command == "predict" and args.preview is not None:
        if args.preview.suffix.lower() != ".png":
            parser.error("--preview requires a filename ending in .png")
    try:
        if args.command == "train":
            train(args.run_dir.resolve())
        elif args.command == "predict":
            predict(args.model, args.sample_index, args.image, args.preview)
        else:
            images, labels = load_dataset()
            print(f"{len(images)} images, {len(np.unique(labels))} subjects, seed {SEED}")
            for name, indices in split_dataset(labels).items():
                print(f"{name}: {len(indices)} images")
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
