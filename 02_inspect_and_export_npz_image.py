"""Inspect the prepared image datasets and save an example image to disk."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


DATA_DIRECTORY = Path(__file__).parent / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"

# These names describe the label values used by the prepared datasets.
DATASETS = {
    "pneumonia": {
        "file": DATA_DIRECTORY / "pneumoniamnist_224.npz",
        "labels": {0: "normal", 1: "pneumonia"},
    },
    "retina": {
        "file": DATA_DIRECTORY / "retinamnist_224.npz",
        "labels": {
            0: "no diabetic retinopathy",
            1: "mild diabetic retinopathy",
            2: "moderate diabetic retinopathy",
            3: "severe diabetic retinopathy",
            4: "proliferative diabetic retinopathy",
        },
    },
}


def inspect_dataset(dataset_name: str) -> None:
    """Print each split's image shape and the number of examples per label."""
    dataset = DATASETS[dataset_name]

    # NPZ files are zip archives of NumPy arrays. allow_pickle=False avoids
    # loading arbitrary Python objects from a file.
    with np.load(dataset["file"], allow_pickle=False) as arrays:
        print(f"\nDataset: {dataset_name} ({dataset['file'].name})")
        for split in ("train", "val", "test"):
            images = arrays[f"{split}_images"]
            labels = arrays[f"{split}_labels"].reshape(-1)
            label_counts = Counter(int(label) for label in labels)

            print(f"  {split}: image shape {images.shape}, data type {images.dtype}")
            for label, count in sorted(label_counts.items()):
                label_name = dataset["labels"].get(label, "unknown label")
                print(f"    label {label} ({label_name}): {count} images")


def load_image_sample(
    dataset_name: str, split: str = "test", image_index: int = 0
) -> tuple[np.ndarray, int]:
    """Load one image and its label from an NPZ split."""
    dataset = DATASETS[dataset_name]
    with np.load(dataset["file"], allow_pickle=False) as arrays:
        images = arrays[f"{split}_images"]
        labels = arrays[f"{split}_labels"].reshape(-1)

        if image_index < 0 or image_index >= len(images):
            raise IndexError(f"Image index must be between 0 and {len(images) - 1}.")

        # Copy the image before closing the NPZ file.
        return images[image_index].copy(), int(labels[image_index])


def save_image_sample(
    dataset_name: str, split: str = "test", image_index: int = 0
) -> Path:
    """Save one selected image under data/test and return its filename."""
    image_array, label = load_image_sample(dataset_name, split, image_index)
    label_name = DATASETS[dataset_name]["labels"].get(label, "unknown")
    safe_label_name = label_name.replace(" ", "_")
    image_path = OUTPUT_DIRECTORY / (
        f"{dataset_name}_{split}_{image_index:04d}_label-{label}_{safe_label_name}.png"
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(image_path)
    print(f"Saved label {label} ({label_name}) to: {image_path}")
    return image_path


if __name__ == "__main__":
    # Each call is a small, notebook-like activity and can also be run on its own.
    inspect_dataset("pneumonia")
    inspect_dataset("retina")
    save_image_sample("pneumonia", split="test", image_index=0)
    save_image_sample("retina", split="test", image_index=0)
