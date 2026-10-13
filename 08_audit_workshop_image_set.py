"""Audit the curated image set and save label-hidden workshop contact sheets."""

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
EXPECTED_EXAMPLES_PER_LABEL = 3
RANDOM_SEED = 42

DATASETS: dict[str, dict[str, Any]] = {
    "pneumoniamnist": {
        "labels": {
            0: ("normal", "*_label_0_normal.jpg"),
            1: ("pneumonia", "*_label_1_pneumonia.jpg"),
        }
    },
    "retinamnist": {
        "labels": {
            0: ("no diabetic retinopathy", "*_label_0_no_diabetic_retinopathy.jpg"),
            1: ("mild diabetic retinopathy", "*_label_1_mild_diabetic_retinopathy.jpg"),
            2: (
                "moderate diabetic retinopathy",
                "*_label_2_moderate_diabetic_retinopathy.jpg",
            ),
            3: ("severe diabetic retinopathy", "*_label_3_severe_diabetic_retinopathy.jpg"),
            4: (
                "proliferative diabetic retinopathy",
                "*_label_4_proliferative_diabetic_retinopathy.jpg",
            ),
        }
    },
}


def collect_examples(dataset_name: str) -> list[dict[str, Any]]:
    """Collect prepared files and their labels for one dataset."""
    dataset = DATASETS[dataset_name]
    examples = []
    for label_index, (label_name, pattern) in dataset["labels"].items():
        paths = sorted((DATA_DIRECTORY / dataset_name).glob(pattern))
        for path in paths:
            examples.append(
                {"label_index": label_index, "label_name": label_name, "path": path}
            )
    return examples


def audit_image_set() -> dict[str, dict[str, int]]:
    """Print the prepared count per class and report missing or unexpected counts."""
    audit = {}
    for dataset_name, dataset in DATASETS.items():
        examples = collect_examples(dataset_name)
        counts = Counter(example["label_index"] for example in examples)
        audit[dataset_name] = {}
        print(f"\n{dataset_name}:")
        for label_index, (label_name, _) in dataset["labels"].items():
            count = counts[label_index]
            audit[dataset_name][label_name] = count
            status = "OK" if count == EXPECTED_EXAMPLES_PER_LABEL else "CHECK"
            print(f"  {label_index}: {label_name}: {count} examples ({status})")
    return audit


def save_unlabeled_grid(dataset_name: str) -> Path:
    """Save a shuffled contact sheet with neutral titles and no label filenames."""
    examples = collect_examples(dataset_name)
    if not examples:
        raise FileNotFoundError(
            f"No prepared images found in {DATA_DIRECTORY / dataset_name}."
        )

    random.Random(RANDOM_SEED).shuffle(examples)
    column_count = min(3, len(examples))
    row_count = math.ceil(len(examples) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4 * column_count, 3.5 * row_count),
        squeeze=False,
    )
    for sample_index, axis in enumerate(axes.flat):
        axis.axis("off")
        if sample_index >= len(examples):
            continue
        with Image.open(examples[sample_index]["path"]) as image:
            axis.imshow(image)
        axis.set_title(f"Sample {sample_index + 1}")

    figure.suptitle(f"{dataset_name}: label-hidden teaching examples")
    figure.tight_layout()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"08_{dataset_name}_unlabeled_grid.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved label-hidden grid to: {output_path}")
    return output_path


def save_label_key() -> Path:
    """Save class names separately so they can be revealed after the exercise."""
    lines = ["Workshop image label key (reveal after participants respond)"]
    for dataset_name, dataset in DATASETS.items():
        lines.append(f"\n{dataset_name}:")
        for label_index, (label_name, _) in dataset["labels"].items():
            lines.append(f"- {label_index}: {label_name}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "08_workshop_image_label_key.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved separate label key to: {output_path}")
    return output_path


if __name__ == "__main__":
    audit_image_set()
    for dataset_name in DATASETS:
        save_unlabeled_grid(dataset_name)
    save_label_key()
