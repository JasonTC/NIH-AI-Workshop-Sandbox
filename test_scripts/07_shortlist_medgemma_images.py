"""Check MedGemma class predictions in small batches and create JPG grids.

Each candidate gets three differently worded prompts. A JPG is saved only when
all three answers match the dataset label. By default, one run checks five
candidates from one dataset/class so the output can be reviewed before continuing.
"""

import argparse
import base64
import json
import os
import random
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

# Save grids without opening a desktop window when the script runs in a terminal.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
GRID_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "medgemma1.5:latest")

DEFAULT_BATCH_SIZE = 5
TARGET_PER_LABEL = 3
SPLIT = "test"
RANDOM_SEED = 42
ATTEMPT_LOG = GRID_DIRECTORY / "medgemma_classification_attempts.jsonl"

DATASETS: dict[str, dict[str, Any]] = {
    "pneumoniamnist": {
        "file": DATA_DIRECTORY / "pneumoniamnist_224.npz",
        "task": "classify this chest X-ray",
        "labels": {0: "normal", 1: "pneumonia"},
    },
    "retinamnist": {
        "file": DATA_DIRECTORY / "retinamnist_224.npz",
        "task": "classify the diabetic retinopathy grade in this retinal image",
        "labels": {
            0: "no diabetic retinopathy",
            1: "mild diabetic retinopathy",
            2: "moderate diabetic retinopathy",
            3: "severe diabetic retinopathy",
            4: "proliferative diabetic retinopathy",
        },
    },
}

# These ask the same classification question in different words, without
# requiring MedGemma to produce a complex JSON structure.
PROMPT_TEMPLATES = [
    "Classify the image into one category using visible image features only. "
    "Choose from: {classes}. Reply with just the exact category name.",
    "Which single dataset label best fits this image? Available labels: "
    "{classes}. Answer with one label name only; do not explain.",
    "Select the best matching class for the image from this list: {classes}. "
    "Return only the selected class name, exactly as written.",
]


def create_ollama_client() -> OpenAI:
    """Connect the OpenAI Python client to the local Ollama server."""
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=180.0,
        max_retries=0,
    )


def load_dataset_arrays(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the test images and their labels from one NPZ dataset."""
    dataset = DATASETS[dataset_name]
    with np.load(dataset["file"], allow_pickle=False) as arrays:
        images = arrays[f"{SPLIT}_images"].copy()
        labels = arrays[f"{SPLIT}_labels"].reshape(-1).copy()
    return images, labels


def image_to_data_url(image_array: np.ndarray) -> str:
    """Encode one NumPy image in memory for the Ollama vision request."""
    image_buffer = BytesIO()
    Image.fromarray(image_array).save(image_buffer, format="PNG")
    encoded_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


def build_prompt(dataset_name: str, attempt_number: int) -> str:
    """Build one of three distinct prompts and rotate the label order."""
    label_items = list(DATASETS[dataset_name]["labels"].items())
    rotation = (attempt_number - 1) % len(label_items)
    label_items = label_items[rotation:] + label_items[:rotation]
    class_list = ", ".join(label_name for _, label_name in label_items)
    prompt = PROMPT_TEMPLATES[attempt_number - 1].format(classes=class_list)
    return f"Task: {DATASETS[dataset_name]['task']}.\n{prompt}"


def normalize_text(text: str) -> str:
    """Lowercase text and replace punctuation with spaces for label matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def get_final_answer(response_text: str) -> str:
    """Remove model-specific text that appears before its final answer marker."""
    if "<unused95>" in response_text:
        return response_text.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return response_text.strip()


def parse_predicted_label(response_text: str, dataset_name: str) -> str | None:
    """Extract a label from JSON or plain text, returning None if ambiguous."""
    answer = get_final_answer(response_text)

    # Some model responses may still use a small JSON object; accept it if present.
    json_match = re.search(r"\{.*?\}", answer, flags=re.DOTALL)
    if json_match:
        try:
            response_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            response_data = {}
        for field_name in ("label", "class", "prediction", "answer"):
            value = response_data.get(field_name)
            if isinstance(value, str):
                normalized_value = normalize_text(value)
                for label_name in DATASETS[dataset_name]["labels"].values():
                    if normalized_value == normalize_text(label_name):
                        return label_name

    normalized_answer = normalize_text(answer)
    labels = DATASETS[dataset_name]["labels"]

    # Flexible keyword parsing for short, natural-language model answers.
    if dataset_name == "pneumoniamnist":
        # Treat an answer beginning with a class name as an explicit choice.
        for label_name in labels.values():
            normalized_name = normalize_text(label_name)
            if normalized_answer == normalized_name or normalized_answer.startswith(
                normalized_name + " "
            ):
                return label_name

        mentions_normal = bool(re.search(r"\bnormal\b", normalized_answer))
        mentions_pneumonia = bool(re.search(r"\bpneumonia\b", normalized_answer))
        negates_pneumonia = bool(
            re.search(
                r"\b(no|without|negative for|absence of)\s+"
                r"(evidence of\s+|signs of\s+)?pneumonia\b",
                normalized_answer,
            )
        )

        if mentions_normal and not mentions_pneumonia:
            return labels[0]
        if negates_pneumonia and not mentions_normal:
            return labels[0]
        if mentions_pneumonia and not mentions_normal and not negates_pneumonia:
            return labels[1]
        return None

    # For the retinal dataset, an exact class phrase is a useful first match.
    exact_matches = [
        label_name
        for label_name in labels.values()
        if normalize_text(label_name) in normalized_answer
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    # For RetinaMNIST, accept a single unambiguous grade keyword. For example,
    # "mild non-proliferative changes" maps to mild, but "mild to moderate" is
    # intentionally treated as ambiguous and is not accepted as a correct match.
    grade_keywords = {
        1: r"\bmild\b",
        2: r"\bmoderate\b",
        3: r"\bsevere\b",
        4: r"\bproliferative\b",
    }
    grade_matches = []
    for label_index, pattern in grade_keywords.items():
        match = re.search(pattern, normalized_answer)
        if not match:
            continue
        preceding_text = normalized_answer[max(0, match.start() - 20) : match.start()]
        if re.search(r"\b(no|not|without|absence of)\s*$", preceding_text):
            continue
        if label_index == 4 and re.search(r"\bnon\s*$", preceding_text):
            continue
        grade_matches.append(label_index)

    if len(grade_matches) == 1:
        return labels[grade_matches[0]]

    no_retinopathy = bool(
        re.search(
            r"\b(no|without|absence of)\s+"
            r"(signs of\s+|evidence of\s+)?"
            r"(diabetic\s+)?retinopathy\b",
            normalized_answer,
        )
        or re.search(r"\b(normal retina|normal fundus)\b", normalized_answer)
    )
    if no_retinopathy and not grade_matches:
        return labels[0]
    return None


def classify_image_once(
    client: OpenAI, dataset_name: str, image_array: np.ndarray, attempt_number: int
) -> tuple[str | None, str]:
    """Ask MedGemma once and return both the parsed label and raw response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(dataset_name, attempt_number)},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_array)},
                    },
                ],
            }
        ],
    )
    raw_answer = response.choices[0].message.content or ""
    prediction = parse_predicted_label(raw_answer, dataset_name)
    return prediction, raw_answer


def log_attempt(
    dataset_name: str,
    label_index: int,
    image_index: int,
    attempt_number: int,
    prediction: str | None,
    raw_answer: str,
) -> None:
    """Append each raw answer to a JSONL file for troubleshooting later."""
    GRID_DIRECTORY.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": dataset_name,
        "split": SPLIT,
        "image_index": image_index,
        "expected_label_index": label_index,
        "expected_label": DATASETS[dataset_name]["labels"][label_index],
        "attempt": attempt_number,
        "parsed_prediction": prediction,
        "raw_response": raw_answer,
    }
    with ATTEMPT_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def classify_consistently(
    client: OpenAI,
    dataset_name: str,
    image_array: np.ndarray,
    label_index: int,
    image_index: int,
) -> list[str] | None:
    """Run all three prompt variants and accept only unanimous correct answers."""
    predictions: list[str] = []
    expected_label = DATASETS[dataset_name]["labels"][label_index]

    # Run every prompt even when an earlier response is wrong so the full batch
    # log shows exactly how the model answered each version of the question.
    for attempt_number in range(1, 4):
        prediction, raw_answer = classify_image_once(
            client, dataset_name, image_array, attempt_number
        )
        log_attempt(
            dataset_name,
            label_index,
            image_index,
            attempt_number,
            prediction,
            raw_answer,
        )

        # Keep raw output visible during the small-batch troubleshooting process.
        preview = raw_answer.strip().replace("\n", " ")[:350]
        print(f"    Prompt {attempt_number} raw: {preview!r}")
        print(f"    Parsed label: {prediction!r}")
        if prediction is not None:
            predictions.append(prediction)

    if len(predictions) == 3 and all(
        normalize_text(prediction) == normalize_text(expected_label)
        for prediction in predictions
    ):
        return predictions
    return None


def safe_label_name(dataset_name: str, label_index: int) -> str:
    """Convert a class label to the underscore-separated filename form."""
    return DATASETS[dataset_name]["labels"][label_index].replace(" ", "_")


def saved_examples(dataset_name: str, label_index: int) -> list[Path]:
    """Find JPGs previously accepted for one dataset label."""
    output_directory = DATA_DIRECTORY / dataset_name
    pattern = f"*_label_{label_index}_{safe_label_name(dataset_name, label_index)}.jpg"
    return sorted(
        output_directory.glob(pattern),
        key=lambda path: int(path.name.split("_label_", maxsplit=1)[0]),
    )


def save_as_jpeg(
    image_array: np.ndarray,
    dataset_name: str,
    image_index: int,
    label_index: int,
) -> Path:
    """Save one accepted image to its dataset directory using the requested name."""
    output_directory = DATA_DIRECTORY / dataset_name
    output_directory.mkdir(parents=True, exist_ok=True)
    image_path = output_directory / (
        f"{image_index}_label_{label_index}_"
        f"{safe_label_name(dataset_name, label_index)}.jpg"
    )
    Image.fromarray(image_array).convert("RGB").save(image_path, format="JPEG", quality=95)
    return image_path


def create_label_grid(dataset_name: str) -> Path:
    """Create a grid of the currently saved examples, grouped by class label."""
    label_map = DATASETS[dataset_name]["labels"]
    figure, axes = plt.subplots(
        nrows=len(label_map),
        ncols=TARGET_PER_LABEL,
        figsize=(4 * TARGET_PER_LABEL, 3.5 * len(label_map)),
        squeeze=False,
    )

    for label_index, label_name in label_map.items():
        examples = saved_examples(dataset_name, label_index)[:TARGET_PER_LABEL]
        for column_index in range(TARGET_PER_LABEL):
            axis = axes[label_index][column_index]
            axis.axis("off")
            if column_index >= len(examples):
                axis.set_title(f"{label_name}\nNo accepted sample yet")
                continue

            image_path = examples[column_index]
            with Image.open(image_path) as image:
                axis.imshow(image)
            image_index = image_path.name.split("_label_", maxsplit=1)[0]
            axis.set_title(f"{label_name}\nimage index {image_index}")

    figure.suptitle(f"{dataset_name}: three-prompt MedGemma matches")
    figure.tight_layout()
    GRID_DIRECTORY.mkdir(parents=True, exist_ok=True)
    grid_path = GRID_DIRECTORY / f"{dataset_name}_medgemma_verified_grid.png"
    figure.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return grid_path


def run_batch(
    dataset_name: str, label_index: int, offset: int, batch_size: int
) -> None:
    """Check one batch of candidates for a single dataset label."""
    if label_index not in DATASETS[dataset_name]["labels"]:
        raise ValueError(f"Unknown label {label_index} for {dataset_name}.")

    images, labels = load_dataset_arrays(dataset_name)
    candidate_indices = [
        index for index, label in enumerate(labels) if int(label) == label_index
    ]
    # Use the same shuffled ordering every time so --offset continues the search.
    seed_offset = sum(ord(character) for character in dataset_name) + label_index
    random.Random(RANDOM_SEED + seed_offset).shuffle(candidate_indices)
    batch = candidate_indices[offset : offset + batch_size]
    label_name = DATASETS[dataset_name]["labels"][label_index]

    print(
        f"{dataset_name}, label {label_index} ({label_name}), {SPLIT} split: "
        f"candidates {offset} to {offset + len(batch) - 1} "
        f"of {len(candidate_indices)}"
    )
    client = create_ollama_client()

    for batch_position, image_index in enumerate(batch, start=offset + 1):
        already_saved = saved_examples(dataset_name, label_index)
        if len(already_saved) >= TARGET_PER_LABEL:
            print(f"Already have {TARGET_PER_LABEL} accepted examples for this label.")
            break

        print(f"\nBatch candidate {batch_position}: image index {image_index}")
        predictions = classify_consistently(
            client, dataset_name, images[image_index], label_index, image_index
        )
        if predictions is None:
            print("  Not accepted: the three prompts did not all match the dataset label.")
            continue

        image_path = save_as_jpeg(
            images[image_index], dataset_name, image_index, label_index
        )
        print(f"  Accepted unanimously: {image_path}")

    grid_path = create_label_grid(dataset_name)
    count = len(saved_examples(dataset_name, label_index))
    next_offset = offset + len(batch)
    print(f"\nAccepted images for {label_name}: {min(count, TARGET_PER_LABEL)}/{TARGET_PER_LABEL}")
    print(f"Updated grid: {grid_path}")
    if count < TARGET_PER_LABEL and next_offset < len(candidate_indices):
        print(
            "Next batch command offset: "
            f"--offset {next_offset} --batch-size {DEFAULT_BATCH_SIZE}"
        )


def main() -> None:
    """Parse batch settings and run one small, inspectable batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="pneumoniamnist")
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    arguments = parser.parse_args()

    if arguments.offset < 0 or arguments.batch_size < 1:
        parser.error("--offset must be zero or greater and --batch-size must be positive.")

    run_batch(
        arguments.dataset,
        arguments.label,
        arguments.offset,
        arguments.batch_size,
    )


if __name__ == "__main__":
    main()
