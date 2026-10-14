"""Classify one chest X-ray and one retinal image with MedGemma."""

import base64
import mimetypes
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"

# Filenames contain the dataset labels. The display function deliberately uses
# neutral captions; reveal EXPECTED_LABEL only after participants have guessed.
CLASSIFICATION_EXAMPLES = [
    {
        "title": "Chest X-ray",
        "pattern": "pneumoniamnist/*_label_1_pneumonia.jpg",
        "choices": "normal, pneumonia",
        "expected_label": "pneumonia",
    },
    {
        "title": "Retinal image",
        "pattern": "retinamnist/*_label_2_moderate_diabetic_retinopathy.jpg",
        "choices": (
            "no diabetic retinopathy, mild diabetic retinopathy, "
            "moderate diabetic retinopathy, severe diabetic retinopathy, "
            "proliferative diabetic retinopathy"
        ),
        "expected_label": "moderate diabetic retinopathy",
    },
]


def load_classification_examples() -> list[dict[str, str | Path]]:
    """Resolve one prepared image for each classification task."""
    examples = []
    for example in CLASSIFICATION_EXAMPLES:
        matches = sorted(DATA_DIRECTORY.glob(str(example["pattern"])))
        if not matches:
            raise FileNotFoundError(
                f"No image matches {example['pattern']!r} under {DATA_DIRECTORY}."
            )
        examples.append({**example, "image_path": matches[0]})
    return examples


def save_classification_examples(examples: list[dict[str, str | Path]]) -> Path:
    """Save a neutral preview grid without opening an interactive figure."""
    figure, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5))
    if len(examples) == 1:
        axes = [axes]
    for axis, example in zip(axes, examples):
        with Image.open(Path(example["image_path"])) as image:
            axis.imshow(image, cmap="gray")
        axis.set_title(str(example["title"]))
        axis.axis("off")
    figure.tight_layout()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "102_classification_examples.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    return output_path


def classify_image(image_path: Path, choices: str, task: str) -> str:
    """Request a single image classification from the model."""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Task: {task}. Choose the single best category from: "
                            f"{choices}. Reply with the category name and a brief "
                            "note if the image is insufficient for classification."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
    )
    answer = response.choices[0].message.content or ""
    if "<unused95>" in answer:
        answer = answer.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return answer or "The model returned an empty response."


def classify_all_examples(
    examples: list[dict[str, str | Path]],
) -> list[dict[str, str]]:
    """Run both image tasks and return neutral titles with model responses."""
    results = []
    for example in examples:
        task = (
            "classify this chest X-ray"
            if example["title"] == "Chest X-ray"
            else "classify the diabetic retinopathy grade in this retinal image"
        )
        answer = classify_image(
            Path(example["image_path"]), str(example["choices"]), task
        )
        results.append({"title": str(example["title"]), "response": answer})
    return results


def display_classification_results(results: list[dict[str, str]]) -> None:
    """Print the model responses without revealing the known labels."""
    for result in results:
        print(f"{result['title']} — model response:\n{result['response']}\n")


def reveal_expected_labels(examples: list[dict[str, str | Path]]) -> None:
    """Reveal dataset labels after participants have reviewed model responses."""
    print("Dataset labels (for comparison, not a clinical reference):")
    for example in examples:
        print(f"{example['title']}: {example['expected_label']}")


if __name__ == "__main__":
    try:
        sample_examples = load_classification_examples()
        save_classification_examples(sample_examples)
        model_results = classify_all_examples(sample_examples)
        display_classification_results(model_results)
        reveal_expected_labels(sample_examples)
    except Exception as error:
        print(f"Classification examples could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
