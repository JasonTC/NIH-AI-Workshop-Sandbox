"""Compare visible patterns across two different chest X-ray examples."""

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
IMAGE_PATTERNS = [
    "pneumoniamnist/*_label_0_normal.jpg",
    "pneumoniamnist/*_label_1_pneumonia.jpg",
]


def find_comparison_images() -> list[Path]:
    """Find one prepared example from each chest X-ray label group."""
    image_paths = []
    for pattern in IMAGE_PATTERNS:
        matches = sorted(DATA_DIRECTORY.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No prepared image matches {pattern!r}.")
        image_paths.append(matches[0])
    return image_paths


def save_comparison_images(image_paths: list[Path]) -> Path:
    """Save a neutral comparison grid and close the Matplotlib figure."""
    figure, axes = plt.subplots(1, len(image_paths), figsize=(10, 5))
    for axis, image_path, title in zip(axes, image_paths, ("Example A", "Example B")):
        with Image.open(image_path) as image:
            axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "106_comparison_examples.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image comparison to: {output_path}")
    return output_path


def compare_images(image_paths: list[Path]) -> str:
    """Ask the model to describe similarities and differences between images."""
    content: list[dict[str, str | dict[str, str]]] = [
        {
            "type": "text",
            "text": (
                "Compare Example A and Example B. Describe visible similarities "
                "and differences only. These are different teaching examples, not "
                "images of the same person. Do not infer progression or recommend "
                "treatment."
            ),
        }
    ]
    for title, image_path in zip(("Example A", "Example B"), image_paths):
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        content.append({"type": "text", "text": title})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
            }
        )
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[{"role": "user", "content": content}],
    )
    answer = response.choices[0].message.content or ""
    if "<unused95>" in answer:
        answer = answer.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return answer or "The model returned an empty response."


if __name__ == "__main__":
    try:
        examples = find_comparison_images()
        save_comparison_images(examples)
        print(compare_images(examples))
    except Exception as error:
        print(f"Image comparison could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
