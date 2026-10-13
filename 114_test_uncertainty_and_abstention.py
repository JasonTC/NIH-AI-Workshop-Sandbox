"""Probe whether MedGemma expresses uncertainty rather than forcing a label."""

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "retinamnist/*_label_2_moderate_diabetic_retinopathy.jpg"
PROMPT = (
    "Assess whether this image alone supports a reliable diabetic retinopathy "
    "grade. If the image is too limited or uncertain, say that you cannot "
    "determine a grade rather than guessing. State the limitation and what "
    "additional information or review may be needed. This is a model-behavior "
    "demonstration, not a clinical assessment."
)


def find_example_image() -> Path:
    """Find one prepared retinal example for the abstention prompt."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_image_preview(image_path: Path) -> Path:
    """Copy the image to a label-neutral preview filename."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "114_uncertainty_example.jpg"
    output_path.write_bytes(image_path.read_bytes())
    print(f"Saved image preview to: {output_path}")
    return output_path


def ask_for_uncertainty(image_path: Path, prompt: str = PROMPT) -> str:
    """Ask the model to state when the provided visual information is insufficient."""
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
                    {"type": "text", "text": prompt},
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


def save_uncertainty_response(response_text: str) -> Path:
    """Save the raw response for participant review."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "114_uncertainty_response.txt"
    output_path.write_text(response_text + "\n", encoding="utf-8")
    print(response_text)
    print(f"Saved response to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        example_image = find_example_image()
        save_image_preview(example_image)
        save_uncertainty_response(ask_for_uncertainty(example_image))
    except Exception as error:
        print(f"Uncertainty example could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
