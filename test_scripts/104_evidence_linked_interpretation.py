"""Ask MedGemma to connect an image classification with visible evidence."""

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
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"
PROMPT = (
    "For this educational image, state a possible classification and list the "
    "visible features that support or do not support it. Separate observations "
    "from interpretation. State what is uncertain and what cannot be determined "
    "from this image alone. Do not provide treatment advice."
)


def find_example_image() -> Path:
    """Find one prepared image for the evidence-linked interpretation activity."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_example_image(image_path: Path) -> Path:
    """Save an image preview and close its Matplotlib figure."""
    with Image.open(image_path) as image:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image, cmap="gray")
        axis.set_title("Teaching image")
        axis.axis("off")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "104_evidence_interpretation_example.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    return output_path


def interpret_with_evidence(image_path: Path, prompt: str = PROMPT) -> str:
    """Request a classification with visible evidence and explicit limitations."""
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


if __name__ == "__main__":
    try:
        sample_image = find_example_image()
        save_example_image(sample_image)
        print(interpret_with_evidence(sample_image))
    except Exception as error:
        print(f"Evidence-linked interpretation could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
