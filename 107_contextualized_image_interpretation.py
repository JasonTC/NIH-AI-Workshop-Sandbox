"""Compare an image with a deliberately constructed fictional clinical vignette."""

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

# This teaching context is intentionally fictional and constructed for the
# selected image. It does not describe a real person or establish real linkage.
FICTIONAL_CASE_CONTEXT = """
Fictional teaching vignette constructed for this exercise: an adult presents
with a three-day cough and fever. Oxygen saturation is reported as 91% on room
air. For this activity only, assume this image is being considered in that
hypothetical vignette; it is not a real patient-image pairing.
""".strip()


def find_example_image() -> Path:
    """Find one prepared chest X-ray example for the fictional vignette."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_image_and_context_preview(image_path: Path, context: str) -> Path:
    """Save the image preview; print the fictional context separately."""
    with Image.open(image_path) as image:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image, cmap="gray")
        axis.set_title("Teaching image")
        axis.axis("off")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "107_contextualized_case_image.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    print("Fictional context supplied to the model:")
    print(context)
    return output_path


def interpret_with_context(image_path: Path, context: str) -> str:
    """Ask how the supplied context affects image interpretation, separating sources."""
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
                            "This is an explicitly fictional teaching vignette. "
                            "First list observations from the image; separately "
                            "list facts supplied in the vignette. Then describe "
                            "whether and how the context may affect interpretation "
                            "or urgency, and state uncertainty. Do not claim that "
                            "the context is verified by the image or give treatment "
                            "instructions.\n\n"
                            f"Hypothetical context:\n{context}"
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


if __name__ == "__main__":
    try:
        sample_image = find_example_image()
        save_image_and_context_preview(sample_image, FICTIONAL_CASE_CONTEXT)
        print(interpret_with_context(sample_image, FICTIONAL_CASE_CONTEXT))
    except Exception as error:
        print(f"Contextual interpretation could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
