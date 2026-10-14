"""Create a draft clinician-facing summary from a teaching image and vignette."""

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"
FICTIONAL_CONTEXT = (
    "Fictional teaching vignette constructed for this exercise: an adult has "
    "a three-day cough and fever. This is not a real patient-image pairing."
)


def find_example_image() -> Path:
    """Find one prepared chest X-ray example."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_image_preview(image_path: Path) -> Path:
    """Copy the image under a label-neutral teaching filename."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "112_handoff_teaching_image.jpg"
    output_path.write_bytes(image_path.read_bytes())
    print(f"Saved image preview to: {output_path}")
    return output_path


def create_clinical_handoff(image_path: Path, context: str) -> str:
    """Request a concise draft handoff with source facts kept distinct."""
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
                            "Draft a short clinician-facing teaching handoff with "
                            "sections for (1) facts supplied in the vignette, (2) "
                            "visible image observations, (3) uncertainty or missing "
                            "information, and (4) items for clinician review. In "
                            "the review section, list only verification questions "
                            "such as confirming source linkage or consulting prior "
                            "images. Do not suggest diagnoses, tests, management, "
                            "or treatment. This pairing is fictional; do not treat "
                            "image observations as confirmed diagnoses.\n\n"
                            f"Fictional context:\n{context}"
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


def save_handoff(handoff: str) -> Path:
    """Save and print the draft handoff."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "112_clinical_handoff_draft.txt"
    output_path.write_text(handoff + "\n", encoding="utf-8")
    print(handoff)
    print(f"Saved draft to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        image_path = find_example_image()
        save_image_preview(image_path)
        save_handoff(create_clinical_handoff(image_path, FICTIONAL_CONTEXT))
    except Exception as error:
        print(f"Clinical handoff example could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
