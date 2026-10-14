"""Turn a model-generated image description into a patient-friendly draft."""

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


def find_example_image() -> Path:
    """Find one prepared image for the two-step explanation activity."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_image_preview(image_path: Path) -> Path:
    """Copy the image under a label-neutral teaching filename."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "113_patient_friendly_teaching_image.jpg"
    output_path.write_bytes(image_path.read_bytes())
    print(f"Saved image preview to: {output_path}")
    return output_path


def draft_visible_findings(image_path: Path) -> str:
    """Ask for a brief image-only description to use as the rewriting source."""
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
                            "Describe only visible features in this teaching image. "
                            "Mention uncertainty; do not diagnose or recommend treatment."
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


def restate_for_patient(findings: str) -> str:
    """Rewrite the supplied draft in plain language without adding new facts."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Rewrite the following draft as clear, respectful, patient-"
                    "friendly language for a fictional teaching example. Refer to "
                    "'the image' rather than 'your image'. Preserve only information "
                    "in the draft; do not add causes, alternative explanations, a "
                    "diagnosis, prognosis, or treatment advice. Preserve uncertainty "
                    "and say the draft requires clinician review.\n\n"
                    f"Draft image description:\n{findings}"
                ),
            }
        ],
    )
    answer = response.choices[0].message.content or ""
    if "<unused95>" in answer:
        answer = answer.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return answer or "The model returned an empty response."


def save_patient_friendly_draft(source_findings: str, rewrite: str) -> Path:
    """Save both stages so reviewers can compare meaning and wording."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "113_patient_friendly_draft.txt"
    output_path.write_text(
        "Model-generated image description (source):\n"
        f"{source_findings}\n\n"
        "Patient-friendly draft (requires clinician review):\n"
        f"{rewrite}\n",
        encoding="utf-8",
    )
    print(rewrite)
    print(f"Saved source and rewrite to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        image_path = find_example_image()
        save_image_preview(image_path)
        findings = draft_visible_findings(image_path)
        save_patient_friendly_draft(findings, restate_for_patient(findings))
    except Exception as error:
        print(f"Patient-friendly rewrite could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
