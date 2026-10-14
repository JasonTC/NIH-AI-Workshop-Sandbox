"""Compare image-only, record-only, and combined model responses."""

import base64
import mimetypes
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
SUMMARY_PATH = DATA_DIRECTORY / "test" / "synthea_patient_summary.txt"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"


def load_teaching_inputs() -> tuple[str, Path]:
    """Load the Synthea summary and an image, which are explicitly unlinked."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary not found at {SUMMARY_PATH}. Run 04_summarize_synthea_fhir.py."
        )
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return SUMMARY_PATH.read_text(encoding="utf-8"), matches[0]


def save_image_preview_and_print_record(image_path: Path, record_summary: str) -> Path:
    """Save the image preview, print the record, and call out their unlinked status."""
    with Image.open(image_path) as image:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image, cmap="gray")
        axis.set_title("Teaching image (unlinked to record)")
        axis.axis("off")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "108_unlinked_modality_comparison.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    print("Synthetic record summary (not linked to the displayed image):")
    print(record_summary)
    return output_path


def ask_about_image(image_path: Path) -> str:
    """Request a brief image-only description."""
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
                    {"type": "text", "text": "Describe visible features only."},
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


def summarize_record_only(record_summary: str) -> str:
    """Ask for a concise summary using only the synthetic record text."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize the supplied synthetic record only. Do not add "
                    "clinical facts that are not stated.\n\n" + record_summary
                ),
            }
        ],
    )
    answer = response.choices[0].message.content or ""
    if "<unused95>" in answer:
        answer = answer.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return answer or "The model returned an empty response."


def ask_about_both(image_path: Path, record_summary: str) -> str:
    """Supply both unlinked sources and ask the model to keep them distinct."""
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
                            "The synthetic record and image are independently "
                            "selected and are NOT linked to the same person. "
                            "Summarize record facts and visible image features in "
                            "separate sections. Do not combine them into a patient "
                            "diagnosis or infer a relationship. State what cannot "
                            "be concluded.\n\n"
                            f"Synthetic record:\n{record_summary}"
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


def compare_modalities(image_path: Path, record_summary: str) -> dict[str, str]:
    """Run all three requests so participants can compare each input condition."""
    return {
        "Image only": ask_about_image(image_path),
        "Record only": summarize_record_only(record_summary),
        "Combined, sources unlinked": ask_about_both(image_path, record_summary),
    }


def display_modality_responses(responses: dict[str, str]) -> None:
    """Print the response from each modality condition under its own heading."""
    for heading, response in responses.items():
        print(f"{heading}:\n{response}\n")


if __name__ == "__main__":
    try:
        summary, image_path = load_teaching_inputs()
        save_image_preview_and_print_record(image_path, summary)
        display_modality_responses(compare_modalities(image_path, summary))
    except Exception as error:
        print(f"Modality comparison could not be completed: {error}")
        print("Check the Synthea summary, prepared image files, and Ollama.")
