"""Show why cross-modal consistency cannot be assessed without patient linkage."""

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


def load_unlinked_inputs() -> tuple[str, Path]:
    """Load the existing synthetic summary and an independently chosen image."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary not found at {SUMMARY_PATH}. Run 04_summarize_synthea_fhir.py."
        )
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return SUMMARY_PATH.read_text(encoding="utf-8"), matches[0]


def save_unlinked_image_and_print_record(record_summary: str, image_path: Path) -> Path:
    """Save the image, print the record, and note that the inputs are unlinked."""
    print("Synthetic record (independent teaching example):")
    print(record_summary)
    with Image.open(image_path) as image:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image, cmap="gray")
        axis.set_title("Independent teaching image")
        axis.axis("off")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "109_unlinked_consistency_image.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    return output_path


def check_cross_modal_consistency(record_summary: str, image_path: Path) -> str:
    """Ask whether consistency can be assessed without assuming a shared patient."""
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
                            "The record and image are independently selected and "
                            "are explicitly NOT linked to the same patient. "
                            "Describe record facts and visible image features "
                            "separately. Then state whether a patient-level "
                            "consistency check is supported, contradicted, or "
                            "cannot be assessed because linkage is unknown. Do "
                            "not infer a diagnosis by combining these sources.\n\n"
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


if __name__ == "__main__":
    try:
        summary, image_path = load_unlinked_inputs()
        save_unlinked_image_and_print_record(summary, image_path)
        print(check_cross_modal_consistency(summary, image_path))
    except Exception as error:
        print(f"Cross-modal consistency example could not be completed: {error}")
        print("Check the Synthea summary, prepared image files, and Ollama.")
