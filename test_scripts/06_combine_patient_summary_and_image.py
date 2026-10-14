"""Combine a Synthea summary and an image in one local VLM request."""

import base64
import os
from pathlib import Path

from openai import OpenAI

ROOT_DIRECTORY = Path(__file__).parent
TEST_DIRECTORY = ROOT_DIRECTORY / "data" / "test"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "medgemma1.5:latest")


def create_ollama_client() -> OpenAI:
    """Create an OpenAI client connected to the local Ollama server."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def load_teaching_inputs() -> tuple[str, Path]:
    """Load the text summary and image created by the earlier activities."""
    summary_path = TEST_DIRECTORY / "synthea_patient_summary.txt"
    image_path = TEST_DIRECTORY / "pneumonia_test_0000_label-1_pneumonia.png"

    if not summary_path.exists():
        raise FileNotFoundError(
            "Run 04_summarize_synthea_fhir.py to create the patient summary first."
        )
    if not image_path.exists():
        raise FileNotFoundError(
            "Run 02_inspect_and_export_npz_image.py to create an image first."
        )

    return summary_path.read_text(encoding="utf-8"), image_path


def image_to_data_url(image_path: Path) -> str:
    """Convert an image file into the data URL format used by the API."""
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


def ask_about_combined_inputs(patient_summary: str, image_path: Path) -> str:
    """Ask the model to keep image observations separate from record facts."""
    client = create_ollama_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a synthetic teaching vignette. The image "
                            "and generated patient record are independently "
                            "selected and do NOT belong to the same person. "
                            "Summarize the supplied record facts and visible "
                            "image features in separate sections. Do not infer "
                            "a diagnosis or claim that the image is linked to "
                            "this patient. State uncertainty.\n\n"
                            f"Synthetic patient record summary:\n{patient_summary}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or "The model returned an empty response."


def run_multimodal_example() -> Path:
    """Print and save the model's response to the two teaching inputs."""
    patient_summary, image_path = load_teaching_inputs()
    response_text = ask_about_combined_inputs(patient_summary, image_path)

    output_path = TEST_DIRECTORY / "multimodal_patient_profile.txt"
    output_path.write_text(
        "Teaching vignette: the patient record and image are not linked.\n"
        f"Image: {image_path.name}\n"
        f"Model: {MODEL_NAME}\n\n"
        f"{response_text}\n",
        encoding="utf-8",
    )
    print(response_text)
    print(f"\nSaved response to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        run_multimodal_example()
    except Exception as error:
        print(f"The multimodal example could not be completed: {error}")
        print("Check that the earlier sample files exist and Ollama is running.")
