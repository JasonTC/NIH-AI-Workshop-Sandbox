"""Send a saved example image to a local vision-language model."""

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


def image_to_data_url(image_path: Path) -> str:
    """Convert an image file into the data URL format used by the API."""
    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded_image}"


def ask_model_to_interpret_image(image_path: Path) -> str:
    """Ask MedGemma for a brief description of the selected image."""
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. Run 02_inspect_and_export_npz_image.py first."
        )

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
                            "Describe only visible features in this educational "
                            "medical image. State uncertainty and do not provide "
                            "a diagnosis or treatment recommendation."
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


def save_interpretation(image_path: Path) -> Path:
    """Print and save the model's response for later inspection."""
    interpretation = ask_model_to_interpret_image(image_path)
    output_path = TEST_DIRECTORY / "medgemma_image_interpretation.txt"
    TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"Model: {MODEL_NAME}\nImage: {image_path.name}\n\n{interpretation}\n",
        encoding="utf-8",
    )
    print(interpretation)
    print(f"Saved response to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Start with the saved pneumonia example; change this path to the retina image
    # produced by script 02 to try the second dataset.
    sample_image = TEST_DIRECTORY / "pneumonia_test_0000_label-1_pneumonia.png"
    try:
        save_interpretation(sample_image)
    except Exception as error:
        print(f"Image interpretation could not be completed: {error}")
        print("Check that Ollama is running and the selected model supports images.")
