"""Ask for possible interpretations and evidence that could distinguish them."""

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
PROMPT = (
    "Describe visible observations, then list a small number of possible "
    "interpretations as hypotheses rather than diagnoses. For each, say what "
    "additional evidence or clinical information would help assess it. State "
    "uncertainty and do not recommend treatment. This is an educational exercise."
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
    output_path = OUTPUT_DIRECTORY / "117_possible_interpretations_image.jpg"
    output_path.write_bytes(image_path.read_bytes())
    print(f"Saved image preview to: {output_path}")
    return output_path


def generate_possible_interpretations(image_path: Path, prompt: str = PROMPT) -> str:
    """Ask MedGemma for possibilities and the evidence needed to assess them."""
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


def save_interpretation_response(response_text: str) -> Path:
    """Save the response for review and discussion."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "117_possible_interpretations.txt"
    output_path.write_text(response_text + "\n", encoding="utf-8")
    print(response_text)
    print(f"Saved response to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        image_path = find_example_image()
        save_image_preview(image_path)
        response = generate_possible_interpretations(image_path)
        save_interpretation_response(response)
    except Exception as error:
        print(f"Possible interpretations could not be generated: {error}")
        print("Check the prepared image files and that Ollama is running.")
