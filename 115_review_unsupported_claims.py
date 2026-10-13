"""Generate a review packet for checking image claims against their source."""

import base64
import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"
REVIEW_QUESTIONS = [
    "Which response statements are directly supported by visible image features?",
    "Which statements are uncertain, speculative, or not assessable from this image?",
    "Did the response add patient history or other facts not supplied?",
    "What would need clinician or source-image review?",
]


def find_example_image() -> Path:
    """Find one prepared chest X-ray to use in the review packet."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def generate_image_response(image_path: Path) -> str:
    """Generate a short image description that participants can audit."""
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
                            "Describe visible features and possible findings in this "
                            "educational image. Distinguish observations from "
                            "interpretation and state uncertainty."
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


def save_review_packet(image_path: Path, response_text: str) -> Path:
    """Save the image, raw response, and blank human-review prompts together."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image_copy = OUTPUT_DIRECTORY / "115_claim_review_image.jpg"
    shutil.copyfile(image_path, image_copy)
    packet: dict[str, Any] = {
        "model": MODEL_NAME,
        "image": image_copy.name,
        "model_response": response_text,
        "review_questions": REVIEW_QUESTIONS,
        "participant_notes": "",
    }
    output_path = OUTPUT_DIRECTORY / "115_unsupported_claims_review.json"
    output_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    print("Model response for human review:")
    print(response_text)
    print(f"Saved review packet to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        image_path = find_example_image()
        save_review_packet(image_path, generate_image_response(image_path))
    except Exception as error:
        print(f"Claim review example could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
