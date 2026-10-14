"""Ask MedGemma what additional information would help assess an image."""

import base64
import json
import mimetypes
import re
from pathlib import Path

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"
PROMPT = (
    "Give 3 to 5 concise clinician follow-up questions to help interpret this "
    "image. Put one question on each line, each starting with 'QUESTION:'. Do "
    "not provide image findings, an impression, a diagnosis, or treatment advice."
)


def find_example_image() -> Path:
    """Find one prepared chest X-ray example."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_image_preview(image_path: Path) -> Path:
    """Copy the example to an unlabeled teaching-preview filename."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "111_follow_up_questions_image.jpg"
    output_path.write_bytes(image_path.read_bytes())
    print(f"Saved image preview to: {output_path}")
    return output_path


def generate_follow_up_questions(image_path: Path, prompt: str = PROMPT) -> str:
    """Ask for relevant follow-up questions without inferring missing history."""
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


def parse_follow_up_questions(response_text: str) -> list[str]:
    """Extract QUESTION-labeled lines while retaining the raw text separately."""
    questions = []
    for line in response_text.splitlines():
        match = re.match(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?QUESTION\s*:\s*(.*)$", line, re.I)
        if match and match.group(1).strip():
            questions.append(match.group(1).strip())
    return questions


def save_follow_up_response(response_text: str, questions: list[str]) -> Path:
    """Save both the raw model response and questions parsed from its labels."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "111_follow_up_questions.json"
    result = {"raw_response": response_text, "parsed_questions": questions}
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Parsed follow-up questions:")
    for question in questions:
        print(f"- {question}")
    if not questions:
        print("No QUESTION-labeled lines were parsed; inspect the raw response.")
    print(f"Saved response to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        example_image = find_example_image()
        save_image_preview(example_image)
        response = generate_follow_up_questions(example_image)
        save_follow_up_response(response, parse_follow_up_questions(response))
    except Exception as error:
        print(f"Follow-up question generation could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
