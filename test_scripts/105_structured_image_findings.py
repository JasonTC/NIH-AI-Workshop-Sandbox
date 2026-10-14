"""Request labeled plain-text image findings and parse the requested headings."""

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"


def find_example_image() -> Path:
    """Find one prepared image for the structured findings activity."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_example_image(image_path: Path) -> Path:
    """Save an image preview and close its Matplotlib figure."""
    with Image.open(image_path) as image:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image, cmap="gray")
        axis.set_title("Teaching image")
        axis.axis("off")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "105_structured_findings_example.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    return output_path


def extract_structured_findings(image_path: Path) -> str:
    """Ask for findings using labeled plain text rather than a strict JSON schema."""
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
                            "Use these plain-text headings exactly, each on its own "
                            "line: Image type:, Visible findings:, Possible "
                            "interpretation:, Uncertainty:, Cannot determine:. Put "
                            "findings and unknowns on separate bullet lines. Use only "
                            "features visible in the image. State uncertainty. Do "
                            "not provide treatment advice."
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
    response_text = response.choices[0].message.content or ""
    if "<unused95>" in response_text:
        response_text = response_text.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return response_text or "The model returned an empty response."


def parse_labeled_findings(response_text: str) -> dict[str, Any]:
    """Parse known headings and bullet lists, retaining the original response."""
    heading_map = {
        "image type": "image_type",
        "visible findings": "visible_findings",
        "possible interpretation": "possible_interpretation",
        "uncertainty": "uncertainty",
        "cannot determine": "cannot_determine",
    }
    list_fields = {"visible_findings", "cannot_determine"}
    findings: dict[str, Any] = {
        "image_type": "",
        "visible_findings": [],
        "possible_interpretation": "",
        "uncertainty": "",
        "cannot_determine": [],
        "raw_response": response_text,
    }
    current_field = ""
    for line in response_text.splitlines():
        heading = re.match(r"^\s*([^:]+):\s*(.*)$", line)
        if heading and heading.group(1).casefold().strip() in heading_map:
            current_field = heading_map[heading.group(1).casefold().strip()]
            value = heading.group(2).strip()
            if value:
                if current_field in list_fields:
                    findings[current_field].append(value.lstrip("-* "))
                else:
                    findings[current_field] = value
            continue

        value = line.strip().lstrip("-*• ").strip()
        if not value or not current_field:
            continue
        if current_field in list_fields:
            findings[current_field].append(value)
        elif not findings[current_field]:
            findings[current_field] = value
        else:
            findings[current_field] += f" {value}"
    return findings


def validate_structured_findings(findings: dict[str, Any]) -> list[str]:
    """Flag labeled fields that are absent or empty after plain-text parsing."""
    expected_fields = (
        "image_type",
        "visible_findings",
        "possible_interpretation",
        "uncertainty",
        "cannot_determine",
    )
    return [
        f"Could not parse a non-empty '{field_name}' field from the response."
        for field_name in expected_fields
        if not findings.get(field_name)
    ]


if __name__ == "__main__":
    try:
        sample_image = find_example_image()
        save_example_image(sample_image)
        response_text = extract_structured_findings(sample_image)
        findings = parse_labeled_findings(response_text)
        print(json.dumps(findings, indent=2, ensure_ascii=False))
        validation_issues = validate_structured_findings(findings)
        if validation_issues:
            print("Basic structure check found issues:")
            for issue in validation_issues:
                print(f"- {issue}")
        else:
            print("All requested headings were parsed from the plain-text response.")
    except Exception as error:
        print(f"Structured image findings could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
