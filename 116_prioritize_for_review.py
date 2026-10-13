"""Demonstrate relative image-review prioritization as an unvalidated output."""

import base64
import json
import mimetypes
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
RANDOM_SEED = 42
RETINA_EXAMPLES = [
    ("No diabetic retinopathy", "*_label_0_no_diabetic_retinopathy.jpg"),
    ("Moderate diabetic retinopathy", "*_label_2_moderate_diabetic_retinopathy.jpg"),
    ("Proliferative diabetic retinopathy", "*_label_4_proliferative_diabetic_retinopathy.jpg"),
]


def load_examples() -> list[dict[str, str | Path]]:
    """Find three retinal examples spanning the supplied label set."""
    examples = []
    for label, pattern in RETINA_EXAMPLES:
        matches = sorted((DATA_DIRECTORY / "retinamnist").glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No retinal example matches {pattern!r}.")
        examples.append({"expected_label": label, "image_path": matches[0]})
    random.Random(RANDOM_SEED).shuffle(examples)
    return examples


def save_unlabeled_grid(examples: list[dict[str, str | Path]]) -> Path:
    """Save the images in shuffled order using neutral captions."""
    figure, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4))
    for index, (axis, example) in enumerate(zip(axes, examples), start=1):
        with Image.open(Path(example["image_path"])) as image:
            axis.imshow(image)
        axis.set_title(f"Example {index}")
        axis.axis("off")
    figure.tight_layout()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "116_retina_prioritization_examples.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    return output_path


def prioritize_for_review(examples: list[dict[str, str | Path]]) -> str:
    """Ask the model to order images for review and explain its uncertainty."""
    content: list[dict[str, str | dict[str, str]]] = [
        {
            "type": "text",
            "text": (
                "Order the three images for clinician review, most to least, using "
                "only visible changes, image quality, and uncertainty. Output one "
                "line in this format only: REVIEW ORDER: Example N > Example N > "
                "Example N. Use each example once. Do not name diagnoses, causes, "
                "systemic diseases, or treatment. This is unvalidated teaching "
                "output, not clinical triage. The images are different people."
            ),
        }
    ]
    for index, example in enumerate(examples, start=1):
        image_path = Path(example["image_path"])
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        content.append({"type": "text", "text": f"Example {index}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
            }
        )
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=200,
        messages=[{"role": "user", "content": content}],
    )
    answer = response.choices[0].message.content or ""
    if "<unused95>" in answer:
        answer = answer.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return answer or "The model returned an empty response."


def parse_priority_lines(response_text: str) -> list[dict[str, str | int]]:
    """Parse a compact review order, with support for labeled priority lines."""
    order_match = re.search(r"REVIEW ORDER\s*:\s*(.+)", response_text, re.I)
    if order_match:
        example_numbers = [
            int(number)
            for number in re.findall(r"Example\s*(\d+)", order_match.group(1), re.I)
        ]
        return [
            {"priority": priority, "example": example, "reason": ""}
            for priority, example in enumerate(example_numbers, start=1)
        ]

    parsed = []
    pattern = re.compile(
        r"PRIORITY\s*(\d+)\s*:\s*Example\s*(\d+)\s*\|\s*REASON\s*:\s*(.*)",
        re.I,
    )
    for line in response_text.splitlines():
        match = pattern.search(line.strip())
        if match:
            parsed.append(
                {
                    "priority": int(match.group(1)),
                    "example": int(match.group(2)),
                    "reason": match.group(3).strip(),
                }
            )
    return parsed


def validate_priority_lines(parsed: list[dict[str, str | int]]) -> list[str]:
    """Flag incomplete or duplicate ranks; this check does not assess reasons."""
    issues = []
    priorities = [int(item["priority"]) for item in parsed]
    examples = [int(item["example"]) for item in parsed]
    if priorities != [1, 2, 3]:
        issues.append("Could not parse a complete ordered list of three priorities.")
    if sorted(examples) != [1, 2, 3]:
        issues.append("The response did not assign each of Examples 1–3 exactly once.")
    return issues


def save_prioritization_response(
    response_text: str,
    parsed_priorities: list[dict[str, str | int]],
    validation_issues: list[str],
) -> Path:
    """Save raw text and parsed rank lines for human review."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "116_prioritization_response.json"
    result = {
        "raw_response": response_text,
        "parsed_priorities": parsed_priorities,
        "format_issues": validation_issues,
        "caveat": "Unvalidated teaching output; not a clinical triage recommendation.",
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(response_text)
    if validation_issues:
        print("Priority format needs review:")
        for issue in validation_issues:
            print(f"- {issue}")
    print(f"Saved response to: {output_path}")
    return output_path


def reveal_dataset_labels(examples: list[dict[str, str | Path]]) -> None:
    """Reveal class labels after reviewing the model's ordering."""
    print("Dataset labels (reveal after the exercise):")
    for index, example in enumerate(examples, start=1):
        print(f"Example {index}: {example['expected_label']}")


if __name__ == "__main__":
    try:
        sample_examples = load_examples()
        save_unlabeled_grid(sample_examples)
        response = prioritize_for_review(sample_examples)
        parsed_priorities = parse_priority_lines(response)
        validation_issues = validate_priority_lines(parsed_priorities)
        save_prioritization_response(response, parsed_priorities, validation_issues)
        reveal_dataset_labels(sample_examples)
    except Exception as error:
        print(f"Prioritization example could not be completed: {error}")
        print("Check the prepared retinal examples and that Ollama is running.")
