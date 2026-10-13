"""Probe image-only and contextual VLM responses for candidate teaching vignettes."""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
SUMMARY_PATH = OUTPUT_DIRECTORY / "synthea_patient_summary.txt"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"

# These vignettes are fictional and intentionally constructed around the image
# for prompt exploration; they are not real patient-image linkages.
CANDIDATE_CONTEXTS = {
    "fictional_respiratory_symptoms": (
        "Fictional teaching vignette constructed for this exercise: an adult "
        "has a three-day cough and fever, with oxygen saturation reported as "
        "91% on room air. Assume the image is being considered in this "
        "hypothetical case only."
    ),
    "fictional_no_respiratory_symptoms": (
        "Fictional teaching vignette constructed for this exercise: an adult "
        "has no cough or fever and normal oxygen saturation. Assume the image "
        "is being considered in this hypothetical case only."
    ),
}


def find_candidate_image() -> Path:
    """Find one prepared chest X-ray candidate without displaying its filename."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_candidate_preview(image_path: Path) -> Path:
    """Copy the probe image to a label-neutral preview filename."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "09_multimodal_probe_image.jpg"
    output_path.write_bytes(image_path.read_bytes())
    print(f"Saved candidate image preview to: {output_path}")
    return output_path


def load_unlinked_synthea_control() -> str | None:
    """Load the synthetic summary as an explicitly unlinked control, if available."""
    if not SUMMARY_PATH.exists():
        return None
    return SUMMARY_PATH.read_text(encoding="utf-8")


def ask_model(image_path: Path, context: str | None = None) -> str:
    """Ask for observations and keep the source and linkage status explicit."""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    if context is None:
        instruction = (
            "Describe visible image features and state uncertainty. Do not infer "
            "a patient history from the image."
        )
    elif context.startswith("UNLINKED CONTROL:"):
        instruction = (
            "The supplied synthetic record is explicitly unlinked from the image. "
            "Summarize the record facts and image observations separately; do not "
            "combine them into a patient-level interpretation.\n\n" + context
        )
    else:
        instruction = (
            "This is a deliberately constructed fictional teaching vignette, not "
            "a real patient-image pairing. Separate visible image findings from "
            "the supplied case facts. Explain whether the context could affect "
            "interpretation or urgency, and state uncertainty. Do not prescribe "
            "treatment.\n\n" + context
        )

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
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


def probe_candidate_pairs(image_path: Path) -> dict[str, str]:
    """Compare image-only with fictional context variants and an unlinked control."""
    results = {"Image only": ask_model(image_path)}
    for case_name, context in CANDIDATE_CONTEXTS.items():
        results[case_name] = ask_model(image_path, context)

    unlinked_record = load_unlinked_synthea_control()
    if unlinked_record is not None:
        results["Synthea summary (unlinked control)"] = ask_model(
            image_path, f"UNLINKED CONTROL:\n{unlinked_record}"
        )
    return results


def save_probe_results(results: dict[str, str]) -> Path:
    """Save raw responses and their input labels for later prompt review."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "09_multimodal_pair_probe.json"
    output: dict[str, Any] = {
        "model": MODEL_NAME,
        "fictional_contexts": CANDIDATE_CONTEXTS,
        "note": (
            "Fictional contexts are constructed teaching prompts. The Synthea "
            "summary, when included, is an unlinked control. This is not clinical validation."
        ),
        "responses": results,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved probe responses to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        candidate_image = find_candidate_image()
        save_candidate_preview(candidate_image)
        candidate_responses = probe_candidate_pairs(candidate_image)
        save_probe_results(candidate_responses)
    except Exception as error:
        print(f"Multimodal candidate probe could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
