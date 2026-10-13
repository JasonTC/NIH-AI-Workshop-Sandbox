"""Extract fictional note facts from labeled plain text and parse them locally."""

import json
import re
from pathlib import Path

from openai import OpenAI


OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
OUTPUT_DIRECTORY = Path(__file__).parent / "data" / "test"

# This example is fictional and is written only to demonstrate information
# extraction. It does not contain information about a real patient.
CLINICAL_NOTE = """
Fictional visit note: On 2025-04-10, the patient reports a dry cough for three
days and a temperature of 38.1 C yesterday evening. They report taking
acetaminophen as needed. The note does not mention allergies.
""".strip()


def create_ollama_client() -> OpenAI:
    """Create an OpenAI client connected to the local Ollama server."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def extract_facts_from_note() -> str:
    """Ask the model for labeled plain-text facts and supporting quotes."""
    client = create_ollama_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only information explicitly stated in the note. "
                    "Do not diagnose or infer missing facts. Return only these "
                    "labeled plain-text lines, not JSON, one per field. Use the "
                    "exact format shown:\n"
                    "DATE: 2025-04-10\n"
                    "SYMPTOM: dry cough for three days\n"
                    "OBSERVATION: temperature 38.1 C yesterday evening\n"
                    "MEDICATION: acetaminophen as needed\n"
                    "ALLERGY STATUS: not stated\n"
                    "Keep each field on a single line. "
                    "Use 'not stated' for missing information; do not turn it into "
                    "a negative finding."
                ),
            },
            {"role": "user", "content": CLINICAL_NOTE},
        ],
    )
    response_text = response.choices[0].message.content or ""
    if "<unused95>" in response_text:
        response_text = response_text.rsplit("<unused95>", maxsplit=1)[-1].strip()
    response_text = response_text.replace("<unused94>", "").strip()
    return response_text or "The model returned an empty response."


def parse_labeled_facts(response_text: str) -> dict[str, object]:
    """Parse common field labels and retain lines that need manual inspection."""
    category_map = {
        "symptom": "symptoms",
        "medication": "medications",
        "date": "dates",
        "observation": "observations",
    }
    extraction: dict[str, object] = {
        "symptoms": [],
        "medications": [],
        "allergy_status": "",
        "dates": [],
        "observations": [],
        "not_stated": [],
        "unparsed_lines": [],
        "raw_response": response_text,
    }
    for line in response_text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        match = re.match(
            r"^(SYMPTOM|MEDICATION|DATE|OBSERVATION|ALLERGY STATUS|NOT STATED)\s*:\s*(.*)$",
            line,
            re.I,
        )
        if not match:
            if line:
                extraction["unparsed_lines"].append(line)
            continue

        category = match.group(1).casefold()
        value_and_quote = match.group(2).strip()
        if category == "allergy status":
            extraction["allergy_status"] = value_and_quote.rstrip(".; ")
            if value_and_quote.casefold().startswith("not stated"):
                extraction["not_stated"].append("allergies")
            continue
        if category == "not stated":
            category_text = re.split(
                r"\|\s*SOURCE QUOTE\s*:", value_and_quote, maxsplit=1, flags=re.I
            )[0]
            extraction["not_stated"].append(category_text.strip().rstrip(".; "))
            continue

        value = re.split(
            r"\|\s*SOURCE QUOTE\s*:", value_and_quote, maxsplit=1, flags=re.I
        )[0].strip().rstrip(".; ")
        extraction[category_map[category]].append(value)
    return extraction


def run_extraction_example() -> Path:
    """Print and save raw labeled text plus its locally parsed fields."""
    response_text = extract_facts_from_note()
    extraction = parse_labeled_facts(response_text)
    result = {
        "source_note": CLINICAL_NOTE,
        "raw_model_response": response_text,
        "extraction": extraction,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "clinical_note_extraction.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved structured extraction to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        run_extraction_example()
    except Exception as error:
        print(f"Extraction could not be completed: {error}")
        print("Check that Ollama is running and the selected model is available.")
