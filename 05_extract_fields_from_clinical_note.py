"""Ask a local model to extract facts from a fictional clinical note as JSON."""

import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "medgemma1.5:latest")
OUTPUT_DIRECTORY = Path(__file__).parent / "data" / "test"

# This example is fictional and is written only to demonstrate information
# extraction. It does not contain information about a real patient.
CLINICAL_NOTE = """
Fictional visit note: On 2025-04-10, the patient reports a dry cough for three
days and a temperature of 38.1 C yesterday evening. They report taking
acetaminophen as needed. The note does not mention allergies.
""".strip()


class ExtractedFact(BaseModel):
    """One extracted fact paired with the words that support it in the note."""

    value: str
    source_quote: str


class ClinicalNoteExtraction(BaseModel):
    """A small, validated structure for information from the sample note."""

    symptoms: list[ExtractedFact]
    medications: list[ExtractedFact]
    allergies: list[ExtractedFact]
    dates: list[ExtractedFact]
    not_stated: list[str]


def create_ollama_client() -> OpenAI:
    """Create an OpenAI client connected to the local Ollama server."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def extract_facts_from_note() -> ClinicalNoteExtraction:
    """Ask the model for JSON and validate it against the Pydantic schema."""
    client = create_ollama_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only information explicitly stated in the note. "
                    "Do not make diagnoses or infer missing facts. For every "
                    "fact, include a short exact supporting source_quote. If a "
                    "category is not mentioned, return an empty list and note "
                    "it in not_stated. Return only a JSON object using this "
                    "exact structure: "
                    '{"symptoms":[{"value":"dry cough",'
                    '"source_quote":"reports a dry cough"}],'
                    '"medications":[],"allergies":[],"dates":[],'
                    '"not_stated":["allergies"]}. If a category list is '
                    "empty because the note does not mention it, include its "
                    "field name in not_stated. Each item in symptoms, "
                    "medications, allergies, and dates must have both value "
                    "and source_quote."
                ),
            },
            {"role": "user", "content": CLINICAL_NOTE},
        ],
    )
    response_text = response.choices[0].message.content or "{}"

    # Pydantic checks that the response has the expected fields and data types.
    return ClinicalNoteExtraction.model_validate_json(response_text)


def run_extraction_example() -> Path:
    """Print and save the source note and its validated structured extraction."""
    extraction = extract_facts_from_note()
    result = {
        "source_note": CLINICAL_NOTE,
        "extraction": extraction.model_dump(),
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
    except ValidationError as error:
        print(f"The model response did not match the expected fields:\n{error}")
    except Exception as error:
        print(f"Extraction could not be completed: {error}")
        print("Check that Ollama is running and the selected model is available.")
