"""Extract note facts in labeled plain text and parse them locally."""

import json
import re
from pathlib import Path

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
OUTPUT_DIRECTORY = ROOT_DIRECTORY / "data" / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
CLINICAL_NOTE = """
Fictional visit note: On April 10, 2025, the patient reports a dry cough for
three days and a temperature of 38.1 C yesterday evening. They report taking
acetaminophen as needed. Allergy information was not discussed in this note.
""".strip()


def extract_clinical_facts(note: str = CLINICAL_NOTE) -> str:
    """Ask the model for labeled plain-text facts with supporting source quotes."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only facts explicitly stated in the note. Do not "
                    "diagnose or infer missing facts. For each extracted fact, "
                    "include a short exact quote copied from the note. Do not "
                    "treat undocumented information as a negative finding. Use "
                    "one plain-text line per fact with these labels: SYMPTOM:, "
                    "MEDICATION:, DATE:, OBSERVATION:, or NOT STATED:. For facts, "
                    "append | SOURCE QUOTE: followed by an exact quote. Use each "
                    "fact once, including symptom duration. For undocumented "
                    "categories, write only NOT STATED: category. Follow this "
                    "format: DATE: 2025-04-10 | SOURCE QUOTE: On April 10, 2025; "
                    "SYMPTOM: dry cough for three days | SOURCE QUOTE: reports a "
                    "dry cough for three days; OBSERVATION: temperature 38.1 C | "
                    "SOURCE QUOTE: temperature of 38.1 C yesterday evening; "
                    "MEDICATION: acetaminophen as needed | SOURCE QUOTE: taking "
                    "acetaminophen as needed; NOT STATED: allergies. Use only "
                    "these labels and do not return JSON."
                ),
            },
            {"role": "user", "content": note},
        ],
    )
    response_text = response.choices[0].message.content or ""
    if "<unused95>" in response_text:
        response_text = response_text.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return response_text or "The model returned an empty response."


def parse_labeled_facts(response_text: str) -> dict[str, object]:
    """Parse category labels and source quotes while preserving the raw text."""
    category_map = {
        "symptom": "symptoms",
        "medication": "medications",
        "date": "dates",
        "observation": "observations",
    }
    parsed: dict[str, object] = {
        "symptoms": [],
        "medications": [],
        "dates": [],
        "observations": [],
        "not_stated": [],
        "unparsed_lines": [],
        "raw_response": response_text,
    }
    for line in response_text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        match = re.match(r"^(SYMPTOM|MEDICATION|DATE|OBSERVATION|NOT STATED)\s*:\s*(.*)$", line, re.I)
        if not match:
            if line:
                parsed["unparsed_lines"].append(line)
            continue

        category = match.group(1).casefold()
        value_and_quote = match.group(2).strip()
        if category == "not stated":
            category_text = re.split(
                r"\|\s*SOURCE QUOTE\s*:", value_and_quote, maxsplit=1, flags=re.I
            )[0]
            parsed["not_stated"].append(category_text.strip().rstrip(".; "))
            continue

        pieces = re.split(r"\|\s*SOURCE QUOTE\s*:\s*", value_and_quote, maxsplit=1, flags=re.I)
        fact = {
            "value": pieces[0].strip(),
            "source_quote": pieces[1].strip().strip('"').rstrip(".; ")
            if len(pieces) > 1
            else "",
        }
        parsed[category_map[category]].append(fact)
    return parsed


def save_extraction(
    note: str, response_text: str, parsed_extraction: dict[str, object]
) -> Path:
    """Save the raw labeled response beside fields parsed from its labels."""
    result = {
        "source_note": note,
        "raw_model_response": response_text,
        "parsed_extraction": parsed_extraction,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "118_clinical_fact_extraction.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved extraction to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        response_text = extract_clinical_facts()
        parsed_extraction = parse_labeled_facts(response_text)
        print(f"Source note:\n{CLINICAL_NOTE}\n")
        print(f"Raw labeled response:\n{response_text}\n")
        print(json.dumps(parsed_extraction, indent=2))
        save_extraction(CLINICAL_NOTE, response_text, parsed_extraction)
    except Exception as error:
        print(f"Fact extraction could not be completed: {error}")
        print("Check that Ollama is running and the model is available.")
