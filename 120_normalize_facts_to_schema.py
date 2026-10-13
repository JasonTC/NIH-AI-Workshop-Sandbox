"""Map fictional note content into a small, explicit structured schema."""

import json
import re
from pathlib import Path

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
OUTPUT_DIRECTORY = ROOT_DIRECTORY / "data" / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
CLINICAL_NOTE = """
Fictional visit note dated April 10, 2025: temperature 100.6 degrees F, pulse
92 beats per minute, and blood pressure 126/78 mmHg. The patient says they take
acetaminophen 650 mg by mouth as needed.
""".strip()


def normalize_facts(note: str = CLINICAL_NOTE) -> str:
    """Ask for labeled plain text that maps note content to normalized fields."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Map only explicitly stated note content into the requested "
                    "fields. Format the date as YYYY-MM-DD. Use the "
                    "provided concept values temperature, pulse, or blood_pressure. "
                    "For each observation, preserve the written number in "
                    "value_as_written, put a normalized numeric string in "
                    "normalized_value, and retain its stated unit; do not convert "
                    "units. For medication, report only stated name, dose, route, "
                    "and frequency; use 'not stated' where needed. Include an "
                    "exact source quote for every observation and medication. "
                    "Return labeled plain-text lines, not JSON. Use one record per "
                    "line in these formats:\n"
                    "ENCOUNTER DATE: 2025-04-10\n"
                    "OBSERVATION: temperature | ORIGINAL VALUE: 100.6 | "
                    "NORMALIZED VALUE: 100.6 | UNIT: degrees F | SOURCE QUOTE: "
                    "temperature 100.6 degrees F\n"
                    "MEDICATION: acetaminophen | DOSE: 650 mg | ROUTE: by mouth | "
                    "FREQUENCY: as needed | SOURCE QUOTE: acetaminophen 650 mg by "
                    "mouth as needed\n"
                    "NOT STATED: allergies"
                ),
            },
            {"role": "user", "content": note},
        ],
    )
    response_text = response.choices[0].message.content or ""
    if "<unused95>" in response_text:
        response_text = response_text.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return response_text or "The model returned an empty response."


def parse_normalized_fields(response_text: str) -> dict[str, object]:
    """Parse the labeled lines without requiring the model to emit JSON."""
    normalized: dict[str, object] = {
        "encounter_date": "",
        "observations": [],
        "medications": [],
        "not_stated": [],
        "unparsed_lines": [],
        "raw_response": response_text,
    }
    for line in response_text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        if re.match(r"^ENCOUNTER DATE\s*:", line, re.I):
            normalized["encounter_date"] = re.split(
                r"^ENCOUNTER DATE\s*:\s*", line, maxsplit=1, flags=re.I
            )[-1].strip()
            continue
        if re.match(r"^NOT STATED\s*:", line, re.I):
            normalized["not_stated"].append(
                re.split(r"^NOT STATED\s*:\s*", line, maxsplit=1, flags=re.I)[-1]
            )
            continue

        segments = [segment.strip() for segment in line.split("|")]
        if not segments or ":" not in segments[0]:
            normalized["unparsed_lines"].append(line)
            continue
        kind, first_value = segments[0].split(":", maxsplit=1)
        kind = kind.casefold().strip()
        record_key = "concept" if kind == "observation" else "name"
        record: dict[str, str] = {record_key: first_value.strip()}
        field_map = {
            "original value": "value_as_written",
            "normalized value": "normalized_value",
            "unit": "unit",
            "dose": "dose_as_written",
            "route": "route",
            "frequency": "frequency",
            "source quote": "source_quote",
        }
        for segment in segments[1:]:
            if ":" not in segment:
                continue
            key, value = segment.split(":", maxsplit=1)
            normalized_key = key.casefold().strip()
            if normalized_key in field_map:
                record[field_map[normalized_key]] = value.strip().strip('"')
        if kind == "observation":
            normalized["observations"].append(record)
        elif kind == "medication":
            normalized["medications"].append(record)
        else:
            normalized["unparsed_lines"].append(line)
    return normalized


def display_normalized_note(
    note: str, response_text: str, normalized: dict[str, object]
) -> None:
    """Display source text, raw labeled output, and locally parsed fields."""
    print(f"Source note:\n{note}\n")
    print(f"Raw labeled response:\n{response_text}\n")
    print(json.dumps(normalized, indent=2))


def save_normalized_note(
    note: str, response_text: str, normalized: dict[str, object]
) -> Path:
    """Save raw text and locally parsed fields for comparison."""
    result = {
        "source_note": note,
        "raw_model_response": response_text,
        "parsed_normalized_fields": normalized,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "120_normalized_clinical_note.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved normalized result to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        response_text = normalize_facts()
        normalized = parse_normalized_fields(response_text)
        display_normalized_note(CLINICAL_NOTE, response_text, normalized)
        save_normalized_note(CLINICAL_NOTE, response_text, normalized)
    except Exception as error:
        print(f"Schema normalization could not be completed: {error}")
        print("Check that Ollama is running and the model is available.")
