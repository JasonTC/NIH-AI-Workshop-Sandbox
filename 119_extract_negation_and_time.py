"""Extract affirmed, denied, historical, current, and unstated note facts."""

import json
import re
from pathlib import Path

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
OUTPUT_DIRECTORY = ROOT_DIRECTORY / "data" / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
CLINICAL_NOTE = """
Fictional follow-up note: The patient reports a cough for four days and denies
shortness of breath today. They took amoxicillin in 2022 but are not taking it
now. No allergy information is documented in this note.
""".strip()


def extract_negation_and_time(note: str = CLINICAL_NOTE) -> str:
    """Ask for labeled plain-text facts that preserve denial and temporal status."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only note-supported facts. Use status present for an "
                    "affirmed fact, denied for an explicit denial, and not_stated "
                    "only for information the note says is undocumented. When a "
                    "medication is described as taken in the past but not taken "
                    "now, represent both facts separately. Use one plain-text line "
                    "per fact in this format: FACT: concept | STATUS: present or "
                    "denied | TIME: current, historical, or unspecified | VALUE: "
                    "short description | SOURCE QUOTE: exact words from the note. "
                    "Use NOT STATED: category for undocumented information. Do not "
                    "return JSON. Include each of these distinctions: "
                    "FACT: cough | STATUS: present | TIME: current | VALUE: cough "
                    "for four days | SOURCE QUOTE: reports a cough for four days; "
                    "FACT: shortness of breath | STATUS: denied | TIME: current | "
                    "VALUE: shortness of breath | SOURCE QUOTE: denies shortness "
                    "of breath today; FACT: amoxicillin | STATUS: present | TIME: "
                    "historical | VALUE: took amoxicillin | SOURCE QUOTE: They took "
                    "amoxicillin in 2022; FACT: amoxicillin | STATUS: denied | "
                    "TIME: current | VALUE: not taking amoxicillin | SOURCE QUOTE: "
                    "not taking it now; NOT STATED: allergies. Put each example "
                    "on its own line."
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
    """Parse labeled fact rows while preserving any lines the parser misses."""
    facts = []
    not_stated = []
    unparsed_lines = []
    for line in response_text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        if re.match(r"^NOT STATED\s*:", line, re.I):
            category = re.split(
                r"^NOT STATED\s*:\s*", line, maxsplit=1, flags=re.I
            )[-1].rstrip(".; ")
            if category not in not_stated:
                not_stated.append(category)
            continue
        if not re.match(r"^FACT\s*:", line, re.I):
            unparsed_lines.append(line)
            continue

        fact = {}
        for segment in line.split("|"):
            if ":" not in segment:
                continue
            key, value = segment.split(":", maxsplit=1)
            key = key.casefold().strip()
            value = value.strip().strip('"')
            if key in {"fact", "concept"}:
                fact["concept"] = value
            elif key == "status":
                fact["status_as_returned"] = value
                fact["status"] = "present" if value.casefold() == "historical" else value
            elif key in {"time", "temporality"}:
                fact["temporality"] = (
                    "historical" if value.casefold() == "past" else value
                )
            elif key == "value":
                fact["value"] = value
            elif key in {"source quote", "quote", "evidence"}:
                if value.casefold() not in {"none", "not stated", "n/a"}:
                    fact["source_quote"] = value.rstrip(".; ")
        if fact:
            facts.append(fact)
            if (
                fact.get("status", "").casefold() == "not_stated"
                or fact.get("status_as_returned", "").casefold() == "not_stated"
            ):
                category = fact.get("concept", "unspecified")
                if category not in not_stated:
                    not_stated.append(category)
        else:
            unparsed_lines.append(line)
    return {
        "facts": facts,
        "not_stated": not_stated,
        "unparsed_lines": unparsed_lines,
        "raw_response": response_text,
    }


def save_extraction(
    note: str, response_text: str, parsed_extraction: dict[str, object]
) -> Path:
    """Save the raw labeled text alongside locally parsed fields."""
    result = {
        "source_note": note,
        "raw_model_response": response_text,
        "parsed_extraction": parsed_extraction,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "119_negation_temporality_extraction.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved extraction to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        response_text = extract_negation_and_time()
        parsed_extraction = parse_labeled_facts(response_text)
        print(f"Source note:\n{CLINICAL_NOTE}\n")
        print(f"Raw labeled response:\n{response_text}\n")
        print(json.dumps(parsed_extraction, indent=2))
        save_extraction(CLINICAL_NOTE, response_text, parsed_extraction)
    except Exception as error:
        print(f"Negation/temporality extraction could not be completed: {error}")
        print("Check that Ollama is running and the model is available.")
