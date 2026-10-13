"""Audit note-extraction quotes and required facts against the source note."""

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIRECTORY = Path(__file__).parent
OUTPUT_DIRECTORY = ROOT_DIRECTORY / "data" / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
CLINICAL_NOTE = """
Fictional visit note: The patient reports a dry cough for three days and fever
of 38.1 C yesterday evening. They take acetaminophen as needed. The note states
that allergy information was not discussed.
""".strip()
REQUIRED_SOURCE_PHRASES = [
    "dry cough",
    "fever of 38.1 C",
    "acetaminophen as needed",
]


def extract_for_audit(note: str = CLINICAL_NOTE) -> str:
    """Ask for labeled plain-text facts with quotes that can be checked locally."""
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract explicit facts as plain-text lines, not JSON. For each "
                    "fact use this format: FACT: category | VALUE: short value | "
                    "SOURCE QUOTE: exact words copied from the note. For information "
                    "explicitly described as not discussed, use NOT STATED: category. "
                    "Include the dry cough, fever, acetaminophen, and undocumented "
                    "allergy information. Do not infer diagnoses."
                ),
            },
            {"role": "user", "content": note},
        ],
    )
    response_text = response.choices[0].message.content or ""
    if "<unused95>" in response_text:
        response_text = response_text.rsplit("<unused95>", maxsplit=1)[-1].strip()
    return response_text or "The model returned an empty response."


def parse_labeled_extraction(response_text: str) -> dict[str, Any]:
    """Parse FACT and NOT STATED lines while retaining any unparsed text."""
    facts = []
    not_stated = []
    unparsed_lines = []
    for line in response_text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        if re.match(r"^NOT STATED\s*:", line, re.I):
            category_text = re.split(
                r"^NOT STATED\s*:\s*", line, maxsplit=1, flags=re.I
            )[-1]
            not_stated.append(category_text.split("|", maxsplit=1)[0].strip().casefold())
            continue
        if not re.match(r"^FACT\s*:", line, re.I):
            unparsed_lines.append(line)
            continue

        fact = {}
        for segment in line.split("|"):
            if ":" not in segment:
                continue
            key, value = segment.split(":", maxsplit=1)
            normalized_key = key.casefold().strip()
            value = value.strip().strip('"').rstrip(".; ")
            if normalized_key in {"fact", "category", "field"}:
                fact["field"] = value
            elif normalized_key == "value":
                fact["value"] = value
            elif normalized_key in {"source quote", "quote", "evidence"}:
                fact["source_quote"] = value
        if fact:
            facts.append(fact)
        else:
            unparsed_lines.append(line)
    return {
        "facts": facts,
        "not_stated": not_stated,
        "unparsed_lines": unparsed_lines,
        "raw_response": response_text,
    }


def audit_extraction(
    note: str, extraction: dict[str, Any] | str
) -> list[str]:
    """Flag quotes not found in the note and expected phrases not extracted."""
    if not isinstance(extraction, dict):
        return ["Response was not parsed into labeled fact fields."]

    facts = extraction.get("facts")
    if not isinstance(facts, list):
        return ["No parsed FACT lines were available to audit."]

    normalized_note = " ".join(note.casefold().split())
    issues = []
    extracted_quotes = []
    for index, fact in enumerate(facts, start=1):
        if not isinstance(fact, dict):
            issues.append(f"Fact {index} is not an object.")
            continue
        source_quote = fact.get("source_quote")
        if not isinstance(source_quote, str) or not source_quote.strip():
            issues.append(f"Fact {index} has no source quote.")
            continue
        extracted_quotes.append(" ".join(source_quote.casefold().split()))
        if " ".join(source_quote.casefold().split()) not in normalized_note:
            issues.append(f"Fact {index} quote is not an exact note substring.")

    for required_phrase in REQUIRED_SOURCE_PHRASES:
        normalized_phrase = " ".join(required_phrase.casefold().split())
        if not any(normalized_phrase in quote for quote in extracted_quotes):
            issues.append(f"Expected source information may be missing: {required_phrase}.")
    return issues


def save_audit_result(
    note: str,
    response_text: str,
    extraction: dict[str, Any],
    issues: list[str],
) -> Path:
    """Save model output and simple audit flags for human follow-up."""
    result = {
        "source_note": note,
        "raw_model_response": response_text,
        "extraction": extraction,
        "audit_status": "needs human review" if issues else "basic checks passed",
        "audit_issues": issues,
        "limitation": (
            "Substring checks do not establish clinical correctness or completeness; "
            "review the source and extraction manually."
        ),
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "121_extraction_audit.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved audit to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        response_text = extract_for_audit()
        extraction = parse_labeled_extraction(response_text)
        issues = audit_extraction(CLINICAL_NOTE, extraction)
        save_audit_result(CLINICAL_NOTE, response_text, extraction, issues)
    except Exception as error:
        print(f"Extraction audit could not be completed: {error}")
        print("Check that Ollama is running and the model is available.")
