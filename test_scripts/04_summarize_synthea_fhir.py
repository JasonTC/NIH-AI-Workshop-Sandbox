"""Read one Synthea FHIR bundle and print a short plain-language summary."""

import json
from collections import Counter
from pathlib import Path
from typing import Any


FHIR_DIRECTORY = Path(__file__).parent / "data" / "synthea_output" / "fhir"
OUTPUT_DIRECTORY = Path(__file__).parent / "data" / "test"


def find_fhir_bundle() -> Path:
    """Choose the first patient bundle in a stable, repeatable order."""
    for bundle_path in sorted(FHIR_DIRECTORY.glob("*.json")):
        with bundle_path.open(encoding="utf-8") as file:
            bundle = json.load(file)
        if bundle.get("resourceType") == "Bundle":
            return bundle_path

    raise FileNotFoundError(f"No FHIR Bundle JSON files found in {FHIR_DIRECTORY}")


def read_resources(bundle_path: Path) -> list[dict[str, Any]]:
    """Read the resource entries from one FHIR Bundle."""
    with bundle_path.open(encoding="utf-8") as file:
        bundle = json.load(file)

    if bundle.get("resourceType") != "Bundle":
        raise ValueError(f"Expected a FHIR Bundle in {bundle_path}")

    # A Bundle groups resources together. Each entry contains one resource.
    return [
        entry["resource"]
        for entry in bundle.get("entry", [])
        if isinstance(entry.get("resource"), dict)
    ]


def first_code_display(resource: dict[str, Any]) -> str:
    """Get a human-readable code label from common FHIR fields."""
    for field_name in ("code", "medicationCodeableConcept", "type", "class"):
        field_value = resource.get(field_name, {})
        # Some FHIR fields (such as Encounter.type) contain a list of concepts.
        concepts = field_value if isinstance(field_value, list) else [field_value]
        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            # Some resources use a Code directly rather than a CodeableConcept.
            if concept.get("display"):
                return str(concept["display"])
            if concept.get("text"):
                return str(concept["text"])
            for coding in concept.get("coding", []):
                if coding.get("display"):
                    return str(coding["display"])
    return "Description not provided"


def describe_observation(observation: dict[str, Any]) -> str:
    """Format an observation's name and value for a human reader."""
    description = first_code_display(observation)
    if "valueQuantity" in observation:
        quantity = observation["valueQuantity"]
        value = quantity.get("value", "")
        unit = quantity.get("unit", "")
        return f"{description}: {value} {unit}".strip()
    if "valueString" in observation:
        return f"{description}: {observation['valueString']}"
    if "valueCodeableConcept" in observation:
        return f"{description}: {first_code_display({'code': observation['valueCodeableConcept']})}"
    return description


def build_patient_summary(resources: list[dict[str, Any]]) -> str:
    """Turn selected FHIR resources into a short, readable teaching summary."""
    resource_counts = Counter(
        resource.get("resourceType", "Unknown") for resource in resources
    )
    patient = next(
        (resource for resource in resources if resource.get("resourceType") == "Patient"),
        {},
    )

    # Names and identifiers are intentionally left out of the teaching summary.
    lines = ["Synthetic patient summary", "(Generated data for education only)"]
    if patient.get("gender"):
        lines.append(f"Administrative gender: {patient['gender']}")
    if patient.get("birthDate"):
        lines.append(f"Birth date: {patient['birthDate']}")

    lines.append("\nFHIR resource counts:")
    for resource_type, count in sorted(resource_counts.items()):
        lines.append(f"- {resource_type}: {count}")

    for resource_type, heading in (
        ("Condition", "Conditions"),
        ("MedicationRequest", "Medication requests"),
        ("AllergyIntolerance", "Allergies"),
        ("Encounter", "Encounters"),
        ("Observation", "Observations"),
    ):
        matching_resources = [
            resource
            for resource in resources
            if resource.get("resourceType") == resource_type
        ]
        if not matching_resources:
            continue

        lines.append(f"\n{heading} (showing up to 5):")
        for resource in matching_resources[:5]:
            if resource_type == "Observation":
                description = describe_observation(resource)
            else:
                description = first_code_display(resource)
            lines.append(f"- {description}")

    return "\n".join(lines)


def summarize_one_patient() -> Path:
    """Read one Synthea bundle, print its summary, and save it under data/test."""
    bundle_path = find_fhir_bundle()
    resources = read_resources(bundle_path)
    summary = build_patient_summary(resources)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "synthea_patient_summary.txt"
    output_path.write_text(summary + "\n", encoding="utf-8")
    print(f"Source bundle: {bundle_path.name}")
    print(summary)
    print(f"\nSaved summary to: {output_path}")
    return output_path


if __name__ == "__main__":
    summarize_one_patient()
