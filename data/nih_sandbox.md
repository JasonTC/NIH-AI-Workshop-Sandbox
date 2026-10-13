# NIH Sandbox

The goal of this repository is to provide demo code to facilitate two workshops:
- Clinical Vision-Language Models (VLMs) for Multimodal Healthcare Data Integration
- Clinical LLMs & Structured Data Extraction

The workshops are for clinicians who have low experience in CS and AI, therefore it is intended to be simple and demonstrate how they can use LLMs for clinical tasks.

`uv` shall be used for code management.

Creation of the workshop is as follows:
- Build proof of concepts first
	- Proof of concepts are to be put into `*.py` scripts.
	- Each activity/section is in a separate script
	- Each function should be self-contained, assume that each function corresponds to a notebook cell
- Build notebook from proof of concepts
- Test notebooks

Gitignore contains `data/test/`. To read directory contents directly, temporarily comment it out from the gitignore. However, make sure to uncomment it afterwards.



## Proof-of-concept activities

Run scripts from the repository root using `uv run python <script>`:

1. `01_ollama_hello_world.py` — check the local model connection with a text prompt.
2. `02_inspect_and_export_npz_image.py` — inspect dataset splits and labels, then
   save one example from each dataset to `data/test/`.
3. `03_interpret_image_with_medgemma.py` — send the saved pneumonia image to the
   vision-language model and save its response under `data/test/`.
4. `04_summarize_synthea_fhir.py` — read one Synthea FHIR bundle and save a short
   plain-language summary under `data/test/`.
5. `05_extract_fields_from_clinical_note.py` — extract facts from a fictional
   note into validated JSON and save the result under `data/test/`.
6. `06_combine_patient_summary_and_image.py` — send the generated record summary
   and a sample image together, and save the response under `data/test/`.
7. `07_shortlist_medgemma_images.py` — check five candidate images at a time
   for one dataset/class using three differently worded prompts per candidate.
   Images with three matching predictions are saved as JPGs under the dataset
   folder; updated Matplotlib grids are saved under `data/test/`.

Run activities 2 and 4 before activity 6 so their sample image and patient
summary are available. Activity 3 uses the pneumonia example by default; edit
the `sample_image` path in that script to try the retina example instead.
For activity 7, provide the dataset and numeric label to check. For example:

```sh
uv run python 07_shortlist_medgemma_images.py --dataset pneumoniamnist --label 0
```

Each run checks five candidates by default and prints the next `--offset` when
more candidates are needed. Continue with the reported offset after reviewing
the raw answers. Images are accepted only when all three prompts match the NPZ
label. The response parser accepts concise natural-language labels as well as
JSON, and logs each raw answer to `data/test/medgemma_classification_attempts.jsonl`.

Example:

```sh
uv run python 02_inspect_and_export_npz_image.py
uv run python 03_interpret_image_with_medgemma.py
```

Generated files are kept in `data/test/` and are ignored by Git. The Synthea
record and image in the multimodal example are independently selected teaching
examples, not data from the same patient. These proof of concepts are for
education and are not intended for clinical decision-making.

---

# Repository/task management

- `.gitignore` is respected in the repository, to access data folder contents, make sure to comment out the data directory gitignore.
- Working notes may be updated to keep information that would be useful to persist across agentic sessions
	- Notes should be concise and terse, focusing on key points to maintain human readability. Assume limited attention would be given to it.
	- The operator will be responsible for moving information from the working notes to task details above. Only edit sections other than the working notes when directly instructed to.

---

# Working notes/Comments

## Checklist

- [ ] Keep activity scripts small and self-contained; functions should map cleanly to notebook cells. Use simple imports and constants, including `MODEL_NAME = "medgemma1.5:latest"` and a fixed Ollama URL; do not configure new activity scripts through environment variables.
- [ ] Continue support/verification scripts from `08_*.py`:
  - [ ] `08_audit_workshop_image_set.py`: check label coverage; load and display prepared images without revealing labels; show the label key separately.
  - [ ] `09_probe_multimodal_pairs.py`: compare image-only and image-plus-context responses to screen and verify a deliberately constructed fictional case.
- [ ] Create VLM activity scripts from `101_*.py`:
  - [ ] `101_describe_medical_image.py`: plain-language image description.
  - [ ] `102_classify_medical_image.py`: classify chest X-ray and retinal examples; display predictions and reveal labels afterward.
  - [ ] `103_visual_question_answering.py`: display image, question, and answer.
  - [ ] `104_evidence_linked_interpretation.py`: visible supporting features, uncertainty, and image limitations.
  - [ ] `105_structured_image_findings.py`: extract and validate a consistent set of findings.
  - [ ] `106_compare_image_examples.py`: compare examples from different cases.
  - [ ] `107_contextualized_image_interpretation.py`: interpret an image with a deliberately linked fictional case.
  - [ ] `108_compare_modalities.py`: compare image-only, record-only, and combined responses.
  - [ ] `109_check_cross_modal_consistency.py`: identify agreement, mismatch, or insufficient linkage.
  - [ ] `110_test_context_relevance.py`: compare relevant context with irrelevant or distracting details.
  - [ ] `111_generate_follow_up_questions.py`: suggest information needed to resolve uncertainty.
  - [ ] `112_create_clinical_handoff.py`: create a concise clinician-facing handoff.
  - [ ] `113_restate_for_patient.py`: restate supplied findings in plain language.
  - [ ] `114_test_uncertainty_and_abstention.py`: probe responses to incomplete or ambiguous input.
  - [ ] `115_review_unsupported_claims.py`: present model claims for human review against source material.
  - [ ] `116_prioritize_for_review.py`: demonstrate relative prioritization, clearly marked as unvalidated.
  - [ ] `117_generate_possible_interpretations.py`: generate possible interpretations and evidence needed to assess them.
- [ ] Create structured-data extraction activity scripts:
  - [ ] `118_extract_clinical_facts.py`: extract facts with exact supporting quotes and validate fields.
  - [ ] `119_extract_negation_and_time.py`: distinguish affirmed, denied, historical, current, and unstated facts.
  - [ ] `120_normalize_facts_to_schema.py`: map explicit note content to a small predefined JSON schema.
  - [ ] `121_audit_extracted_data.py`: review source support, omissions, and unsupported extracted values.
- [ ] Run script syntax checks and local MedGemma examples; note model, prompt, and data limitations for notebook use.

## Comments and remarks

- The two workshops total 120 minutes (90-minute VLM workshop and 30-minute structured extraction workshop); activity categories need not be a strict boundary. Prepare more optional activities than can fit and let the facilitator select.
- The prepared image set is being handled separately. Hide labels during prediction; `07_shortlist_medgemma_images.py` selects examples using MedGemma agreement, so those samples are curated demonstrations, not a representative accuracy sample.
- Existing Synthea summaries and MedMNIST images are independently selected, not the same patient. For case-level multimodal interpretation, construct and clearly identify a fictional linked vignette, then verify the candidate pairing and responses with MedGemma. Use unlinked examples only to discuss linkage/consistency limits.
- Prefer a small custom JSON schema with source quotes and explicit “not stated” handling for the 30-minute extraction workshop; avoid full FHIR authoring unless the activity scope changes. Existing scripts `04_summarize_synthea_fhir.py` and `05_extract_fields_from_clinical_note.py` are reusable starting points.
- First-pass observations / issues to add after trying the activities:
  - Model or prompt behavior:
  - Data or example suitability:
  - Setup, latency, or notebook usability:
  - Facilitator/participant feedback:
