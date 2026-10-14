"""Compare model responses with and without incidental fictional context."""

import base64
import mimetypes
from pathlib import Path

import matplotlib.pyplot as plt
from openai import OpenAI
from PIL import Image


ROOT_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_DIRECTORY = DATA_DIRECTORY / "test"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "medgemma1.5:latest"
IMAGE_PATTERN = "pneumoniamnist/*_label_1_pneumonia.jpg"
RELEVANT_CONTEXT = (
    "Fictional teaching note: the person has had a cough for three days and "
    "reports fever. This note is constructed for the exercise."
)
CONTEXT_WITH_INCIDENTAL_DETAILS = (
    "Fictional teaching note: the person has had a cough for three days and "
    "reports fever. The person works as a librarian and enjoys gardening. "
    "This note is constructed for the exercise."
)


def find_example_image() -> Path:
    """Find one prepared image to use with both context variants."""
    matches = sorted(DATA_DIRECTORY.glob(IMAGE_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No image matches {IMAGE_PATTERN!r}.")
    return matches[0]


def save_image_and_print_contexts(image_path: Path) -> Path:
    """Save the shared image preview and print both context variants."""
    with Image.open(image_path) as image:
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.imshow(image, cmap="gray")
        axis.set_title("Same teaching image for both requests")
        axis.axis("off")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "110_context_relevance_image.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved image preview to: {output_path}")
    print(f"Relevant context only:\n{RELEVANT_CONTEXT}\n")
    print(f"Same context plus incidental details:\n{CONTEXT_WITH_INCIDENTAL_DETAILS}")
    return output_path


def ask_with_context(image_path: Path, context: str) -> str:
    """Ask for image observations and keep non-image context identified as text."""
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=180.0)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe visible image features separately from the "
                            "fictional context. State whether the context is "
                            "relevant to interpreting the image, and do not use "
                            "incidental personal details as clinical evidence. "
                            "State uncertainty.\n\n"
                            f"Fictional context:\n{context}"
                        ),
                    },
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


def compare_context_variants(image_path: Path) -> dict[str, str]:
    """Run the same task with relevant context, then with incidental details added."""
    return {
        "Relevant context": ask_with_context(image_path, RELEVANT_CONTEXT),
        "Relevant plus incidental context": ask_with_context(
            image_path, CONTEXT_WITH_INCIDENTAL_DETAILS
        ),
    }


def display_context_responses(responses: dict[str, str]) -> None:
    """Print the two outputs for participant comparison."""
    for heading, response in responses.items():
        print(f"{heading}:\n{response}\n")


if __name__ == "__main__":
    try:
        sample_image = find_example_image()
        save_image_and_print_contexts(sample_image)
        display_context_responses(compare_context_variants(sample_image))
    except Exception as error:
        print(f"Context relevance example could not be completed: {error}")
        print("Check the prepared image files and that Ollama is running.")
