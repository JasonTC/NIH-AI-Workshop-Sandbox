"""Send a simple greeting to the local Ollama model."""

import os

from openai import OpenAI


# These defaults match a local Ollama installation. They can be changed without
# editing the script by setting OLLAMA_BASE_URL and OLLAMA_MODEL.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "medgemma1.5:latest")


def create_ollama_client() -> OpenAI:
    """Create an OpenAI client that sends requests to Ollama."""
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        # Ollama does not require an API key, but the OpenAI client expects one.
        api_key="ollama",
    )


def ask_for_a_greeting() -> str:
    """Ask the selected local model for a short hello-world response."""
    client = create_ollama_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    "For this API connectivity test, reply with exactly: "
                    "Hello, world!"
                ),
            }
        ],
    )
    answer = response.choices[0].message.content or ""

    # Some Ollama model templates include an internal section before this
    # marker. Keep the simple demo focused on the user-facing answer.
    if "<unused95>" in answer:
        answer = answer.rsplit("<unused95>", maxsplit=1)[-1]

    return answer.strip() or "The model returned an empty response."


if __name__ == "__main__":
    print(f"Ollama model: {MODEL_NAME}")
    try:
        print(ask_for_a_greeting())
    except Exception as error:
        print(f"Could not contact Ollama: {error}")
        print("Check that Ollama is running and the model has been downloaded.")
