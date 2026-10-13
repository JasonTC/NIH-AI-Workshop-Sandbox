# NIH AI Workshop Sandbox

Beginner-friendly proof-of-concept scripts for the clinical AI workshops. Each
script is a small activity, and the functions inside it are intended to be read
and run one at a time, like notebook cells.

## Setup

Prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download)

Install the Python dependencies and download the example model:

```sh
uv sync
ollama pull medgemma1.5:latest
```

Ollama should be running locally before running a model activity. By default,
the scripts use `http://localhost:11434/v1` and `medgemma1.5:latest`. Override
these values with `OLLAMA_BASE_URL` and `OLLAMA_MODEL` if needed.

