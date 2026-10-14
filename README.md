# NIH AI Workshop Sandbox

NIH AI Workshop Sandbox repository containing activities for the following workshops:
- Workshop 1: Clinical Vision-Language Models (VLMs) for Multimodal Healthcare Data Integration
- Workshop 2: Clinical LLMs & Structured Data Extraction

Repository QR Code:  
![Repository QR Code](<data/NIH Workshop Repository.png>)

## Setup

Prerequisites:
- [uv](https://docs.astral.sh/uv/getting-started/installation/): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- [Ollama](https://ollama.com/download): `irm https://ollama.com/install.ps1 | iex`

Install the Python dependencies and download the example model:

```sh
uv sync
ollama pull medgemma1.5:latest
```

Ollama should be running locally before running the workshop activities. By default, the scripts use `http://localhost:11434/v1` and `medgemma1.5:latest`. Override these values with `OLLAMA_BASE_URL` and `OLLAMA_MODEL` if needed.

Launch the Jupyter notebook bu running: `jupyter lab`
