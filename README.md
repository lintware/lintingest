# 📄 LintIngest

**Unified locally-run agentic document retrieval and answering system.**

Ingest any document, ask any question. LintIngest reads your files, builds a knowledge base, and answers questions using recursive deep research — all running locally on your hardware.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

## What It Does

1. **Ingest** — Drop in PDFs, DOCX, TXT, MD, HTML, images, spreadsheets. LintIngest parses, chunks, and embeds them.
2. **Retrieve** — Semantic search over your document corpus with hybrid retrieval (vector + BM25).
3. **Answer** — A recursive deep research agent reads relevant chunks, identifies gaps, fetches more context, and synthesizes a final answer.

No API keys. No cloud. Everything runs on your machine.

## Architecture

```
Documents → Parser → Chunker → Embeddings → Vector Store (ChromaDB)
                                                    ↓
                                    WebUI → Agent (Qwen3.5) → Answer
                                              ↑         ↓
                                              ← Recursive Retrieval ←
```

The agent uses a **deep research loop**:
1. Receives your question
2. Retrieves relevant chunks
3. Evaluates if it has enough context
4. If not → reformulates query → retrieves again
5. Repeats until confident (max depth configurable)
6. Synthesizes final answer with citations

## Models

| Component | Default Model | Alternatives |
|-----------|--------------|--------------|
| **Reasoning** | Qwen3.5-7B | Qwen3.5-2B (light), Qwen3.5-9B (heavy) |
| **Embeddings** | Qwen3-Embedding-0.6B | Any sentence-transformers model |
| **OCR** | (planned) | Qwen-VL, GOT-OCR |

All models run locally via [Ollama](https://ollama.com) or [vLLM](https://github.com/vllm-project/vllm). Fully configurable in `config.yaml`.

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/lintware/lintingest.git
cd lintingest
cp config.example.yaml config.yaml  # edit model settings

docker compose up -d
```

Open `http://localhost:7600` — done.

### Local Install

```bash
git clone https://github.com/lintware/lintingest.git
cd lintingest

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Ollama with your model
ollama pull qwen3.5:7b

# Run
python -m lintingest serve
```

## Usage

### WebUI

Navigate to `http://localhost:7600`. Drag and drop documents, ask questions.

### CLI

```bash
# Ingest documents
lintingest ingest ./documents/

# Ask a question
lintingest ask "What are the key findings from the Q3 report?"

# Deep research mode (recursive, thorough)
lintingest ask --deep "Compare the financial performance across all quarterly reports"
```

### API

```bash
# Ingest
curl -X POST http://localhost:7600/api/ingest \
  -F "files=@report.pdf"

# Ask
curl -X POST http://localhost:7600/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the main points", "deep": true}'

# List documents
curl http://localhost:7600/api/documents
```

## Configuration

```yaml
# config.yaml
model:
  provider: ollama          # ollama | vllm | llamacpp
  name: qwen3.5:7b          # model name
  base_url: http://localhost:11434

embeddings:
  model: Qwen/Qwen3-Embedding-0.6B
  device: auto               # cpu | cuda | mps

retrieval:
  chunk_size: 512
  chunk_overlap: 64
  top_k: 10
  hybrid: true               # vector + BM25

agent:
  max_depth: 5               # recursive retrieval depth
  confidence_threshold: 0.7  # stop when confident enough
  thinking: true             # show reasoning steps

ocr:
  enabled: false             # enable for scanned PDFs / images
  model: null                # coming soon

server:
  host: 0.0.0.0
  port: 7600

storage:
  vector_db: chromadb
  persist_dir: ./data
```

## Supported Formats

| Format | Status |
|--------|--------|
| PDF | ✅ |
| DOCX / DOC | ✅ |
| TXT / MD | ✅ |
| HTML | ✅ |
| CSV / XLSX | ✅ |
| Images (OCR) | 🔜 Planned |
| Audio (transcription) | 🔜 Planned |

## Hardware Requirements

| Model | VRAM | RAM | Notes |
|-------|------|-----|-------|
| Qwen3.5-2B | 2 GB | 4 GB | Fast, good for simple docs |
| Qwen3.5-7B | 5 GB | 8 GB | Recommended default |
| Qwen3.5-9B | 7 GB | 12 GB | Best quality |

Runs on CPU too (slower). Apple Silicon (MPS) and NVIDIA CUDA supported.

## Project Structure

```
lintingest/
├── lintingest/
│   ├── __init__.py
│   ├── server.py          # FastAPI server + WebUI
│   ├── agent.py           # Deep research agent
│   ├── retriever.py       # Hybrid retrieval (vector + BM25)
│   ├── ingest.py          # Document parsing & chunking
│   ├── embeddings.py      # Embedding model wrapper
│   ├── ocr.py             # OCR pipeline (planned)
│   └── config.py          # Configuration loader
├── webui/
│   ├── index.html         # Single-page app
│   ├── style.css
│   └── app.js
├── docker-compose.yaml
├── Dockerfile
├── config.example.yaml
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Document ingestion (PDF, DOCX, TXT, MD, HTML, CSV)
- [x] Hybrid retrieval (vector + BM25)
- [x] Recursive deep research agent
- [x] Built-in WebUI
- [x] Docker support
- [ ] OCR pipeline (Qwen-VL / GOT-OCR)
- [ ] Audio transcription & ingestion
- [ ] Multi-collection support
- [ ] Conversation memory
- [ ] Export answers as reports
- [ ] Plugin system for custom parsers

## Contributing

PRs welcome. Keep it simple, keep it local.

## License

MIT

---

Built by [Lintware](https://github.com/lintware)
