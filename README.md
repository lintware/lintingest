# LintIngest

**Lightweight, locally-run agentic document retrieval.**

Index a directory, ask questions about its contents. LintIngest builds a file map and answers using recursive tool-calling with a small local model — all on your hardware, no API keys, no cloud.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## How It Works

```
Target Dir (~/Desktop)
        |
   [Indexing Pass]  ← Agent maps all files, builds index.json
        |
   [Query Pass]     ← User asks question
        |
   4x parallel model workers
        |
   Tool calls: glob_search, content_search, file_read, note_store
        |
   Compaction: summarize + trim context after each retrieval cycle
        |
   Final answer with citations
```

1. **Index** — Walks a target directory, records file metadata and previews into `data/index.json`
2. **Query** — Dispatches up to 4 parallel file reads, compacts results, synthesizes an answer with citations
3. **Interactive** — REPL mode for continuous Q&A over your indexed files

## Model Backends

LintIngest auto-detects your platform and picks the right backend:

| Backend | Platform | Model | How it runs |
|---------|----------|-------|-------------|
| **MLX** | macOS Apple Silicon | `Qwen3.5-0.8B-MLX-8bit` | `mlx_vlm.server` |
| **llama.cpp** | Windows / Linux / any | `Qwen3.5-0.8B-GGUF` (Q4_K_M) | `llama-server` |

Both expose the same OpenAI-compatible API on `localhost:8080`. The agent code is backend-agnostic.

## Quick Start

```bash
git clone https://github.com/lintware/lintingest.git
cd lintingest

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### macOS (Apple Silicon) — uses MLX automatically

```bash
python cli.py interactive
```

### Windows / Linux — requires llama.cpp

Install [llama.cpp](https://github.com/ggerganov/llama.cpp#build) and ensure `llama-server` is on your PATH. The GGUF model is downloaded automatically on first run.

```bash
python cli.py interactive
```

### Manual backend override

```bash
python cli.py interactive --backend llamacpp   # force llama.cpp on macOS
python cli.py interactive --backend mlx        # force MLX (Apple Silicon only)
```

## CLI Usage

```bash
# Index a directory
python cli.py index --target ~/Documents

# Ask a question
python cli.py query "What files mention API keys?"

# Parallel query (reads multiple files at once)
python cli.py query -p "Summarize the project structure"

# Interactive mode (auto-indexes on first run)
python cli.py interactive --target ~/Desktop

# Start model server only
python cli.py server --port 8080
```

### Interactive commands

- `/index` — Re-index the target directory
- `/parallel` — Toggle parallel file reading
- `/quit` — Exit

## Project Structure

```
lintingest/
├── agent/                  # Agent implementation
│   ├── core.py            # Main agent loop (indexing + query)
│   ├── tools.py           # Tool definitions and execution
│   ├── compaction.py      # Context compaction logic
│   ├── parallel.py        # Parallel worker dispatch
│   ├── index.py           # Directory indexing logic
│   └── config.py          # Agent configuration
├── data/                   # Agent working directory (read/write)
│   ├── index.json         # File index
│   ├── notes/             # Agent notes
│   └── cache/             # Context cache
├── server/                 # Model server management
│   ├── backend.py         # Auto-detect platform, unified start/stop
│   ├── mlx_runner.py      # MLX server (macOS Apple Silicon)
│   └── llamacpp_runner.py # llama.cpp server (Windows/Linux)
├── cli.py                  # CLI entry point
├── requirements.txt
├── CLAUDE.md              # Development instructions
├── LICENSE                # MIT
└── README.md
```

## Hardware

Qwen3.5-0.8B is small — runs comfortably on any modern machine:

- **Apple Silicon**: ~1 GB memory, ~244 tok/s aggregate with 4 parallel workers
- **CPU (any platform)**: Works fine, slower inference
- **GPU (CUDA/Vulkan via llama.cpp)**: Fastest on non-Apple hardware

## License

MIT

---

Built by [Lintware](https://github.com/lintware)
