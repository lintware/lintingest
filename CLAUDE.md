# LintIngest - Project Instructions

## Overview

LintIngest is a lightweight, locally-run agentic document retrieval system built on top of ZeroClaw's agent framework with Qwen3.5-0.8B running via MLX on Apple Silicon. It indexes a target directory (default: `~/Desktop`), builds a structured file map, and answers natural language questions about the contents using parallel model execution and recursive tool-calling.

## Architecture

```
Target Dir (~/Desktop)
        |
   [Indexing Pass]  ← Agent maps all files, builds index.json
        |
   [Query Pass]     ← User asks question
        |
   4x parallel model workers (MLX server, concurrency=4)
        |
   Tool calls: glob_search, content_search, file_read, note_store
        |
   Compaction: summarize + trim context after each retrieval cycle
        |
   Final answer with citations
```

### Two-Phase Agent Design

**Phase 1 - Indexing Pass:**
- Walks the target directory recursively
- Records: file path, type, size, modified date, first-line preview
- For text files: extracts summary snippet (first 200 chars)
- Stores result in `data/index.json`
- Runs once per session or on-demand refresh

**Phase 2 - Query Pass:**
- Receives user question
- Consults `data/index.json` for candidate files
- Dispatches up to 4 parallel read/search operations via MLX
- Each worker reads a file chunk and extracts relevant info
- Agent compacts intermediate results (summarize + drop irrelevant)
- Can make notes in `data/notes/` for future queries
- Synthesizes final answer with file path citations

## Model Configuration

- **Model:** `mlx-community/Qwen3.5-0.8B-MLX-8bit` (vision-language model)
- **Server:** `mlx_vlm.server` on `http://127.0.0.1:8080/v1`
- **Concurrency:** 4 parallel requests (optimal for M4 Mac Mini)
- **Max tokens per call:** 256 (keep fast, use multiple calls)
- **Temperature:** 0.3 (factual retrieval, low creativity)

## Backend Support

LintIngest supports two model backends that expose the same OpenAI-compatible API on `localhost:8080`:

| Backend | Platform | Model Format | Runner |
|---------|----------|-------------|--------|
| **MLX VLM** | macOS Apple Silicon | `mlx-community/Qwen3.5-0.8B-MLX-8bit` | `server/mlx_runner.py` |
| **llama.cpp** | Windows / Linux / any | `unsloth/Qwen3.5-0.8B-GGUF` (Q4_K_M) | `server/llamacpp_runner.py` |

- **Auto-detection** (`server/backend.py`): Apple Silicon macOS → MLX, everything else → llama.cpp
- **Manual override:** `--backend mlx` or `--backend llamacpp` on any CLI command
- The agent code is backend-agnostic — it only talks to `http://127.0.0.1:8080/v1`
- llama.cpp requires `llama-server` on PATH (not bundled)
- GGUF model is auto-downloaded via `huggingface_hub` on first run

## Security Constraints - MANDATORY

These rules are non-negotiable and must be enforced at every layer:

### File Operations
- **NO FILE DELETION** - The agent must NEVER delete any file, anywhere, under any circumstances
- **NO FILE MODIFICATION** of source documents - target directory files are READ-ONLY
- **WRITE ONLY** to `{project_root}/data/` directory - all agent outputs (index, notes, cache) go here
- **READ SCOPE:** The target directory (default `~/Desktop`) is read-only accessible
- **BLOCKED PATHS:** `/etc`, `/usr`, `/System`, `/Library`, `/bin`, `/sbin`, `~/.ssh`, `~/.aws`, `~/.gnupg`, any dotfiles outside the project

### Agent Behavior
- **Shell access:** The agent has a `shell_exec` tool for running read-only shell commands in the target directory. Destructive commands (`rm`, `mv`, `cp`, `chmod`, `sudo`, etc.) and network commands (`curl`, `wget`, `ssh`, etc.) are blocked. Scripting interpreters (`python`, `node`, `bash`, etc.) are also blocked to prevent arbitrary code execution.
- No network access (all processing is local)
- No git operations on external repos
- No process spawning beyond the model server (MLX or llama.cpp)
- Rate limit: 200 tool calls per session maximum

## Tool Definitions

The agent has access to these tools:

### `shell_exec`
- **Purpose:** Run shell commands in the target directory for flexible file inspection
- **Input:** `{ "command": string, "timeout": number }`
- **Output:** Command stdout/stderr
- **Security:** Runs in target dir only. Blocked: `rm`, `mv`, `cp`, `chmod`, `sudo`, `curl`, `wget`, `ssh`, `python`, `node`, `bash`, and other destructive/network/scripting commands. No subshells (`$()`, backticks). Max 60s timeout. Stdout capped at 4KB.
- **Use cases:** `ls -la`, `cat file.txt`, `head -50 file.py`, `find . -name '*.py' | wc -l`, `grep -r "pattern" .`, `wc -l *.txt`, `file *.pdf`, `du -sh *`, `sort`, `awk`, `sed` (read-only), `diff file1 file2`

### `index_directory`
- **Purpose:** Walk target dir, build file index
- **Input:** `{ "path": string, "max_depth": number }`
- **Output:** JSON array of file entries
- **Security:** Read-only, respects blocked paths

### `file_read`
- **Purpose:** Read file contents (text files only)
- **Input:** `{ "path": string, "offset": number, "limit": number }`
- **Output:** File content with line numbers
- **Security:** Read-only, max 10MB, target dir + project data dir only

### `glob_search`
- **Purpose:** Find files by pattern
- **Input:** `{ "pattern": string, "directory": string }`
- **Output:** List of matching file paths
- **Security:** Target dir + project data dir only, max 100 results

### `content_search`
- **Purpose:** Grep-like text search across files
- **Input:** `{ "pattern": string, "directory": string, "max_results": number }`
- **Output:** Matching lines with file paths and line numbers
- **Security:** Target dir + project data dir only

### `note_store`
- **Purpose:** Store a note for future reference
- **Input:** `{ "key": string, "content": string, "category": string }`
- **Output:** Confirmation
- **Security:** Writes ONLY to `data/notes/`

### `note_recall`
- **Purpose:** Search stored notes
- **Input:** `{ "query": string, "limit": number }`
- **Output:** Matching notes ranked by relevance

### `read_index`
- **Purpose:** Read the file index without re-scanning
- **Input:** `{}`
- **Output:** Contents of `data/index.json`

### `image_describe`
- **Purpose:** Describe an image file using the VLM
- **Input:** `{ "path": string, "question": string }`
- **Output:** VLM-generated description of the image
- **Security:** Read-only, target dir + project data dir only, image files only

## Compaction Strategy

To keep context small and the agent fast on a 0.8B model:

1. **Max context per turn:** 1500 tokens (model has limited context window)
2. **After each tool result:** Summarize the result into 2-3 key sentences before adding to context
3. **Rolling window:** Keep only the last 3 tool call results in context, summarize older ones into a single "findings so far" block
4. **Index compression:** The file index uses abbreviated entries (path + type + size only), full content loaded on demand
5. **Final synthesis:** Combine all compacted findings into the answer, max 500 tokens

## Parallel Execution Model

The MLX server handles 4 concurrent requests at ~244 tok/s aggregate:

1. **During indexing:** Process 4 files simultaneously for preview extraction
2. **During query:** Dispatch 4 file reads in parallel, each with a focused sub-question
3. **Worker pattern:** Main agent splits work into sub-tasks, dispatches to workers, collects results
4. **No cross-worker dependency:** Each parallel call is independent

## Project Structure

```
lintingest/
├── CLAUDE.md              # This file - project instructions
├── README.md              # Project overview
├── LICENSE                 # MIT
├── agent/                  # Agent implementation
│   ├── __init__.py
│   ├── core.py            # Main agent loop (indexing + query)
│   ├── tools.py           # Tool definitions and execution
│   ├── compaction.py      # Context compaction logic
│   ├── parallel.py        # Parallel worker dispatch
│   ├── index.py           # Directory indexing logic
│   └── config.py          # Agent configuration
├── data/                   # Agent working directory (ONLY writable location)
│   ├── index.json         # File index from indexing pass
│   ├── notes/             # Agent notes for future queries
│   └── cache/             # Compacted context cache
├── server/                 # Model server management
│   ├── backend.py         # Auto-detect platform, unified start/stop
│   ├── mlx_runner.py      # Start/stop MLX server (macOS Apple Silicon)
│   └── llamacpp_runner.py # Start/stop llama.cpp server (Windows/Linux)
├── cli.py                  # CLI entry point
├── requirements.txt
└── .venv/                  # Python virtual environment
```

## Development Rules

- Use structured output schemas for all AI outputs (JSON schema, not prompt instructions)
- Keep individual functions under 50 lines where possible
- All file path operations must go through a `safe_path()` validator
- Log all tool calls to `data/agent.log` with timestamps
- Test with `~/Desktop` as the default target directory
- The 0.8B model is small - keep prompts concise and focused
- Prefer multiple short calls over one long call
- Always validate model output before using it (the model will make mistakes)
