"""llama.cpp server management - start/stop the local model server."""

import shutil
import subprocess
import time
import sys
import httpx


GGUF_REPO = "unsloth/Qwen3.5-0.8B-GGUF"
GGUF_FILENAME = "Qwen3.5-0.8B-Q4_K_M.gguf"


def _find_llama_server() -> str | None:
    """Locate the llama-server binary on PATH."""
    return shutil.which("llama-server")


def _ensure_model() -> str:
    """Download the GGUF model if not cached and return its path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)
    return path


def start_server(
    port: int = 8080,
    **kwargs,
) -> subprocess.Popen | None:
    """Start the llama.cpp server if not already running."""
    # Check if already running
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
        if resp.status_code == 200:
            print(f"llama.cpp server already running on port {port}")
            return None
    except Exception:
        pass

    binary = _find_llama_server()
    if not binary:
        print(
            "ERROR: llama-server not found on PATH.\n"
            "Install llama.cpp: https://github.com/ggerganov/llama.cpp#build",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = _ensure_model()
    cmd = [
        binary,
        "--model", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--ctx-size", "2048",
        "--parallel", "4",
        "--jinja",
        "--chat-template", "chatml",
    ]
    print(f"Starting llama.cpp server: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(60):
        time.sleep(1)
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
            if resp.status_code == 200:
                print(f"llama.cpp server ready on port {port}")
                return proc
        except Exception:
            pass

    print("WARNING: llama.cpp server may not be ready")
    return proc


def stop_server(proc: subprocess.Popen | None):
    """Stop the llama.cpp server."""
    if proc:
        proc.terminate()
        proc.wait(timeout=5)
        print("llama.cpp server stopped")
