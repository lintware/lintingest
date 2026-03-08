"""MLX VLM server management - start/stop the local vision-language model server."""

import subprocess
import time
import sys
import httpx


def start_server(
    model: str = "mlx-community/Qwen3.5-0.8B-MLX-8bit",
    port: int = 8080,
    venv_python: str | None = None,
) -> subprocess.Popen | None:
    """Start the MLX VLM server if not already running."""
    # Check if already running
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
        if resp.status_code == 200:
            print(f"MLX VLM server already running on port {port}")
            return None
    except Exception:
        pass

    python_cmd = venv_python or sys.executable
    cmd = [python_cmd, "-m", "mlx_vlm.server", "--port", str(port)]
    print(f"Starting MLX VLM server: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
            if resp.status_code == 200:
                print(f"MLX VLM server ready on port {port}")
                _warmup_model(port, model)
                return proc
        except Exception:
            pass

    print("WARNING: MLX VLM server may not be ready")
    return proc


def _warmup_model(port: int, model: str):
    """Send a short request to trigger model loading before real queries."""
    print(f"Loading model {model}...")
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
            timeout=120.0,
        )
        if resp.status_code == 200:
            print("Model loaded and ready")
        else:
            print(f"Warmup got status {resp.status_code} (model may still be loading)")
    except Exception as e:
        print(f"Warmup request failed: {e} (model may still be loading)")


def stop_server(proc: subprocess.Popen | None):
    """Stop the MLX VLM server."""
    if proc:
        proc.terminate()
        proc.wait(timeout=5)
        print("MLX VLM server stopped")
