"""MLX server management - start/stop the local model server."""

import subprocess
import time
import sys
import httpx


def start_server(
    model: str = "mlx-community/Qwen3.5-0.8B-MLX-8bit",
    port: int = 8080,
    venv_python: str | None = None,
) -> subprocess.Popen | None:
    """Start the MLX server if not already running."""
    # Check if already running
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
        if resp.status_code == 200:
            print(f"MLX server already running on port {port}")
            return None
    except Exception:
        pass

    python_cmd = venv_python or sys.executable
    cmd = [python_cmd, "-m", "mlx_lm.server", "--model", model, "--port", str(port)]
    print(f"Starting MLX server: {' '.join(cmd)}")

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
                print(f"MLX server ready on port {port}")
                return proc
        except Exception:
            pass

    print("WARNING: MLX server may not be ready")
    return proc


def stop_server(proc: subprocess.Popen | None):
    """Stop the MLX server."""
    if proc:
        proc.terminate()
        proc.wait(timeout=5)
        print("MLX server stopped")
