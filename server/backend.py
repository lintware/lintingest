"""Backend auto-detection and unified server interface."""

import platform
import subprocess
import sys

from server import mlx_runner, llamacpp_runner


def detect_backend() -> str:
    """Return 'mlx' on Apple Silicon macOS, 'llamacpp' everywhere else."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "llamacpp"


def start_server(
    backend: str | None = None,
    port: int = 8080,
    **kwargs,
) -> subprocess.Popen | None:
    """Start the appropriate model server.

    Args:
        backend: 'mlx', 'llamacpp', or None for auto-detect.
        port: Port to serve on.
    """
    backend = backend or detect_backend()

    if backend == "mlx":
        return mlx_runner.start_server(port=port, **kwargs)
    elif backend == "llamacpp":
        return llamacpp_runner.start_server(port=port, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected 'mlx' or 'llamacpp')")


def stop_server(proc: subprocess.Popen | None):
    """Stop whichever server process is running."""
    if proc:
        proc.terminate()
        proc.wait(timeout=5)
        print("Server stopped")
