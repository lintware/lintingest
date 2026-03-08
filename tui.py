#!/usr/bin/env python3
"""LintIngest TUI - interactive terminal interface using prompt_toolkit."""

import asyncio
import sys
from pathlib import Path

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML

from agent.config import AgentConfig
from agent.core import Agent
from server.backend import start_server, stop_server


def make_toolbar(target_dir: Path, parallel: bool, backend: str | None):
    def toolbar():
        parts = [
            f"<b>Target:</b> {target_dir}",
            f"<b>Parallel:</b> {'on' if parallel else 'off'}",
            f"<b>Backend:</b> {backend or 'auto'}",
        ]
        return HTML("  |  ".join(parts))
    return toolbar


def _ensure_server(backend: str | None) -> "subprocess.Popen | None":
    """Start the model server if it isn't already running. Returns the process (or None if it was already up)."""
    try:
        resp = httpx.get("http://127.0.0.1:8080/v1/models", timeout=2.0)
        if resp.status_code == 200:
            print("Model server already running.")
            return None
    except Exception:
        pass

    print("Starting model server...")
    proc = start_server(backend=backend, port=8080, venv_python=sys.executable)
    return proc


async def tui_main(
    target: str | None = None,
    backend: str | None = None,
):
    target_dir = Path(target or str(Path.home() / "Desktop")).expanduser().resolve()

    # Auto-start model server
    server_proc = _ensure_server(backend)

    config = AgentConfig()
    config.target_dir = target_dir
    agent = Agent(config)
    use_parallel = True

    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
    )

    print("LintIngest TUI")
    print("Commands: /target <path>, /index, /parallel, /quit")
    print("-" * 50)

    # Auto-index if needed
    index_path = config.data_dir / "index.json"
    if not index_path.exists():
        print(f"No index found. Indexing {target_dir}...")
        await agent.index()

    try:
        while True:
            try:
                toolbar = make_toolbar(config.target_dir, use_parallel, backend)
                text = await session.prompt_async(
                    "lintingest> ",
                    bottom_toolbar=toolbar,
                )
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            text = text.strip()
            if not text:
                continue

            if text == "/quit":
                print("Bye!")
                break

            if text == "/index":
                print(f"Re-indexing {config.target_dir}...")
                await agent.index()
                continue

            if text == "/parallel":
                use_parallel = not use_parallel
                print(f"Parallel mode: {'on' if use_parallel else 'off'}")
                continue

            if text.startswith("/target"):
                parts = text.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"Current target: {config.target_dir}")
                    print("Usage: /target <path>")
                    continue
                new_target = Path(parts[1]).expanduser().resolve()
                if not new_target.is_dir():
                    print(f"Not a directory: {new_target}")
                    continue
                config.target_dir = new_target
                agent = Agent(config)
                print(f"Target set to {new_target}. Indexing...")
                await agent.index()
                continue

            # It's a query
            try:
                if use_parallel:
                    await agent.parallel_query(text)
                else:
                    await agent.query(text)
            except httpx.ConnectError:
                print(
                    "\nError: Cannot connect to model server at "
                    f"{config.base_url}\n"
                    "The server may have crashed. Restart with: "
                    ".venv/bin/python cli.py tui"
                )
            except (RuntimeError, httpx.HTTPStatusError) as e:
                print(f"\nError: {e}")
            except Exception as e:
                print(f"\nUnexpected error: {type(e).__name__}: {e}")
    finally:
        # Stop the server if we started it
        if server_proc:
            print("Stopping model server...")
            stop_server(server_proc)


def run_tui(target: str | None = None, backend: str | None = None):
    asyncio.run(tui_main(target=target, backend=backend))
