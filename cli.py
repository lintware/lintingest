#!/usr/bin/env python3
"""LintIngest CLI - agentic document retrieval."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from agent.config import AgentConfig
from agent.core import Agent
from server.mlx_runner import start_server, stop_server


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).parent / "data" / "agent.log",
                mode="a",
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="LintIngest - Local Document Agent")
    sub = parser.add_subparsers(dest="command")

    # Index command
    idx = sub.add_parser("index", help="Index a directory")
    idx.add_argument("--target", default=str(Path.home() / "Desktop"), help="Directory to index")
    idx.add_argument("--verbose", "-v", action="store_true")

    # Query command
    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", nargs="+", help="The question to ask")
    q.add_argument("--parallel", "-p", action="store_true", help="Use parallel file reading")
    q.add_argument("--verbose", "-v", action="store_true")

    # Interactive mode
    i = sub.add_parser("interactive", help="Interactive Q&A mode")
    i.add_argument("--target", default=str(Path.home() / "Desktop"), help="Directory to index")
    i.add_argument("--verbose", "-v", action="store_true")

    # Server command
    s = sub.add_parser("server", help="Start MLX server only")
    s.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    verbose = getattr(args, "verbose", False)
    setup_logging(verbose)

    config = AgentConfig()
    if hasattr(args, "target"):
        config.target_dir = Path(args.target)

    if args.command == "server":
        proc = start_server(port=args.port, venv_python=sys.executable)
        if proc:
            try:
                print("Press Ctrl+C to stop")
                proc.wait()
            except KeyboardInterrupt:
                stop_server(proc)
        return

    if args.command == "index":
        agent = Agent(config)
        asyncio.run(agent.index(args.target))

    elif args.command == "query":
        question = " ".join(args.question)
        agent = Agent(config)
        if args.parallel:
            asyncio.run(agent.parallel_query(question))
        else:
            asyncio.run(agent.query(question))

    elif args.command == "interactive":
        asyncio.run(interactive_mode(config))


async def interactive_mode(config: AgentConfig):
    agent = Agent(config)

    # Auto-index if no index exists
    index_path = config.data_dir / "index.json"
    if not index_path.exists():
        print("No index found. Running initial indexing...")
        await agent.index()
    else:
        print(f"Using existing index ({index_path})")

    print("\nLintIngest Interactive Mode")
    print("Commands: /index (re-index), /parallel (toggle), /quit")
    print("-" * 40)

    use_parallel = True

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question == "/quit":
            break
        if question == "/index":
            await agent.index()
            continue
        if question == "/parallel":
            use_parallel = not use_parallel
            print(f"Parallel mode: {'on' if use_parallel else 'off'}")
            continue

        if use_parallel:
            await agent.parallel_query(question)
        else:
            await agent.query(question)


if __name__ == "__main__":
    main()
