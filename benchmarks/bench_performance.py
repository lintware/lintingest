#!/usr/bin/env python3
"""Benchmark script for LintIngest non-LLM components.
Measures: directory indexing, tool execution, context compaction,
JSON I/O, and module import/startup time.

Does NOT require the model server to be running."""

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(label: str, func, iterations: int = 1):
    """Run func `iterations` times, print min/avg/max."""
    times = []
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mn, avg, mx = min(times), sum(times) / len(times), max(times)
    if iterations == 1:
        print(f"  {label}: {avg*1000:.2f} ms")
    else:
        print(f"  {label}: min={mn*1000:.2f} ms  avg={avg*1000:.2f} ms  max={mx*1000:.2f} ms  ({iterations} runs)")
    return result


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. Import / Startup Time
# ---------------------------------------------------------------------------
def bench_imports():
    section("1. Import & Startup Time")

    # Measure cold import via subprocess
    import subprocess
    script = (
        "import time; t0=time.perf_counter(); "
        "from agent.config import AgentConfig; "
        "from agent.tools import ToolExecutor, TOOL_SCHEMAS; "
        "from agent.compaction import ContextCompactor; "
        "from agent.core import Agent; "
        "t1=time.perf_counter(); "
        "print(f'{(t1-t0)*1000:.2f}')"
    )
    # Run 3 times to get a range
    cold_times = []
    for _ in range(3):
        p = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        if p.returncode == 0:
            cold_times.append(float(p.stdout.strip()))
    if cold_times:
        print(f"  Cold import (subprocess): min={min(cold_times):.2f} ms  avg={sum(cold_times)/len(cold_times):.2f} ms  max={max(cold_times):.2f} ms  (3 runs)")

    # Warm import (already loaded in this process)
    def warm_import():
        # Force re-execution of module-level code paths we care about
        from agent.config import AgentConfig
        from agent.tools import ToolExecutor, TOOL_SCHEMAS
        from agent.compaction import ContextCompactor
        _ = AgentConfig()
    timeit("Warm import + AgentConfig init", warm_import, iterations=10)

    # Agent construction (no server needed)
    from agent.config import AgentConfig
    from agent.core import Agent
    cfg = AgentConfig()
    timeit("Agent() construction", lambda: Agent(cfg), iterations=10)


# ---------------------------------------------------------------------------
# 2. Directory Indexing Speed
# ---------------------------------------------------------------------------
def bench_indexing():
    section("2. Directory Indexing Speed")

    from agent.config import AgentConfig
    from agent.tools import ToolExecutor

    config = AgentConfig()
    # Use a temp data dir to avoid overwriting real index
    tmp_data = Path(tempfile.mkdtemp(prefix="lintingest_bench_"))
    config.data_dir = tmp_data
    (tmp_data / "notes").mkdir(exist_ok=True)
    (tmp_data / "cache").mkdir(exist_ok=True)

    tools = ToolExecutor(config)

    # Benchmark indexing the project directory itself (small, predictable)
    project_dir = str(Path(__file__).parent.parent)
    # Need to allow reading the project dir
    config.target_dir = Path(project_dir)

    def index_project():
        tools.call_count = 0  # Reset rate limit
        return tools.execute("index_directory", {"path": project_dir, "max_depth": 3})

    result = timeit("Index project dir (depth=3)", index_project, iterations=5)
    if result and result.get("success"):
        m = re.search(r"<indexed>(\d+)</indexed>", result["output"])
        count = int(m.group(1)) if m else "?"
        print(f"    -> {count} entries indexed")

    # Benchmark indexing ~/Desktop
    desktop = str(Path.home() / "Desktop")
    if Path(desktop).is_dir():
        config.target_dir = Path(desktop)
        def index_desktop():
            tools.call_count = 0
            return tools.execute("index_directory", {"path": desktop, "max_depth": 5})
        result = timeit("Index ~/Desktop (depth=5)", index_desktop, iterations=3)
        if result and result.get("success"):
            m = re.search(r"<indexed>(\d+)</indexed>", result["output"])
            count = int(m.group(1)) if m else "?"
            print(f"    -> {count} entries indexed")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_data, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Tool Execution Overhead
# ---------------------------------------------------------------------------
def bench_tools():
    section("3. Tool Execution Overhead")

    from agent.config import AgentConfig
    from agent.tools import ToolExecutor

    config = AgentConfig()
    project_dir = Path(__file__).parent.parent
    config.target_dir = project_dir
    tools = ToolExecutor(config)

    # Ensure we have an index to work with
    tools.execute("index_directory", {"path": str(project_dir), "max_depth": 3})

    # file_read
    cli_path = str(project_dir / "cli.py")
    def read_file():
        tools.call_count = 0
        return tools.execute("file_read", {"path": cli_path, "offset": 0, "limit": 50})
    timeit("file_read (cli.py, 50 lines)", read_file, iterations=20)

    # glob_search
    def glob_py():
        tools.call_count = 0
        return tools.execute("glob_search", {"pattern": "*.py", "directory": str(project_dir)})
    result = timeit("glob_search (*.py)", glob_py, iterations=10)

    # content_search
    def search_import():
        tools.call_count = 0
        return tools.execute("content_search", {"pattern": "import", "directory": str(project_dir), "max_results": 20})
    timeit("content_search ('import')", search_import, iterations=5)

    # read_index
    def read_idx():
        tools.call_count = 0
        return tools.execute("read_index", {})
    timeit("read_index (XML)", read_idx, iterations=20)

    # note_store
    def store_note():
        tools.call_count = 0
        return tools.execute("note_store", {
            "key": "bench_test",
            "content": "This is a benchmark test note with some content to write.",
            "category": "benchmark",
        })
    timeit("note_store", store_note, iterations=10)

    # note_recall
    def recall_note():
        tools.call_count = 0
        return tools.execute("note_recall", {"query": "benchmark test", "limit": 5})
    timeit("note_recall", recall_note, iterations=10)

    # safe_path validation overhead
    def safe_path_check():
        tools.safe_read_path(cli_path)
    timeit("safe_read_path validation", safe_path_check, iterations=100)


# ---------------------------------------------------------------------------
# 4. Context Compaction Speed
# ---------------------------------------------------------------------------
def bench_compaction():
    section("4. Context Compaction Speed")

    from agent.compaction import ContextCompactor

    # Simulate a realistic workload: adding results and getting context
    sample_outputs = [
        ("read_index", "<index base=\"/Users/admin/Desktop\">\n" + "\n".join(
            f'<f p="file_{i}.txt" t=".txt" s="1024">Preview of file {i}</f>'
            for i in range(50)
        ) + "\n</index>"),
        ("file_read", "\n".join(f"{i}: line content here with some text" for i in range(50))),
        ("glob_search", "<glob>\n" + "\n".join(f'<m p="/path/to/file_{i}.py"/>' for i in range(30)) + "\n</glob>"),
        ("content_search", "<search>\n" + "\n".join(
            f'<hit f="/path/file_{i}.py" l="{i}">matching line content</hit>'
            for i in range(20)
        ) + "\n</search>"),
        ("file_read", "\n".join(f"{i}: another file content line" for i in range(50))),
    ]

    def run_compaction():
        c = ContextCompactor(max_history=3, max_chars=3000)
        for tool_name, output in sample_outputs:
            c.add_result(tool_name, output)
        ctx = c.get_context()
        return ctx

    result = timeit("Add 5 results + get_context", run_compaction, iterations=100)
    print(f"    -> Context length: {len(result)} chars")

    # Stress test: many results
    def stress_compaction():
        c = ContextCompactor(max_history=3, max_chars=3000)
        for i in range(50):
            c.add_result(f"tool_{i % 5}", f"Result data block {i} " * 50)
        return c.get_context()

    result = timeit("Add 50 results + get_context (stress)", stress_compaction, iterations=50)
    print(f"    -> Context length: {len(result)} chars")


# ---------------------------------------------------------------------------
# 5. JSON Parsing / Serialization
# ---------------------------------------------------------------------------
def bench_json():
    section("5. JSON Parsing & Serialization")

    # Use real index.json if available, else generate synthetic data
    real_index = Path(__file__).parent.parent / "data" / "index.json"
    if real_index.exists():
        raw = real_index.read_text()
        data = json.loads(raw)
        print(f"  Using real index.json: {len(raw)} bytes, {len(data)} entries")
    else:
        # Generate synthetic index
        data = []
        for i in range(200):
            data.append({
                "path": f"/Users/admin/Desktop/folder/subfolder/file_{i}.txt",
                "name": f"file_{i}.txt",
                "type": ".txt",
                "size": 1024 * (i + 1),
                "modified": "2026-03-08T18:29:00",
                "preview": f"This is the preview content of file {i}. " * 3,
            })
        raw = json.dumps(data, indent=2, default=str)
        print(f"  Using synthetic index: {len(raw)} bytes, {len(data)} entries")

    # Parse
    timeit("JSON parse (json.loads)", lambda: json.loads(raw), iterations=100)

    # Serialize
    timeit("JSON serialize (json.dumps, indent=2)", lambda: json.dumps(data, indent=2, default=str), iterations=100)

    # Serialize compact
    timeit("JSON serialize (json.dumps, compact)", lambda: json.dumps(data, default=str), iterations=100)

    # Full round-trip: read file + parse
    if real_index.exists():
        def read_and_parse():
            text = real_index.read_text()
            return json.loads(text)
        timeit("Read file + parse index.json", read_and_parse, iterations=50)

        # Write round-trip
        tmp = Path(tempfile.mktemp(suffix=".json"))
        def write_index():
            tmp.write_text(json.dumps(data, indent=2, default=str))
        timeit("Serialize + write index.json", write_index, iterations=50)
        tmp.unlink(missing_ok=True)

    # XML generation from entries (used by _entries_to_xml)
    from agent.tools import ToolExecutor
    from agent.config import AgentConfig
    cfg = AgentConfig()
    cfg.target_dir = Path(__file__).parent.parent
    te = ToolExecutor(cfg)

    def gen_xml():
        return te._entries_to_xml(data)
    timeit("XML generation (_entries_to_xml)", gen_xml, iterations=50)


# ---------------------------------------------------------------------------
# 6. Misc: _parse_tool_call, _build_tool_descriptions, _build_index_summary
# ---------------------------------------------------------------------------
def bench_misc():
    section("6. Misc Agent Methods")

    from agent.config import AgentConfig
    from agent.core import Agent

    cfg = AgentConfig()
    agent = Agent(cfg)

    # _build_tool_descriptions
    timeit("_build_tool_descriptions", agent._build_tool_descriptions, iterations=100)

    # _parse_tool_call with valid JSON
    valid_response = '{"tool": "file_read", "args": {"path": "/Users/admin/Desktop/test.txt", "offset": 0, "limit": 50}}'
    timeit("_parse_tool_call (valid JSON)", lambda: agent._parse_tool_call(valid_response), iterations=100)

    # _parse_tool_call with embedded JSON in text
    messy_response = 'I think we should read the file. {"tool": "content_search", "args": {"pattern": "hello", "directory": "/Users/admin/Desktop"}} Let me do that.'
    timeit("_parse_tool_call (embedded JSON)", lambda: agent._parse_tool_call(messy_response), iterations=100)

    # _parse_tool_call with no valid call
    no_call = "I don't know the answer. Let me think about this more carefully."
    timeit("_parse_tool_call (no match)", lambda: agent._parse_tool_call(no_call), iterations=100)

    # _build_index_summary
    real_index = Path(__file__).parent.parent / "data" / "index.json"
    if real_index.exists():
        files = json.loads(real_index.read_text())
        timeit("_build_index_summary", lambda: agent._build_index_summary(files), iterations=100)
        summary = agent._build_index_summary(files)
        print(f"    -> Summary length: {len(summary)} chars")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("LintIngest Benchmark - Non-LLM Components")
    print(f"Python {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"CWD: {os.getcwd()}")

    bench_imports()
    bench_indexing()
    bench_tools()
    bench_compaction()
    bench_json()
    bench_misc()

    section("Done")
    print("  All benchmarks complete.\n")


if __name__ == "__main__":
    main()
