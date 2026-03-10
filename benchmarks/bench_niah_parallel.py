#!/usr/bin/env python3
"""NIAH benchmark using parallel LLM extraction.

Instead of a sequential tool-calling loop, this variant:
1. Greps for keywords (no LLM)
2. Reads top matching files (no LLM)
3. Sends files to 4 parallel LLM workers for extraction
4. Synthesizes a final answer from the findings

Requires the model server to be running (MLX or llama.cpp).

Usage:
    python -m benchmarks.bench_niah_parallel [--sizes small,medium,large]
"""

import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import AgentConfig
from agent.core import Agent
from benchmarks.bench_niah import (
    NEEDLES,
    HAYSTACK_SIZES,
    DEPTH_POSITIONS,
    BenchmarkResults,
    TrialResult,
    build_haystack,
    check_answer,
    print_summary,
    save_results,
)


async def run_niah_parallel_benchmark(
    sizes: list[str] | None = None,
    depths: list[str] | None = None,
    needle_ids: list[str] | None = None,
    trials_per: int = 1,
) -> BenchmarkResults:
    """Run the NIAH benchmark using parallel_niah_query."""
    sizes = sizes or ["small", "medium"]
    depths = depths or list(DEPTH_POSITIONS.keys())
    needles = NEEDLES
    if needle_ids:
        needles = [n for n in NEEDLES if n["id"] in needle_ids]

    results = BenchmarkResults(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="niah_bench_"))

    try:
        total_runs = len(sizes) * len(depths) * len(needles) * trials_per
        run_num = 0

        for size_name in sizes:
            haystack_cfg = HAYSTACK_SIZES[size_name]

            for depth_name in depths:
                for needle in needles:
                    for trial in range(trials_per):
                        run_num += 1
                        label = (
                            f"[{run_num}/{total_runs}] "
                            f"size={size_name} depth={depth_name} "
                            f"needle={needle['id']}"
                        )
                        if trials_per > 1:
                            label += f" trial={trial+1}"
                        print(f"\n{'='*60}")
                        print(f"  {label}")
                        print(f"{'='*60}")

                        hay_dir = build_haystack(
                            tmpdir, haystack_cfg, needle, depth_name,
                        )

                        config = AgentConfig()
                        config.target_dir = hay_dir
                        config.skip_summaries = True
                        bench_data = tmpdir / f"data_{size_name}_{depth_name}_{needle['id']}_{trial}"
                        config.data_dir = bench_data
                        bench_data.mkdir(parents=True, exist_ok=True)
                        (bench_data / "notes").mkdir(exist_ok=True)
                        (bench_data / "cache").mkdir(exist_ok=True)

                        agent = Agent(config)
                        await agent.index(str(hay_dir))

                        answer = ""
                        latency = 0.0
                        for attempt in range(3):
                            try:
                                if attempt > 0 or run_num > 1:
                                    await asyncio.sleep(2)
                                t0 = time.perf_counter()
                                answer = await agent.parallel_niah_query(needle["question"])
                                latency = time.perf_counter() - t0
                                break
                            except Exception as e:
                                print(f"  Attempt {attempt+1} failed: {e}")
                                if attempt == 2:
                                    answer = f"ERROR: {e}"
                                    latency = 0.0
                                await asyncio.sleep(5)

                        found = check_answer(answer, needle["expected"])
                        tool_calls = agent.tools.call_count

                        status = "PASS" if found else "FAIL"
                        print(f"\n  Result: {status}")
                        print(f"  Answer: {answer[:200]}")
                        print(f"  Expected: {needle['expected']}")
                        print(f"  Latency: {latency:.1f}s | Tools: {tool_calls}")

                        results.trials.append(TrialResult(
                            needle_id=needle["id"],
                            haystack_size=size_name,
                            depth=depth_name,
                            found=found,
                            answer=answer[:500],
                            expected=needle["expected"],
                            latency_s=round(latency, 2),
                            tool_calls=tool_calls,
                        ))

                        shutil.rmtree(hay_dir, ignore_errors=True)
                        shutil.rmtree(bench_data, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NIAH Parallel Benchmark for LintIngest")
    parser.add_argument(
        "--sizes", default="small,medium",
        help="Comma-separated haystack sizes: small,medium,large (default: small,medium)",
    )
    parser.add_argument(
        "--depths", default=None,
        help="Comma-separated depths: start,early,middle,late,end (default: all)",
    )
    parser.add_argument(
        "--needles", default=None,
        help="Comma-separated needle IDs to test (default: all)",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials per configuration (default: 1)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    depths = [d.strip() for d in args.depths.split(",")] if args.depths else None
    needle_ids = [n.strip() for n in args.needles.split(",")] if args.needles else None

    print("LintIngest NIAH Benchmark (Parallel)")
    print(f"  Sizes: {sizes}")
    print(f"  Depths: {depths or 'all'}")
    print(f"  Needles: {needle_ids or 'all'}")
    print(f"  Trials per config: {args.trials}")

    results = asyncio.run(run_niah_parallel_benchmark(
        sizes=sizes,
        depths=depths,
        needle_ids=needle_ids,
        trials_per=args.trials,
    ))

    print_summary(results)

    output = Path(args.output) if args.output else Path(__file__).parent / "results" / "niah_results_parallel.json"
    save_results(results, output)


if __name__ == "__main__":
    main()
