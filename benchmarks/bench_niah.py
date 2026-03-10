#!/usr/bin/env python3
"""Needle In A Haystack (NIAH) benchmark for LintIngest.

Tests the agent's ability to find a specific fact ("needle") hidden among
many distractor files ("haystack") in an indexed directory.

Inspired by gkamradt/LLMTest_NeedleInAHaystack.

Requires the model server to be running (MLX or llama.cpp).

Usage:
    python -m benchmarks.bench_niah [--sizes small,medium,large] [--trials 3]
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import AgentConfig
from agent.core import Agent


# ---------------------------------------------------------------------------
# Needle / Haystack definitions
# ---------------------------------------------------------------------------

NEEDLES = [
    {
        "id": "capital_zephyria",
        "fact": "The capital city of the fictional country Zephyria is Luminara.",
        "question": "What is the capital of Zephyria?",
        "expected": ["luminara"],
    },
    {
        "id": "project_aurora_budget",
        "fact": "Project Aurora was allocated exactly $4,271,038 in Q3 funding.",
        "question": "How much funding was allocated to Project Aurora in Q3?",
        "expected": ["4,271,038", "4271038"],
    },
    {
        "id": "server_password",
        "fact": "The emergency override code for server room B7 is: DELTA-9-FOXTROT-22.",
        "question": "What is the emergency override code for server room B7?",
        "expected": ["delta-9-foxtrot-22"],
    },
    {
        "id": "recipe_secret",
        "fact": "Chef Montague's secret ingredient in the award-winning bisque is smoked paprika and star anise.",
        "question": "What is Chef Montague's secret ingredient in the bisque?",
        "expected": ["smoked paprika", "star anise"],
    },
]

# Haystack filler paragraphs — generic text that buries the needle
FILLER_PARAGRAPHS = [
    "Market analysis indicates a steady growth trajectory in the consumer electronics sector, "
    "driven primarily by increased demand for wearable technology and smart home devices. "
    "Industry analysts project continued expansion throughout the fiscal year.",
    "The quarterly review highlighted improvements in operational efficiency across all "
    "departments. Supply chain optimizations reduced average delivery times by 12%, while "
    "inventory management systems reported a 7% decrease in carrying costs.",
    "Recent developments in renewable energy infrastructure suggest a transition toward "
    "decentralized power generation. Solar panel installations increased by 34% year over year, "
    "while battery storage capacity doubled in several key markets.",
    "Human resources reported a 15% increase in employee satisfaction scores following the "
    "implementation of flexible work arrangements. Retention rates improved across technical "
    "and creative teams, with voluntary turnover dropping to its lowest level in five years.",
    "The engineering team completed the migration to a microservices architecture, resulting "
    "in improved system reliability and faster deployment cycles. Average response times "
    "decreased by 40%, and uptime reached 99.97% for the quarter.",
    "Customer feedback surveys indicated high satisfaction with recent product updates. "
    "The new user interface received positive reception, with 89% of respondents rating "
    "the experience as improved compared to the previous version.",
    "Financial projections for the upcoming quarter remain optimistic, with revenue targets "
    "set 8% above the current period. Cost reduction initiatives in procurement and logistics "
    "are expected to contribute an additional 3% margin improvement.",
    "The research division published findings on advanced materials for thermal management "
    "applications. Preliminary results suggest a novel composite material could improve heat "
    "dissipation by up to 60% compared to conventional aluminum heat sinks.",
    "Regional sales data shows strong performance in the Asia-Pacific market, with a 22% "
    "increase in unit sales. European markets remain stable, while North American growth "
    "was driven primarily by enterprise-level contracts.",
    "The security audit identified no critical vulnerabilities in the production environment. "
    "Minor recommendations included updating TLS configurations and implementing additional "
    "logging for administrative access events.",
    "Training programs for new hires were expanded to include hands-on workshops and "
    "mentorship pairings. Feedback from recent cohorts indicates faster onboarding times "
    "and improved confidence in role-specific tasks.",
    "Warehouse automation projects reached the pilot phase in two distribution centers. "
    "Robotic picking systems demonstrated a 3x throughput improvement during initial "
    "testing, with full rollout planned for the next quarter.",
]

# Haystack file topics for generating distinct files
FILE_TOPICS = [
    ("quarterly_report_q1.txt", "Q1 Quarterly Business Report"),
    ("quarterly_report_q2.txt", "Q2 Quarterly Business Report"),
    ("marketing_strategy.txt", "Marketing Strategy Document"),
    ("product_roadmap.txt", "Product Development Roadmap"),
    ("budget_overview.txt", "Annual Budget Overview"),
    ("team_updates.txt", "Team Status Updates"),
    ("client_meeting_notes.txt", "Client Meeting Notes"),
    ("technical_specs.txt", "Technical Specifications"),
    ("hr_policy_update.txt", "HR Policy Updates"),
    ("vendor_contracts.txt", "Vendor Contract Summary"),
    ("sales_forecast.txt", "Sales Forecast Analysis"),
    ("research_notes.txt", "Research Division Notes"),
    ("operations_log.txt", "Operations Log"),
    ("compliance_report.txt", "Compliance Review"),
    ("training_materials.txt", "Employee Training Guide"),
    ("infrastructure_plan.txt", "Infrastructure Planning"),
    ("customer_feedback.txt", "Customer Feedback Summary"),
    ("project_timeline.txt", "Project Timeline Overview"),
    ("risk_assessment.txt", "Risk Assessment Report"),
    ("board_minutes.txt", "Board Meeting Minutes"),
    ("innovation_lab.txt", "Innovation Lab Updates"),
    ("sustainability_report.txt", "Sustainability Initiatives"),
    ("supply_chain_review.txt", "Supply Chain Review"),
    ("legal_summary.txt", "Legal Department Summary"),
    ("data_analytics.txt", "Data Analytics Report"),
    ("partnership_proposals.txt", "Partnership Proposals"),
    ("quality_assurance.txt", "QA Testing Results"),
    ("incident_report.txt", "Incident Response Report"),
    ("capacity_planning.txt", "Capacity Planning Document"),
    ("process_improvement.txt", "Process Improvement Plans"),
]


# ---------------------------------------------------------------------------
# Haystack sizes
# ---------------------------------------------------------------------------

@dataclass
class HaystackConfig:
    name: str
    num_files: int
    paragraphs_per_file: int  # filler paragraphs per file


HAYSTACK_SIZES = {
    "small": HaystackConfig(name="small", num_files=5, paragraphs_per_file=3),
    "medium": HaystackConfig(name="medium", num_files=15, paragraphs_per_file=6),
    "large": HaystackConfig(name="large", num_files=30, paragraphs_per_file=10),
}


# ---------------------------------------------------------------------------
# Needle depth positions
# ---------------------------------------------------------------------------

DEPTH_POSITIONS = {
    "start": 0.0,    # needle in the first file
    "early": 0.25,   # 25% through
    "middle": 0.5,   # 50% through
    "late": 0.75,    # 75% through
    "end": 1.0,      # last file
}


# ---------------------------------------------------------------------------
# Benchmark result schema
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    needle_id: str
    haystack_size: str
    depth: str
    found: bool
    answer: str
    expected: list[str]
    latency_s: float
    tool_calls: int


@dataclass
class BenchmarkResults:
    timestamp: str = ""
    trials: list[TrialResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.trials:
            return {"total": 0, "accuracy": 0.0}
        total = len(self.trials)
        correct = sum(1 for t in self.trials if t.found)
        avg_latency = sum(t.latency_s for t in self.trials) / total

        # Breakdown by size
        by_size = {}
        for t in self.trials:
            by_size.setdefault(t.haystack_size, []).append(t)

        size_acc = {}
        for size, trials in by_size.items():
            size_acc[size] = sum(1 for t in trials if t.found) / len(trials)

        # Breakdown by depth
        by_depth = {}
        for t in self.trials:
            by_depth.setdefault(t.depth, []).append(t)

        depth_acc = {}
        for depth, trials in by_depth.items():
            depth_acc[depth] = sum(1 for t in trials if t.found) / len(trials)

        return {
            "total_trials": total,
            "accuracy": correct / total,
            "correct": correct,
            "avg_latency_s": round(avg_latency, 2),
            "by_size": {k: round(v, 3) for k, v in size_acc.items()},
            "by_depth": {k: round(v, 3) for k, v in depth_acc.items()},
        }


# ---------------------------------------------------------------------------
# Haystack builder
# ---------------------------------------------------------------------------

def build_haystack(
    tmpdir: Path,
    haystack_cfg: HaystackConfig,
    needle: dict,
    depth: str,
) -> Path:
    """Create a temp directory with haystack files and a hidden needle."""
    hay_dir = tmpdir / f"haystack_{haystack_cfg.name}_{depth}"
    hay_dir.mkdir(parents=True, exist_ok=True)

    num_files = haystack_cfg.num_files
    topics = FILE_TOPICS[:num_files]

    # Determine which file gets the needle
    depth_frac = DEPTH_POSITIONS[depth]
    needle_file_idx = min(int(depth_frac * (num_files - 1)), num_files - 1)

    # Determine where within that file the needle goes
    paragraphs_per = haystack_cfg.paragraphs_per_file
    needle_para_idx = paragraphs_per // 2  # middle of the file

    for i, (filename, title) in enumerate(topics):
        lines = [f"# {title}\n"]

        for j in range(paragraphs_per):
            # Insert needle at the right spot
            if i == needle_file_idx and j == needle_para_idx:
                lines.append(needle["fact"] + "\n")

            # Add filler paragraph (cycle through available fillers)
            filler_idx = (i * paragraphs_per + j) % len(FILLER_PARAGRAPHS)
            lines.append(FILLER_PARAGRAPHS[filler_idx] + "\n")

        (hay_dir / filename).write_text("\n".join(lines))

    return hay_dir


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_answer(answer: str, expected: list[str]) -> bool:
    """Check if the answer contains any of the expected substrings."""
    answer_lower = answer.lower()
    return any(exp.lower() in answer_lower for exp in expected)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_niah_benchmark(
    sizes: list[str] | None = None,
    depths: list[str] | None = None,
    needle_ids: list[str] | None = None,
    trials_per: int = 1,
) -> BenchmarkResults:
    """Run the NIAH benchmark suite."""
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

                        # Build haystack
                        hay_dir = build_haystack(
                            tmpdir, haystack_cfg, needle, depth_name,
                        )

                        # Configure agent to use this haystack
                        config = AgentConfig()
                        config.target_dir = hay_dir
                        config.skip_summaries = True
                        bench_data = tmpdir / f"data_{size_name}_{depth_name}_{needle['id']}_{trial}"
                        config.data_dir = bench_data
                        bench_data.mkdir(parents=True, exist_ok=True)
                        (bench_data / "notes").mkdir(exist_ok=True)
                        (bench_data / "cache").mkdir(exist_ok=True)

                        agent = Agent(config)

                        # Index the haystack
                        await agent.index(str(hay_dir))

                        # Query for the needle (with retry on timeout)
                        answer = ""
                        latency = 0.0
                        for attempt in range(3):
                            try:
                                # Brief pause between runs to let server recover
                                if attempt > 0 or run_num > 1:
                                    await asyncio.sleep(2)
                                t0 = time.perf_counter()
                                answer = await agent.query(needle["question"])
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

                        # Cleanup this haystack
                        shutil.rmtree(hay_dir, ignore_errors=True)
                        shutil.rmtree(bench_data, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def print_summary(results: BenchmarkResults):
    """Print a formatted summary of benchmark results."""
    summary = results.summary()

    print(f"\n{'='*60}")
    print("  NIAH Benchmark Results")
    print(f"{'='*60}")
    print(f"  Timestamp: {results.timestamp}")
    print(f"  Total trials: {summary['total_trials']}")
    print(f"  Overall accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total_trials']})")
    print(f"  Avg latency: {summary['avg_latency_s']}s")

    if summary.get("by_size"):
        print(f"\n  Accuracy by haystack size:")
        for size, acc in summary["by_size"].items():
            print(f"    {size:>8s}: {acc:.1%}")

    if summary.get("by_depth"):
        print(f"\n  Accuracy by needle depth:")
        for depth, acc in summary["by_depth"].items():
            print(f"    {depth:>8s}: {acc:.1%}")

    # Per-trial details
    print(f"\n  {'Needle':<25s} {'Size':<8s} {'Depth':<8s} {'Found':<6s} {'Latency':<8s}")
    print(f"  {'-'*55}")
    for t in results.trials:
        status = "YES" if t.found else "NO"
        print(f"  {t.needle_id:<25s} {t.haystack_size:<8s} {t.depth:<8s} {status:<6s} {t.latency_s:>5.1f}s")


def save_results(results: BenchmarkResults, output_path: Path):
    """Save results to JSON."""
    data = {
        "timestamp": results.timestamp,
        "summary": results.summary(),
        "trials": [
            {
                "needle_id": t.needle_id,
                "haystack_size": t.haystack_size,
                "depth": t.depth,
                "found": t.found,
                "answer": t.answer,
                "expected": t.expected,
                "latency_s": t.latency_s,
                "tool_calls": t.tool_calls,
            }
            for t in results.trials
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NIAH Benchmark for LintIngest")
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
        help="Path to save JSON results (default: benchmarks/niah_results.json)",
    )
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    depths = [d.strip() for d in args.depths.split(",")] if args.depths else None
    needle_ids = [n.strip() for n in args.needles.split(",")] if args.needles else None

    print("LintIngest NIAH Benchmark")
    print(f"  Sizes: {sizes}")
    print(f"  Depths: {depths or 'all'}")
    print(f"  Needles: {needle_ids or 'all'}")
    print(f"  Trials per config: {args.trials}")

    results = asyncio.run(run_niah_benchmark(
        sizes=sizes,
        depths=depths,
        needle_ids=needle_ids,
        trials_per=args.trials,
    ))

    print_summary(results)

    output = Path(args.output) if args.output else Path(__file__).parent / "results" / "niah_results.json"
    save_results(results, output)


if __name__ == "__main__":
    main()
