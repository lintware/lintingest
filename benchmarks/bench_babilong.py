#!/usr/bin/env python3
"""BABILong benchmark for LintIngest.

Tests the agent's ability to find and reason over facts scattered across
many files in a directory, using the BABILong dataset (bAbI facts embedded
in PG19 distractor text).

Adapted from: https://github.com/booydar/babilong
Dataset: https://huggingface.co/datasets/RMT-team/babilong

Requires:
  - The model server to be running (MLX or llama.cpp)
  - `datasets` package (`pip install datasets`)

Usage:
    python -m benchmarks.bench_babilong [--tasks qa1,qa2] [--lengths 0k,1k,4k]
    python -m benchmarks.bench_babilong --lengths 3M,5M,10M --max-samples 3  # massive scale
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import AgentConfig
from agent.core import Agent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AVAILABLE_TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5",
                   "qa6", "qa7", "qa8", "qa9", "qa10"]

AVAILABLE_LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k",
                     "64k", "128k", "512k", "1M",
                     # Synthetic (self-generated, not from HF dataset):
                     "3M", "5M", "10M"]

TASK_DESCRIPTIONS = {
    "qa1": "Single supporting fact",
    "qa2": "Two supporting facts",
    "qa3": "Three supporting facts",
    "qa4": "Two arg relations",
    "qa5": "Three arg relations",
    "qa6": "Yes/no questions",
    "qa7": "Counting",
    "qa8": "Lists/sets",
    "qa9": "Simple negation",
    "qa10": "Indefinite knowledge",
}

# How many files to split the context into, by context length
FILES_PER_LENGTH = {
    "0k": 1,
    "1k": 3,
    "2k": 5,
    "4k": 10,
    "8k": 20,
    "16k": 40,
    "32k": 80,
    "64k": 150,
    "128k": 300,
    "512k": 800,
    "1M": 1500,
    "3M": 4000,
    "5M": 6000,
    "10M": 10000,
}

# Nested folder depth by context length (simulates real directory structures)
NESTING_DEPTH = {
    "0k": 0,
    "1k": 0,
    "2k": 1,
    "4k": 1,
    "8k": 2,
    "16k": 2,
    "32k": 3,
    "64k": 3,
    "128k": 4,
    "512k": 4,
    "1M": 5,
    "3M": 5,
    "5M": 6,
    "10M": 6,
}

# Synthetic lengths that we generate ourselves (not from HF dataset)
SYNTHETIC_LENGTHS = {"3M", "5M", "10M"}

# Folder names used to create nested structures
FOLDER_NAMES = [
    "reports", "data", "documents", "archive", "notes",
    "research", "logs", "drafts", "records", "memos",
    "analysis", "reviews", "summaries", "exports", "imports",
]


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    task: str
    context_length: str
    sample_idx: int
    found: bool
    answer: str
    expected: str
    latency_s: float
    tool_calls: int
    num_files: int
    input_chars: int


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

        by_task = {}
        for t in self.trials:
            by_task.setdefault(t.task, []).append(t)
        task_acc = {
            task: sum(1 for t in trials if t.found) / len(trials)
            for task, trials in by_task.items()
        }

        by_length = {}
        for t in self.trials:
            by_length.setdefault(t.context_length, []).append(t)
        length_acc = {
            length: sum(1 for t in trials if t.found) / len(trials)
            for length, trials in by_length.items()
        }

        return {
            "total_trials": total,
            "accuracy": correct / total,
            "correct": correct,
            "avg_latency_s": round(avg_latency, 2),
            "by_task": {k: round(v, 3) for k, v in task_acc.items()},
            "by_length": {k: round(v, 3) for k, v in length_acc.items()},
        }


# ---------------------------------------------------------------------------
# bAbI fact templates (for synthetic generation at 3M/5M/10M scale)
# ---------------------------------------------------------------------------

BABI_FACT_TEMPLATES = {
    "qa1": {
        "facts": [
            ("{name} went to the {place}.",),
        ],
        "question": "Where is {name}?",
        "names": ["Mary", "John", "Sandra", "Daniel", "Fred"],
        "places": ["bathroom", "kitchen", "garden", "office", "bedroom",
                    "hallway", "park", "school"],
    },
    "qa2": {
        "facts": [
            ("{name} picked up the {object}.",
             "{name} went to the {place}."),
        ],
        "question": "Where is the {object}?",
        "names": ["Mary", "John", "Sandra", "Daniel"],
        "objects": ["football", "apple", "milk", "book"],
        "places": ["bathroom", "kitchen", "garden", "office", "bedroom"],
    },
    "qa3": {
        "facts": [
            ("{name} went to the {place1}.",
             "{name} picked up the {object}.",
             "{name} went to the {place2}."),
        ],
        "question": "Where is the {object}?",
        "names": ["Mary", "John", "Sandra", "Daniel"],
        "objects": ["football", "apple", "milk"],
        "places1": ["kitchen", "garden", "office"],
        "places2": ["bathroom", "bedroom", "hallway"],
    },
    "qa6": {
        "facts": [
            ("{name} is in the {place}.",),
        ],
        "question": "Is {name} in the {place}?",
        "names": ["Mary", "John", "Sandra", "Daniel"],
        "places": ["bathroom", "kitchen", "garden", "office"],
    },
}


def _generate_babi_sample(task: str, rng) -> dict:
    """Generate a single bAbI-style fact + question + answer."""
    import random

    if task == "qa1":
        tpl = BABI_FACT_TEMPLATES["qa1"]
        name = rng.choice(tpl["names"])
        place = rng.choice(tpl["places"])
        fact = f"{name} went to the {place}."
        question = f"Where is {name}?"
        target = place
        return {"fact_sentences": [fact], "question": question, "target": target}

    elif task == "qa2":
        tpl = BABI_FACT_TEMPLATES["qa2"]
        name = rng.choice(tpl["names"])
        obj = rng.choice(tpl["objects"])
        place = rng.choice(tpl["places"])
        facts = [f"{name} picked up the {obj}.", f"{name} went to the {place}."]
        question = f"Where is the {obj}?"
        return {"fact_sentences": facts, "question": question, "target": place}

    elif task == "qa3":
        tpl = BABI_FACT_TEMPLATES["qa3"]
        name = rng.choice(tpl["names"])
        obj = rng.choice(tpl["objects"])
        place1 = rng.choice(tpl["places1"])
        place2 = rng.choice(tpl["places2"])
        facts = [
            f"{name} went to the {place1}.",
            f"{name} picked up the {obj}.",
            f"{name} went to the {place2}.",
        ]
        question = f"Where is the {obj}?"
        return {"fact_sentences": facts, "question": question, "target": place2}

    elif task == "qa6":
        tpl = BABI_FACT_TEMPLATES["qa6"]
        name = rng.choice(tpl["names"])
        place = rng.choice(tpl["places"])
        fact = f"{name} is in the {place}."
        # 50% yes, 50% no (ask about wrong place)
        if rng.random() < 0.5:
            question = f"Is {name} in the {place}?"
            target = "yes"
        else:
            wrong = rng.choice([p for p in tpl["places"] if p != place])
            question = f"Is {name} in the {wrong}?"
            target = "no"
        return {"fact_sentences": [fact], "question": question, "target": target}

    else:
        # Fallback to qa1 pattern for unsupported tasks in synthetic mode
        return _generate_babi_sample("qa1", rng)


# ---------------------------------------------------------------------------
# PG19 filler text for synthetic large-scale generation
# ---------------------------------------------------------------------------

_PG19_CACHE: list[str] | None = None


def _load_pg19_filler(target_chars: int) -> str:
    """Load PG19 book text from HuggingFace as filler material.

    Caches the result so we only download once per session.
    Returns a string of at least `target_chars` characters.
    """
    global _PG19_CACHE
    from datasets import load_dataset

    if _PG19_CACHE is None:
        print("  Loading PG19 corpus for filler text (first time only)...")
        # Load a subset of PG19 test split — enough for 10M+ chars
        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        texts = []
        total = 0
        # Collect up to 15M chars (enough for the largest benchmark)
        for row in ds:
            text = row["text"]
            texts.append(text)
            total += len(text)
            if total >= 15_000_000:
                break
        _PG19_CACHE = texts
        print(f"  PG19 loaded: {len(texts)} books, {total:,} chars total")

    # Concatenate enough text to reach target
    result = []
    total = 0
    idx = 0
    while total < target_chars:
        book = _PG19_CACHE[idx % len(_PG19_CACHE)]
        result.append(book)
        total += len(book)
        idx += 1
    return "\n\n".join(result)[:target_chars]


def _generate_synthetic_sample(
    task: str,
    target_chars: int,
    rng,
) -> dict:
    """Generate a synthetic BABILong-style sample at arbitrary scale.

    Creates bAbI facts and injects them into PG19 filler text at
    random positions, then returns the combined text + question + answer.
    """
    babi = _generate_babi_sample(task, rng)
    filler = _load_pg19_filler(target_chars)

    facts = babi["fact_sentences"]

    # Split filler into sentences (rough split on ". ")
    filler_sentences = filler.split(". ")

    # Insert facts at spread-out positions (like the real BABILong dataset)
    total_sentences = len(filler_sentences)
    if total_sentences < len(facts) + 1:
        # Filler too short — just prepend facts
        combined = " ".join(facts) + " " + filler
    else:
        # Spread facts across the middle 80% of the text
        start_pct = 0.1
        end_pct = 0.9
        positions = []
        for i, fact in enumerate(facts):
            pct = start_pct + (end_pct - start_pct) * (i / max(len(facts) - 1, 1))
            pos = int(pct * total_sentences)
            positions.append(pos)

        # Insert facts (in reverse to preserve indices)
        for pos, fact in sorted(zip(positions, facts), reverse=True):
            filler_sentences.insert(pos, fact)

        combined = ". ".join(filler_sentences)

    return {
        "input": combined[:target_chars],
        "question": babi["question"],
        "target": babi["target"],
    }


# ---------------------------------------------------------------------------
# Length string to character count
# ---------------------------------------------------------------------------

def _length_to_chars(length: str) -> int:
    """Convert a length string like '4k' or '3M' to approximate char count.

    BABILong uses token counts; we approximate 1 token ~ 4 chars.
    """
    length = length.strip()
    if length.endswith("M"):
        return int(length[:-1]) * 1_000_000 * 4  # tokens -> chars
    elif length.endswith("k"):
        return int(length[:-1]) * 1_000 * 4
    return int(length) * 4


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_babilong_samples(
    task: str,
    context_length: str,
    max_samples: int = 10,
) -> list[dict]:
    """Load samples from the BABILong HuggingFace dataset, or generate
    synthetic samples for very large context lengths (3M, 5M, 10M).

    Returns list of {"input": str, "question": str, "target": str}.
    """
    import random

    # For synthetic lengths, generate our own samples
    if context_length in SYNTHETIC_LENGTHS:
        target_chars = _length_to_chars(context_length)
        print(f"  Generating synthetic BABILong samples: "
              f"{context_length} (~{target_chars:,} chars) / {task} ...")
        rng = random.Random(42)  # deterministic
        samples = []
        for i in range(max_samples):
            sample = _generate_synthetic_sample(task, target_chars, rng)
            samples.append(sample)
            print(f"    Sample {i}: {len(sample['input']):,} chars, "
                  f"Q: {sample['question']}, A: {sample['target']}")
        return samples

    # For standard lengths, load from HuggingFace
    from datasets import load_dataset

    print(f"  Loading BABILong dataset: {context_length}/{task} ...")
    ds = load_dataset("RMT-team/babilong", context_length, trust_remote_code=True)

    if task not in ds:
        print(f"  WARNING: task {task} not found in {context_length} split. "
              f"Available: {list(ds.keys())}")
        return []

    split = ds[task]
    samples = []
    for i, row in enumerate(split):
        if i >= max_samples:
            break
        samples.append({
            "input": row["input"],
            "question": row["question"],
            "target": row["target"],
        })

    print(f"  Loaded {len(samples)} samples "
          f"(avg {sum(len(s['input']) for s in samples) // max(len(samples), 1)} chars)")
    return samples


# ---------------------------------------------------------------------------
# Haystack builder — split BABILong text into files in nested dirs
# ---------------------------------------------------------------------------

def build_file_haystack(
    tmpdir: Path,
    sample: dict,
    context_length: str,
    sample_idx: int,
) -> tuple[Path, int]:
    """Split a BABILong sample's input text across multiple files in nested dirs.

    Returns (haystack_dir, num_files_created).
    """
    hay_dir = tmpdir / f"haystack_{context_length}_{sample_idx}"
    hay_dir.mkdir(parents=True, exist_ok=True)

    text = sample["input"]
    num_files = FILES_PER_LENGTH.get(context_length, 5)
    depth = NESTING_DEPTH.get(context_length, 0)

    # Split text into roughly equal chunks
    chunk_size = max(len(text) // num_files, 1)
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    # If we ended up with more chunks than intended, merge the last ones
    while len(chunks) > num_files and len(chunks) > 1:
        chunks[-2] += chunks[-1]
        chunks.pop()

    # Create nested folder structure
    folders = _build_folder_tree(hay_dir, depth, len(chunks))

    # Distribute chunks across folders as files
    for i, chunk in enumerate(chunks):
        folder = folders[i % len(folders)]
        filename = _generate_filename(i, len(chunks))
        filepath = folder / filename
        filepath.write_text(chunk, encoding="utf-8")

    return hay_dir, len(chunks)


def _build_folder_tree(root: Path, depth: int, num_files: int) -> list[Path]:
    """Create a nested folder structure and return all leaf directories."""
    if depth == 0:
        return [root]

    folders = [root]
    current_level = [root]

    for d in range(depth):
        next_level = []
        # Spread folders across current level
        folders_per_parent = max(2, min(4, num_files // max(len(current_level), 1)))
        for parent in current_level:
            for j in range(folders_per_parent):
                idx = (len(next_level)) % len(FOLDER_NAMES)
                name = FOLDER_NAMES[idx]
                # Add numeric suffix to avoid collisions
                if j > 0:
                    name = f"{name}_{j}"
                child = parent / name
                child.mkdir(parents=True, exist_ok=True)
                next_level.append(child)
        current_level = next_level
        folders = current_level

    return folders


def _generate_filename(index: int, total: int) -> str:
    """Generate a realistic-looking filename."""
    templates = [
        "report_{i:03d}.txt",
        "document_{i:03d}.txt",
        "notes_{i:03d}.txt",
        "memo_{i:03d}.txt",
        "log_{i:03d}.txt",
        "entry_{i:03d}.txt",
        "record_{i:03d}.txt",
        "data_{i:03d}.txt",
        "summary_{i:03d}.txt",
        "draft_{i:03d}.txt",
    ]
    template = templates[index % len(templates)]
    return template.format(i=index)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_answer(answer: str, target: str) -> bool:
    """Check if the answer contains the target.

    BABILong targets are short (single word or phrase like "bathroom",
    "yes", "Daniel"). We do case-insensitive substring matching.
    """
    answer_lower = answer.lower().strip()
    target_lower = target.lower().strip()

    # Exact match
    if target_lower in answer_lower:
        return True

    # For multi-word targets, check each word is present
    target_words = target_lower.split()
    if len(target_words) > 1:
        return all(w in answer_lower for w in target_words)

    return False


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def run_babilong_benchmark(
    tasks: list[str] | None = None,
    lengths: list[str] | None = None,
    max_samples: int = 10,
) -> BenchmarkResults:
    """Run the BABILong benchmark suite."""
    tasks = tasks or ["qa1"]
    lengths = lengths or ["0k", "1k", "4k"]

    results = BenchmarkResults(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # Pre-load all datasets
    all_samples: dict[tuple[str, str], list[dict]] = {}
    for length in lengths:
        for task in tasks:
            samples = load_babilong_samples(task, length, max_samples)
            if samples:
                all_samples[(task, length)] = samples

    total_runs = sum(len(v) for v in all_samples.values())
    run_num = 0

    tmpdir = Path(tempfile.mkdtemp(prefix="babilong_bench_"))

    try:
        for (task, length), samples in all_samples.items():
            for sample_idx, sample in enumerate(samples):
                run_num += 1
                label = (
                    f"[{run_num}/{total_runs}] "
                    f"task={task} ({TASK_DESCRIPTIONS.get(task, '?')}) "
                    f"length={length} sample={sample_idx}"
                )
                print(f"\n{'='*60}")
                print(f"  {label}")
                print(f"{'='*60}")

                # Build file haystack from BABILong sample
                hay_dir, num_files = build_file_haystack(
                    tmpdir, sample, length, sample_idx,
                )

                print(f"  Files: {num_files} | "
                      f"Input: {len(sample['input']):,} chars | "
                      f"Target: {sample['target']}")

                # Configure agent
                config = AgentConfig()
                config.target_dir = hay_dir
                config.skip_summaries = True
                bench_data = tmpdir / f"data_{task}_{length}_{sample_idx}"
                config.data_dir = bench_data
                bench_data.mkdir(parents=True, exist_ok=True)
                (bench_data / "notes").mkdir(exist_ok=True)
                (bench_data / "cache").mkdir(exist_ok=True)

                agent = Agent(config)

                # Index
                await agent.index(str(hay_dir))

                # Query with retry
                answer = ""
                latency = 0.0
                for attempt in range(3):
                    try:
                        if attempt > 0 or run_num > 1:
                            await asyncio.sleep(2)
                        t0 = time.perf_counter()
                        answer = await agent.query(sample["question"])
                        latency = time.perf_counter() - t0
                        break
                    except Exception as e:
                        print(f"  Attempt {attempt+1} failed: {e}")
                        if attempt == 2:
                            answer = f"ERROR: {e}"
                            latency = 0.0
                        await asyncio.sleep(5)

                found = check_answer(answer, sample["target"])
                tool_calls = agent.tools.call_count

                status = "PASS" if found else "FAIL"
                print(f"\n  Result: {status}")
                print(f"  Answer: {answer[:200]}")
                print(f"  Expected: {sample['target']}")
                print(f"  Latency: {latency:.1f}s | Tools: {tool_calls}")

                results.trials.append(TrialResult(
                    task=task,
                    context_length=length,
                    sample_idx=sample_idx,
                    found=found,
                    answer=answer[:500],
                    expected=sample["target"],
                    latency_s=round(latency, 2),
                    tool_calls=tool_calls,
                    num_files=num_files,
                    input_chars=len(sample["input"]),
                ))

                # Cleanup this haystack
                shutil.rmtree(hay_dir, ignore_errors=True)
                shutil.rmtree(bench_data, ignore_errors=True)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: BenchmarkResults):
    """Print a formatted summary of benchmark results."""
    summary = results.summary()

    print(f"\n{'='*60}")
    print("  BABILong Benchmark Results")
    print(f"{'='*60}")
    print(f"  Timestamp: {results.timestamp}")
    print(f"  Total trials: {summary['total_trials']}")
    print(f"  Overall accuracy: {summary['accuracy']:.1%} "
          f"({summary['correct']}/{summary['total_trials']})")
    print(f"  Avg latency: {summary['avg_latency_s']}s")

    if summary.get("by_task"):
        print(f"\n  Accuracy by task:")
        for task, acc in sorted(summary["by_task"].items()):
            desc = TASK_DESCRIPTIONS.get(task, "")
            print(f"    {task:>5s} ({desc:<25s}): {acc:.1%}")

    if summary.get("by_length"):
        print(f"\n  Accuracy by context length:")
        for length, acc in sorted(summary["by_length"].items(),
                                   key=lambda x: _length_sort_key(x[0])):
            print(f"    {length:>6s}: {acc:.1%}")

    # Per-trial details
    print(f"\n  {'Task':<6s} {'Length':<8s} {'#':<4s} {'Found':<6s} "
          f"{'Latency':<8s} {'Files':<6s} {'Calls':<6s}")
    print(f"  {'-'*50}")
    for t in results.trials:
        status = "YES" if t.found else "NO"
        print(f"  {t.task:<6s} {t.context_length:<8s} {t.sample_idx:<4d} "
              f"{status:<6s} {t.latency_s:>5.1f}s  {t.num_files:<6d} "
              f"{t.tool_calls:<6d}")


def _length_sort_key(length: str) -> int:
    """Sort context lengths numerically."""
    num = length.replace("k", "000").replace("M", "000000")
    try:
        return int(num)
    except ValueError:
        return 0


def save_results(results: BenchmarkResults, output_path: Path):
    """Save results to JSON."""
    data = {
        "timestamp": results.timestamp,
        "summary": results.summary(),
        "trials": [
            {
                "task": t.task,
                "context_length": t.context_length,
                "sample_idx": t.sample_idx,
                "found": t.found,
                "answer": t.answer,
                "expected": t.expected,
                "latency_s": t.latency_s,
                "tool_calls": t.tool_calls,
                "num_files": t.num_files,
                "input_chars": t.input_chars,
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

    parser = argparse.ArgumentParser(
        description="BABILong Benchmark for LintIngest",
    )
    parser.add_argument(
        "--tasks", default="qa1",
        help="Comma-separated tasks: qa1-qa10 (default: qa1)",
    )
    parser.add_argument(
        "--lengths", default="0k,1k,4k",
        help="Comma-separated context lengths: "
             "0k,1k,2k,4k,8k,16k,32k,64k,128k,512k,1M (from HF dataset) "
             "or 3M,5M,10M (synthetic generation with PG19 filler) "
             "(default: 0k,1k,4k)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10,
        help="Max samples per task/length combination (default: 10)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results "
             "(default: benchmarks/babilong_results.json)",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    lengths = [l.strip() for l in args.lengths.split(",")]

    # Validate
    for t in tasks:
        if t not in AVAILABLE_TASKS:
            print(f"ERROR: Unknown task '{t}'. Available: {AVAILABLE_TASKS}")
            sys.exit(1)
    for l in lengths:
        if l not in AVAILABLE_LENGTHS:
            print(f"ERROR: Unknown length '{l}'. Available: {AVAILABLE_LENGTHS}")
            sys.exit(1)

    print("LintIngest BABILong Benchmark")
    print(f"  Tasks: {tasks}")
    print(f"  Context lengths: {lengths}")
    print(f"  Max samples per config: {args.max_samples}")

    results = asyncio.run(run_babilong_benchmark(
        tasks=tasks,
        lengths=lengths,
        max_samples=args.max_samples,
    ))

    print_summary(results)

    output = (Path(args.output) if args.output
              else Path(__file__).parent / "babilong_results.json")
    save_results(results, output)


if __name__ == "__main__":
    main()
