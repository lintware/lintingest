# NIAH Benchmark Results

Needle In A Haystack (NIAH) benchmark for LintIngest. Tests the agent's ability to find a specific fact hidden among distractor files.

**Model:** `mlx-community/Qwen3.5-0.8B-MLX-8bit` on Apple Silicon (M4 Mac Mini)
**Server:** `mlx_vlm.server` on `http://127.0.0.1:8080/v1`

## Results Summary

| Version | Accuracy | Avg Latency | Query Method |
|---------|----------|-------------|--------------|
| v1 | 76.7% (46/60) | 8.76s | Sequential with auto content_search |
| v2 | 90.0% (54/60) | 8.6s | Sequential with auto content_search + auto file_read |
| v3 | 80.0% (48/60) | 2.64s | Parallel extraction (4 concurrent LLM workers) |

## v1 — Sequential with auto content_search (76.7%)

**File:** `niah_results_v1_76pct.json`

**Config:**
- `max_context_tokens`: 2500
- `max_tokens`: 384
- `max_history_results`: 5
- `skip_summaries`: True
- Compaction truncation: 800 chars per result

**Changes from baseline (0% accuracy):**
1. Rewrote system prompt to instruct model to search files before answering
2. Added auto `content_search` with keyword extraction before the LLM loop
3. Increased compaction budget (1500 → 2500 tokens, 3 → 5 history results)
4. Increased `max_tokens` (256 → 384) for better JSON formatting
5. Suppressed `note_store`/`note_recall` during search to avoid wasted iterations

**Command:**
```bash
.venv/bin/python -u -m benchmarks.bench_niah \
  --sizes small,medium,large \
  --depths start,early,middle,late,end \
  --needles capital_zephyria,project_aurora_budget,server_password,recipe_secret
```

**Breakdown:**
| Size | Accuracy |
|------|----------|
| small | 65.0% |
| medium | 80.0% |
| large | 85.0% |

## v2 — Sequential with auto file_read (90.0%)

**File:** `niah_results_v2_90pct.json`

**Config:**
- `max_context_tokens`: 10000
- `max_tokens`: 384
- `max_history_results`: 5
- `skip_summaries`: True
- Compaction truncation: 800 chars per result

**Changes from v1:**
1. After auto `content_search`, automatically read the top 3 matching files via `file_read`
2. Increased `max_context_tokens` (2500 → 10000) to accommodate full file contents

**Command:** Same as v1.

**Breakdown:**
| Size | Accuracy |
|------|----------|
| small | 90.0% |
| medium | 80.0% |
| large | 100.0% |

## v3 — Parallel extraction (80.0%)

**File:** `niah_results_parallel.json`

**Config:** Same as v2, plus `parallel_workers=4` for concurrent LLM extraction.

**Changes from v2:**
1. New `parallel_niah_query()` method bypasses the sequential tool-calling loop
2. After `content_search`, reads top 4 matching files
3. Sends files to 4 parallel LLM workers via `parallel_tool_reads()` for extraction
4. Single final LLM call synthesizes findings into the answer
5. Falls back to sequential `query()` if parallel pass finds nothing

**Command:**
```bash
.venv/bin/python -u -m benchmarks.bench_niah_parallel \
  --sizes small,medium,large \
  --depths start,early,middle,late,end \
  --needles capital_zephyria,project_aurora_budget,server_password,recipe_secret
```

**Breakdown:**
| Size | Accuracy |
|------|----------|
| small | 75.0% |
| medium | 85.0% |
| large | 80.0% |

**Tradeoff:** 3.3x faster than v2 (2.64s vs 8.6s avg) but 10% lower accuracy. The 0.8B extraction workers sometimes mark relevant content as "NOT_RELEVANT", particularly for `project_aurora_budget` where keywords match many filler files. When parallel extraction fails, the fallback to sequential `query()` recovers most cases but adds latency.

## Reproducing

1. Start the model server:
   ```bash
   .venv/bin/python -m mlx_vlm.server --port 8080
   ```

2. Run the benchmark (sequential):
   ```bash
   .venv/bin/python -u -m benchmarks.bench_niah --sizes small,medium,large
   ```

3. Run the benchmark (parallel):
   ```bash
   .venv/bin/python -u -m benchmarks.bench_niah_parallel --sizes small,medium,large
   ```

Results are saved to `benchmarks/results/`.
