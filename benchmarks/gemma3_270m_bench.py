"""Benchmark all Gemma 3 270M MLX variants for tokens/sec."""

import time
import json
import subprocess
import sys
import os
import resource

# All Gemma 3 270M variants from HuggingFace mlx-community
MODELS = [
    # Instruct variants
    "mlx-community/gemma-3-270m-it-4bit",
    "mlx-community/gemma-3-270m-it-5bit",
    "mlx-community/gemma-3-270m-it-6bit",
    "mlx-community/gemma-3-270m-it-8bit",
    "mlx-community/gemma-3-270m-it-bf16",
    # Instruct QAT variants
    "mlx-community/gemma-3-270m-it-qat-4bit",
    "mlx-community/gemma-3-270m-it-qat-5bit",
    "mlx-community/gemma-3-270m-it-qat-6bit",
    "mlx-community/gemma-3-270m-it-qat-8bit",
    "mlx-community/gemma-3-270m-it-qat-bf16",
    # Base variants
    "mlx-community/gemma-3-270m-4bit",
    "mlx-community/gemma-3-270m-5bit",
    "mlx-community/gemma-3-270m-8bit",
    "mlx-community/gemma-3-270m-bf16",
    # Base QAT variants
    "mlx-community/gemma-3-270m-qat-4bit",
    "mlx-community/gemma-3-270m-qat-5bit",
    "mlx-community/gemma-3-270m-qat-6bit",
    "mlx-community/gemma-3-270m-qat-8bit",
    "mlx-community/gemma-3-270m-qat-bf16",
]

PROMPT = "Explain what a file system is in three sentences."
MAX_TOKENS = 100


def get_memory_mb():
    """Get current process memory in MB."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_maxrss / (1024 * 1024)  # macOS returns bytes


def benchmark_model(model_name):
    """Run a single model benchmark and return results."""
    try:
        from mlx_lm import load, generate
        import mlx.core as mx

        print(f"\n{'='*60}")
        print(f"Loading: {model_name}")

        mem_before = get_memory_mb()
        t0 = time.perf_counter()

        model, tokenizer = load(model_name)

        load_time = time.perf_counter() - t0
        mem_after = get_memory_mb()

        print(f"  Load time: {load_time:.2f}s")
        print(f"  Memory delta: {mem_after - mem_before:.1f} MB")

        # Check if it's an instruct model
        is_instruct = "-it-" in model_name or "-it" in model_name

        if is_instruct and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": PROMPT}]
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = PROMPT
        else:
            prompt_text = PROMPT

        # Warmup
        _ = generate(model, tokenizer, prompt=prompt_text, max_tokens=5, verbose=False)
        mx.metal.clear_cache()

        # Benchmark: 3 runs
        tok_per_sec_runs = []
        outputs = []
        for run in range(3):
            input_ids = tokenizer.encode(prompt_text)
            num_prompt_tokens = len(input_ids)

            t_start = time.perf_counter()
            output = generate(
                model, tokenizer, prompt=prompt_text,
                max_tokens=MAX_TOKENS, verbose=False
            )
            t_end = time.perf_counter()

            output_ids = tokenizer.encode(output)
            # Approximate generated tokens (output includes prompt echo sometimes)
            gen_tokens = len(output_ids) - num_prompt_tokens
            if gen_tokens <= 0:
                gen_tokens = len(tokenizer.encode(output))

            elapsed = t_end - t_start
            tps = gen_tokens / elapsed if elapsed > 0 else 0
            tok_per_sec_runs.append(tps)
            outputs.append(output)

        avg_tps = sum(tok_per_sec_runs) / len(tok_per_sec_runs)
        peak_tps = max(tok_per_sec_runs)

        # Get model size on disk
        from huggingface_hub import scan_cache_dir
        try:
            cache_info = scan_cache_dir()
            model_size_mb = 0
            short_name = model_name.split("/")[-1]
            for repo in cache_info.repos:
                if short_name in str(repo.repo_id):
                    model_size_mb = repo.size_on_disk / (1024 * 1024)
                    break
        except Exception:
            model_size_mb = 0

        result = {
            "model": model_name,
            "avg_tok_per_sec": round(avg_tps, 1),
            "peak_tok_per_sec": round(peak_tps, 1),
            "load_time_sec": round(load_time, 2),
            "mem_delta_mb": round(mem_after - mem_before, 1),
            "model_size_mb": round(model_size_mb, 1),
            "sample_output": outputs[-1][:200],
        }

        print(f"  Avg tok/s: {avg_tps:.1f}  |  Peak: {peak_tps:.1f}")

        # Cleanup
        del model, tokenizer
        mx.metal.clear_cache()

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "model": model_name,
            "error": str(e),
        }


def main():
    results = []

    print("Gemma 3 270M MLX Benchmark")
    print(f"Prompt: '{PROMPT}'")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Runs per model: 3")

    for model_name in MODELS:
        result = benchmark_model(model_name)
        results.append(result)

    # Save results
    os.makedirs("/Users/admin/lintingest/benchmarks", exist_ok=True)
    out_path = "/Users/admin/lintingest/benchmarks/gemma3_270m_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Avg tok/s':>10} {'Peak tok/s':>11} {'Load(s)':>8} {'Size(MB)':>9}")
    print(f"{'-'*45} {'-'*10} {'-'*11} {'-'*8} {'-'*9}")

    for r in sorted(results, key=lambda x: x.get("avg_tok_per_sec", 0), reverse=True):
        if "error" in r:
            print(f"{r['model']:<45} {'ERROR':>10}")
        else:
            name = r["model"].replace("mlx-community/", "")
            print(f"{name:<45} {r['avg_tok_per_sec']:>10.1f} {r['peak_tok_per_sec']:>11.1f} {r['load_time_sec']:>8.2f} {r['model_size_mb']:>9.1f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
