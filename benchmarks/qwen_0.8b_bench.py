"""Benchmark Qwen3.5-0.8B MLX variants for tokens/sec comparison."""

import time
import json
import os
import resource

MODELS = [
    "mlx-community/Qwen3.5-0.8B-MLX-8bit",
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "mlx-community/Qwen3.5-0.8B-MLX-bf16",
]

# Also try common Qwen3 0.5B-1.5B range if they exist
EXTRA_MODELS = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-0.6B-8bit",
    "mlx-community/Qwen3-0.6B-bf16",
]

PROMPT = "Explain what a file system is in three sentences."
MAX_TOKENS = 100


def get_memory_mb():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_maxrss / (1024 * 1024)


def benchmark_model(model_name):
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

        if hasattr(tokenizer, "apply_chat_template"):
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
        mx.clear_cache()

        # Benchmark: 3 runs
        tok_per_sec_runs = []
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
            gen_tokens = len(output_ids) - num_prompt_tokens
            if gen_tokens <= 0:
                gen_tokens = len(tokenizer.encode(output))

            elapsed = t_end - t_start
            tps = gen_tokens / elapsed if elapsed > 0 else 0
            tok_per_sec_runs.append(tps)

        avg_tps = sum(tok_per_sec_runs) / len(tok_per_sec_runs)
        peak_tps = max(tok_per_sec_runs)

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
        }

        print(f"  Avg tok/s: {avg_tps:.1f}  |  Peak: {peak_tps:.1f}")

        del model, tokenizer
        mx.clear_cache()

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": model_name, "error": str(e)}


def main():
    results = []

    print("Qwen 0.8B MLX Benchmark")
    print(f"Prompt: '{PROMPT}'")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Runs per model: 3")

    for model_name in MODELS + EXTRA_MODELS:
        result = benchmark_model(model_name)
        results.append(result)

    out_path = "/Users/admin/lintingest/benchmarks/qwen_0.8b_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("QWEN SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Avg tok/s':>10} {'Peak tok/s':>11} {'Load(s)':>8} {'Size(MB)':>9}")
    print(f"{'-'*45} {'-'*10} {'-'*11} {'-'*8} {'-'*9}")

    for r in sorted(results, key=lambda x: x.get("avg_tok_per_sec", 0), reverse=True):
        if "error" in r:
            print(f"{r['model']:<45} {'ERROR':>10}  {r.get('error','')[:30]}")
        else:
            name = r["model"].replace("mlx-community/", "")
            print(f"{name:<45} {r['avg_tok_per_sec']:>10.1f} {r['peak_tok_per_sec']:>11.1f} {r['load_time_sec']:>8.2f} {r['model_size_mb']:>9.1f}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
