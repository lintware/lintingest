"""Parallel worker dispatch for concurrent model calls."""

import asyncio
import json
import httpx


async def parallel_completions(
    base_url: str,
    model_id: str,
    prompts: list[str],
    max_tokens: int = 256,
    temperature: float = 0.3,
    max_concurrent: int = 4,
) -> list[str]:
    """Send multiple prompts in parallel to the MLX server.
    Returns list of responses in same order as prompts."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _complete(client: httpx.AsyncClient, prompt: str) -> str:
        async with semaphore:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            try:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=30.0,
                )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # Strip thinking tags if present (Qwen3.5 uses <think>)
                if "<think>" in content:
                    parts = content.split("</think>")
                    content = parts[-1].strip() if len(parts) > 1 else content
                return content
            except Exception as e:
                return f"[error: {e}]"

    async with httpx.AsyncClient() as client:
        tasks = [_complete(client, p) for p in prompts]
        return await asyncio.gather(*tasks)


async def parallel_tool_reads(
    base_url: str,
    model_id: str,
    file_contents: list[dict],
    question: str,
    max_concurrent: int = 4,
    max_tokens: int = 256,
    temperature: float = 0.3,
) -> list[dict]:
    """Read multiple files in parallel via LLM and extract relevant info.

    file_contents: list of {"path": str, "content": str}
    Returns: list of {"path": str, "finding": str}
    """
    prompts = []
    for fc in file_contents:
        prompt = (
            f"Given this file content from '{fc['path']}':\n"
            f"---\n{fc['content'][:1500]}\n---\n"
            f"Question: {question}\n"
            f"Extract ONLY the parts relevant to the question. "
            f"If nothing is relevant, say 'NOT_RELEVANT'. Be concise."
        )
        prompts.append(prompt)

    results = await parallel_completions(
        base_url, model_id, prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        max_concurrent=max_concurrent,
    )

    findings = []
    for fc, result in zip(file_contents, results):
        if "NOT_RELEVANT" not in result.upper():
            findings.append({"path": fc["path"], "finding": result})
    return findings
