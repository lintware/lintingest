"""Core agent loop - orchestrates indexing and query passes."""

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from agent.config import AgentConfig
from agent.tools import ToolExecutor, TOOL_SCHEMAS
from agent.compaction import ContextCompactor
from agent.parallel import parallel_completions, parallel_tool_reads

logger = logging.getLogger("lintingest.core")

SYSTEM_PROMPT = """You search files and answer questions. You have these tools:
{tool_descriptions}

To use a tool, respond with JSON only:
{{"tool": "tool_name", "args": {{"key": "value"}}}}

When you have enough info, respond with ANSWER: followed by your answer in plain language."""

SUMMARIZE_PROMPT = """Summarize this tool result in 2-3 short sentences, keeping only information relevant to the question.
Question: {question}
Tool result:
{result}
Summary:"""


class Agent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.tools = ToolExecutor(self.config)
        self.compactor = ContextCompactor(
            max_history=self.config.max_history_results,
            max_chars=self.config.max_context_tokens * 4,  # rough char estimate
        )

    async def _llm_call(
        self, prompt: str, max_tokens: int | None = None, images: list[str] | None = None,
    ) -> str:
        """Single LLM call to the MLX VLM server. Optionally include images."""
        if images:
            content = [{"type": "text", "text": prompt}]
            for img_path in images:
                content.append({"type": "image_url", "image_url": {"url": img_path}})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=120.0,
            )
            if resp.status_code != 200:
                logger.error(f"LLM call failed ({resp.status_code}): {resp.text[:500]}")
                raise RuntimeError(f"Model server error {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            if "choices" not in data:
                logger.error(f"Unexpected response: {json.dumps(data)[:500]}")
                raise RuntimeError(f"No choices in response: {json.dumps(data)[:200]}")
            content = data["choices"][0]["message"]["content"]
            if "<think>" in content:
                parts = content.split("</think>")
                content = parts[-1].strip() if len(parts) > 1 else content
            return content

    def _build_tool_descriptions(self) -> str:
        descs = []
        for name, schema in TOOL_SCHEMAS.items():
            props = schema.get("properties", {})
            params = ", ".join(f'{k}: {v.get("type", "any")}' for k, v in props.items())
            descs.append(f"- {name}({params})")
        return "\n".join(descs)

    async def index(self, target_dir: str | None = None) -> dict:
        """Phase 1: Index the target directory."""
        target = target_dir or str(self.config.target_dir)
        logger.info(f"Indexing: {target}")
        start = time.time()

        result = self.tools.execute("index_directory", {"path": target})
        elapsed = time.time() - start
        logger.info(f"Indexing completed in {elapsed:.1f}s")

        if result["success"]:
            # Parse count from XML: <result><indexed>N</indexed></result>
            import re
            m = re.search(r"<indexed>(\d+)</indexed>", result["output"])
            count = int(m.group(1)) if m else 0
            print(f"Indexed {count} files in {elapsed:.1f}s")

            # Generate summaries for text files in parallel
            await self._parallel_preview_summaries()
            return {"files_indexed": count}
        else:
            print(f"Indexing failed: {result.get('error')}")
            return {"error": result.get("error")}

    async def _parallel_preview_summaries(self):
        """Use parallel workers to generate quick summaries of text files."""
        index_path = self.config.data_dir / "index.json"
        if not index_path.exists():
            return

        index = json.loads(index_path.read_text())
        text_files = [
            e for e in index
            if e.get("preview") and e.get("type") in (
                ".txt", ".md", ".py", ".js", ".ts", ".json", ".html"
            )
        ]

        if not text_files:
            return

        # Process in batches of 4
        for i in range(0, len(text_files), self.config.parallel_workers):
            batch = text_files[i:i + self.config.parallel_workers]
            prompts = [
                f"Summarize in one sentence what this file is about:\n{e['preview']}"
                for e in batch
            ]
            summaries = await parallel_completions(
                self.config.base_url,
                self.config.model_id,
                prompts,
                max_tokens=60,
                temperature=0.2,
                max_concurrent=self.config.parallel_workers,
            )
            for entry, summary in zip(batch, summaries):
                entry["summary"] = summary

        index_path.write_text(json.dumps(index, indent=2, default=str))
        logger.info(f"Generated summaries for {len(text_files)} text files")

    async def query(self, question: str) -> str:
        """Phase 2: Answer a question using agentic tool-calling loop."""
        self.compactor.reset()
        tool_descs = self._build_tool_descriptions()
        system = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)

        max_iterations = 5
        iteration = 0
        tools_called = []  # Track what we've already called
        read_index_done = False  # Hard-block re-reads
        _seen_calls: set[str] = set()  # Track exact call signatures to block duplicates

        print(f"\nQuery: {question}")
        print("-" * 40)

        # Step 0: Always read index first — no model decision needed
        index_result = self.tools.execute("read_index", {})
        if index_result["success"]:
            self.compactor.add_result("read_index", index_result["output"])
            read_index_done = True
            tools_called.append("read_index")
            print(f"  [0] read_index (auto)")
        else:
            # Index missing — run indexing pass, then read
            await self.index()
            index_result = self.tools.execute("read_index", {})
            if index_result["success"]:
                self.compactor.add_result("read_index", index_result["output"])
                read_index_done = True
                tools_called.append("read_index")
                print(f"  [0] read_index (auto, after re-index)")

        stall_count = 0  # Track consecutive blocked/duplicate calls

        while iteration < max_iterations:
            iteration += 1
            context = self.compactor.get_context()

            hint = ""
            if len(tools_called) >= 4:
                hint = "\nYou have enough data. Write ANSWER: followed by your response."

            prompt = (
                f"{system}\n\n"
                f"Findings so far:\n{context}\n\n"
                f"User question: {question}\n"
                f"{hint}\n"
                f"Call a tool or write ANSWER:"
            )

            response = await self._llm_call(prompt)
            logger.info(f"Iteration {iteration}: {response[:200]}")

            # Check if it's a final answer
            if "ANSWER:" in response:
                answer = response.split("ANSWER:", 1)[1].strip()
                print(f"\nAnswer: {answer}")
                self.tools.execute("note_store", {
                    "key": f"query_{int(time.time())}",
                    "content": f"Q: {question}\nA: {answer}",
                    "category": "query_history",
                })
                return answer

            # Try to parse tool call
            tool_call = self._parse_tool_call(response)
            if not tool_call:
                # Model didn't format correctly — treat as answer
                print(f"\nAnswer: {response}")
                return response

            tool_name, args = tool_call

            # Hard-block read_index after first call — don't waste an iteration
            if tool_name == "read_index" and read_index_done:
                stall_count += 1
                iteration -= 1  # Don't count this as a real iteration
                if stall_count >= 2:
                    # Model is stuck — force synthesis
                    break
                self.compactor.add_result(
                    "system",
                    "You already have the file index. Use file_read, glob_search, or content_search.",
                )
                continue

            # Block exact duplicate tool calls (same name + same args)
            call_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            if call_key in _seen_calls:
                stall_count += 1
                iteration -= 1
                if stall_count >= 2:
                    break
                self.compactor.add_result(
                    "system",
                    f"You already called {tool_name} with those arguments. Try a different tool or write ANSWER:",
                )
                continue

            _seen_calls.add(call_key)
            stall_count = 0  # Reset on successful new call
            tools_called.append(tool_name)
            print(f"  [{iteration}] {tool_name}({json.dumps(args)[:80]})")

            # Execute tool
            result = self.tools.execute(tool_name, args)

            if result["success"]:
                output = result["output"]

                # Handle VLM image requests from image_describe tool
                if tool_name == "image_describe":
                    try:
                        vlm_req = json.loads(output)
                        if vlm_req.get("__vlm_request__"):
                            img_path = vlm_req["image_path"]
                            img_question = vlm_req["question"]
                            print(f"    -> VLM analyzing: {Path(img_path).name}")
                            description = await self._llm_call(
                                img_question,
                                max_tokens=200,
                                images=[img_path],
                            )
                            self.compactor.add_result(
                                tool_name,
                                f"Image {Path(img_path).name}: {description}",
                            )
                            continue
                    except (json.JSONDecodeError, KeyError):
                        pass
                    self.compactor.add_result(tool_name, output)
                # If we got file content, try parallel extraction
                elif tool_name == "file_read" and len(output) > 500:
                    summary = await self._llm_call(
                        SUMMARIZE_PROMPT.format(question=question, result=output[:2000]),
                        max_tokens=100,
                    )
                    self.compactor.add_result(tool_name, output, summary)
                elif tool_name == "read_index":
                    # XML index is already compact, pass through
                    self.compactor.add_result(tool_name, output)
                elif tool_name in ("glob_search", "content_search"):
                    # XML output is already compact, summarize only if large
                    if len(output) > 1500:
                        summary = await self._llm_call(
                            SUMMARIZE_PROMPT.format(question=question, result=output[:2000]),
                            max_tokens=100,
                        )
                        self.compactor.add_result(tool_name, output, summary)
                    else:
                        self.compactor.add_result(tool_name, output)
                else:
                    self.compactor.add_result(tool_name, output)
            else:
                self.compactor.add_result(
                    tool_name, f"ERROR: {result.get('error', 'unknown')}"
                )

        # Exhausted iterations — synthesize a final answer
        context = self.compactor.get_context()
        final_prompt = (
            f"You searched the user's files and found this:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Write a helpful answer in plain language. "
            f"Mention specific file names. Be concise."
        )
        answer = await self._llm_call(final_prompt, max_tokens=self.config.max_answer_tokens)
        print(f"\nAnswer: {answer}")
        return answer

    async def parallel_query(self, question: str) -> str:
        """Enhanced query that uses parallel file reading."""
        self.compactor.reset()

        # Step 1: Read index from JSON file directly
        json_path = self.config.data_dir / "index.json"
        if not json_path.exists():
            await self.index()
        try:
            files = json.loads(json_path.read_text())
        except (json.JSONDecodeError, TypeError, FileNotFoundError):
            return await self.query(question)

        file_list = "\n".join(
            f"- {f['name']} ({f.get('type', '?')}, {f.get('size', '?')}b)"
            for f in files[:50]
        )
        select_prompt = (
            f"Given these files:\n{file_list}\n\n"
            f"Question: {question}\n\n"
            f"List up to 8 filenames most likely to contain the answer, one per line. "
            f"Just the filenames, nothing else."
        )
        selection = await self._llm_call(select_prompt, max_tokens=100)
        selected_names = [
            line.strip().lstrip("- ").strip()
            for line in selection.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        # Step 3: Read selected files in parallel batches
        file_contents = []
        for name in selected_names[:8]:
            matching = [f for f in files if f["name"] == name or name in f.get("path", "")]
            if matching:
                path = matching[0]["path"]
                read_result = self.tools.execute("file_read", {"path": path, "limit": 100})
                if read_result["success"]:
                    file_contents.append({"path": path, "content": read_result["output"]})

        if not file_contents:
            return await self.query(question)

        # Step 4: Parallel LLM extraction
        print(f"\nSearching {len(file_contents)} files in parallel...")
        findings = await parallel_tool_reads(
            self.config.base_url,
            self.config.model_id,
            file_contents,
            question,
            max_concurrent=self.config.parallel_workers,
        )

        if not findings:
            return await self.query(question)

        # Step 5: Synthesize
        findings_text = "\n\n".join(
            f"From {f['path']}:\n{f['finding']}" for f in findings
        )
        synth_prompt = (
            f"Based on these findings from the user's files:\n{findings_text}\n\n"
            f"Answer concisely: {question}\n"
            f"Cite file paths."
        )
        answer = await self._llm_call(synth_prompt, max_tokens=self.config.max_answer_tokens)
        print(f"\nAnswer: {answer}")

        # Store note
        self.tools.execute("note_store", {
            "key": f"pquery_{int(time.time())}",
            "content": f"Q: {question}\nA: {answer}\nFiles: {[f['path'] for f in findings]}",
            "category": "query_history",
        })
        return answer

    def _parse_tool_call(self, response: str) -> tuple[str, dict] | None:
        """Try to extract a tool call from the LLM response."""
        # Look for JSON in the response
        import re
        patterns = [
            r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}[^{}]*\}',
            r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*[^{}]*\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    tool_name = parsed.get("tool")
                    args = parsed.get("args", {})
                    if tool_name and tool_name in TOOL_SCHEMAS:
                        return (tool_name, args)
                except json.JSONDecodeError:
                    continue

        # Try to find tool name mentioned directly
        for tool_name in TOOL_SCHEMAS:
            if tool_name in response:
                return (tool_name, {})

        return None
