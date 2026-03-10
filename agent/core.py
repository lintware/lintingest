"""Core agent loop - orchestrates indexing and query passes."""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import httpx

from agent.config import AgentConfig
from agent.tools import ToolExecutor, TOOL_SCHEMAS
from agent.compaction import ContextCompactor
from agent.parallel import parallel_completions, parallel_tool_reads, parallel_vlm_descriptions

logger = logging.getLogger("lintingest.core")

SYSTEM_PROMPT = """You are a file search assistant. You MUST search files before answering.
NEVER answer from your own knowledge. ONLY answer from file contents.

Steps:
1. Use content_search to grep for keywords from the question
2. Use file_read to read matching files
3. Write ANSWER: with the exact information you found

Tools:
{tool_descriptions}

Respond with JSON only: {{"tool": "tool_name", "args": {{"key": "value"}}}}
When done: ANSWER: your answer (quote the exact text you found)"""

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
        self._conversation: list[dict] = []  # recent Q&A pairs for follow-ups
        self._index_cache: list[dict] | None = None  # cached parsed index

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
        self._invalidate_index_cache()
        target = target_dir or str(self.config.target_dir)
        logger.info(f"Indexing: {target}")
        start = time.time()

        result = self.tools.execute("index_directory", {"path": target})
        elapsed = time.time() - start
        logger.info(f"Indexing completed in {elapsed:.1f}s")

        if result["success"]:
            # Parse count from XML: <result><indexed>N</indexed></result>
            m = re.search(r"<indexed>(\d+)</indexed>", result["output"])
            count = int(m.group(1)) if m else 0
            print(f"Indexed {count} files in {elapsed:.1f}s")

            if not self.config.skip_summaries:
                # Generate summaries for text files in parallel
                await self._parallel_preview_summaries()
                # Describe images via VLM
                await self._parallel_image_descriptions()
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

    async def _parallel_image_descriptions(self):
        """Use the VLM to describe image files during indexing."""
        index_path = self.config.data_dir / "index.json"
        if not index_path.exists():
            return

        index = json.loads(index_path.read_text())
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
                      ".webp", ".heic", ".heif", ".ico"}
        # Skip .svg — not a raster image the VLM can read
        image_files = [
            e for e in index
            if e.get("type", "").lower() in image_exts
            and e.get("size", 0) < self.config.max_file_size
        ]

        if not image_files:
            return

        print(f"Describing {len(image_files)} images via VLM...")

        # Process in batches of 2 (VLM calls are heavier than text)
        batch_size = max(1, self.config.parallel_workers // 2)
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            descriptions = await parallel_vlm_descriptions(
                self.config.base_url,
                self.config.model_id,
                batch,
                max_tokens=80,
                temperature=0.2,
                max_concurrent=batch_size,
            )
            for entry, desc in zip(batch, descriptions):
                entry["preview"] = desc
                entry["vlm_description"] = desc
            print(f"  Described {min(i + batch_size, len(image_files))}/{len(image_files)} images")

        index_path.write_text(json.dumps(index, indent=2, default=str))
        logger.info(f"Generated VLM descriptions for {len(image_files)} images")

    async def query(self, question: str) -> str:
        """Phase 2: Answer a question using agentic tool-calling loop."""
        self.compactor.reset()
        tool_descs = self._build_tool_descriptions()

        # Include conversation history in system prompt
        conv_ctx = self._get_conversation_context()
        system = SYSTEM_PROMPT.format(tool_descriptions=tool_descs)
        if conv_ctx:
            system = f"{system}\n\n{conv_ctx}"

        max_iterations = 10
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

        # Step 0b: Auto content_search with question keywords
        keywords = self._extract_keywords(question)
        if keywords:
            search_result = self.tools.execute("content_search", {
                "pattern": keywords,
                "directory": str(self.config.target_dir),
                "max_results": 10,
            })
            if search_result["success"] and search_result["output"].strip():
                self.compactor.add_result("content_search", search_result["output"])
                tools_called.append("content_search")
                _seen_calls.add(
                    f"content_search:{json.dumps({'pattern': keywords, 'directory': str(self.config.target_dir), 'max_results': 10}, sort_keys=True)}"
                )
                print(f"  [0b] content_search (auto, keywords: {keywords[:60]})")

                # Step 0c: Auto file_read on top matching files
                hit_files = re.findall(r'<hit f="([^"]+)"', search_result["output"])
                seen_files = set()
                for fpath in hit_files:
                    if fpath in seen_files:
                        continue
                    seen_files.add(fpath)
                    if len(seen_files) > 3:
                        break
                    read_result = self.tools.execute("file_read", {
                        "path": fpath, "offset": 0, "limit": 100,
                    })
                    if read_result["success"]:
                        self.compactor.add_result("file_read", read_result["output"])
                        tools_called.append("file_read")
                        print(f"  [0c] file_read (auto, {Path(fpath).name})")

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
                self._add_to_conversation(question, answer)
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
                self._add_to_conversation(question, response)
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

            # Suppress note_store/note_recall during search — don't waste iterations
            if tool_name in ("note_store", "note_recall"):
                stall_count += 1
                iteration -= 1
                if stall_count >= 2:
                    break
                self.compactor.add_result(
                    "system",
                    "Do not store notes yet. Use content_search or file_read to find the answer first.",
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
        self._add_to_conversation(question, answer)
        return answer

    def _load_index(self) -> list[dict]:
        """Load and cache the file index."""
        if self._index_cache is not None:
            return self._index_cache
        json_path = self.config.data_dir / "index.json"
        if not json_path.exists():
            return []
        try:
            self._index_cache = json.loads(json_path.read_text())
            return self._index_cache
        except (json.JSONDecodeError, TypeError):
            return []

    def _invalidate_index_cache(self):
        self._index_cache = None

    def _build_index_summary(self, files: list[dict]) -> str:
        """Build a factual summary of the index for the LLM."""
        if not files:
            return "No files found."

        # Count by type
        type_counts: dict[str, int] = {}
        dirs = 0
        total_size = 0
        for f in files:
            ftype = f.get("type", "unknown")
            if ftype == "dir":
                dirs += 1
            else:
                type_counts[ftype] = type_counts.get(ftype, 0) + 1
            total_size += f.get("size", 0)

        # Build listing
        lines = [f"Total: {len(files)} entries ({dirs} folders, {len(files) - dirs} files)"]
        lines.append(f"Total size: {total_size:,} bytes")

        # Type breakdown
        if type_counts:
            lines.append("File types:")
            for ext, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {ext}: {count}")

        # File list
        lines.append("\nFiles:")
        for f in files:
            if f.get("type") == "dir":
                continue
            preview = ""
            # Prefer VLM description for images, then preview
            if f.get("vlm_description"):
                preview = f" - {f['vlm_description'][:80]}"
            elif f.get("preview"):
                preview = f" - {f['preview'][:50]}"
            lines.append(f"  {f['name']} ({f.get('type', '?')}, {f.get('size', '?')}b){preview}")

        return "\n".join(lines)

    def _add_to_conversation(self, question: str, answer: str):
        """Store Q&A pair for follow-up context."""
        self._conversation.append({"q": question, "a": answer})
        # Keep only last 3 exchanges
        if len(self._conversation) > 3:
            self._conversation = self._conversation[-3:]

    def _get_conversation_context(self) -> str:
        """Build conversation history string."""
        if not self._conversation:
            return ""
        lines = ["Previous conversation:"]
        for ex in self._conversation:
            lines.append(f"User: {ex['q']}")
            lines.append(f"Assistant: {ex['a']}")
        return "\n".join(lines)

    async def parallel_query(self, question: str) -> str:
        """Enhanced query: answers from index data when possible, falls back to tool loop."""
        files = self._load_index()
        if not files:
            await self.index()
            files = self._load_index()
        if not files:
            return "No files found in the target directory. Try running /index first."

        # Build a factual index summary (computed, not LLM-guessed)
        index_summary = self._build_index_summary(files)
        conversation_ctx = self._get_conversation_context()

        # Ask the LLM to answer using the pre-computed summary
        prompt_parts = []
        if conversation_ctx:
            prompt_parts.append(conversation_ctx)
        prompt_parts.append(f"Here is a factual summary of the user's directory:\n{index_summary}")
        prompt_parts.append(f"\nUser question: {question}")
        prompt_parts.append(
            "\nIf you can answer from the summary above, write your answer directly."
            "\nIf you need to read file contents to answer, write NEED_FILES: followed by filenames."
        )

        response = await self._llm_call("\n".join(prompt_parts), max_tokens=400)

        # If the model needs file contents, fall back to the agentic tool loop
        if "NEED_FILES:" in response:
            answer = await self.query(question)
            self._add_to_conversation(question, answer)
            return answer

        # Model answered directly — clean it up
        answer = response.strip()
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:", 1)[1].strip()

        print(f"\nAnswer: {answer}")
        self._add_to_conversation(question, answer)
        self.tools.execute("note_store", {
            "key": f"pquery_{int(time.time())}",
            "content": f"Q: {question}\nA: {answer}",
            "category": "query_history",
        })
        return answer

    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "from", "by", "about", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "over", "under", "again", "further", "then", "once",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "because", "if", "when", "where", "how", "what",
        "which", "who", "whom", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his", "she", "her",
        "it", "its", "they", "them", "their", "there", "here", "up", "down",
        "file", "files", "find", "tell", "show", "give", "get", "list",
        "many", "much", "also", "like",
    })

    def _extract_keywords(self, question: str) -> str:
        """Extract search keywords from a question. Returns regex OR pattern."""
        words = re.findall(r"[a-zA-Z0-9_]+", question)
        keywords = [w for w in words if w.lower() not in self._STOP_WORDS and len(w) > 1]
        if not keywords:
            return ""
        return "|".join(keywords)

    def _parse_tool_call(self, response: str) -> tuple[str, dict] | None:
        """Try to extract a tool call from the LLM response."""
        # Look for JSON in the response
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
