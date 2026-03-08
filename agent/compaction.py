"""Context compaction logic.
Keeps the agent's context small enough for the 0.8B model."""

import json
from dataclasses import dataclass, field


@dataclass
class CompactedResult:
    tool_name: str
    summary: str
    relevance: float = 1.0


class ContextCompactor:
    """Manages compaction of tool results to keep context within budget."""

    def __init__(self, max_history: int = 3, max_chars: int = 3000):
        self.max_history = max_history
        self.max_chars = max_chars
        self.results: list[CompactedResult] = []
        self.findings_summary: str = ""

    def add_result(self, tool_name: str, output: str, llm_summary: str | None = None):
        """Add a tool result, compacting if needed."""
        summary = llm_summary if llm_summary else self._truncate(output, 400)
        self.results.append(CompactedResult(tool_name=tool_name, summary=summary))

        if len(self.results) > self.max_history:
            old = self.results[:-self.max_history]
            old_text = "\n".join(f"- [{r.tool_name}]: {r.summary}" for r in old)
            if self.findings_summary:
                self.findings_summary = f"{self.findings_summary}\n{old_text}"
            else:
                self.findings_summary = old_text
            self.findings_summary = self._truncate(self.findings_summary, 800)
            self.results = self.results[-self.max_history:]

    def get_context(self) -> str:
        """Build the compacted context string for the LLM."""
        parts = []
        if self.findings_summary:
            parts.append(f"Previous findings:\n{self.findings_summary}")
        for r in self.results:
            parts.append(f"[{r.tool_name}]: {r.summary}")
        return "\n\n".join(parts)

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def reset(self):
        self.results.clear()
        self.findings_summary = ""
