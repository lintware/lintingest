from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    # Model
    model_id: str = "mlx-community/Qwen3.5-0.8B-MLX-8bit"
    base_url: str = "http://127.0.0.1:8080/v1"
    max_tokens: int = 256
    temperature: float = 0.3
    parallel_workers: int = 4

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    target_dir: Path = field(default_factory=lambda: Path.home() / "Desktop")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # Security
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_tool_calls: int = 200
    max_search_results: int = 100
    blocked_paths: list = field(default_factory=lambda: [
        "/etc", "/usr", "/System", "/Library", "/bin", "/sbin",
        str(Path.home() / ".ssh"),
        str(Path.home() / ".aws"),
        str(Path.home() / ".gnupg"),
    ])

    # Compaction
    max_context_tokens: int = 1500
    max_history_results: int = 3
    max_answer_tokens: int = 500

    # Indexing
    max_depth: int = 10
    preview_chars: int = 200

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "notes").mkdir(exist_ok=True)
        (self.data_dir / "cache").mkdir(exist_ok=True)
