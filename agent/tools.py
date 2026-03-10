"""Tool definitions and execution for the agent.
All file operations go through safe_path() validation.
NO deletion. Writes ONLY to data/ directory.
Index and tool outputs use XML for token efficiency."""

import json
import os
import re
import fnmatch
import logging
import subprocess
import shlex
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

from agent.config import AgentConfig

logger = logging.getLogger("lintingest.tools")

TOOL_SCHEMAS = {
    "shell_exec": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run (e.g. ls -la, cat file.txt, find . -name '*.py' | wc -l, head -20 file.txt)"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
        },
        "required": ["command"],
    },
    "index_directory": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to index"},
            "max_depth": {"type": "integer", "description": "Max recursion depth", "default": 10},
        },
        "required": ["path"],
    },
    "file_read": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "offset": {"type": "integer", "description": "Start line (0-indexed)", "default": 0},
            "limit": {"type": "integer", "description": "Max lines to read", "default": 50},
        },
        "required": ["path"],
    },
    "glob_search": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.txt)"},
            "directory": {"type": "string", "description": "Directory to search in"},
        },
        "required": ["pattern"],
    },
    "content_search": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "directory": {"type": "string", "description": "Directory to search in"},
            "max_results": {"type": "integer", "description": "Max results", "default": 20},
        },
        "required": ["pattern"],
    },
    "note_store": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Note identifier"},
            "content": {"type": "string", "description": "Note content"},
            "category": {"type": "string", "description": "Category tag", "default": "general"},
        },
        "required": ["key", "content"],
    },
    "note_recall": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query for notes"},
            "limit": {"type": "integer", "description": "Max notes to return", "default": 5},
        },
        "required": ["query"],
    },
    "read_index": {
        "type": "object",
        "properties": {},
    },
    "image_describe": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to image file to describe"},
            "question": {"type": "string", "description": "What to look for in the image", "default": "Describe this image"},
        },
        "required": ["path"],
    },
}


class ToolExecutor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.call_count = 0

    def safe_read_path(self, path_str: str) -> Path:
        """Validate a path is safe to READ. Must be in target_dir or data_dir."""
        path = Path(path_str).expanduser().resolve()
        target = self.config.target_dir.resolve()
        data = self.config.data_dir.resolve()

        for blocked in self.config.blocked_paths:
            if str(path).startswith(str(Path(blocked).resolve())):
                raise PermissionError(f"Blocked path: {path}")

        if not (str(path).startswith(str(target)) or str(path).startswith(str(data))):
            raise PermissionError(
                f"Path {path} outside allowed dirs ({target}, {data})"
            )
        return path

    def safe_write_path(self, path_str: str) -> Path:
        """Validate a path is safe to WRITE. Must be in data_dir ONLY."""
        path = Path(path_str).expanduser().resolve()
        data = self.config.data_dir.resolve()

        if not str(path).startswith(str(data)):
            raise PermissionError(f"Write path {path} outside data dir ({data})")
        return path

    def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a tool call and return result."""
        self.call_count += 1
        if self.call_count > self.config.max_tool_calls:
            return {"success": False, "error": "Max tool calls exceeded"}

        logger.info(f"Tool call #{self.call_count}: {tool_name}({json.dumps(args)[:200]})")

        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if not handler:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
            result = handler(args)
            return {"success": True, "output": result}
        except PermissionError as e:
            return {"success": False, "error": f"Permission denied: {e}"}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

    # Commands that are never allowed (destructive / dangerous)
    BLOCKED_COMMANDS = {
        "rm", "rmdir", "mkfs", "dd", "shutdown", "reboot", "halt",
        "systemctl", "launchctl", "kill", "killall", "pkill",
        "chmod", "chown", "chgrp", "sudo", "su", "doas",
        "curl", "wget", "ssh", "scp", "sftp", "nc", "ncat",
        "python", "python3", "node", "ruby", "perl", "bash", "zsh", "sh",
        "pip", "npm", "brew", "apt", "yum", "dnf",
        "mv", "cp",  # block moves/copies to prevent overwriting source files
        "open",  # block opening apps
    }

    def _tool_shell_exec(self, args: dict) -> str:
        """Execute a shell command within the target directory.

        Full read access to the target directory. No destructive commands.
        No network access. No writes outside data/.
        """
        command = args["command"].strip()
        timeout = min(args.get("timeout", 30), 60)  # cap at 60s

        if not command:
            return "<error>Empty command</error>"

        # Block dangerous commands by checking the first token of each
        # pipe segment and any $() or `` subshells
        try:
            # Check for subshell attempts
            if "`" in command or "$(" in command:
                return "<error>Subshell execution not allowed</error>"

            # Check each pipe segment
            segments = command.split("|")
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                # Also split on && and ;
                for sub in re.split(r'[;&]', segment):
                    sub = sub.strip()
                    if not sub:
                        continue
                    first_token = sub.split()[0] if sub.split() else ""
                    # Strip path prefix (e.g. /usr/bin/rm -> rm)
                    base_cmd = os.path.basename(first_token)
                    if base_cmd in self.BLOCKED_COMMANDS:
                        return f"<error>Blocked command: {base_cmd}</error>"
        except Exception:
            return "<error>Could not parse command</error>"

        # Validate working directory is target_dir
        cwd = str(self.config.target_dir.resolve())

        # Check cwd isn't in blocked paths
        for blocked in self.config.blocked_paths:
            if cwd.startswith(str(Path(blocked).resolve())):
                return f"<error>Target directory is in a blocked path</error>"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={
                    **os.environ,
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "HOME": str(Path.home()),
                },
            )
            stdout = result.stdout[:4000] if result.stdout else ""
            stderr = result.stderr[:1000] if result.stderr else ""

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"STDERR: {stderr}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")

            output = "\n".join(output_parts) if output_parts else "(no output)"
            return f"<shell>{xml_escape(output)}</shell>"

        except subprocess.TimeoutExpired:
            return f"<error>Command timed out after {timeout}s</error>"
        except Exception as e:
            return f"<error>{type(e).__name__}: {e}</error>"

    def _tool_index_directory(self, args: dict) -> str:
        path = self.safe_read_path(args.get("path", str(self.config.target_dir)))
        max_depth = args.get("max_depth", self.config.max_depth)
        entries = []
        self._walk_dir(path, entries, 0, max_depth)

        # Store raw data as JSON for internal use
        index_path = self.config.data_dir / "index.json"
        index_path.write_text(json.dumps(entries, indent=2, default=str))

        # Store XML version for LLM consumption
        xml_path = self.config.data_dir / "index.xml"
        xml_path.write_text(self._entries_to_xml(entries))

        return f"<result><indexed>{len(entries)}</indexed></result>"

    def _entries_to_xml(self, entries: list) -> str:
        """Convert file entries to compact XML format.
        Uses base path attribute to avoid repeating common prefix."""
        if not entries:
            return "<index/>"
        # Find common prefix to shorten paths
        paths = [e["path"] for e in entries]
        base = os.path.commonpath(paths) if len(paths) > 1 else str(Path(paths[0]).parent)
        base = base.rstrip("/")

        lines = [f'<index base="{xml_escape(base)}">']
        for e in entries:
            # Relative path from base
            rel = e["path"][len(base):].lstrip("/")
            tag = "d" if e["type"] == "dir" else "f"
            attrs = f'p="{xml_escape(rel)}" t="{e["type"]}" s="{e["size"]}"'
            if e.get("preview"):
                preview = xml_escape(e["preview"][:100].replace("\n", " "))
                lines.append(f"<{tag} {attrs}>{preview}</{tag}>")
            else:
                lines.append(f"<{tag} {attrs}/>")
        lines.append("</index>")
        return "\n".join(lines)

    def _walk_dir(self, directory: Path, entries: list, depth: int, max_depth: int):
        if depth > max_depth:
            return
        try:
            for item in sorted(directory.iterdir()):
                if item.name.startswith("."):
                    continue
                try:
                    blocked = any(
                        str(item.resolve()).startswith(str(Path(b).resolve()))
                        for b in self.config.blocked_paths
                    )
                    if blocked:
                        continue
                except (OSError, ValueError):
                    continue

                stat = item.stat()
                entry = {
                    "path": str(item),
                    "name": item.name,
                    "type": "dir" if item.is_dir() else item.suffix or "file",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }

                if item.is_file() and stat.st_size < self.config.max_file_size:
                    if item.suffix in (".txt", ".md", ".py", ".js", ".ts", ".json",
                                       ".csv", ".html", ".css", ".yaml", ".yml",
                                       ".toml", ".xml", ".sh", ".swift", ".rs",
                                       ".go", ".java", ".c", ".cpp", ".h", ".rb",
                                       ".log", ".cfg", ".ini", ".env.example"):
                        try:
                            text = item.read_text(errors="replace")
                            entry["preview"] = text[:self.config.preview_chars]
                        except Exception:
                            pass
                    elif item.suffix.lower() in self.IMAGE_EXTENSIONS:
                        try:
                            from PIL import Image
                            with Image.open(item) as img:
                                entry["preview"] = f"image {img.width}x{img.height} {img.mode}"
                        except Exception:
                            entry["preview"] = "image file"

                entries.append(entry)
                if item.is_dir():
                    self._walk_dir(item, entries, depth + 1, max_depth)
        except PermissionError:
            pass

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
                        ".webp", ".heic", ".heif", ".ico", ".svg"}

    def _tool_file_read(self, args: dict) -> str:
        path = self.safe_read_path(args["path"])
        if not path.is_file():
            return f"Not a file: {path}"
        if path.stat().st_size > self.config.max_file_size:
            return f"File too large: {path.stat().st_size} bytes"

        # Handle image files — return metadata; actual vision analysis happens in core
        if path.suffix.lower() in self.IMAGE_EXTENSIONS:
            return self._read_image_metadata(path)


        offset = args.get("offset", 0)
        limit = args.get("limit", 50)

        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception as e:
            return f"Cannot read: {e}"

        selected = lines[offset:offset + limit]
        numbered = [f"{i + offset + 1}: {line}" for i, line in enumerate(selected)]
        return "\n".join(numbered)

    def _read_image_metadata(self, path: Path) -> str:
        """Extract basic metadata from an image file."""
        stat = path.stat()
        info = {
            "file": path.name,
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

        # Try to get dimensions via PIL if available
        try:
            from PIL import Image
            with Image.open(path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["mode"] = img.mode
                if hasattr(img, "info"):
                    exif_keys = [k for k in img.info if isinstance(k, str)]
                    if exif_keys:
                        info["metadata_keys"] = exif_keys[:10]
        except ImportError:
            info["note"] = "Install Pillow for image dimensions"
        except Exception:
            pass

        lines = [f"<image path=\"{xml_escape(str(path))}\">"]
        for k, v in info.items():
            lines.append(f"  <{k}>{xml_escape(str(v))}</{k}>")
        lines.append("</image>")
        return "\n".join(lines)

    def _tool_glob_search(self, args: dict) -> str:
        directory = args.get("directory", str(self.config.target_dir))
        directory = self.safe_read_path(directory)
        pattern = args["pattern"]
        matches = []
        for p in directory.rglob(pattern):
            if len(matches) >= self.config.max_search_results:
                break
            try:
                blocked = any(
                    str(p.resolve()).startswith(str(Path(b).resolve()))
                    for b in self.config.blocked_paths
                )
                if blocked:
                    continue
            except (OSError, ValueError):
                continue
            matches.append(str(p))
        results = matches[:self.config.max_search_results]
        lines = ["<glob>"] + [f'<m p="{xml_escape(p)}"/>' for p in results] + ["</glob>"]
        return "\n".join(lines)

    def _tool_content_search(self, args: dict) -> str:
        directory = args.get("directory", str(self.config.target_dir))
        directory = self.safe_read_path(directory)
        pattern = args["pattern"]
        max_results = args.get("max_results", 20)
        regex = re.compile(pattern, re.IGNORECASE)
        results = []

        for p in directory.rglob("*"):
            if len(results) >= max_results:
                break
            if not p.is_file() or p.stat().st_size > self.config.max_file_size:
                continue
            if p.suffix not in (".txt", ".md", ".py", ".js", ".ts", ".json",
                                ".csv", ".html", ".css", ".yaml", ".yml",
                                ".toml", ".xml", ".sh", ".swift", ".rs",
                                ".go", ".java", ".c", ".cpp", ".h", ".rb",
                                ".log", ".cfg", ".ini"):
                continue
            try:
                for i, line in enumerate(p.read_text(errors="replace").splitlines()):
                    if regex.search(line):
                        results.append((str(p), i + 1, line.strip()[:200]))
                        if len(results) >= max_results:
                            break
            except Exception:
                continue

        lines = ["<search>"]
        for filepath, lineno, text in results:
            lines.append(f'<hit f="{xml_escape(filepath)}" l="{lineno}">{xml_escape(text)}</hit>')
        lines.append("</search>")
        return "\n".join(lines)

    def _tool_note_store(self, args: dict) -> str:
        key = args["key"].replace("/", "_").replace("..", "_")
        content = args["content"]
        category = args.get("category", "general")

        note_dir = self.config.data_dir / "notes"
        note_path = self.safe_write_path(str(note_dir / f"{key}.json"))

        note = {
            "key": key,
            "content": content,
            "category": category,
            "created": datetime.now().isoformat(),
        }
        note_path.write_text(json.dumps(note, indent=2))
        return f"<ok>stored {xml_escape(key)}</ok>"

    def _tool_note_recall(self, args: dict) -> str:
        query = args["query"].lower()
        limit = args.get("limit", 5)
        note_dir = self.config.data_dir / "notes"
        results = []

        for note_file in note_dir.glob("*.json"):
            try:
                note = json.loads(note_file.read_text())
                text = f"{note.get('key', '')} {note.get('content', '')} {note.get('category', '')}".lower()
                score = sum(1 for word in query.split() if word in text)
                if score > 0:
                    results.append((score, note))
            except Exception:
                continue

        results.sort(key=lambda x: x[0], reverse=True)
        lines = ["<notes>"]
        for _, note in results[:limit]:
            lines.append(
                f'<note k="{xml_escape(note.get("key", ""))}" '
                f'cat="{xml_escape(note.get("category", ""))}">'
                f'{xml_escape(note.get("content", ""))}</note>'
            )
        lines.append("</notes>")
        return "\n".join(lines)

    def _tool_read_index(self, args: dict) -> str:
        xml_path = self.config.data_dir / "index.xml"
        if xml_path.exists():
            return xml_path.read_text()
        # Fallback: build XML from JSON
        json_path = self.config.data_dir / "index.json"
        if not json_path.exists():
            return "<error>No index found. Run index_directory first.</error>"
        entries = json.loads(json_path.read_text())
        xml = self._entries_to_xml(entries)
        xml_path.write_text(xml)
        return xml

    def _tool_image_describe(self, args: dict) -> str:
        """Validate image path and return a VLM marker for the core loop."""
        path = self.safe_read_path(args["path"])
        if not path.is_file():
            return f"Not a file: {path}"
        if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
            return f"Not an image file: {path.name}"
        if path.stat().st_size > self.config.max_file_size:
            return f"File too large: {path.stat().st_size} bytes"

        question = args.get("question", "Describe this image in detail")
        # Return a JSON marker that core.py intercepts to make a VLM call
        return json.dumps({
            "__vlm_request__": True,
            "image_path": str(path),
            "question": question,
        })
