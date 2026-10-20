#!/usr/bin/env python3
"""
H2C Extract — Extract conversation skeletons from AI coding sessions.

Supports both Claude Code and Codex CLI session formats.
Reads .jsonl session files, extracts useful signal (user text,
assistant text, tool metadata), and outputs readable markdown
skeletons to ~/.hermes/inbox/ for Hermes consumption.

Commands:
    python3 h2c_extract.py sync              # Incremental sync (all sources)
    python3 h2c_extract.py sync --source cc  # Only Claude Code
    python3 h2c_extract.py sync --source codex  # Only Codex
    python3 h2c_extract.py sync --force      # Re-process all files
    python3 h2c_extract.py status            # Show sync status
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ── Constants ──

HERMES_DIR = Path.home() / ".hermes"
INBOX_DIR = HERMES_DIR / "inbox"
ARCHIVE_DIR = HERMES_DIR / "inbox-archive"
STATE_FILE = HERMES_DIR / "h2c-state.json"

CC_SESSIONS_DIR = Path.home() / ".claude" / "projects"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
CODEX_ARCHIVE_DIR = Path.home() / ".codex" / "archived_sessions"

MIN_MESSAGES = 3
STALE_SECONDS = 300
TEXT_KEEP_THRESHOLD = 500
TEXT_HEAD_CHARS = 300
TEXT_TAIL_CHARS = 200
MAX_SKELETON_CHARS = 20_000
SAVE_INTERVAL = 50
CODE_BLOCK_PLACEHOLDER = "[code: ~{n} lines]"

_USERNAME = getpass.getuser()
_PROJECT_PREFIXES = (
    f"-Users-{_USERNAME}-coding-",
    f"-Users-{_USERNAME}-",
)

# ── Sensitive Data Patterns ──

SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[a-zA-Z0-9_-]{20,}"), "[API_KEY_REDACTED]"),
    (re.compile(r"key-[a-zA-Z0-9_-]{20,}"), "[API_KEY_REDACTED]"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "[GITHUB_TOKEN_REDACTED]"),
    (re.compile(r"xox[bp]-[a-zA-Z0-9-]+"), "[SLACK_TOKEN_REDACTED]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[AWS_KEY_REDACTED]"),
    (re.compile(r"eyJ[a-zA-Z0-9_-]{50,}\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"), "[JWT_REDACTED]"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_.-]{20,}"), "[BEARER_TOKEN_REDACTED]"),
    (re.compile(r"(?i)password\s*[=:]\s*[\"']?[^\s\"']{8,}"), "[PASSWORD_REDACTED]"),
    (re.compile(r"(?i)secret\s*[=:]\s*[\"']?[^\s\"']{8,}"), "[SECRET_REDACTED]"),
]

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

# ── System Tag Stripping (Claude Code specific) ──

SYSTEM_TAGS = [
    "system-reminder", "command-message", "command-name",
    "observed_from_primary_session", "user-prompt-submit-hook",
    "task-notification", "task-id",
    "EXTREMELY_IMPORTANT", "SUBAGENT-STOP", "EXTREMELY-IMPORTANT",
    "local-command-caveat", "objective", "execution_context", "process",
]


def _build_system_tag_re() -> re.Pattern[str]:
    """Build regex that strips system tags, one pattern per tag."""
    patterns: list[str] = []
    for tag in SYSTEM_TAGS:
        patterns.append(rf"<{tag}\b[^>]*>[\s\S]*?</{tag}>")
        patterns.append(rf"<{tag}\b[^>]*/>")
        patterns.append(rf"<{tag}\b[^>]*>(?:(?!</).)*$")
    return re.compile("|".join(patterns), re.MULTILINE)


SYSTEM_TAG_RE = _build_system_tag_re()

# ── Codex System Tag Stripping ──

CODEX_SYSTEM_TAGS = [
    "permissions", "app-context", "environment_context",
    "collaboration_mode", "INSTRUCTIONS",
]


def _build_codex_tag_re() -> re.Pattern[str]:
    patterns: list[str] = []
    for tag in CODEX_SYSTEM_TAGS:
        patterns.append(rf"<{tag}\b[^>]*>[\s\S]*?</{tag}>")
        patterns.append(rf"<{tag}\b[^>]*/>")
    return re.compile("|".join(patterns), re.MULTILINE)


CODEX_TAG_RE = _build_codex_tag_re()

# ── Tagging ──

CODE_CHANGE_TOOLS = frozenset({"Edit", "Write", "MultiEdit"})
CODEX_CODE_CHANGE_TOOLS = frozenset({"write_file", "edit_file", "apply_diff"})
DEBUG_KEYWORDS = frozenset({"error", "bug", "fail", "fix", "crash", "traceback", "exception"})

LOW_VALUE_RE = re.compile(
    r"^(?:Unknown skill:|usage:?$|/help$|help$)",
    re.IGNORECASE,
)

# ── Boilerplate Prefixes (shared) ──

_BOILERPLATE_PREFIXES = (
    "Hello memory agent",
    "PROGRESS SUMMARY CHECKPOINT",
    "You are a Claude-Mem",
    "Continue the conversation from where",
    "This session is being continued",
    "Respond in this XML format",
    "IMPORTANT! DO NOT do any work",
    "Base directory for this skill:",
    "Tell your human partner that this command",
    "# AGENTS.md instructions for",
)

_BOILERPLATE_TAG_INDICATORS = (
    "<local-command-caveat>", "<command-message>", "<objective>",
    "<permissions instructions>", "<app-context>", "<environment_context>",
    "<collaboration_mode>",
)


# ── Data Models ──


@dataclass(frozen=True)
class ToolCall:
    name: str
    file_path: str | None


@dataclass(frozen=True)
class DialogTurn:
    role: str  # "user" | "assistant"
    text: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class SessionSkeleton:
    session_id: str
    date: str
    project: str
    source: str  # "cc" | "codex"
    tags: tuple[str, ...]
    files_touched: tuple[str, ...]
    turns: tuple[DialogTurn, ...]


# ── State Management ──


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"processed": {}}


def save_state(state: dict) -> None:
    HERMES_DIR.mkdir(parents=True, exist_ok=True)
    content = json.dumps(state, ensure_ascii=False, indent=2)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(STATE_FILE)


# ── Text Processing ──


def redact_secrets(text: str) -> str:
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def count_code_lines(code_block: str) -> int:
    lines = code_block.strip().split("\n")
    return max(len(lines) - 2, 1)


def truncate_text(text: str) -> str:
    if len(text) <= TEXT_KEEP_THRESHOLD:
        return text

    def replace_code(match: re.Match[str]) -> str:
        n = count_code_lines(match.group(0))
        return CODE_BLOCK_PLACEHOLDER.format(n=n)

    compressed = CODE_BLOCK_RE.sub(replace_code, text)
    if len(compressed) <= TEXT_KEEP_THRESHOLD:
        return compressed

    head = compressed[:TEXT_HEAD_CHARS]
    tail = compressed[-TEXT_TAIL_CHARS:]
    omitted = len(compressed) - TEXT_HEAD_CHARS - TEXT_TAIL_CHARS
    return f"{head}\n[...omitted {omitted} chars...]\n{tail}"


def is_boilerplate(text: str) -> bool:
    """Check if a message is system boilerplate (shared across sources)."""
    if any(text.startswith(p) for p in _BOILERPLATE_PREFIXES):
        return True
    head = text[:150]
    if any(tag in head for tag in _BOILERPLATE_TAG_INDICATORS):
        return True
    return False


def is_low_value_session(turns: list[DialogTurn]) -> bool:
    """Check if all user messages are low-value noise."""
    user_texts = [t.text for t in turns if t.role == "user"]
    if not user_texts:
        return True
    return all(LOW_VALUE_RE.match(t) for t in user_texts)


# ── Session Parser Protocol ──


class SessionParser(ABC):
    """Base class for parsing different AI CLI session formats."""

    source_name: str  # "cc" or "codex"
    assistant_label: str  # display label in skeleton output

    @abstractmethod
    def discover_files(self) -> list[Path]:
        """Find all session .jsonl files, sorted by mtime."""

    @abstractmethod
    def parse(self, filepath: Path) -> list[DialogTurn]:
        """Parse a session file into dialog turns."""

    @abstractmethod
    def get_session_id(self, filepath: Path) -> str:
        """Extract session ID from filepath."""

    @abstractmethod
    def get_project_name(self, filepath: Path) -> str:
        """Extract human-readable project name."""

    def get_session_date(self, filepath: Path) -> str:
        try:
            mtime = filepath.stat().st_mtime
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except OSError:
            return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


# ── Claude Code Parser ──


class ClaudeCodeParser(SessionParser):
    source_name = "cc"
    assistant_label = "CC"

    def discover_files(self) -> list[Path]:
        if not CC_SESSIONS_DIR.exists():
            return []
        return sorted(
            (
                p for p in CC_SESSIONS_DIR.rglob("*.jsonl")
                if "subagents" not in p.parts
            ),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        )

    def get_session_id(self, filepath: Path) -> str:
        return filepath.stem

    def get_project_name(self, filepath: Path) -> str:
        name = filepath.parent.name
        for prefix in _PROJECT_PREFIXES:
            name = name.replace(prefix, "")
        return "home" if name == "-" else name

    def parse(self, filepath: Path) -> list[DialogTurn]:
        turns: list[DialogTurn] = []
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type")
                    if msg_type not in ("user", "assistant"):
                        continue

                    msg = data.get("message", {})
                    if not isinstance(msg, dict):
                        continue

                    content = msg.get("content", "")

                    if msg_type == "user":
                        text = self._extract_text(content)
                        if not text or len(text) < 5:
                            continue
                        cleaned = self._sanitize(text)
                        if len(cleaned) < 5 or is_boilerplate(cleaned):
                            continue
                        turns.append(DialogTurn(role="user", text=cleaned))

                    elif msg_type == "assistant":
                        tool_calls = self._extract_tool_calls(content)
                        text = self._extract_text(content)
                        cleaned = self._sanitize(text) if text else ""
                        if cleaned or tool_calls:
                            turns.append(DialogTurn(
                                role="assistant",
                                text=cleaned,
                                tool_calls=tuple(tool_calls),
                            ))

        except (OSError, UnicodeDecodeError) as exc:
            print(f"  Warning: failed to read {filepath.name}: {exc}", file=sys.stderr)
        return turns

    def _extract_text(self, content) -> str | None:
        if isinstance(content, str):
            return content.strip() or None
        if not isinstance(content, list):
            return None
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts) if texts else None

    def _extract_tool_calls(self, content) -> list[ToolCall]:
        if not isinstance(content, list):
            return []
        calls = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "unknown")
            inputs = block.get("input", {})
            file_path = (
                inputs.get("file_path")
                or inputs.get("path")
                or (inputs.get("command", "")[:80] if name == "Bash" else None)
            )
            calls.append(ToolCall(name=name, file_path=file_path))
        return calls

    def _sanitize(self, text: str) -> str:
        text = SYSTEM_TAG_RE.sub("", text)
        text = redact_secrets(text)
        return text.strip()


# ── Codex Parser ──


class CodexParser(SessionParser):
    source_name = "codex"
    assistant_label = "Codex"

    def __init__(self) -> None:
        self._project_cache: dict[str, str] = {}

    def discover_files(self) -> list[Path]:
        files: list[Path] = []
        for directory in (CODEX_SESSIONS_DIR, CODEX_ARCHIVE_DIR):
            if directory.exists():
                files.extend(directory.rglob("*.jsonl"))
        return sorted(files, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)

    _UUID_RE = re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        re.IGNORECASE,
    )

    def get_session_id(self, filepath: Path) -> str:
        m = self._UUID_RE.search(filepath.stem)
        return m.group(0) if m else filepath.stem[:12]

    def get_project_name(self, filepath: Path) -> str:
        key = str(filepath)
        if key in self._project_cache:
            return self._project_cache[key]

        project = "unknown"
        try:
            with open(filepath, encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    if data.get("type") == "session_meta":
                        cwd = data.get("payload", {}).get("cwd", "")
                        if cwd:
                            project = Path(cwd).name
        except (OSError, json.JSONDecodeError):
            pass

        self._project_cache[key] = project
        return project

    def get_session_date(self, filepath: Path) -> str:
        # Try to extract date from filename first: rollout-2026-03-03T...
        name = filepath.stem
        match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if match:
            return match.group(1)
        return super().get_session_date(filepath)

    def parse(self, filepath: Path) -> list[DialogTurn]:
        turns: list[DialogTurn] = []
        try:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") != "response_item":
                        continue

                    payload = data.get("payload", {})
                    item_type = payload.get("type")
                    role = payload.get("role")

                    if item_type == "message" and role == "user":
                        text = self._extract_text(payload)
                        if not text or len(text) < 5:
                            continue
                        cleaned = self._sanitize(text)
                        if len(cleaned) < 5 or is_boilerplate(cleaned):
                            continue
                        turns.append(DialogTurn(role="user", text=cleaned))

                    elif item_type == "message" and role == "assistant":
                        text = self._extract_text(payload)
                        cleaned = self._sanitize(text) if text else ""
                        if cleaned:
                            turns.append(DialogTurn(
                                role="assistant", text=cleaned,
                            ))

                    elif item_type == "function_call":
                        tc = self._extract_function_call(payload)
                        if tc:
                            # Attach to last assistant turn, or create one
                            if turns and turns[-1].role == "assistant":
                                last = turns[-1]
                                turns[-1] = DialogTurn(
                                    role="assistant",
                                    text=last.text,
                                    tool_calls=last.tool_calls + (tc,),
                                )
                            else:
                                turns.append(DialogTurn(
                                    role="assistant", text="",
                                    tool_calls=(tc,),
                                ))

        except (OSError, UnicodeDecodeError) as exc:
            print(f"  Warning: failed to read {filepath.name}: {exc}", file=sys.stderr)
        return turns

    def _extract_text(self, payload: dict) -> str | None:
        content = payload.get("content", [])
        if isinstance(content, str):
            return content.strip() or None
        if not isinstance(content, list):
            return None
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("input_text", "output_text", "text"):
                text = block.get("text", "").strip()
                if text:
                    texts.append(text)
        return "\n".join(texts) if texts else None

    def _extract_function_call(self, payload: dict) -> ToolCall | None:
        name = payload.get("name", "").strip()
        if not name:
            return None
        args_raw = payload.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}

        file_path = None
        if name in ("exec_command", "shell"):
            file_path = str(args.get("cmd", ""))[:80] or None
        elif name in ("read_file", "write_file", "edit_file", "apply_diff"):
            file_path = args.get("path") or args.get("file_path")

        return ToolCall(name=name, file_path=file_path)

    def _sanitize(self, text: str) -> str:
        text = CODEX_TAG_RE.sub("", text)
        text = redact_secrets(text)
        return text.strip()


# ── Session Analysis (shared) ──


def compute_tags(turns: list[DialogTurn], source: str) -> list[str]:
    """Auto-tag a session based on its content."""
    tags: list[str] = []
    all_tool_names = {tc.name for t in turns for tc in t.tool_calls}
    all_text = " ".join(t.text.lower() for t in turns if t.text)

    change_tools = CODE_CHANGE_TOOLS if source == "cc" else CODEX_CODE_CHANGE_TOOLS
    if all_tool_names & change_tools:
        tags.append("code-change")

    if not all_tool_names:
        tags.append("discussion-only")

    if any(kw in all_text for kw in DEBUG_KEYWORDS):
        tags.append("debugging")

    agent_tools = {"Agent"} if source == "cc" else set()
    if all_tool_names & agent_tools:
        tags.append("multi-agent")

    return tags


def collect_files_touched(turns: list[DialogTurn], source: str) -> list[str]:
    """Collect unique file paths from tool calls."""
    skip_tools = {"Bash"} if source == "cc" else {"exec_command", "shell"}
    seen: set[str] = set()
    files: list[str] = []
    for turn in turns:
        for tc in turn.tool_calls:
            if tc.file_path and tc.name not in skip_tools and tc.file_path not in seen:
                seen.add(tc.file_path)
                files.append(tc.file_path)
    return files


def build_skeleton(
    parser: SessionParser,
    filepath: Path,
    turns: list[DialogTurn],
) -> SessionSkeleton | None:
    """Build a session skeleton from parsed turns."""
    user_turns = [t for t in turns if t.role == "user"]
    if len(user_turns) < MIN_MESSAGES:
        return None
    if is_low_value_session(turns):
        return None

    session_id = parser.get_session_id(filepath)
    date = parser.get_session_date(filepath)
    project = parser.get_project_name(filepath)
    tags = compute_tags(turns, parser.source_name)
    files_touched = collect_files_touched(turns, parser.source_name)

    return SessionSkeleton(
        session_id=session_id,
        date=date,
        project=project,
        source=parser.source_name,
        tags=tuple(tags),
        files_touched=tuple(files_touched),
        turns=tuple(turns),
    )


# ── Markdown Output ──


_SOURCE_LABELS = {"cc": "CC", "codex": "Codex"}


def format_skeleton(skeleton: SessionSkeleton) -> str:
    """Format a session skeleton as readable markdown."""
    assistant_label = _SOURCE_LABELS.get(skeleton.source, skeleton.source)

    lines: list[str] = [
        "---",
        f"session: {skeleton.session_id[:8]}",
        f"date: {skeleton.date}",
        f"project: {skeleton.project}",
        f"source: {skeleton.source}",
        f"tags: [{', '.join(skeleton.tags)}]",
    ]

    if skeleton.files_touched:
        shown = skeleton.files_touched[:20]
        files_str = ", ".join(shown)
        if len(skeleton.files_touched) > 20:
            files_str += f", +{len(skeleton.files_touched) - 20} more"
        lines.append(f"files_touched: [{files_str}]")
    else:
        lines.append("files_touched: []")

    lines.extend(["---", ""])

    for turn in skeleton.turns:
        if turn.role == "user":
            text = truncate_text(turn.text)
            lines.append(f"**User**: {text}")
            lines.append("")
        elif turn.role == "assistant":
            if turn.text:
                text = truncate_text(turn.text)
                lines.append(f"**{assistant_label}**: {text}")
                lines.append("")
            if turn.tool_calls:
                tool_summary = _summarize_tool_calls(turn.tool_calls)
                if tool_summary:
                    lines.append(f"*[Tools: {tool_summary}]*")
                    lines.append("")

    return "\n".join(lines)


def _summarize_tool_calls(calls: tuple[ToolCall, ...]) -> str:
    """Summarize tool calls into a compact string."""
    groups: dict[str, list[str | None]] = defaultdict(list)
    for tc in calls:
        groups[tc.name].append(tc.file_path)

    shell_tools = {"Bash", "exec_command", "shell"}
    parts: list[str] = []
    for name, paths in groups.items():
        valid_paths = [p for p in paths if p]
        if valid_paths and name not in shell_tools:
            short_paths = [Path(p).name for p in valid_paths[:3]]
            suffix = f"+{len(valid_paths) - 3}" if len(valid_paths) > 3 else ""
            parts.append(f"{name}({', '.join(short_paths)}{suffix})")
        else:
            parts.append(f"{name} x{len(paths)}")
    return ", ".join(parts)


def _trim_skeleton_content(content: str) -> str:
    """Trim skeleton content to fit within MAX_SKELETON_CHARS."""
    if len(content) <= MAX_SKELETON_CHARS:
        return content

    lines = content.split("\n")

    fm_end = 0
    dashes_seen = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            dashes_seen += 1
            if dashes_seen == 2:
                fm_end = i + 1
                break

    if dashes_seen < 2:
        return content[:MAX_SKELETON_CHARS] + "\n*[...trimmed]*"

    header = "\n".join(lines[:fm_end])
    body_lines = lines[fm_end:]

    budget = MAX_SKELETON_CHARS - len(header) - 100
    if budget <= 0:
        return header[:MAX_SKELETON_CHARS]

    # Keep lines from the end (most recent context is most valuable)
    trimmed: list[str] = []
    for line in reversed(body_lines):
        if budget - len(line) - 1 < 0:
            break
        trimmed.append(line)
        budget -= len(line) + 1

    trimmed.reverse()
    trim_notice = f"\n*[...trimmed: kept last {len(trimmed)}/{len(body_lines)} lines]*\n"
    return header + trim_notice + "\n".join(trimmed)


def write_skeleton(skeleton: SessionSkeleton) -> Path | None:
    """Write a skeleton to the inbox directory."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "cx" if skeleton.source == "codex" else "cc"
    filename = f"{skeleton.date}_{prefix}_{skeleton.session_id[:8]}.md"
    filepath = INBOX_DIR / filename
    content = format_skeleton(skeleton)
    content = _trim_skeleton_content(content)

    tmp = filepath.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(filepath)
        return filepath
    except OSError as exc:
        print(f"  Warning: failed to write {filename}: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return None


# ── Session Discovery ──


def find_pending_sessions(
    parser: SessionParser, state: dict,
) -> list[Path]:
    """Find sessions that haven't been processed yet."""
    processed = state.get("processed", {})
    now = datetime.now(tz=timezone.utc).timestamp()
    pending: list[Path] = []

    for filepath in parser.discover_files():
        key = str(filepath)
        try:
            mtime = filepath.stat().st_mtime
        except OSError:
            continue

        if now - mtime < STALE_SECONDS:
            continue

        if processed.get(key) == str(mtime):
            continue

        pending.append(filepath)

    return pending


# ── Commands ──


def _get_parsers(source: str | None) -> list[SessionParser]:
    """Get parser(s) based on source filter."""
    all_parsers: list[SessionParser] = [ClaudeCodeParser(), CodexParser()]
    if source is None:
        return all_parsers
    matched = [p for p in all_parsers if p.source_name == source]
    if not matched:
        print(f"Error: unknown source '{source}'", file=sys.stderr)
        sys.exit(1)
    return matched


def cmd_sync(args: argparse.Namespace) -> None:
    state = load_state()

    if args.force:
        state = {"processed": {}}
        print("Force mode: re-processing all sessions.")

    parsers = _get_parsers(args.source)
    total_written = 0
    total_skipped = 0

    for parser in parsers:
        pending = find_pending_sessions(parser, state)
        if not pending:
            print(f"[{parser.source_name}] No new sessions.")
            continue

        print(f"[{parser.source_name}] Found {len(pending)} new session(s).")
        written = 0
        skipped = 0

        for i, filepath in enumerate(pending):
            turns = parser.parse(filepath)
            skeleton = build_skeleton(parser, filepath, turns)

            if skeleton is None:
                skipped += 1
            else:
                out_path = write_skeleton(skeleton)
                if out_path is not None:
                    written += 1
                    tag_str = ", ".join(skeleton.tags) if skeleton.tags else "none"
                    print(f"  + {out_path.name} [{tag_str}]")

            try:
                mtime = str(filepath.stat().st_mtime)
            except OSError:
                mtime = "deleted"
            state.setdefault("processed", {})[str(filepath)] = mtime

            if (i + 1) % SAVE_INTERVAL == 0:
                save_state(state)

        save_state(state)
        print(f"[{parser.source_name}] Done: {written} written, {skipped} skipped.")
        total_written += written
        total_skipped += skipped

    print(f"\nTotal: {total_written} skeleton(s) written, {total_skipped} skipped.")


def cmd_status(args: argparse.Namespace) -> None:
    state = load_state()
    processed_count = len(state.get("processed", {}))

    parsers = _get_parsers(None)
    total_files = 0
    total_pending = 0

    for parser in parsers:
        files = parser.discover_files()
        pending = find_pending_sessions(parser, state)
        total_files += len(files)
        total_pending += len(pending)
        print(f"[{parser.source_name}] Sessions: {len(files)}, Pending: {len(pending)}")

    inbox_count = len(list(INBOX_DIR.glob("*.md"))) if INBOX_DIR.exists() else 0
    archive_count = len(list(ARCHIVE_DIR.glob("*.md"))) if ARCHIVE_DIR.exists() else 0

    print(f"\nOverall:")
    print(f"  Total sessions:     {total_files}")
    print(f"  Processed:          {processed_count}")
    print(f"  Pending:            {total_pending}")
    print(f"  Inbox skeletons:    {inbox_count}")
    print(f"  Archived:           {archive_count}")
    print(f"  State file:         {STATE_FILE}")
    print(f"  Inbox dir:          {INBOX_DIR}")


# ── Entry Point ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2C Extract — conversation skeletons from AI coding sessions",
    )
    sub = parser.add_subparsers(dest="command")

    sync_parser = sub.add_parser("sync", help="Incremental sync new sessions")
    sync_parser.add_argument("--force", action="store_true", help="Re-process all")
    sync_parser.add_argument(
        "--source", choices=["cc", "codex"], default=None,
        help="Only sync from specific source (default: all)",
    )

    sub.add_parser("status", help="Show sync status")

    args = parser.parse_args()

    commands = {
        "sync": cmd_sync,
        "status": cmd_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
