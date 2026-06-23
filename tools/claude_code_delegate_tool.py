"""Claude Code cascade delegation tool for Hermes.

Spawns a Claude Code CLI subprocess (``claude -p``) that inherits the
operator's OAuth credentials and native tool/MCP/skill surface.  The main
Hermes agent exposes this meta-tool on its direct Anthropic OAuth path; the
subprocess does the real tool-heavy work and bills as a first-party Claude
Code request.

Why this exists
---------------
Anthropic's OAuth subscription classifies some request shapes as "third-party
extra usage" based on wire-format signals (tool name prefixes, system prompt
size, total tool-schema bulk).  When the extra-usage bucket is empty the
request fails with HTTP 400.  Claude Code CLI requests are classified as
first-party regardless of payload shape.  This tool routes around the billing
classifier by delegating tool-heavy turns to a CLI subprocess.

The tool is intentionally small: Hermes' direct OAuth request only needs
enough surface to ask Claude Code to do the real work.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home, get_skills_dir
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 900
DEFAULT_BUDGET_USD = 1.0
MAX_BUDGET_USD = 10.0
DEFAULT_MODEL = "sonnet"
MAX_RESULT_CHARS = 6000


def _get_log_root() -> Path:
    return get_hermes_home() / "claude_delegate_logs"


DELEGATE_TO_CLAUDE_CODE_SCHEMA = {
    "name": "delegate_to_claude_code",
    "description": (
        "Delegate a tool-heavy request to a Claude Code subprocess with the "
        "operator's full Claude Code tool surface, MCP servers, skills, and "
        "OAuth subscription billing. Use this whenever the user asks to browse "
        "the web, take a screenshot, run shell commands, inspect or write "
        "files, or do any work that needs external tools. "
        "DO NOT delegate for: "
        "(a) self-introspection ('what voices/skills/model do you have'), "
        "(b) self-configuration ('switch voice', 'enable/disable X'), "
        "(c) requests that map to a slash command "
        "(/tts-voice, /skills, /voice, /commands). "
        "DO NOT delegate for simple chat or casual responses."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The complete task for Claude Code. Include the user's exact "
                    "request and any relevant conversation context."
                ),
            },
            "timeout_s": {
                "type": "integer",
                "description": (
                    "Hard wall-clock timeout in seconds. "
                    f"Default {DEFAULT_TIMEOUT_SECONDS}; maximum {MAX_TIMEOUT_SECONDS}."
                ),
                "default": DEFAULT_TIMEOUT_SECONDS,
                "minimum": 5,
                "maximum": MAX_TIMEOUT_SECONDS,
            },
            "max_budget_usd": {
                "type": "number",
                "description": (
                    "Claude Code --max-budget-usd cap. For subscription OAuth this "
                    "is an API-equivalent guardrail, not API-key spend. Default 1.0."
                ),
                "default": DEFAULT_BUDGET_USD,
                "minimum": 0.01,
                "maximum": MAX_BUDGET_USD,
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Optional working directory for Claude Code. Defaults to the "
                    "operator's home directory."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Claude Code model alias or full model id. Default: sonnet."
                ),
                "default": DEFAULT_MODEL,
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
}


def _get_claude_bin() -> str:
    return os.environ.get("HERMES_CLAUDE_CODE_BIN") or "claude"


def check_claude_code_available() -> bool:
    """Return True if the Claude Code CLI is installed and on PATH."""
    return shutil.which(_get_claude_bin()) is not None


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(maximum, coerced))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        coerced = default
    return max(minimum, min(maximum, coerced))


def _resolve_cwd(raw_cwd: Any) -> Path:
    if isinstance(raw_cwd, str) and raw_cwd.strip():
        candidate = Path(raw_cwd).expanduser()
    else:
        candidate = Path.home()
    try:
        candidate = candidate.resolve()
    except OSError:
        candidate = Path.home()
    if not candidate.exists() or not candidate.is_dir():
        return Path.home()
    return candidate


def _existing_add_dirs() -> list[str]:
    """Return ``--add-dir`` paths that actually exist on this host.

    These give the subprocess visibility into Hermes' skill registry.
    Paths that don't exist are silently skipped so the tool works on
    fresh installs without special setup.
    """
    candidates = [
        get_skills_dir(),       # ~/.hermes/skills (canonical)
        # Do NOT add get_hermes_home() -- it contains config.yaml with
        # secrets that a model-controlled prompt could exfiltrate.
    ]
    # Additional user-configured directories via env var (colon-separated).
    extra = os.environ.get("HERMES_DELEGATE_ADD_DIRS", "").strip()
    if extra:
        for raw in extra.split(":"):
            raw = raw.strip()
            if raw:
                candidates.append(Path(raw).expanduser())

    dirs: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except OSError:
            continue
        if path.is_dir() and resolved not in seen:
            dirs.append(resolved)
            seen.add(resolved)
    return dirs


def _wrap_prompt(prompt: str) -> str:
    """Wrap the user's task in runtime context for the Claude Code subprocess."""
    skills_dir = get_skills_dir()
    lines = [
        "You are Claude Code running as Hermes' delegated tool executor.",
        "Complete the task end-to-end and return a concise result.",
        "",
        "# Hermes Runtime Context",
        f"- Hermes skills directory: {skills_dir}",
        "- For browser/screenshot requests, use Claude Code's native browser "
        "or shell-accessible screenshot tooling.",
        "- Do not ask Hermes to do another tool call. You are the tool-capable "
        "worker.",
        "",
        "# User Task",
        prompt.strip(),
        "",
        "# Response Contract",
        "Return only the useful result. Include file paths for created notes, "
        "screenshots, or other artifacts.",
    ]
    return "\n".join(lines)


def _command_for(
    prompt: str,
    max_budget_usd: float,
    model: str,
) -> list[str]:
    claude = _get_claude_bin()
    skip_permissions = os.environ.get(
        "HERMES_DELEGATE_SKIP_PERMISSIONS", "0",
    ).lower() in ("1", "true", "yes", "on")

    cmd = [
        claude,
        "-p", prompt,
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--no-session-persistence",
        "--max-budget-usd", f"{max_budget_usd:.4f}",
        "--model", model or DEFAULT_MODEL,
    ]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    for add_dir in _existing_add_dirs():
        cmd.extend(["--add-dir", add_dir])
    return cmd


def _parse_stream_json(raw_stdout: str) -> Dict[str, Any]:
    """Parse Claude Code's stream-json output into a structured result."""
    final_result: Dict[str, Any] | None = None
    init_event: Dict[str, Any] | None = None
    tool_calls: list[Dict[str, Any]] = []
    text_fragments: list[str] = []
    parse_errors = 0
    bad_lines: list[str] = []

    for line in raw_stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            parse_errors += 1
            if len(bad_lines) < 5:
                bad_lines.append(line[:200])
            continue

        etype = event.get("type")
        if etype == "system" and event.get("subtype") == "init":
            init_event = event
        elif etype == "assistant":
            message = event.get("message") or {}
            content = message.get("content") or []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_use":
                        tool_calls.append({
                            "name": block.get("name"),
                            "input": block.get("input") or {},
                        })
                    elif block.get("type") == "text" and block.get("text"):
                        text_fragments.append(str(block["text"]))
        elif etype == "result":
            final_result = event

    return {
        "init": init_event or {},
        "final": final_result or {},
        "tool_calls": tool_calls,
        "text": "\n".join(t for t in text_fragments if t).strip(),
        "parse_errors": parse_errors,
        "bad_lines": bad_lines,
    }


def _redacted_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    if "-p" in redacted:
        idx = redacted.index("-p")
        if idx + 1 < len(redacted):
            redacted[idx + 1] = "<prompt>"
    return redacted


def delegate_to_claude_code(args: dict, **kwargs) -> str:
    """Execute a Claude Code subprocess and return the structured result."""
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return tool_error("prompt is required")

    timeout_s = _as_int(
        args.get("timeout_s"), DEFAULT_TIMEOUT_SECONDS, 5, MAX_TIMEOUT_SECONDS,
    )
    max_budget_usd = _as_float(
        args.get("max_budget_usd"), DEFAULT_BUDGET_USD, 0.01, MAX_BUDGET_USD,
    )
    model = str(args.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    cwd = _resolve_cwd(args.get("cwd"))
    wrapped_prompt = _wrap_prompt(prompt)

    log_root = _get_log_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = log_root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.txt").write_text(wrapped_prompt, encoding="utf-8")

    cmd = _command_for(wrapped_prompt, max_budget_usd, model)
    (log_dir / "command.json").write_text(
        json.dumps(_redacted_command(cmd), indent=2), encoding="utf-8",
    )

    # Inherit the full parent environment so the subprocess can access the
    # operator's OAuth credentials and tool configuration.
    env = os.environ.copy()
    # Ensure node (required by Claude Code) is on PATH even if the gateway
    # process was started with a minimal environment.
    node_bin = str(get_hermes_home() / "node" / "bin")
    env["PATH"] = node_bin + os.pathsep + env.get("PATH", "")

    stdout_path = log_dir / "stdout.jsonl"
    stderr_path = log_dir / "stderr.log"
    started = time.monotonic()
    timed_out = False

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            raw_stdout, raw_stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            raw_stdout, raw_stderr = proc.communicate(timeout=10)
        exit_code = proc.returncode
    except FileNotFoundError:
        result: Dict[str, Any] = {
            "success": False,
            "status": "missing_claude_cli",
            "error": (
                "Claude Code CLI not found on PATH; "
                "install @anthropic-ai/claude-code or set HERMES_CLAUDE_CODE_BIN."
            ),
            "log_dir": str(log_dir),
        }
        (log_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8",
        )
        return json.dumps(result, ensure_ascii=False)

    duration_s = round(time.monotonic() - started, 2)
    stdout_path.write_text(raw_stdout or "", encoding="utf-8")
    stderr_path.write_text(raw_stderr or "", encoding="utf-8")

    parsed = _parse_stream_json(raw_stdout or "")
    final = parsed.get("final") or {}
    final_text = str(final.get("result") or parsed.get("text") or "").strip()
    truncated = len(final_text) > MAX_RESULT_CHARS
    if truncated:
        final_text = final_text[:MAX_RESULT_CHARS]
    status = "timeout" if timed_out else "completed"
    if not timed_out and exit_code != 0:
        status = "error"
    if final.get("is_error"):
        status = "error"

    result = {
        "success": status == "completed",
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": duration_s,
        "model_requested": model,
        "model_used": (parsed.get("init") or {}).get("model"),
        "session_id": (
            final.get("session_id")
            or (parsed.get("init") or {}).get("session_id")
        ),
        "num_turns": final.get("num_turns"),
        "total_cost_usd": final.get("total_cost_usd"),
        "result": final_text,
        "truncated": truncated,
        "tool_calls": (parsed.get("tool_calls") or [])[:20],
        "log_dir": str(log_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_tail": (raw_stderr or "")[-2000:],
        "parse_errors": parsed.get("parse_errors", 0),
        "bad_lines": parsed.get("bad_lines", []),
    }
    (log_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    logger.info(
        "delegate_to_claude_code: status=%s exit=%s duration=%.1fs model=%s cost=%s",
        status, exit_code, duration_s, model,
        final.get("total_cost_usd", "?"),
    )
    return json.dumps(result, ensure_ascii=False)


registry.register(
    name="delegate_to_claude_code",
    toolset="claude_code_delegation",
    schema=DELEGATE_TO_CLAUDE_CODE_SCHEMA,
    handler=delegate_to_claude_code,
    description=DELEGATE_TO_CLAUDE_CODE_SCHEMA["description"],
    emoji="\u2197",  # ↗
    check_fn=check_claude_code_available,
)
