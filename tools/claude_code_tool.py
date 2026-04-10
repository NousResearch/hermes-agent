#!/usr/bin/env python3
"""
Claude Code Tool -- Delegate tasks to Claude Code CLI.

Allows Hermes to offload complex tasks (deep code analysis, refactoring,
multi-file edits, architecture review) to Claude Code via its print mode
(``claude -p``).  This uses the user's existing Claude subscription --
no API key or extra usage credits required.

Features:
  - Session persistence: each Hermes session gets a Claude session that
    can be resumed across calls (``--resume <session_id>``).
  - JSON output: structured results with session_id, cost, duration.
  - Streaming support: real-time output via ``--output-format stream-json``.

Requirements:
  - Claude Code CLI installed and authenticated (``claude`` on PATH)
"""

import json
import logging
import os
import shutil
import subprocess
import threading
from typing import Dict, Any, Optional, Callable

from tools.registry import registry

logger = logging.getLogger(__name__)

TOOL_NAME = "claude_code"
TOOLSET = "claude_code"

DEFAULT_TIMEOUT = 300  # 5 minutes max
MAX_PROMPT_LENGTH = 100_000
MAX_OUTPUT_LENGTH = 50_000

# Session store: maps hermes session key -> claude session_id
_session_store: Dict[str, str] = {}
_session_lock = threading.Lock()


def check_claude_code_available() -> bool:
    """Return True when the ``claude`` CLI binary is on PATH."""
    return shutil.which("claude") is not None


def get_claude_session(hermes_session_key: str) -> Optional[str]:
    """Get the Claude session ID for a Hermes session."""
    with _session_lock:
        return _session_store.get(hermes_session_key)


def set_claude_session(hermes_session_key: str, claude_session_id: str) -> None:
    """Store the Claude session ID for a Hermes session."""
    with _session_lock:
        _session_store[hermes_session_key] = claude_session_id


def clear_claude_session(hermes_session_key: str) -> None:
    """Clear the Claude session for a Hermes session."""
    with _session_lock:
        _session_store.pop(hermes_session_key, None)


def _build_claude_command(
    prompt: str,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    session_id: Optional[str] = None,
    output_format: str = "json",
) -> list:
    """Build the claude CLI command list."""
    cmd = ["claude", "-p"]

    if model:
        cmd.extend(["--model", model])

    if max_turns and max_turns > 0:
        cmd.extend(["--max-turns", str(max_turns)])

    if session_id:
        cmd.extend(["--resume", session_id])

    cmd.extend(["--output-format", output_format])

    cmd.append(prompt)
    return cmd


def _parse_json_output(raw_output: str) -> Dict[str, Any]:
    """Parse Claude CLI JSON output and extract result + session_id."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return {"result_text": raw_output.strip(), "session_id": None, "cost_usd": None}

    # JSON output is a list of events
    if isinstance(data, list):
        result_text = ""
        session_id = None
        cost_usd = None
        duration_ms = None

        for event in data:
            event_type = event.get("type", "")

            if event_type == "system" and event.get("subtype") == "init":
                session_id = event.get("session_id")

            elif event_type == "assistant":
                msg = event.get("message", {})
                for content in msg.get("content", []):
                    if content.get("type") == "text":
                        result_text += content.get("text", "")

            elif event_type == "result":
                result_text = event.get("result", result_text)
                session_id = session_id or event.get("session_id")
                cost_usd = event.get("total_cost_usd")
                duration_ms = event.get("duration_ms")

        return {
            "result_text": result_text.strip(),
            "session_id": session_id,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
        }

    return {"result_text": str(data), "session_id": None, "cost_usd": None}


def claude_code(
    prompt: str,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a task via Claude Code CLI in print mode.

    Args:
        prompt:      The task description / prompt for Claude.
        model:       Optional model override (e.g. "sonnet", "opus").
        max_turns:   Max agentic turns Claude can take.
        cwd:         Working directory for Claude (defaults to current dir).
        timeout:     Max seconds to wait (default: 300).
        session_key: Hermes session key for session persistence.

    Returns:
        Dict with "success", "output", "session_id", "cost_usd", and optional "error" keys.
    """
    if not prompt or not prompt.strip():
        return {"success": False, "output": "", "error": "Prompt cannot be empty."}

    if len(prompt) > MAX_PROMPT_LENGTH:
        return {
            "success": False,
            "output": "",
            "error": f"Prompt too long ({len(prompt)} chars, max {MAX_PROMPT_LENGTH}).",
        }

    if not check_claude_code_available():
        return {
            "success": False,
            "output": "",
            "error": "Claude Code CLI not found. Install it: npm install -g @anthropic-ai/claude-code",
        }

    effective_timeout = min(timeout or DEFAULT_TIMEOUT, 600)
    work_dir = os.path.realpath(cwd) if cwd else os.getcwd()

    if not os.path.isdir(work_dir):
        return {
            "success": False,
            "output": "",
            "error": f"Working directory does not exist: {work_dir}",
        }

    # Resume existing Claude session if available
    resume_session_id = get_claude_session(session_key) if session_key else None

    cmd = _build_claude_command(
        prompt,
        model=model,
        max_turns=max_turns,
        session_id=resume_session_id,
        output_format="json",
    )

    logger.info(
        "Running Claude Code: cwd=%s, model=%s, resume=%s, timeout=%ds",
        work_dir, model or "default", resume_session_id or "new", effective_timeout,
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=work_dir,
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONINTERACTIVE_HINT": "1"},
        )

        stderr = result.stderr.strip()

        if result.returncode != 0:
            error_detail = stderr or f"Exit code {result.returncode}"
            logger.warning("Claude Code returned non-zero: %s", error_detail)
            return {
                "success": False,
                "output": "",
                "error": f"Claude Code failed: {error_detail}",
            }

        parsed = _parse_json_output(result.stdout)
        output = parsed["result_text"]

        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n\n[Output truncated]"

        # Persist session for future calls
        if session_key and parsed.get("session_id"):
            set_claude_session(session_key, parsed["session_id"])

        logger.info(
            "Claude Code completed: %d chars, session=%s, cost=$%.4f",
            len(output),
            parsed.get("session_id", "?"),
            parsed.get("cost_usd") or 0,
        )

        return {
            "success": True,
            "output": output,
            "session_id": parsed.get("session_id"),
            "cost_usd": parsed.get("cost_usd"),
            "duration_ms": parsed.get("duration_ms"),
        }

    except subprocess.TimeoutExpired:
        logger.error("Claude Code timed out after %ds", effective_timeout)
        return {
            "success": False,
            "output": "",
            "error": f"Claude Code timed out after {effective_timeout} seconds.",
        }
    except Exception as e:
        logger.error("Claude Code execution error: %s", e, exc_info=True)
        return {"success": False, "output": "", "error": f"Execution error: {e}"}


def claude_code_streaming(
    prompt: str,
    on_text: Callable[[str], None],
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    session_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Claude Code with streaming output.

    Streams partial results via the ``on_text`` callback as they arrive.

    Args:
        prompt:      The task description for Claude.
        on_text:     Callback invoked with each text chunk.
        model:       Optional model override.
        max_turns:   Max agentic turns.
        cwd:         Working directory.
        timeout:     Max seconds to wait.
        session_key: Hermes session key for session persistence.

    Returns:
        Dict with "success", "output", "session_id", "cost_usd", and optional "error" keys.
    """
    if not prompt or not prompt.strip():
        return {"success": False, "output": "", "error": "Prompt cannot be empty."}

    if not check_claude_code_available():
        return {
            "success": False,
            "output": "",
            "error": "Claude Code CLI not found.",
        }

    effective_timeout = min(timeout or DEFAULT_TIMEOUT, 600)
    work_dir = os.path.realpath(cwd) if cwd else os.getcwd()

    if not os.path.isdir(work_dir):
        return {"success": False, "output": "", "error": f"Directory not found: {work_dir}"}

    resume_session_id = get_claude_session(session_key) if session_key else None

    cmd = _build_claude_command(
        prompt,
        model=model,
        max_turns=max_turns,
        session_id=resume_session_id,
        output_format="stream-json",
    )
    cmd.insert(2, "--include-partial-messages")

    logger.info("Running Claude Code (streaming): cwd=%s, resume=%s", work_dir, resume_session_id or "new")

    collected_text = []
    session_id = None
    cost_usd = None
    duration_ms = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONINTERACTIVE_HINT": "1"},
        )

        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "system" and event.get("subtype") == "init":
                    session_id = event.get("session_id")

                elif event_type == "assistant":
                    msg = event.get("message", {})
                    for content in msg.get("content", []):
                        if content.get("type") == "text":
                            text = content.get("text", "")
                            if text:
                                collected_text.append(text)
                                on_text(text)

                elif event_type == "result":
                    session_id = session_id or event.get("session_id")
                    cost_usd = event.get("total_cost_usd")
                    duration_ms = event.get("duration_ms")
                    final_result = event.get("result", "")
                    if final_result and not collected_text:
                        collected_text.append(final_result)

            proc.wait(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"success": False, "output": "", "error": f"Timed out after {effective_timeout}s."}

        if session_key and session_id:
            set_claude_session(session_key, session_id)

        full_output = "".join(collected_text).strip()

        if proc.returncode != 0:
            stderr = proc.stderr.read().strip() if proc.stderr else ""
            return {
                "success": False,
                "output": full_output,
                "error": f"Exit code {proc.returncode}: {stderr}",
            }

        return {
            "success": True,
            "output": full_output,
            "session_id": session_id,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
        }

    except Exception as e:
        logger.error("Claude Code streaming error: %s", e, exc_info=True)
        return {"success": False, "output": "", "error": f"Execution error: {e}"}


# ---------------------------------------------------------------------------
# Tool handler (called by model_tools dispatch)
# ---------------------------------------------------------------------------

def _handle_claude_code_dispatch(args: dict, **kwargs) -> str:
    """Tool handler entry point (called by registry dispatch with args dict).

    Registry passes ``task_id`` and ``user_task`` as kwargs (see model_tools.py).
    We use ``task_id`` as the session key so Claude sessions are scoped to
    the Hermes session/task.
    """
    session_key = kwargs.get("task_id")

    result = claude_code(
        prompt=args.get("prompt", ""),
        model=args.get("model", "") or None,
        max_turns=args.get("max_turns", 0) or None,
        cwd=args.get("cwd", "") or None,
        timeout=args.get("timeout", 0) or None,
        session_key=session_key,
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema + Registration
# ---------------------------------------------------------------------------

CLAUDE_CODE_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Delegate a task to Claude Code (Anthropic's coding agent). "
        "ONLY use this tool when the user explicitly asks to use Claude, "
        "or when a task clearly requires capabilities you lack (e.g. multi-file "
        "refactoring across 5+ files, complex architecture redesign, or debugging "
        "issues you failed to solve after 2+ attempts). "
        "Do NOT use this for simple questions, single-file edits, or tasks you can handle. "
        "Claude runs locally using the user's existing subscription. "
        "Sessions are preserved across calls. "
        "Provide a clear, detailed prompt with file paths and context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Detailed task description for Claude. Include relevant context: "
                    "file paths, code snippets, expected behavior, and specific instructions."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model override: 'sonnet', 'opus', or 'haiku'. Leave empty for default.",
                "default": "",
            },
            "max_turns": {
                "type": "integer",
                "description": "Max agentic turns (0 = default). Set higher for complex multi-step tasks.",
                "default": 0,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for Claude. Leave empty for current directory.",
                "default": "",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default 300, max 600).",
                "default": 0,
            },
        },
        "required": ["prompt"],
    },
}

registry.register(
    name=TOOL_NAME,
    toolset=TOOLSET,
    schema=CLAUDE_CODE_SCHEMA,
    handler=_handle_claude_code_dispatch,
    check_fn=check_claude_code_available,
    requires_env=[],
    description="Delegate complex tasks to Claude Code CLI",
    emoji="",
)
