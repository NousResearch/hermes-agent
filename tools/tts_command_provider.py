"""Shared runner for user-configured shell ("command") TTS/STT providers.

``tts.providers.<name>: {type: command, command: "piper -f {output_path} < {input_path}"}``
(and the ``stt.`` twin): ``{placeholders}`` are shell-quoted for their surrounding quote
context, ``{{``/``}}`` stay literal. Owns the quote-aware rendering, the idle-timeout
process runner and the generic ``<section>.providers.<name>`` readers, re-imported by
``tts_tool``/``transcription_tools`` under their historical private names. TTS placeholders:
``{input_path}``/``{text_path}``, ``{output_path}``, ``{format}``, ``{voice}``, ``{model}``,
``{speed}``. Built-in provider names always win over a same-named ``providers`` entry.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

from utils import is_truthy_value


def shell_quote_context(command_template: str, position: int) -> Optional[str]:
    """Return the shell quote char (``'``/``"``) active right before *position*, or None."""
    quote: Optional[str] = None
    escaped = False
    i = 0
    while i < position:
        char = command_template[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "\\":
            i += 1
        i += 1
    return quote


def quote_command_placeholder(value: str, quote_context: Optional[str]) -> str:
    """Quote a placeholder value for its position in a shell command template."""
    if quote_context == "'":
        return value.replace("'", r"'\''")
    if quote_context == '"':
        return value.replace("\\", "\\\\").replace('"', r'\"').replace("$", r"\$").replace("`", r"\`")
    return subprocess.list2cmdline([value]) if os.name == "nt" else shlex.quote(value)


def render_command_template(command_template: str, placeholders: Dict[str, str]) -> str:
    """Replace ``{name}`` placeholders (quote-aware) while preserving ``{{``/``}}``."""
    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})")
    replacements: list[tuple[str, str]] = []

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_CMD_PLACEHOLDER_{len(replacements)}__"
        quoted = quote_command_placeholder(placeholders[name], shell_quote_context(command_template, match.start()))
        replacements.append((token, quoted))
        return token

    rendered = pattern.sub(replace_match, command_template).replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


def _signal_process_tree(psutil: Any, proc: subprocess.Popen, method: str) -> None:
    """Apply ``terminate``/``kill`` to *proc* and all descendants (best effort)."""
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    try:
        for child in parent.children(recursive=True):
            try:
                getattr(child, method)()
            except psutil.NoSuchProcess:
                pass
        getattr(parent, method)()
    except psutil.NoSuchProcess:
        pass
    except Exception:
        getattr(proc, method)()


def terminate_command_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, stdin=subprocess.DEVNULL)
        except Exception:
            proc.kill()
        return
    # Prefer os.killpg to signal the entire process group at once.  The caller
    # (run_command_provider) creates processes with start_new_session=True, so the
    # launcher is both session leader and PGID leader.  Even if the launcher has
    # already exited, orphaned workers remain in the original PGID until they
    # explicitly call setsid().  os.killpg reaches them; walking the process tree
    # via psutil misses them because psutil.Process(parent_pid) raises
    # NoSuchProcess once the launcher is gone, and children() returns an empty
    # list after reparenting.
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None
    if pgid is not None and pgid == proc.pid:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        return
    # Fallback for non-group-leaders: walk the process tree via psutil.
    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None
    # Without psutil only the shell itself is signalled (children may survive).
    signal_fn = ((lambda m: getattr(proc, m)()) if psutil is None
                 else (lambda m: _signal_process_tree(psutil, proc, m)))
    signal_fn("terminate")
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        signal_fn("kill")
