"""Optional terminal output optimization inspired by RTK.

This module is intentionally conservative:
- disabled by default via config,
- never rewrites or re-executes commands,
- never changes exit codes/errors,
- fails open to the original output,
- preserves the sanitized raw output before returning a compact view.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import stat
from dataclasses import dataclass
from typing import Any


DEFAULT_CONFIG = {
    "enabled": False,
    "min_chars": 4000,
    "target_chars": 12000,
    "raw_output": "preserve",  # preserve | off
    "storage_dir": "/tmp/hermes-terminal-raw",
}


@dataclass(frozen=True)
class OptimizationResult:
    output: str
    optimized: bool
    reason: str = ""
    raw_path: str = ""
    original_chars: int = 0
    optimized_chars: int = 0
    command_class: str = "generic"


def _load_optimizer_config(config_override: dict[str, Any] | None = None) -> dict[str, Any]:
    if config_override is not None:
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config_override or {})
        return cfg
    try:
        from hermes_cli.config import load_config

        root = load_config() or {}
        user_cfg = (((root.get("terminal") or {}).get("output_optimizer")) or {})
    except Exception:
        user_cfg = {}
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(user_cfg, dict):
        cfg.update(user_cfg)
    return cfg


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def classify_command(command: str) -> str:
    """Return a coarse command class for output filtering only.

    This is not a shell parser and must not be used for approval or command
    rewrite decisions. It only selects a reducer after the command has already
    run.
    """
    text = (command or "").strip()
    if not text:
        return "generic"
    # Lightweight segment scan for compound commands. This remains output-only
    # classification, not a rewrite/safety parser.
    if re.search(r"(^|[;&|()\s])git\s+status(\s|$)", text):
        return "git_status"
    if re.search(r"(^|[;&|()\s])git\s+(diff|show)(\s|$)", text):
        return "git_diff"
    if re.search(r"(^|[;&|()\s])git\s+log(\s|$)", text):
        return "git_log"
    if re.search(r"(^|[;&|()\s])(pytest|py\.test|vitest|jest)(\s|$)", text) or re.search(r"(^|[;&|()\s])(npm|pnpm|yarn|bun|cargo|go)\s+test(\s|$)", text):
        return "test"
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        tokens = text.split()
    if not tokens:
        return "generic"

    # Skip leading environment assignments and common wrappers.
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i]):
        i += 1
    while i < len(tokens) and tokens[i] in {"sudo", "env", "command", "time"}:
        i += 1
    if i >= len(tokens):
        return "generic"

    cmd = os.path.basename(tokens[i])
    rest = tokens[i + 1 :]
    joined = " ".join([cmd] + rest)

    if cmd == "git":
        if "status" in rest:
            return "git_status"
        if "diff" in rest or "show" in rest:
            return "git_diff"
        if "log" in rest:
            return "git_log"
        return "git"
    if cmd in {"pytest", "py.test"} or "pytest" in joined:
        return "test"
    if cmd == "cargo" and "test" in rest:
        return "test"
    if cmd in {"npm", "pnpm", "yarn", "bun"} and any(t in rest for t in {"test", "vitest", "jest"}):
        return "test"
    if cmd in {"vitest", "jest", "go"} and (cmd != "go" or "test" in rest):
        return "test"
    if cmd in {"rg", "grep"}:
        return "search"
    return "generic"


def _collapse_repeated_lines(lines: list[str], max_repeats: int = 3) -> list[str]:
    if not lines:
        return lines
    out: list[str] = []
    prev = None
    count = 0
    for line in lines:
        if line == prev:
            count += 1
            if count <= max_repeats:
                out.append(line)
            continue
        if prev is not None and count > max_repeats:
            out.append(f"[... repeated {count - max_repeats} more times ...]")
        prev = line
        count = 1
        out.append(line)
    if prev is not None and count > max_repeats:
        out.append(f"[... repeated {count - max_repeats} more times ...]")
    return out


def _head_tail(lines: list[str], *, head: int, tail: int) -> str:
    if len(lines) <= head + tail:
        return "\n".join(lines)
    omitted = len(lines) - head - tail
    return "\n".join(lines[:head] + [f"... [{omitted} lines omitted] ..."] + lines[-tail:])


def _reduce_generic(output: str, target_chars: int) -> str:
    lines = _collapse_repeated_lines(output.splitlines())
    if len("\n".join(lines)) <= target_chars:
        return "\n".join(lines)
    head = max(20, target_chars // 220)
    tail = max(40, target_chars // 140)
    return _head_tail(lines, head=head, tail=tail)


def _reduce_git_status(output: str, target_chars: int) -> str:
    lines = output.splitlines()
    if len(output) <= target_chars:
        return output
    important = [ln for ln in lines if ln.startswith(("##", " M", "M ", " A", "A ", " D", "D ", " R", "R ", " C", "C ", "??", "UU"))]
    if important:
        body = _head_tail(important, head=80, tail=80)
        return "[git status compact view]\n" + body
    return _reduce_generic(output, target_chars)


def _reduce_git_log(output: str, target_chars: int) -> str:
    lines = output.splitlines()
    return _head_tail(lines, head=120, tail=20) if len(output) > target_chars else output


def _reduce_git_diff(output: str, target_chars: int) -> str:
    if len(output) <= target_chars:
        return output
    kept: list[str] = []
    for line in output.splitlines():
        if line.startswith(("diff --git", "+++ ", "--- ", "@@", "+", "-")):
            kept.append(line)
    compact = _head_tail(kept, head=160, tail=120) if kept else _reduce_generic(output, target_chars)
    return "[git diff compact view: headers/hunks preserved, context reduced]\n" + compact


def _reduce_test_output(output: str, target_chars: int) -> str:
    if len(output) <= target_chars:
        return output
    lines = output.splitlines()
    patterns = re.compile(
        r"(FAILED|FAILURES|ERROR|Error|Traceback|AssertionError|E\s+|F\s+|=+.*(failed|error|passed)|short test summary|FAIL|panic|thread '.*' panicked)",
        re.IGNORECASE,
    )
    important = [ln for ln in lines if patterns.search(ln)]
    if important:
        body = _head_tail(important, head=120, tail=120)
        tail = "\n".join(lines[-40:])
        return "[test output compact view: failures/errors/summary preserved]\n" + body + "\n\n[tail]\n" + tail
    return _head_tail(lines, head=80, tail=80)


def _reduce_output(command_class: str, output: str, target_chars: int) -> str:
    if command_class == "git_status":
        return _reduce_git_status(output, target_chars)
    if command_class == "git_log":
        return _reduce_git_log(output, target_chars)
    if command_class == "git_diff":
        return _reduce_git_diff(output, target_chars)
    if command_class == "test":
        return _reduce_test_output(output, target_chars)
    return _reduce_generic(output, target_chars)


def _write_raw_output(output: str, command: str, env, storage_dir: str) -> str:
    digest = hashlib.sha256((command + "\0" + output).encode("utf-8", errors="replace")).hexdigest()[:16]
    remote_path = f"{storage_dir.rstrip('/')}/terminal-{digest}.txt"
    if env is None:
        os.makedirs(storage_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(storage_dir, 0o700)
        except OSError:
            pass
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(remote_path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(output)
        finally:
            try:
                os.chmod(remote_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        return remote_path
    try:
        storage = shlex.quote(storage_dir.rstrip("/"))
        target = shlex.quote(remote_path)
        cmd = f"umask 077 && mkdir -p {storage} && chmod 700 {storage} && cat > {target} && chmod 600 {target}"
        result = env.execute(cmd, timeout=30, stdin_data=output)
        if result.get("returncode", 1) == 0:
            return remote_path
    except Exception:
        return ""
    return ""


def optimize_terminal_output(
    *,
    command: str,
    output: str,
    returncode: int = 0,
    env=None,
    config: dict[str, Any] | None = None,
) -> OptimizationResult:
    """Return optimized output when enabled and safe, else original output."""
    cfg = _load_optimizer_config(config)
    if not _as_bool(cfg.get("enabled")):
        return OptimizationResult(output=output, optimized=False, reason="disabled")
    if not output:
        return OptimizationResult(output=output, optimized=False, reason="empty")

    min_chars = _as_int(cfg.get("min_chars"), DEFAULT_CONFIG["min_chars"], minimum=0)
    target_chars = _as_int(cfg.get("target_chars"), DEFAULT_CONFIG["target_chars"], minimum=1000)
    if len(output) < min_chars:
        return OptimizationResult(output=output, optimized=False, reason="below_min_chars")

    command_class = classify_command(command)
    try:
        compact = _reduce_output(command_class, output, target_chars)
    except Exception:
        return OptimizationResult(output=output, optimized=False, reason="reducer_error")

    # Do not add metadata or raw files when the reducer did not materially help.
    if len(compact) >= len(output) * 0.95 and len(compact) <= len(output):
        return OptimizationResult(output=output, optimized=False, reason="not_beneficial")
    if len(compact) >= len(output):
        return OptimizationResult(output=output, optimized=False, reason="expanded")

    raw_path = ""
    if str(cfg.get("raw_output", "preserve")).lower() == "preserve":
        raw_path = _write_raw_output(output, command, env, str(cfg.get("storage_dir") or DEFAULT_CONFIG["storage_dir"]))
        if not raw_path:
            return OptimizationResult(output=output, optimized=False, reason="raw_preserve_failed")

    saved = len(output) - len(compact)
    metadata = [
        "<terminal-output-optimized>",
        f"command_class: {command_class}",
        f"exit_code: {returncode}",
        f"original_chars: {len(output)}",
        f"optimized_chars: {len(compact)}",
        f"saved_chars: {saved}",
    ]
    if raw_path:
        metadata.append(f"raw_sanitized_output: {raw_path}")
    metadata.append("Raw file contains the complete post-ANSI-strip, post-secret-redaction output before optimization.")
    metadata.append("</terminal-output-optimized>")
    final = "\n".join(metadata) + "\n" + compact
    return OptimizationResult(
        output=final,
        optimized=True,
        raw_path=raw_path,
        original_chars=len(output),
        optimized_chars=len(final),
        command_class=command_class,
    )
