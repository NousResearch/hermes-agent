"""Spill oversized hook-injected context to disk with a preview placeholder.

Ported from openai/codex PR #21069 (``Spill large hook outputs from context``).

Background
----------
Both shell hooks (``agent/shell_hooks.py``) and Python plugins
(``pre_llm_call`` hook in ``run_agent.py``) can return ``{"context": "..."}``
which gets concatenated into the current turn's user message on EVERY
subsequent API call. If a hook emits a large blob (e.g. a debug dump, a
full file, or a runaway prompt-engineering script), that blob inflates
every turn of the session and blows out the prompt cache prefix the
moment it's appended.

This mirrors what Codex does for its ``PreToolUse``/``Stop``/feedback
hooks: once the injected text exceeds a configured budget, write the
full content to a per-session directory on disk and replace the in-prompt
payload with a head/tail preview plus the saved path. The model can still
inspect the full content via ``read_file`` or ``terminal`` if it needs to.

Config (``config.yaml``)::

    hooks:
      output_spill:
        enabled: true          # default: true; set false to disable spilling
        max_chars: 10000       # default; context above this is spilled
        preview_head: 500      # chars shown at the start of the preview
        preview_tail: 500      # chars shown at the end of the preview
        directory: null        # default: <HERMES_HOME>/hook_outputs

Design invariants
-----------------
* Behaviour-preserving when ``enabled: false`` or when content is under
  the cap — return the input string unchanged.
* Never raises. Any I/O error (disk full, permission denied, missing
  HERMES_HOME, etc.) falls back to a byte-length truncation with an
  in-prompt notice — the hook context still reaches the model, just
  bounded in size.
* Spill files are grouped by session so a ``/new`` session doesn't grow
  them forever in one directory.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_MAX_CHARS = 10_000
DEFAULT_PREVIEW_HEAD = 500
DEFAULT_PREVIEW_TAIL = 500
DEFAULT_ENABLED = True


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def _coerce_non_negative_int(value: Any, default: int) -> int:
    """Like ``_coerce_positive_int`` but allows zero (e.g. empty tail)."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv < 0:
        return default
    return iv


def get_spill_config() -> Dict[str, Any]:
    """Return resolved hook output-spill config. Never raises."""
    section: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        hooks = cfg.get("hooks") if isinstance(cfg, dict) else None
        if isinstance(hooks, dict):
            sub = hooks.get("output_spill")
            if isinstance(sub, dict):
                section = sub
    except Exception:
        section = {}

    enabled_raw = section.get("enabled", DEFAULT_ENABLED)
    enabled = bool(enabled_raw) if enabled_raw is not None else DEFAULT_ENABLED

    directory = section.get("directory")
    if directory is not None and not isinstance(directory, str):
        directory = None

    return {
        "enabled": enabled,
        "max_chars": _coerce_positive_int(section.get("max_chars"), DEFAULT_MAX_CHARS),
        "preview_head": _coerce_non_negative_int(
            section.get("preview_head"), DEFAULT_PREVIEW_HEAD
        ),
        "preview_tail": _coerce_non_negative_int(
            section.get("preview_tail"), DEFAULT_PREVIEW_TAIL
        ),
        "directory": directory,
    }


def _resolve_spill_dir(directory_override: Optional[str], session_id: Optional[str]) -> Path:
    """Return the directory where spill files for this session live."""
    if directory_override:
        base = Path(os.path.expanduser(directory_override))
    else:
        try:
            from hermes_constants import get_hermes_home
            base = Path(get_hermes_home()) / "hook_outputs"
        except Exception:
            # Last-resort fallback: HERMES_HOME env var, then ~/.hermes
            home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
            base = Path(home) / "hook_outputs"

    # Group by session so spills are contained per conversation.
    session_segment = session_id or "no-session"
    # Defensive: strip path separators so a weird session id can't
    # escape the directory.
    session_segment = session_segment.replace("/", "_").replace("\\", "_").replace("..", "_")
    return base / session_segment


def _build_preview(
    text: str,
    head: int,
    tail: int,
    saved_path: Optional[str],
    *,
    source: str,
) -> str:
    """Assemble the in-prompt preview with head/tail and saved-path footer."""
    total = len(text)
    head_chunk = text[:head] if head > 0 else ""
    tail_chunk = text[-tail:] if tail > 0 and total > head else ""

    parts = [
        f"[{source} output truncated — {total:,} chars; full content "
        + (f"saved to {saved_path}]" if saved_path else "unavailable — spill write failed]"),
    ]
    if head_chunk:
        parts.append("--- head ---")
        parts.append(head_chunk)
    if tail_chunk:
        parts.append("--- tail ---")
        parts.append(tail_chunk)
    return "\n".join(parts)


def spill_if_oversized(
    text: str,
    *,
    session_id: Optional[str] = None,
    source: str = "hook",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Spill ``text`` to disk if it exceeds the configured cap.

    Returns either ``text`` unchanged (when under the cap, disabled, or
    empty) or a preview string with a filesystem path pointing at the
    full content.

    Parameters
    ----------
    text:
        The raw injected-context string from a hook. Non-string inputs
        are coerced with ``str()``.
    session_id:
        Used to group spill files by conversation. Falls back to
        ``"no-session"`` if missing.
    source:
        Human-readable label used in the preview header (``"hook"``,
        ``"plugin hook"``, ``"shell hook"``, etc.). Free-form.
    config:
        Optional override for tests; normally resolved from
        ``config.yaml``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    cfg = config if config is not None else get_spill_config()
    if not cfg.get("enabled", True):
        return text

    max_chars = int(cfg.get("max_chars") or DEFAULT_MAX_CHARS)
    if len(text) <= max_chars:
        return text

    head = int(cfg.get("preview_head") or 0)
    tail = int(cfg.get("preview_tail") or 0)
    directory_override = cfg.get("directory")

    # Try to write the spill file. If that fails we still need to return
    # something bounded — never let a disk failure blow up the turn.
    saved_path: Optional[str] = None
    try:
        spill_dir = _resolve_spill_dir(directory_override, session_id)
        spill_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.txt"
        spill_path = spill_dir / filename
        # Write the raw text plus a trailing newline so tail readers
        # (``tail -f``, editors) don't report "missing newline".
        spill_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        saved_path = str(spill_path)
    except Exception as exc:
        logger.warning("hook output spill failed: %s", exc)
        saved_path = None

    return _build_preview(text, head, tail, saved_path, source=source)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_PREVIEW_HEAD",
    "DEFAULT_PREVIEW_TAIL",
    "DEFAULT_ENABLED",
    "get_spill_config",
    "spill_if_oversized",
]
