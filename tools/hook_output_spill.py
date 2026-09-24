"""Spill oversized hook-injected context to disk with a preview placeholder.

Hook ``{"context": ...}`` output rides EVERY subsequent API call, so a large blob
inflates every turn and breaks the prompt-cache prefix. Above
``hooks.output_spill.max_chars`` (default 10000) the text is written under
``hooks.output_spill.directory`` (default ``<HERMES_HOME>/hook_outputs/<session>``)
and the payload becomes a ``preview_head``/``preview_tail`` excerpt plus the path.
``enabled: false`` disables. Never raises: an I/O failure still returns a preview.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from tools.tool_output_limits import _coerce_int, _coerce_positive_int

logger = logging.getLogger(__name__)


DEFAULT_MAX_CHARS = 10_000
DEFAULT_PREVIEW_HEAD = 500
DEFAULT_PREVIEW_TAIL = 500
DEFAULT_ENABLED = True


def get_spill_config() -> Dict[str, Any]:
    """Return resolved hook output-spill config. Never raises."""
    section: Dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        hooks = cfg.get("hooks") if isinstance(cfg, dict) else None
        if isinstance(hooks, dict) and isinstance(hooks.get("output_spill"), dict):
            section = hooks["output_spill"]
    except Exception:
        section = {}
    enabled_raw = section.get("enabled", DEFAULT_ENABLED)
    directory = section.get("directory")
    return {
        "enabled": bool(enabled_raw) if enabled_raw is not None else DEFAULT_ENABLED,
        "max_chars": _coerce_positive_int(section.get("max_chars"), DEFAULT_MAX_CHARS),
        # head/tail allow zero (empty tail), max_chars must be positive.
        "preview_head": _coerce_int(section.get("preview_head"), DEFAULT_PREVIEW_HEAD, 0),
        "preview_tail": _coerce_int(section.get("preview_tail"), DEFAULT_PREVIEW_TAIL, 0),
        "directory": directory if isinstance(directory, str) else None,
    }


def _resolve_spill_dir(directory_override: Optional[str], session_id: Optional[str]) -> Path:
    """Per-session spill directory; session id is sanitised so it can't escape ``base``."""
    if directory_override:
        base = Path(os.path.expanduser(directory_override))
    else:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home()) / "hook_outputs"
    session_segment = (session_id or "no-session").replace("/", "_").replace("\\", "_").replace("..", "_")
    return base / session_segment


def spill_if_oversized(
    text: str, *, session_id: Optional[str] = None, source: str = "hook", config: Optional[Dict[str, Any]] = None,
) -> str:
    """Spill ``text`` to disk if it exceeds the configured cap.

    Returns ``text`` unchanged (under cap, disabled, or empty) or a preview
    string pointing at the full content. Non-string input is ``str()``-coerced;
    ``source`` labels the preview header; ``config`` overrides config.yaml.
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
    if len(text) <= int(cfg.get("max_chars") or DEFAULT_MAX_CHARS):
        return text
    head = int(cfg.get("preview_head") or 0)
    tail = int(cfg.get("preview_tail") or 0)

    # A disk failure must never blow up the turn — fall through to a preview
    # without a saved path.
    saved_path: Optional[str] = None
    try:
        spill_dir = _resolve_spill_dir(cfg.get("directory"), session_id)
        from tools.spill_safety import ensure_spill_dir, write_text_exclusive
        # Hook context may embed raw secrets: private perms + exclusive,
        # symlink-refusing create (the per-session dir is predictable).
        ensure_spill_dir(spill_dir, private=True)
        spill_path = spill_dir / f"{uuid.uuid4().hex}.txt"
        # Trailing newline so tail readers don't report "missing newline".
        write_text_exclusive(spill_path, text if text.endswith("\n") else text + "\n", private=True)
        saved_path = str(spill_path)
    except Exception as exc:
        logger.warning("hook output spill failed: %s", exc)

    total = len(text)
    parts = [
        f"[{source} output truncated — {total:,} chars; full content "
        + (f"saved to {saved_path}]" if saved_path else "unavailable — spill write failed]"),
    ]
    if head > 0 and text[:head]:
        parts.extend(["--- head ---", text[:head]])
    if tail > 0 and total > head:
        parts.extend(["--- tail ---", text[-tail:]])
    return "\n".join(parts)


__all__ = ["DEFAULT_MAX_CHARS", "DEFAULT_PREVIEW_HEAD", "DEFAULT_PREVIEW_TAIL", "DEFAULT_ENABLED",
           "get_spill_config", "spill_if_oversized"]
