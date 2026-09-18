"""History-reuse budgets for native vision embeds (config section ``vision``).

A native ``vision_analyze`` result bakes the image into conversation history, where it is
re-sent on every later API call. Two knobs bound that cost: ``vision.embed_target_bytes``
(how large one embed may be) and ``vision.max_calls_per_image`` (how often the same image
may be embedded per session). See #112095: a delegated subagent re-loaded five screenshots
158 times in 15 minutes because nothing refused the repeat.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Optional

from tools.registry import tool_error

# 256 KB keeps a 1568px screenshot cheap enough to ride the session (#92699); the clamp keeps one
# setting from turning every later request into a multi-megabyte resend or a useless thumbnail.
_DEFAULT_EMBED_TARGET_BYTES = 256 * 1024
_MIN_EMBED_TARGET_BYTES = 64 * 1024
_MAX_EMBED_TARGET_BYTES = 4 * 1024 * 1024

# Delegated subagents run unattended and cannot be steered mid-loop from the CLI, so they get a
# cap by default; the main agent stays unlimited unless the user sets ``vision.max_calls_per_image``.
_SUBAGENT_REPEAT_CAP = 3
_REPEAT_COUNTS_MAX_KEYS = 4096

_repeat_counts: dict[tuple[str, str], int] = {}
_repeat_lock = threading.Lock()


def _cfg_vision(key: str, default=None):
    """``vision.<key>`` from config.yaml; ``default`` when config is unavailable."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return cfg_get(load_config(), "vision", key, default=default)
    except Exception:
        return default


def resolve_embed_target_bytes() -> int:
    """``vision.embed_target_bytes`` clamped to 64 KiB..4 MiB; the 256 KB default on a bad value."""
    raw = _cfg_vision("embed_target_bytes", default=_DEFAULT_EMBED_TARGET_BYTES)
    try:
        if isinstance(raw, bool):
            raise ValueError("boolean is not a byte budget")
        target = int(raw)
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_EMBED_TARGET_BYTES
    return min(max(target, _MIN_EMBED_TARGET_BYTES), _MAX_EMBED_TARGET_BYTES)


def resolve_repeat_cap() -> int:
    """Per-image embed cap for this session; 0 = unlimited.

    ``vision.max_calls_per_image`` unset (or unparseable) → ``_SUBAGENT_REPEAT_CAP`` inside a
    delegated subagent, unlimited for the main agent. An explicit value applies everywhere.
    """
    raw = _cfg_vision("max_calls_per_image")
    try:
        if raw is not None and raw != "" and not isinstance(raw, bool):
            return max(int(raw), 0)
    except (TypeError, ValueError):
        pass
    from agent.delegation_context import is_delegated_child_process_context
    return _SUBAGENT_REPEAT_CAP if is_delegated_child_process_context() else 0


def _image_key(image_url: str) -> str:
    """Stable identity for an image source: local paths resolve (symlinks, ``~``, ``file://``),
    URLs drop their fragment, data URLs hash. Region crops are NOT part of the key — the incident
    loop alternated full loads and crops of the same files, and both re-embed the image."""
    if image_url.startswith("data:"):
        return "data:" + hashlib.sha256(image_url.encode("utf-8", "ignore")).hexdigest()[:32]
    stripped = image_url.split("#", 1)[0].removeprefix("file://")
    if "://" in stripped:
        return "url:" + stripped
    return "file:" + str(Path(os.path.expanduser(stripped)).resolve())


def _count_key(image_url: str) -> tuple[str, str]:
    from gateway.session_context import get_session_env
    return get_session_env("HERMES_SESSION_ID", ""), _image_key(image_url)


def repeat_refusal(image_url: str) -> Optional[str]:
    """Reserve one native embed of ``image_url`` for the current session; tool-error JSON when the
    per-session cap is already spent, else ``None``. Check and count are ONE lock section: a
    parallel tool batch on the same image (the incident's 4 concurrent calls) must not all pass a
    check taken before any of them recorded. Callers ``release_embed`` when the embed then fails."""
    cap = resolve_repeat_cap()
    if cap <= 0:
        return None
    key = _count_key(image_url)
    with _repeat_lock:
        count = _repeat_counts.get(key, 0)
        if count < cap:
            _record_embed_locked(key)
            return None
    return tool_error(
        f"vision_analyze refused: this image has already been loaded into context {count} time(s) "
        "in this session (region crops of the same file count too), and every native load re-sends "
        "the full image on each later API call. Answer from what you can already see, or ask the "
        f"user. (vision.max_calls_per_image = {cap}; 0 = unlimited)",
        success=False,
    )


def _record_embed_locked(key: tuple[str, str]) -> None:
    _repeat_counts[key] = _repeat_counts.get(key, 0) + 1
    # Bound long-lived gateway memory: evict the oldest (session, image) entries.
    while len(_repeat_counts) > _REPEAT_COUNTS_MAX_KEYS:
        _repeat_counts.pop(next(iter(_repeat_counts)))


def record_embed(image_url: str) -> None:
    """Count one successful native embed of ``image_url`` for the current session (uncapped
    sessions only — capped ones are counted by the reservation in :func:`repeat_refusal`)."""
    with _repeat_lock:
        _record_embed_locked(_count_key(image_url))


def release_embed(image_url: str) -> None:
    """Give back a slot reserved by :func:`repeat_refusal` when the embed did not happen."""
    key = _count_key(image_url)
    with _repeat_lock:
        count = _repeat_counts.get(key, 0)
        if count > 1:
            _repeat_counts[key] = count - 1
        else:
            _repeat_counts.pop(key, None)
