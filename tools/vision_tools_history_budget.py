"""History-reuse budgets for native vision embeds (config section ``vision``).

A native ``vision_analyze`` result bakes the image into conversation history, where it is
re-sent on every later API call. Two knobs bound that cost: ``vision.embed_target_bytes``
(how large one embed may be) and ``vision.max_calls_per_image`` (how often the same image
may be embedded per session). See #112095: a delegated subagent re-loaded five screenshots
158 times in 15 minutes because nothing refused the repeat.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Optional

from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from hermes_constants import hermes_home_key
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

_repeat_counts: dict[tuple[str, str, str], int] = {}
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


def _file_uri_to_local_path(raw: str) -> str:
    parsed = urlparse(raw)
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        return f"//{parsed.netloc}{path}"
    return path


def _safe_local_key(raw_path: str) -> str:
    """Resolve a local path safely without touching NT/device namespaces (Windows SMB/NTLM leak)
    or raising on NUL bytes / symlink loops."""
    if "\x00" in raw_path:
        return "raw:" + hashlib.sha256(raw_path.encode("utf-8", "ignore")).hexdigest()[:32]
    from agent.file_safety import is_nt_namespace_path
    if is_nt_namespace_path(raw_path):
        return "nt:" + hashlib.sha256(raw_path.encode("utf-8", "ignore")).hexdigest()[:32]
    try:
        return "file:" + str(Path(os.path.expanduser(raw_path)).resolve())
    except (OSError, RuntimeError, ValueError):
        return "file:" + raw_path


def _image_key(image_url: str) -> str:
    """Stable identity for a media source: local paths resolve (symlinks, ``~``, ``file://``),
    YouTube URLs canonicalize, other URLs drop their fragment, data URLs hash. Region crops
    are NOT part of the key — the incident loop alternated full loads and crops of the same files."""
    if image_url.lower().startswith("data:"):
        return "data:" + hashlib.sha256(image_url.encode("utf-8", "ignore")).hexdigest()[:32]
    from agent.gemini_urls import normalize_youtube_url
    if yt_url := normalize_youtube_url(image_url):
        return "url:" + yt_url
    stripped = image_url.split("#", 1)[0].strip()
    if stripped.lower().startswith("file://"):
        return _safe_local_key(_file_uri_to_local_path(stripped))
    if "://" in stripped:
        return "url:" + stripped
    return _safe_local_key(stripped)


def _count_key(image_url: str) -> tuple[str, str, str]:
    from gateway.session_context import get_session_env
    return hermes_home_key(), get_session_env("HERMES_SESSION_ID", ""), _image_key(image_url)


def reserve_embed(
    image_url: str, *, tool_name: str = "vision_analyze", media_label: str = "image",
) -> tuple[Optional[str], Optional[tuple[str, str, str]]]:
    """Atomically check cap and reserve one embed slot; returns ``(refusal_json, reserved_key)``.
    Computing cap and key once prevents config-reload or symlink-unlink leaks on release."""
    cap = resolve_repeat_cap()
    if cap <= 0:
        return None, None
    key = _count_key(image_url)
    with _repeat_lock:
        count = _repeat_counts.get(key, 0)
        if count < cap:
            _record_embed_locked(key)
            return None, key
    extra = " (region crops of the same file count too)" if media_label == "image" else ""
    refusal = tool_error(
        f"{tool_name} refused: this {media_label} has already been loaded into context {count} time(s) "
        f"in this session{extra}, and every native load re-sends "
        f"the full {media_label} on each later API call. Answer from what you already have, or ask the "
        f"user. (vision.max_calls_per_image = {cap}; 0 = unlimited)",
        success=False,
    )
    return refusal, None


def release_embed_key(key: Optional[tuple[str, str, str]]) -> None:
    """Undo a reservation returned by :func:`reserve_embed` when the embed fails before returning."""
    if key is None:
        return
    with _repeat_lock:
        count = _repeat_counts.get(key, 0)
        if count <= 1:
            _repeat_counts.pop(key, None)
        else:
            _repeat_counts[key] = count - 1


def repeat_refusal(
    image_url: str, *, tool_name: str = "vision_analyze", media_label: str = "image",
) -> Optional[str]:
    """Reserve one native embed of ``image_url`` for the current session; tool-error JSON when the
    per-session cap is already spent, else ``None``. Callers ``release_embed`` when the embed fails."""
    refusal, _ = reserve_embed(image_url, tool_name=tool_name, media_label=media_label)
    return refusal


def _record_embed_locked(key: tuple[str, str, str]) -> None:
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


# Images already riding the ACTIVE user turn as native content parts (#76411). Every surface that
# attaches natively (gateway, CLI, TUI, delegated children) goes through
# ``image_routing.build_native_content_parts``, which writes one ``[Image attached at: <path>]`` /
# ``[Image attached: <url>]`` handle per image into the text part; ``conversation_loop.run_conversation``
# scopes those handles here for the turn and the tool inherits them through contextvars.
_NATIVE_HANDLE_RE = re.compile(r"^\[Image attached(?: at)?: (.+?)\]\s*$", re.MULTILINE)
_native_turn_images: ContextVar[frozenset[str]] = ContextVar("vision_native_turn_images", default=frozenset())


def _native_turn_keys(user_message: Any) -> frozenset[str]:
    if not isinstance(user_message, list) or not any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in user_message
    ):
        return frozenset()
    text = "\n".join(p.get("text", "") for p in user_message if isinstance(p, dict) and p.get("type") == "text")
    return frozenset(_image_key(m.strip()) for m in _NATIVE_HANDLE_RE.findall(text) if m.strip())


@contextlib.contextmanager
def native_turn_images(user_message: Any) -> Iterator[None]:
    """Scope the images natively attached to ``user_message`` to the running turn."""
    token = _native_turn_images.set(_native_turn_keys(user_message))
    try:
        yield
    finally:
        _native_turn_images.reset(token)


def native_turn_duplicate(image_url: str, region: Optional[list]) -> Optional[str]:
    """Text tool result when ``image_url`` already rides the active user turn natively, else
    ``None``. A native re-embed would put the identical pixels into the same request twice
    (the Telegram case in #76411); a ``region`` crop still embeds because it returns new detail."""
    if region is not None or _image_key(image_url) not in _native_turn_images.get():
        return None
    return json.dumps({
        "success": True,
        "already_in_context": True,
        "message": (
            "This image is already attached natively to the current user message — you can see it "
            "now. Answer with your built-in vision; pass a `region` to zoom into part of it."),
    })
