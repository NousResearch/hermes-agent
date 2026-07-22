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
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_MAX_CHARS = 10_000
DEFAULT_PREVIEW_HEAD = 500
DEFAULT_PREVIEW_TAIL = 500
DEFAULT_ENABLED = True
DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_FILES_PER_SESSION = 100
_OWNERSHIP_MARKER = ".hermes-managed"
_OWNERSHIP_VALUE = "hook_outputs"
_DELETE_QUARANTINE_TOKEN = ".hermes-delete-"


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
        "retention_seconds": _coerce_positive_int(
            section.get("retention_seconds"), DEFAULT_RETENTION_SECONDS
        ),
        "max_files_per_session": _coerce_positive_int(
            section.get("max_files_per_session"), DEFAULT_MAX_FILES_PER_SESSION
        ),
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


def _is_owned_spill_dir(path: Path) -> bool:
    """Return whether a session directory has exact, non-symlink ownership."""
    marker = path / _OWNERSHIP_MARKER
    try:
        path_attrs = int(getattr(path.lstat(), "st_file_attributes", 0))
        marker_attrs = int(getattr(marker.lstat(), "st_file_attributes", 0))
        return (
            path.is_dir()
            and not path.is_symlink()
            and not (path_attrs & 0x400)
            and marker.is_file()
            and not marker.is_symlink()
            and not (marker_attrs & 0x400)
            and marker.read_text(encoding="utf-8").strip() == _OWNERSHIP_VALUE
        )
    except (OSError, UnicodeError):
        return False


def _prepare_owned_spill_dir(path: Path) -> bool:
    """Create and mark a new session dir, or verify an existing owned one."""
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Never adopt a pre-existing directory by stamping a marker into it.
        return _is_owned_spill_dir(path)
    except OSError:
        return False

    marker = path / _OWNERSHIP_MARKER
    try:
        with marker.open("x", encoding="utf-8") as marker_file:
            marker_file.write(f"{_OWNERSHIP_VALUE}\n")
    except (FileExistsError, OSError):
        try:
            path.rmdir()
        except OSError:
            pass
        return False
    if _is_owned_spill_dir(path):
        return True
    try:
        path.rmdir()
    except OSError:
        pass
    return False


def _atomic_unlink_spill(path: Path, expected: os.stat_result) -> bool:
    """Unlink the exact opened spill object, never a pathname replacement."""
    fd: Optional[int] = None
    quarantine = path.with_name(f".{path.name}.hermes-delete-{uuid.uuid4().hex}")
    try:
        current = path.lstat()
        attributes = int(getattr(current, "st_file_attributes", 0))
        if (
            _DELETE_QUARANTINE_TOKEN in path.name
            or path.is_symlink()
            or attributes & 0x400
            or not stat.S_ISREG(current.st_mode)
            or not os.path.samestat(expected, current)
        ):
            return False
        if os.name == "nt":
            import ctypes
            import msvcrt

            create_file = ctypes.windll.kernel32.CreateFileW
            create_file.argtypes = (
                ctypes.c_wchar_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
            )
            create_file.restype = ctypes.c_void_p
            handle = create_file(
                str(path),
                0x80000000,  # GENERIC_READ
                0x00000001 | 0x00000002 | 0x00000004,  # SHARE_READ|WRITE|DELETE
                None,
                3,  # OPEN_EXISTING
                0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
                None,
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError()
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
            except BaseException:
                ctypes.windll.kernel32.CloseHandle(handle)
                raise
        else:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(path, flags)
        opened = os.fstat(fd)
        current = path.lstat()
        current_attributes = int(getattr(current, "st_file_attributes", 0))
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or path.is_symlink()
            or current_attributes & 0x400
            or not os.path.samestat(expected, current)
            or not os.path.samestat(opened, current)
        ):
            return False
        os.replace(path, quarantine)
        if not os.path.samestat(opened, quarantine.lstat()):
            try:
                if os.path.lexists(path):
                    raise FileExistsError(str(path))
                os.replace(quarantine, path)
            except OSError:
                pass
            return False
        try:
            quarantine.unlink()
        except OSError:
            # The exact spill remains preserved for operator inspection. Its
            # quarantine name is permanently excluded from automatic pruning.
            return False
        return True
    except OSError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def prune_spill_files(
    base_directory: str | os.PathLike[str],
    *,
    retention_seconds: int = DEFAULT_RETENTION_SECONDS,
    max_files_per_session: int = DEFAULT_MAX_FILES_PER_SESSION,
) -> int:
    """Prune only owned ``*.txt`` spill files below session directories.

    The spill root is a Hermes-owned namespace, but individual files may be
    concurrently written by another session.  Deletion is therefore limited
    to regular, non-symlink files that are either older than the retention
    window or beyond the deterministic newest-N bound; any stat/unlink error
    preserves the file and is merely logged.
    """
    try:
        root = Path(base_directory)
        if not root.is_dir() or root.is_symlink():
            return 0
        retention = max(1, int(retention_seconds))
        maximum = max(1, int(max_files_per_session))
    except (OSError, TypeError, ValueError):
        return 0

    now = time.time()
    removed = 0
    try:
        session_dirs = sorted(
            (p for p in root.iterdir() if _is_owned_spill_dir(p)),
            key=lambda p: p.name,
        )
    except OSError:
        return 0

    for session_dir in session_dirs:
        try:
            files = []
            for path in session_dir.iterdir():
                if _DELETE_QUARANTINE_TOKEN in path.name or path.suffix != ".txt":
                    continue
                selected = path.lstat()
                attributes = int(getattr(selected, "st_file_attributes", 0))
                if (
                    path.is_symlink()
                    or attributes & 0x400
                    or not stat.S_ISREG(selected.st_mode)
                ):
                    continue
                files.append((path, selected))
            files.sort(
                key=lambda item: (item[1].st_mtime, item[0].name),
                reverse=True,
            )
        except OSError:
            continue
        for index, (path, selected) in enumerate(files):
            expired = now - selected.st_mtime > retention
            if not expired and index < maximum:
                continue
            try:
                if _atomic_unlink_spill(path, selected):
                    removed += 1
            except OSError:
                logger.debug("hook output spill prune skipped %s", path, exc_info=True)
    return removed


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
        prune_spill_files(
            spill_dir.parent,
            retention_seconds=cfg.get("retention_seconds", DEFAULT_RETENTION_SECONDS),
            max_files_per_session=cfg.get(
                "max_files_per_session", DEFAULT_MAX_FILES_PER_SESSION
            ),
        )
        if not _prepare_owned_spill_dir(spill_dir):
            raise OSError(f"spill directory is not exclusively Hermes-owned: {spill_dir}")
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
    "DEFAULT_RETENTION_SECONDS",
    "DEFAULT_MAX_FILES_PER_SESSION",
    "get_spill_config",
    "prune_spill_files",
    "spill_if_oversized",
]
