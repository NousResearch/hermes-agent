"""Small pure helpers shared by the tools.mcp_tool_* modules: SDK 1.x/2.x field access,
error-text sanitising, numeric/bool coercion, timeouts and jitter. No origin state."""

import logging
import math
import os
from pathlib import Path
import random
import re
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger("tools.mcp_tool")


class _OriginProxy:
    """Attribute proxy for ``tools.mcp_tool`` resolved at access time. The split modules read
    origin state (``_servers``, ``_lock``, SDK symbols, patchable helpers) through this so
    ``mock.patch("tools.mcp_tool.X")`` and origin-side rebinds stay effective, and so no split
    module needs the origin imported first (the origin imports them while initialising)."""

    __slots__ = ()

    def __getattr__(self, name: str):
        from tools import mcp_tool
        return getattr(mcp_tool, name)


_core = _OriginProxy()
_MISSING = object()


def mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across the 1.x -> 2.x rename to snake_case. Pydantic aliases
    don't apply to attribute access, so ``getattr(result, "isError", False)`` silently returns
    the default on 2.x — failed calls read as successful, schemas as empty."""
    value = getattr(obj, snake, _MISSING)
    if value is _MISSING:
        value = getattr(obj, camel, _MISSING)
    return default if value is _MISSING else value


_DEFAULT_TOOL_TIMEOUT = 300      # seconds for tool calls


def _resolve_tool_timeout(config: dict) -> float:
    """Per-server tool-call timeout. Precedence: ``mcp_servers.<name>.timeout`` >
    ``timeouts.mcp.tool_call`` > the 300s default; values are platform-clamped by
    ``resolve_timeout``."""
    per_server = config.get("timeout")
    if per_server is not None:
        return per_server
    try:
        from agent.deadline import resolve_timeout
        resolved = resolve_timeout("mcp.tool_call", default=_DEFAULT_TOOL_TIMEOUT)
        if resolved is not None:
            return resolved
    except Exception:
        logger.debug("mcp.tool_call timeout resolution failed", exc_info=True)
    return _DEFAULT_TOOL_TIMEOUT


# Jitter on reconnect backoff so servers that lost the same backend don't retry in lockstep.
_BACKOFF_JITTER = 0.2            # +/-20%


def _jittered(seconds: float) -> float:
    """``seconds`` with +/-20% uniform jitter, floored at 0."""
    return max(0.0, seconds * random.uniform(1.0 - _BACKOFF_JITTER, 1.0 + _BACKOFF_JITTER))


# Credential patterns to strip from error messages: GitHub PAT, OpenAI-style key, Bearer token,
# and ``token= / key= / API_KEY= / password= / secret=`` assignments.
_CREDENTIAL_PATTERN = re.compile(
    r"(?:ghp_[A-Za-z0-9_]{1,255}|sk-[A-Za-z0-9_-](?:\.?[A-Za-z0-9_-]){0,254}|Bearer\s+\S+"
    r"|(?:token|key|API_KEY|password|secret)=[^\s&,;\"']{1,255})", re.IGNORECASE)


def _env_ref_name(ref: str) -> str:
    """Bare env-var name from a ``${...}`` body; strips a Cursor-style ``env:`` prefix."""
    ref = ref.strip()
    if ref.startswith("env:"):
        ref = ref[len("env:"):].strip()
    return ref


def _sanitize_error(text: str) -> str:
    """Replace credential-like patterns with [REDACTED] before text reaches the LLM."""
    from tools.ansi_strip import strip_unicode_tags
    return strip_unicode_tags(_CREDENTIAL_PATTERN.sub("[REDACTED]", text))


def exact_mcp_identifier(value, what: str):
    """*value* unchanged: a name/uri the server matches a later request against must round-trip
    byte-exact, so a hidden TAG/bidi char in it is logged rather than stripped (stripping would make
    the prompt, resource or tool uncallable)."""
    from tools.ansi_strip import strip_tool_metadata_text
    if isinstance(value, str) and strip_tool_metadata_text(value) != value:
        logger.warning("MCP %s %r carries invisible characters; kept as-is (the server matches it)", what, value)
    return value


def _exc_str(exc: BaseException) -> str:
    """Non-empty string for *exc*: some exceptions (``anyio.ClosedResourceError``) carry no
    message, so fall back to ``repr`` to keep diagnostics."""
    text = str(exc).strip()
    return text or repr(exc)


def _prepend_path(env: dict, directory: str) -> dict:
    """Prepend *directory* to env PATH if it is not already present."""
    updated = dict(env or {})
    if directory:
        parts = [part for part in updated.get("PATH", "").split(os.pathsep) if part]
        if directory not in parts:
            parts = [directory, *parts]
        updated["PATH"] = os.pathsep.join(parts) if parts else directory
    return updated


def _safe_numeric(value, default, coerce=int, minimum=1):
    """Coerce a config value (YAML strings included) to a number, clamped to *minimum*;
    *default* on failure or non-finite floats."""
    try:
        result = coerce(value)
        if isinstance(result, float) and not math.isfinite(result):
            return default
        return max(result, minimum)
    except (TypeError, ValueError, OverflowError):
        return default


_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})


def _parse_boolish(value: Any, default: bool = True) -> bool:
    """Parse a bool-like config value with safe fallback (YAML ``0``/``1`` are numbers, not words)."""
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
    logger.warning("MCP config expected a boolean-ish value, got %r; using default=%s", value, default)
    return default


def mcp_server_enabled(cfg: dict) -> bool:
    """Whether ``mcp_servers.<name>`` is on. The ONE reader of the ``enabled`` key: the MCP client,
    the toolset resolver, the profile editor, and every list/status surface call it, so a value
    can never be on for one surface and off for another. Absent, ``null`` or unparseable = on.
    A legacy ``disabled: true`` (pre-v46 profile editor) also means off and wins over ``enabled``,
    as v46 resolves it: a file stamped by a newer build runs no step, so v46 may not have run."""
    return (_parse_boolish(cfg.get("enabled", True), default=True)
            and not _parse_boolish(cfg.get("disabled", False), default=False))


def write_mcp_server_enabled(cfg: dict, on: bool) -> None:
    """Switch ``mcp_servers.<name>`` on or off for every writer. A legacy ``disabled`` is dropped
    in the same write: the reader above lets it win, so leaving it would undo an enable."""
    cfg["enabled"] = bool(on)
    cfg.pop("disabled", None)


def drop_legacy_disabled_for_key(located) -> bool:
    """A generic write of ``mcp_servers.<name>.enabled`` (``hermes config set``) also drops the
    legacy ``disabled``, as :func:`write_mcp_server_enabled` does, or an enable would not take.
    *located*: ``hermes_cli.config._locate_nested`` of the written key, so a server name with a
    dot resolves to the same entry the write landed on. True when it dropped one."""
    if located is None:
        return False
    parents, entry, key = located
    if key != "enabled" or len(parents) != 2 or parents[0][1] != "mcp_servers" or not isinstance(entry, dict):
        return False
    return entry.pop("disabled", None) is not None


def _get_lifecycle_seconds(config: dict, key: str) -> Optional[float]:
    """Optional positive lifecycle timeout from top-level/nested ``lifecycle`` config (``0``
    disables; negatives and non-numbers are warned about and ignored)."""
    raw = config.get(key)
    if raw is None and isinstance(config.get("lifecycle"), dict):
        raw = config["lifecycle"].get(key)
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning("MCP config %s must be a number of seconds; ignoring %r", key, raw)
        return None
    if seconds < 0:
        logger.warning("MCP config %s must be positive; ignoring %r", key, raw)
        return None
    return seconds or None


_MIN_MCP_MEMORY_MAX_MB = 64
_MCP_UNIT_NAME_MAX_CHARS = 40
_MCP_UNIT_BAD_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _get_lifecycle_memory_max_mb(config: dict) -> Optional[int]:
    """Optional per-server ``memory_max_mb`` from top-level or nested ``lifecycle`` config.

    ``0`` or omitted returns ``None`` (use the host-safe default bound); values below
    ``_MIN_MCP_MEMORY_MAX_MB`` (64 MiB), booleans, or non-numbers log a warning and are ignored.
    """
    raw = config.get("memory_max_mb")
    if raw is None and isinstance(config.get("lifecycle"), dict):
        raw = config["lifecycle"].get("memory_max_mb")
    if raw is None:
        return None
    if isinstance(raw, bool):
        logger.warning("MCP config memory_max_mb must be an integer number of MiB; ignoring %r", raw)
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("MCP config memory_max_mb must be an integer number of MiB; ignoring %r", raw)
        return None
    if not math.isfinite(value) or not value.is_integer():
        logger.warning("MCP config memory_max_mb must be an integer number of MiB; ignoring %r", raw)
        return None
    if value == 0:
        return None
    if value < _MIN_MCP_MEMORY_MAX_MB:
        logger.warning(
            "MCP config memory_max_mb must be at least %d MiB; ignoring %r",
            _MIN_MCP_MEMORY_MAX_MB, raw,
        )
        return None
    return int(value)


def _ancestor_cgroup_memory_max_bytes(
    cgroup_text: Optional[str], cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> Optional[int]:
    """Smallest finite cgroup-v2 ``memory.max`` across ancestors that also enclose a
    ``systemd-run --user --scope`` unit (the root cgroup plus ``user.slice``/``user@*.service``/
    ``app.slice``), ignoring sibling ``.service``/``.scope`` leaves like ``hermes-gateway.service``."""
    if not cgroup_text:
        return None
    rel = None
    for line in cgroup_text.splitlines():
        if line.startswith("0::"):
            rel = line.partition("::")[2].strip().strip("/")
            break
    if rel is None:
        return None
    dirs = [cgroup_root]
    parts = [part for part in rel.split("/") if part]
    if parts and parts[0] == "user.slice":
        cur = cgroup_root / parts[0]
        dirs.append(cur)
        if len(parts) >= 2 and parts[1].startswith("user-") and parts[1].endswith(".slice"):
            cur = cur / parts[1]
            dirs.append(cur)
            if len(parts) >= 3 and parts[2].startswith("user@") and parts[2].endswith(".service"):
                cur = cur / parts[2]
                dirs.append(cur)
                if len(parts) >= 4 and parts[3] == "app.slice":
                    dirs.append(cur / parts[3])
    min_bytes = _MIN_MCP_MEMORY_MAX_MB * 1024 * 1024
    limits: list[int] = []
    for directory in dirs:
        try:
            raw = (directory / "memory.max").read_text(encoding="utf-8-sig").strip()
        except OSError:
            continue
        if raw.isdigit():
            val = int(raw)
            if val >= min_bytes:
                limits.append(val)
    return min(limits) if limits else None


def resolve_mcp_memory_max_bytes(
    explicit_memory_max_mb: Optional[int], enclosing_cgroup_max_bytes: Optional[int],
) -> Optional[int]:
    """Byte ceiling for an MCP server's systemd scope, or ``None`` for the default worker bound.

    An explicit ``memory_max_mb`` can tighten below 4 GiB on small VMs or widen on large hosts,
    still clamped by any finite ``memory.max`` on an enclosing slice/container cgroup.
    """
    if explicit_memory_max_mb is None:
        return None
    requested = explicit_memory_max_mb * 1024 * 1024
    if enclosing_cgroup_max_bytes is not None:
        return min(requested, enclosing_cgroup_max_bytes)
    return requested


def _sanitize_mcp_unit_component(server_name: str) -> str:
    """Safe, bounded systemd unit-name token derived from *server_name*."""
    cleaned = _MCP_UNIT_BAD_CHARS.sub("-", str(server_name or "")).strip("-")
    cleaned = cleaned[:_MCP_UNIT_NAME_MAX_CHARS].rstrip("-")
    return cleaned or "server"


def should_isolate_mcp_stdio_in_scope(
    *,
    is_linux: bool,
    environ: Mapping[str, str],
    is_supervised_gateway: bool,
) -> bool:
    """Whether a stdio MCP child should be placed in its own transient systemd scope (#159117).

    Isolates on Linux under systemd (``INVOCATION_ID`` present) whenever this process is the
    supervised gateway or belongs to a supervised gateway tree (``_HERMES_GATEWAY=1`` or
    ``HERMES_SUPERVISED_CHILD`` — covering ``agent.managed_worker`` and cron/Kanban workers,
    since ``systemd-run --user --scope`` creates a flat sibling ``hermes-mcp-*.scope`` under
    ``app.slice`` so a runaway MCP child cannot OOM-kill the gateway or the worker).
    Interactive ``hermes chat`` sessions in desktop terminals carry neither gateway marker and
    are left untouched.
    """
    if not is_linux or environ.get("INVOCATION_ID", "").strip() == "":
        return False
    if is_supervised_gateway:
        return True
    return (
        environ.get("_HERMES_GATEWAY") == "1"
        or environ.get("HERMES_SUPERVISED_CHILD", "").strip() != ""
    )
