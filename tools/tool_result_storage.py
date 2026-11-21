"""Tool result persistence -- preserves large outputs instead of truncating. Layers against
context overflow: (1) per-tool caps inside each tool; (2) ``maybe_persist_tool_result`` —
output over the tool's threshold is persisted and replaced by a preview + path; canonical home
is ALWAYS host-side ``$HERMES_HOME/cache/spillover/{id}.txt`` (works for sessions that never
ran a terminal), remote backends get the translated in-sandbox path (probed for readability)
else a copy in the sandbox temp dir; (3) ``enforce_turn_budget``."""

import hashlib
import json
import logging
import math
import os
import re
import shlex
import tempfile
import threading
import time
from typing import Any, NamedTuple

from session_rules.media import PRUNED_SUBDIR, SPILLOVER_SUBDIR, archive_refs_in_text
from tools.budget_config import DEFAULT_PREVIEW_SIZE_CHARS, BudgetConfig, DEFAULT_BUDGET

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = os.path.join(tempfile.gettempdir(), "hermes-results")
_ARCHIVE_SUBDIRS = (SPILLOVER_SUBDIR, PRUNED_SUBDIR)
# Fallback spillover age only (unreadable config); the real default is the session
# retention window — see _spillover_retention_hours().
SPILLOVER_MAX_AGE_HOURS = 24
_BUDGET_TOOL_NAME = "__budget_enforcement__"
# The exact key set tools/mcp_tool_handlers.py::_render_call_tool_result emits. A JSON object whose
# keys stay inside this set is that handler's own envelope, never an arbitrary tool's JSON payload.
_MCP_ENVELOPE_KEYS = frozenset({"result", "structuredContent", "_meta"})
_ENVELOPE_METADATA_TAG = "<mcp-result-metadata>"
_ENVELOPE_METADATA_CLOSING_TAG = "</mcp-result-metadata>"
_TOOL_METADATA_TAG = "<tool-result-metadata>"
_TOOL_METADATA_CLOSING_TAG = "</tool-result-metadata>"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120

_spillover_prune_lock = threading.Lock()
_spillover_pruned_homes: set = set()  # profile home keys already swept this process


def get_spillover_dir():
    """Return $HERMES_HOME/cache/spillover as a Path (not created)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / SPILLOVER_SUBDIR


def _spillover_retention_hours() -> int:
    """Default spillover age = the session retention window (``sessions.retention_days``).
    Spill files are archives the durable transcript points at, not a cache: a session resumed
    days later must still find what the ``<persisted-output>`` pointer promises (#126351).
    Falls back to the legacy 24h cache age when the config is unreadable."""
    try:
        from hermes_cli.config import load_config_readonly
        days = float((load_config_readonly().get("sessions") or {}).get("retention_days", 90))
        if not math.isfinite(days):  # YAML ``.inf`` / ``.nan`` parse as floats; int() would raise
            return SPILLOVER_MAX_AGE_HOURS
    except Exception:  # health: allow BLE001 -- any config read/parse failure means the legacy floor
        return SPILLOVER_MAX_AGE_HOURS
    # ``retention_days: 0`` is legal (ended sessions pruned at once) but session prune never
    # touches OPEN sessions, while a zero-hour spill cutoff deletes the file the first spill
    # just wrote (the pointer then names a missing path). Floor at the legacy cache age.
    return max(SPILLOVER_MAX_AGE_HOURS, int(days * 24))


# How an archive is named inside transcript text, and the set the transcript holds: session_rules.media
# (the store reads the same rules when a delete retires a session's archives).


_ARCHIVE_PATH_RE = re.compile(r"(?:^|[/\\])(?:spillover|hermes-results|pruned)[/\\][^/\\]+$")


def expired_archive_fields(path: str) -> dict:
    """Extra read_file not-found fields for a persisted-tool-result path: the archive was pruned,
    so the model gets "expired" instead of a generic miss it might take for a typo (#126351)."""
    if not isinstance(path, str) or not _ARCHIVE_PATH_RE.search(path):
        return {}
    return {"expired": True, "hint": (
            "This was a persisted tool result archive and it has expired (pruned with its "
            "session or by the cache sweep); the full output is no longer recoverable. Inspect "
            "the current state before re-running the original tool — the call that produced it "
            "may already have taken effect.")}


def referenced_spillover_names(spill_dir) -> set | None:
    """Basenames under *spill_dir* (``cache/spillover`` or ``cache/pruned``) still referenced by any
    message in the home's ``state.db``; ``None`` when the reference set is unknown (see
    :func:`referenced_archive_refs`)."""
    refs = referenced_archive_refs(spill_dir.parent.parent)
    prefix = f"cache/{spill_dir.name}/"
    return None if refs is None else {ref[len(prefix):] for ref in refs if ref.startswith(prefix)}


def referenced_archive_refs(home) -> set | None:
    """``cache/<dir>/<name>`` archive references held by any message in *home*'s ``state.db``.

    ``set()`` when no database exists (no sessions → nothing references anything), ``None``
    when the reference set is UNKNOWN (unreadable database/query) so callers fail closed
    and keep the files; a read-only handle is enough, the write path owns the locks."""
    from persistence.core.errors import STORE_FAILURES
    from persistence_assembly import open_persistence_reader

    db_path = home / "state.db"
    if not db_path.exists():
        return set()
    try:
        refs: set = set()
        with open_persistence_reader(db_path) as store:
            for content, api_content in store.messages.spillover_reference_rows():
                refs.update(archive_refs_in_text(content, api_content))
    except (OSError, *STORE_FAILURES):
        return None
    return refs


def cleanup_spillover_cache(max_age_hours: int | None = None) -> int:
    """Delete archive files (``cache/spillover`` and the prune originals in ``cache/pruned``) older
    than *max_age_hours*; returns count removed (same
    contract as the ``cleanup_*_cache`` helpers the gateway housekeeping loop runs hourly).
    A ``None`` age means the session retention window, not a cache TTL — an explicit
    ``max_age_hours`` (tests, legacy callers) still wins. An expired file is deleted only
    when no message in the home's state.db references it: the transcript's
    ``<persisted-output>`` promise outlives the archive's mtime exactly as long as the
    owning session does (#126351) — referenced archives leave with their session, via the
    unlink SessionDB delete/prune performs, not with the age sweep."""
    age_hours = _spillover_retention_hours() if max_age_hours is None else max_age_hours
    cutoff = time.time() - (age_hours * 3600)
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return sum(_sweep_archive_dir(home / subdir, cutoff) for subdir in _ARCHIVE_SUBDIRS)


def _sweep_archive_dir(archive_dir, cutoff: float) -> int:
    """Unlink files in *archive_dir* older than *cutoff* that no state.db message references."""
    expired = []
    try:
        entries = list(archive_dir.iterdir())
    except OSError:
        return 0
    for f in entries:
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                expired.append(f)
        except OSError:
            pass
    if not expired:
        return 0
    referenced = referenced_spillover_names(archive_dir)
    removed = 0
    for f in expired:
        if referenced is None or f.name in referenced:  # unknown set: fail closed, keep
            continue
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def _prune_spillover_once() -> None:
    """Best-effort prune, at most once per process PER PROFILE HOME (CLI-only installs never run
    housekeeping; a multiplexed gateway must sweep every profile's ``cache/spillover``, not just the
    first one that spilled)."""
    from hermes_constants import hermes_home_key
    home_key = hermes_home_key()
    with _spillover_prune_lock:
        if home_key in _spillover_pruned_homes:
            return
        _spillover_pruned_homes.add(home_key)
    try:
        if removed := cleanup_spillover_cache():
            logger.debug("Pruned %d expired spillover file(s)", removed)
    except Exception as exc:
        logger.debug("Spillover prune failed: %s", exc)


def _is_host_side_env(env) -> bool:
    """True when this process should write the spill file directly: ``env=None`` (no sandbox
    yet) or a backend whose files are the host's. Others resolve ``read_file`` inside the sandbox."""
    if env is None:
        return True
    from tools.environments.base import env_uses_host_filesystem

    return env_uses_host_filesystem(env)


def _write_to_spillover(content: str, filename: str, subdir: str = SPILLOVER_SUBDIR, exclusive: bool = False):
    """Write host-side to $HERMES_HOME/<subdir> (default cache/spillover); returns path str or None.

    The write is read back and compared byte-for-byte before the caller tells the model "Full
    output saved": a partially-flushed or corrupted file (quota, ENOSPC race) fails closed to the
    bounded inline truncation instead of referencing an archive that silently lost bytes.
    """
    data = content.encode("utf-8", errors="replace")
    created = False
    try:
        from hermes_constants import get_hermes_home
        spill_dir = get_hermes_home() / subdir
        spill_dir.mkdir(parents=True, exist_ok=True)
        path = spill_dir / filename
        flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
        descriptor = os.open(path, flags, 0o600)
        created = True
        with os.fdopen(descriptor, "wb") as fh:
            os.chmod(path, 0o600)
            fh.write(data)
        persisted_size = os.stat(path).st_size
        # Size first (cheap), then the bytes: the content is already in memory.
        exact = persisted_size == len(data) and path.read_bytes() == data
    except OSError as exc:
        if exclusive and created:
            try:
                path.unlink()
            except OSError as cleanup_error:
                logger.warning("Partial archive cleanup failed for %s: %s", path, cleanup_error)
        logger.warning("Archive write to %s failed for %s: %s", subdir, filename, exc)
        return None
    if not exact:
        logger.warning(
            "Spillover write for %s is not lossless (%d bytes on disk, "
            "expected %d, or the bytes differ) — discarding archive",
            filename, persisted_size, len(data),
        )
        try:
            path.unlink()
        except OSError:
            pass
        return None
    _prune_spillover_once()
    return str(path)


def _sandbox_visible_spillover_path(host_path: str, env) -> str | None:
    """Path where a remote backend can read *host_path*, or None. Translates via the image
    tools' helper, forces a sync for synced backends, then PROBES readability — a persistent
    container created before spillover joined the mount list lacks the bind mount and must
    fall back to the in-sandbox write."""
    try:
        from tools.credential_files import to_agent_visible_cache_path
        visible = to_agent_visible_cache_path(host_path)
    except Exception as exc:
        logger.debug("Spillover path translation failed: %s", exc)
        return None
    try:
        if (sync_manager := getattr(env, "_sync_manager", None)) is not None:
            sync_manager.sync(force=True)
    except Exception as exc:
        logger.debug("Spillover sync failed: %s", exc)
    try:
        if env.execute(f"test -r {shlex.quote(visible)}", timeout=15).get("returncode", 1) == 0:
            return visible
    except Exception as exc:
        logger.debug("Spillover readability probe failed: %s", exc)
    return None


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    get_temp_dir = getattr(env, "get_temp_dir", None)
    temp_dir = None
    if callable(get_temp_dir):
        try:
            temp_dir = get_temp_dir()
        except Exception as exc:
            logger.debug("Could not resolve env temp dir: %s", exc)
    return f"{temp_dir.rstrip('/') or '/'}/hermes-results" if temp_dir else STORAGE_DIR


def _safe_result_filename(tool_use_id: str, ext: str = "txt") -> str:
    """Return a single safe filename for a tool result id."""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id
    safe_stem = safe_stem or "tool_result"
    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"
    return f"{safe_stem}.{ext}"


# Built-in backends that publish ``cache/pruned`` (``tools.credential_files._CACHE_DIRS``): docker bind-mounts
# it, modal/ssh/daytona/vercel_sandbox file-sync it, at the path ``to_agent_visible_cache_path`` names.
_ARCHIVE_PUBLISHING_BACKENDS = frozenset({"docker", "modal", "ssh", "daytona", "vercel_sandbox"})


def _archive_backend() -> str | None:
    """The active terminal backend when ``read_file`` there can open a ``cache/pruned`` archive: ``"local"``
    (host paths) or a backend that publishes the folder. None keeps every result whole: a plugin or unknown
    backend (nothing says it mounts the folder), a hosted scope (its own sandbox, no host cache), or an
    unresolvable policy (refusal scope). Decided on the backend: the prune holds no environment to probe."""
    from tools.environments.sandbox_scope import hosted_scope
    from tools.terminal_scope import TerminalPolicyUnavailable, terminal_env

    if hosted_scope() is not None:
        return None
    try:
        backend = terminal_env("TERMINAL_ENV", "local").strip().lower() or "local"
    except TerminalPolicyUnavailable:
        return None
    return backend if backend == "local" or backend in _ARCHIVE_PUBLISHING_BACKENDS else None


def _backend_reads_host_files() -> bool:
    """True only on the ``local`` backend (outside a hosted scope): the one place a host path is the model's."""
    return _archive_backend() == "local"


def _archive_stub_path(host_path: str) -> str:
    """Where the model's ``read_file`` finds the archive at *host_path*: the ``~/``-relative display path on
    the local backend, else the mounted/synced path (``/root/.hermes/...`` or ``~/.hermes/...``). Always
    ``/``-separated, so the reference scan finds ``cache/pruned/<name>`` in either form."""
    from pathlib import Path, PurePath

    from hermes_constants import display_hermes_home, get_hermes_home

    if _archive_backend() != "local":
        from tools.credential_files import to_agent_visible_cache_path

        return to_agent_visible_cache_path(host_path)
    rel = Path(host_path).relative_to(get_hermes_home())
    return f"{PurePath(display_hermes_home()).as_posix()}/{rel.as_posix()}"


def write_pruned_original(content: str, session_id: str, tool_name: str, tool_call_id: str) -> tuple[str, str] | None:
    """Archive one tool result's full original before the proactive prune shortens it.

    Writes ``$HERMES_HOME/cache/pruned/<session>_<UTC stamp>_<tool>_<short id>.<json|txt>`` (``.json``
    when the body parses as JSON), read back and compared byte-for-byte like a spill. Returns
    ``(host path, stub path)``; the stub path is the ``~/``-relative display path, always ``/``-separated so
    the reference scan finds ``cache/pruned/<name>``; on a sandboxed backend it is the path the sandbox reads
    (:func:`_archive_stub_path`). None when the backend cannot open the archive or the write failed: the
    caller then must keep that result unpruned.
    """
    try:
        json.loads(content)
        ext = "json"
    except ValueError:
        ext = "txt"
    from hermes_constants import get_hermes_home
    if _archive_backend() is None:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short_id = hashlib.sha256(str(tool_call_id or "").encode("utf-8")).hexdigest()[:8]
    archive_dir = get_hermes_home() / PRUNED_SUBDIR
    # Resumed histories can repeat a tool_call_id: suffix until the name is free, then create exclusively.
    filename = next(
        name for n in range(1_000)
        if not (archive_dir / (name := _safe_result_filename(
            f"{session_id}_{stamp}_{tool_name}_{short_id}" + (f"_{n}" if n else ""), ext))).exists())
    host_path = str(archive_dir / filename)
    if _write_to_spillover(content, filename, PRUNED_SUBDIR, exclusive=True) is None:
        return None
    return host_path, _archive_stub_path(host_path)


# ── Archive-before-stub: compaction / proactive-prune originals under cache/pruned/ ──
# Every tool result a prune or compaction is about to shorten is first written, byte-verified, to
# ``$HERMES_HOME/cache/pruned/`` and its stub names that file, so ``read_file`` gets the exact output
# back. A result whose write fails keeps its full body; the files share the spillover sweeper above.

ARCHIVE_POINTER_SEP = " — full result: "
_ARCHIVE_POINTER_RE = re.compile(r" — full result: \S*cache/pruned/[A-Za-z0-9_.-]+")
_ARCHIVE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")  # the names the writer makes and the sweeper finds


def has_archive_pointer(content: str) -> bool:
    """True for a prune stub that names its archived original: already shortened, never re-summarized
    (the pointer is the only way back to the body)."""
    return len(content) < 2_000 and ARCHIVE_POINTER_SEP in content and _ARCHIVE_POINTER_RE.search(content) is not None


def _with_pointer(content: Any, display_path: str) -> Any:
    """Stub *content* with the archive pointer, kept before an attached Skill Safety Rule."""
    pointer = f"{ARCHIVE_POINTER_SEP}{display_path}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": pointer.strip(" —")}]
    from agent.context_compressor_tail import _SKILL_SAFETY_SUFFIX

    head, sep, rule = str(content).partition(_SKILL_SAFETY_SUFFIX)
    return f"{head}{pointer}{sep}{rule}"


def _readback_archive_path(tool_name: str, tool_args: str) -> str | None:
    """The stub path of the archive a ``read_file`` call read back, or None.

    Only when the call's ``path`` is absolute (after ``~``) and names an existing file directly inside
    this home's ``cache/pruned``, on a backend that reads host files: that result's original is already
    archived, so the prune points at it again instead of writing a copy of the rendering. Anything else
    (a look-alike elsewhere, a missing file, a relative path, a sandboxed backend) takes the usual path."""
    if tool_name != "read_file":
        return None
    from pathlib import Path, PurePath

    from agent.context_compressor import _json_dict
    from hermes_constants import display_hermes_home, get_hermes_home

    path = _json_dict(tool_args).get("path")
    if not isinstance(path, str) or not path or not _backend_reads_host_files():
        return None
    try:
        candidate = Path(os.path.expanduser(path))
        if not candidate.is_absolute():
            return None
        candidate = candidate.resolve()
        if (candidate.parent != (get_hermes_home() / PRUNED_SUBDIR).resolve() or not candidate.is_file()
                or not _ARCHIVE_NAME_RE.fullmatch(candidate.name)):
            return None
    except (OSError, ValueError, RuntimeError):  # unreadable path, embedded NUL, symlink loop
        return None
    return f"{PurePath(display_hermes_home()).as_posix()}/{PRUNED_SUBDIR}/{candidate.name}"


def _calls_for_rows(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(tool_name, raw_arguments)`` of the call each row answers, by row index: the nearest PRECEDING assistant
    call with that id. A resumed history can repeat an id, and a whole-transcript map would hand an older result a
    later call's arguments (a later archive read-back would then skip saving that result's own original)."""
    from agent.context_compressor import _tc_get

    seen: dict[str, tuple[str, str]] = {}
    out: list[tuple[str, str]] = []
    for msg in messages:
        out.append(seen.get(str(msg.get("tool_call_id") or ""), ("unknown", "")) if isinstance(msg, dict) else ("unknown", ""))
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = _tc_get(tc, "function", {})
                seen[_tc_get(tc, "id") or ""] = (_tc_get(fn, "name", "unknown"), _tc_get(fn, "arguments"))
    return out


def archive_pruned_originals(
    messages: list[dict[str, Any]], pruned: list[dict[str, Any]], session_id: str,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Write the original of every tool row *pruned* changed; returns ``(rows, written host paths, reverted)``.

    A changed row whose archive write fails is put back to its original (``reverted`` counts them),
    so the caller re-checks its reclaim gate and never commits a stub without a file behind it. A row
    that already names its archive is left as the prune made it: its original is on disk.
    """
    call_at = _calls_for_rows(messages)
    rows, written, reverted = list(pruned), [], 0
    for i, (original, replacement) in enumerate(zip(messages, pruned)):
        if (not isinstance(original, dict) or original.get("role") != "tool"
                or not isinstance(replacement, dict) or replacement.get("content") == original.get("content")):
            continue
        body = original.get("content")
        if isinstance(body, str) and has_archive_pointer(body):
            continue
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        call_id = str(original.get("tool_call_id") or "")
        tool_name, tool_args = call_at[i]
        if (read_back := _readback_archive_path(tool_name, tool_args)) is not None:
            rows[i] = {**replacement, "content": _with_pointer(replacement.get("content"), read_back)}
            continue
        archived = write_pruned_original(text, session_id or "session", tool_name, call_id)
        if archived is None:
            rows[i], reverted = dict(original), reverted + 1
            continue
        written.append(archived[0])
        rows[i] = {**replacement, "content": _with_pointer(replacement.get("content"), archived[1])}
    return rows, written, reverted


def archive_compaction_drops(
    messages: list[dict[str, Any]], dropped: list[dict[str, Any]], session_id: str,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """:func:`archive_pruned_originals` for a compaction's tool-result drops (window prune, lean tail).

    A compaction must still reclaim where ``read_file`` cannot open an archive (plugin backends, a hosted
    scope), so there the plain stubs stand and nothing is written; elsewhere a result whose write fails
    keeps its original, and the compaction proceeds with the rest.
    """
    if _archive_backend() is None:
        return dropped, [], 0
    return archive_pruned_originals(messages, dropped, session_id)


ARCHIVE_LIST_HEADING = "Archived tool output (exact originals; read_file a path to recover one):"


def _row_text(row: Any) -> str:
    content = row.get("content") if isinstance(row, dict) else None
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


def pruned_archive_refs(rows: Any) -> set[str]:
    """``cache/pruned/<name>`` archives the message *rows* point at."""
    return {ref for ref in archive_refs_in_text(*map(_row_text, rows)) if ref.startswith(PRUNED_SUBDIR + "/")}


def archive_list_for_summary(named: set[str], kept: list[dict[str, Any]]) -> str:
    """The *named* archives no row of *kept* (the committed transcript, summary included) points at, as a
    deterministic list to append to the committed summary; ``""`` when none. The generated summary may drop
    a pointer: this list keeps the model's recovery path, and the reference the retention sweep reads,
    whatever the summarizer wrote."""
    from hermes_constants import get_hermes_home

    missing = named - pruned_archive_refs(kept)
    if not missing:
        return ""
    home = get_hermes_home()
    return "\n\n" + ARCHIVE_LIST_HEADING + "".join(
        f"\n- {_archive_stub_path(str(home / ref))}" for ref in sorted(missing))


def discard_archives(paths: list[str]) -> None:
    """Unlink archives written for a prune that did not commit (nothing points at them)."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass  # unreferenced: the retention sweep reclaims it


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    last_nl = content.rfind("\n", 0, max_chars)
    return content[:last_nl + 1 if last_nl > max_chars // 2 else max_chars], True


class _Pageable(NamedTuple):
    """A spill file laid out as ``lead + text + trail``; only ``text`` is the result's own text."""

    text: str
    lead: str = ""
    trail: str = ""
    command_output: bool = False

    @property
    def content(self) -> str:
        return f"{self.lead}{self.text}{self.trail}"


def _tool_metadata_block(fields: dict) -> str:
    metadata = json.dumps(fields, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    return f"{_TOOL_METADATA_TAG}\n{metadata}\n{_TOOL_METADATA_CLOSING_TAG}"


def _pageable_tool_json(content: str, payload: dict) -> _Pageable:
    strings = [(key, value) for key, value in payload.items() if isinstance(value, str)]
    if not strings:
        return _Pageable(content)
    key, text = strings[0]
    for name, value in strings[1:]:
        if len(value) > len(text):
            key, text = name, value
    if "\n" not in text or "\x00" in text or len(text) < len(content) / 2:
        return _Pageable(content)
    header, trailer = {"text_field": key}, {}
    for name, value in payload.items():
        if name != key:
            (trailer if isinstance(value, (dict, list)) else header)[name] = value
    trail = f"\n\n{_tool_metadata_block(trailer)}\n" if trailer else ""
    return _Pageable(text, lead=f"{_tool_metadata_block(header)}\n\n", trail=trail,
                     command_output="exit_code" in payload)


def _pageable_text(content: str) -> _Pageable:
    """Split a JSON tool result into the text it wraps and its metadata, else keep *content* whole.

    Persisting a JSON result verbatim put the whole document on ONE line with escaped newlines,
    so the ``read_file`` offset/limit pagination the ``<persisted-output>`` block recommends could
    not be used (#90426 for MCP envelopes; built-in tools such as terminal, execute_code,
    read_file and skill_view return JSON objects with the same problem).

    Recognized by SHAPE, not by tool name: ``enforce_turn_budget`` persists under
    ``_BUDGET_TOOL_NAME``, so a tool-name test would miss exactly the results it has to fix.
    The MCP envelope (keys a subset of ``{"result", "structuredContent", "_meta"}``) unwraps
    ``result`` and appends its sibling members after the text, because they are the structured
    payloads ``_render_call_tool_result`` deliberately keeps for the model (#115430). Any other
    JSON object unwraps its longest string only when that string is multi-line, NUL-free (a NUL
    makes ``read_file`` treat the file as binary) and at least half the payload, so small or
    opaque JSON stays verbatim. Its scalar keys (``status``, ``exit_code``, ``error``, notes,
    paths) go in a metadata block BEFORE the text, so they stay in the inline preview; its lists
    and dicts go in a second block AFTER it, so they cannot push the text out of the preview (a
    ``skill_view`` result's ``linked_files`` and ``metadata`` filled all of it). Either way
    nothing is dropped: the spill file is the only copy left once the result is replaced by the
    preview.
    """
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return _Pageable(content)
    if not isinstance(payload, dict) or not payload:
        return _Pageable(content)
    if "result" not in payload or not set(payload) <= _MCP_ENVELOPE_KEYS:
        return _pageable_tool_json(content, payload)
    text = payload.get("result")
    if not isinstance(text, str) or not text:
        return _Pageable(content)
    extras = {key: value for key, value in payload.items() if key != "result"}
    if not extras:
        return _Pageable(text)
    try:
        metadata = json.dumps(extras, ensure_ascii=False, indent=1, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return _Pageable(text)
    return _Pageable(
        text, trail=f"\n\n{_ENVELOPE_METADATA_TAG}\n{metadata}\n{_ENVELOPE_METADATA_CLOSING_TAG}\n")


_PRUNED_ARCHIVE_RE = re.compile(r"(?:^|[/\\])cache[/\\]pruned[/\\][^/\\]+\.(?:json|txt)$")


def is_pruned_archive(path: str) -> bool:
    """True for a prune archive (``cache/pruned/<name>.json|.txt``): ``read_file`` pages its text."""
    return isinstance(path, str) and _PRUNED_ARCHIVE_RE.search(path) is not None


def _json_archive_text(raw: str) -> str:
    """A JSON archive pages by the text it wraps; a top-level array is printed one item per line."""
    if raw.lstrip().startswith("["):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, list):
            return json.dumps(payload, ensure_ascii=False, indent=2)
    return _pageable_text(raw).content


def pageable_archive_text(raw: str, path: str, max_line_length: int) -> tuple[str, bool]:
    """The pageable rendering of a prune archive, nothing dropped, and whether long lines were wrapped.

    A JSON archive pages by the text it wraps (the spill files' own rendering). Any line still longer than
    *max_line_length* is split into consecutive lines of that width, so ``offset``/``limit`` reach every
    character: the per-line cap otherwise cut a one-line 300K-char archive at its first 2,000."""
    text = _json_archive_text(raw) if path.endswith(".json") else raw
    width = max(1, max_line_length)
    lines = text.split("\n")
    if all(len(line) <= width for line in lines):
        return text, False
    return "\n".join(
        chunk for line in lines for chunk in ([line[i:i + width] for i in range(0, len(line), width)] or [""])
    ), True


_PREVIEW_TAIL_CHARS = 400
_PREVIEW_MARKER_SLACK = 96


def _head_tail_preview(page: _Pageable, max_chars: int):
    """Lead + head + elision marker + tail preview for command output (a result with ``exit_code``).

    ``generate_preview`` is a head truncation: even with ``exit_code``/``error`` in the lead, a
    failing build's error line sits at the END of a large output and still needed ``read_file``
    (#126444). Compose the preview within *max_chars* so it also carries a tail window of the
    text, never of the metadata block after it; the spillover file keeps the full output -- the
    elision is preview-only. ``None`` when *page* is not command output, fits whole, or has no
    room for a meaningful tail."""
    if not page.command_output:
        return None
    lead, output = page.lead, page.text
    if len(output) <= max_chars - len(lead):
        return None
    tail = output[-min(len(output), max(_PREVIEW_TAIL_CHARS, (max_chars - len(lead)) // 5)):]
    if "\n" in tail[:-1]:
        tail = tail[tail.index("\n") + 1:]
    head_budget = max_chars - len(lead) - len(tail) - _PREVIEW_MARKER_SLACK
    if head_budget <= 0 or len(output) - len(tail) <= head_budget:
        return None
    head = output[:head_budget]
    last_nl = head.rfind("\n")
    if last_nl > head_budget // 2:
        head = head[:last_nl + 1]
    hidden = len(output) - len(head) - len(tail)
    # Neutral on purpose: the inline fallback for a failed save reuses this preview.
    marker = f"\n[... {hidden:,} chars elided ...]\n"
    return f"{lead}{head}{marker}{tail}", True


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(); True on success. Content goes through
    stdin, not the command string: Linux ``MAX_ARG_STRLEN`` caps one argv element at 128 KB,
    so a heredoc-in-command silently failed for exactly the oversized results this handles.

    The write is round-trip verified with ``wc -c`` (one extra exec RTT per oversized result):
    a zero exit from ``cat`` does not prove the bytes landed (quota/ENOSPC races, API-body
    truncation on payload backends). A measured mismatch removes the archive and fails closed;
    an unprobeable backend (no ``wc``, exec error, unparseable output) stays best-effort success."""
    storage_dir = os.path.dirname(remote_path)
    # Private dir: archived results carry tool output (can hold secrets) under a
    # shared temp root on remote backends. The umask also covers the cat redirect.
    from tools.code_execution_rpc import _private_dirs_cmd
    cmd = (f"{_private_dirs_cmd(storage_dir)} "
           f"&& cat > {shlex.quote(remote_path)}")
    if env.execute(cmd, timeout=30, stdin_data=content).get("returncode", 1) != 0:
        return False

    expected = len(content.encode("utf-8", errors="replace"))
    try:
        probe = env.execute(f"wc -c < {shlex.quote(remote_path)}", timeout=15)
    except Exception as exc:
        logger.debug("Sandbox size probe failed for %s: %s", remote_path, exc)
        return True
    if probe.get("returncode", 1) != 0:
        return True
    raw = str(probe.get("output", "") or "").strip().split()
    if not raw or not raw[-1].isdigit():
        return True
    persisted_size = int(raw[-1])
    if persisted_size == expected:
        return True
    logger.warning("Sandbox spill for %s is not lossless (%d bytes in sandbox, expected %d) — discarding archive",
                   remote_path, persisted_size, expected)
    try:
        env.execute(f"rm -f {shlex.quote(remote_path)}", timeout=15)
    except Exception:
        pass
    return False


def _build_persisted_message(preview: str, has_more: bool, original_size: int,
                             file_path: str) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    return (
        f"{PERSISTED_OUTPUT_TAG}\n"
        f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
        f"Full output saved to: {file_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n"
        "Recovery: page through the saved file with read_file (offset/limit) or "
        "process it with execute_code — do NOT re-request the same data from the "
        "remote API; the archive is kept while this session's transcript references "
        "it. If it is missing, inspect the current state before retrying — the call "
        "that produced it may already have taken effect.\n\n"
        f"Preview (first {len(preview)} chars):\n"
        + preview + ("\n..." if has_more else "")
        + f"\n{PERSISTED_OUTPUT_CLOSING_TAG}")


_PERSISTED_PATH_RE = re.compile(r"^Full output saved to: (.+)$", re.MULTILINE)


def extract_persisted_path(content: str) -> str | None:
    """File path from a <persisted-output> block, or None (lets the result-reference stubbing
    guard in agent/tool_guardrails.py carry the spillover path instead of leaving it dangling)."""
    match = (_PERSISTED_PATH_RE.search(content)
             if isinstance(content, str) and PERSISTED_OUTPUT_TAG in content else None)
    return match.group(1).strip() if match else None


def maybe_persist_tool_result(content: str, tool_name: str, tool_use_id: str, env=None,
                              config: BudgetConfig = DEFAULT_BUDGET,
                              threshold: float | None = None) -> str:
    """Layer 2: persist an oversized result, return preview + path. ``threshold`` overrides
    ``config.resolve_threshold(tool_name)``; falls back to inline truncation when no write
    location succeeds."""
    if threshold is None:
        threshold = config.resolve_threshold(tool_name)
    if threshold == float("inf") or len(content) <= threshold:
        return content
    # The size decision above stays on the raw inline result (that is what cost context); the file
    # and the preview carry the pageable text inside an MCP envelope (#90426).
    page = _pageable_text(content)
    persisted_content = page.content
    filename = _safe_result_filename(tool_use_id)
    preview, has_more = _head_tail_preview(page, config.preview_size) or generate_preview(
        persisted_content, max_chars=config.preview_size)

    def _persisted(path: str, host_suffix: str = "") -> str:
        logger.info("Persisted large tool result: %s (%s, %d chars -> %s%s)",
                    tool_name, tool_use_id, len(persisted_content), path, host_suffix)
        return _build_persisted_message(preview, has_more, len(persisted_content), path)

    if (hosted_path := _persist_hosted(persisted_content, tool_use_id, env)) is not None:
        return _persisted(hosted_path) if hosted_path else _inline_truncated(preview, tool_name, content)
    # Always persist host-side first: cache/spillover is the single canonical home.
    host_path = _write_to_spillover(persisted_content, filename)
    host_side = _is_host_side_env(env)
    if host_side and host_path is not None:
        return _persisted(host_path)
    if not host_side:
        # Remote backend: reference the mounted/synced path when the sandbox can actually read
        # it, else write into the sandbox temp dir (containers without the spillover mount).
        visible = _sandbox_visible_spillover_path(host_path, env) if host_path else None
        if visible is not None:
            return _persisted(visible, f" [host: {host_path}]")
        remote_path = f"{_resolve_storage_dir(env)}/{filename}"
        try:
            if _write_to_sandbox(persisted_content, remote_path, env):
                return _persisted(remote_path)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)
    return _inline_truncated(preview, tool_name, content)


def _inline_truncated(preview: str, tool_name: str, content: str) -> str:
    logger.info("Inline-truncating large tool result: %s (%d chars, no sandbox write)",
                tool_name, len(content))
    return (f"{preview}\n\n[Truncated: tool response was {len(content):,} chars. "
            "Full output could not be saved to sandbox.]")


def _persist_hosted(content: str, tool_use_id: str, env) -> str | None:
    """In a hosted scope, write the result straight into the bound agent's own sandbox and return
    its path there (``""`` when that is impossible: no sandbox yet, another agent's environment, or a
    failed write — the caller truncates inline). Nothing is written on this host, and the file name
    carries the sandbox identity, so two agents' equal tool-use ids never share a file. ``None``
    outside a hosted scope: the caller keeps the host spillover."""
    from tools.environments.sandbox_scope import (
        CrossAgentHandleRefused, admit_env, current_sandbox_key)
    sandbox_key = current_sandbox_key()
    if not sandbox_key:
        return None
    try:
        env = admit_env(env)
    except CrossAgentHandleRefused:
        logger.warning("Refused to persist %s into another agent's sandbox", tool_use_id)
        return ""
    if env is None:
        return ""
    stem = hashlib.sha256(sandbox_key.encode("utf-8")).hexdigest()[:16]
    remote_path = f"{_resolve_storage_dir(env)}/{stem}-{_safe_result_filename(tool_use_id)}"
    try:
        return remote_path if _write_to_sandbox(content, remote_path, env) else ""
    except Exception:  # health: allow BLE001 -- a backend's failure falls back to inline truncation, logged with its traceback
        logger.warning("Sandbox write failed for %s", tool_use_id, exc_info=True)
        return ""


def _content_text_size(content) -> int:
    """Characters of TEXT a tool message carries: a string's length, or the summed text parts of a multimodal
    part list (image parts are governed by the vision embed budget, and a list's ``len`` counts parts)."""
    if isinstance(content, list):
        return sum(len(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return len(content or "")


def _spill_largest_text_part(parts: list, tool_use_id: str, env, config: BudgetConfig) -> tuple[list, int]:
    """Persist the largest non-persisted text part of a multimodal list; ``(new parts, chars saved)``."""
    texts = [(i, p["text"]) for i, p in enumerate(parts) if isinstance(p, dict) and p.get("type") == "text"
             and isinstance(p.get("text"), str) and PERSISTED_OUTPUT_TAG not in p["text"]]
    if not texts:
        return parts, 0
    idx, text = max(texts, key=lambda t: len(t[1]))
    replacement = maybe_persist_tool_result(content=text, tool_name=_BUDGET_TOOL_NAME,
                                            tool_use_id=f"{tool_use_id}_part{idx}", env=env, config=config, threshold=0)
    if replacement == text:
        return parts, 0
    return [*parts[:idx], {**parts[idx], "text": replacement}, *parts[idx + 1:]], len(text) - len(replacement)


def enforce_turn_budget(tool_messages: list[dict], env=None,
                        config: BudgetConfig = DEFAULT_BUDGET) -> list[dict]:
    """Layer 3: persist the largest non-persisted results first until the turn's aggregate is
    under budget. Mutates the list in-place and returns it. A multimodal result (part list) counts its
    text characters and spills its largest text part; its images stay inline."""
    sizes = [_content_text_size(msg.get("content", "")) for msg in tool_messages]
    total_size = sum(sizes)
    candidates = [(i, size) for i, size in enumerate(sizes)
                  if isinstance(tool_messages[i].get("content"), list)
                  or PERSISTED_OUTPUT_TAG not in tool_messages[i].get("content", "")]
    if total_size <= config.turn_budget:
        return tool_messages
    for idx, size in sorted(candidates, key=lambda x: x[1], reverse=True):
        if total_size <= config.turn_budget:
            break
        content = tool_messages[idx]["content"]
        tool_use_id = tool_messages[idx].get("tool_call_id", f"budget_{idx}")
        if isinstance(content, list):
            parts, saved = _spill_largest_text_part(content, tool_use_id, env, config)
            if saved:
                total_size -= saved
                tool_messages[idx]["content"] = parts
                logger.info("Budget enforcement: persisted a text part of tool result %s (%d chars)", tool_use_id, saved)
            continue
        replacement = maybe_persist_tool_result(
            content=content, tool_name=_BUDGET_TOOL_NAME, tool_use_id=tool_use_id,
            env=env, config=config, threshold=0)
        if replacement != content:
            total_size += len(replacement) - size
            tool_messages[idx]["content"] = replacement
            logger.info("Budget enforcement: persisted tool result %s (%d chars)",
                        tool_use_id, size)
    return tool_messages
