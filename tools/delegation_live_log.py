"""Live, tail-able transcripts for delegated subagents.

One append-only log per child under ``<hermes_home>/cache/delegation/live/
<delegation_id>/task-<n>.log``, pre-created with a header at dispatch (so
``tail -f`` attaches immediately); paths are returned from ``delegate_task``.
``cache/delegation`` is mounted read-only into remote terminal backends, so
every line written here must be credential-redacted. Never raises into the
agent loop; append mode per write (close() is the flush); 7-day retention.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LIVE_RETENTION_DAYS = 7

# Per-line truncation budgets (chars): the .log is a compact operational view;
# the child's SessionDB transcript and summary spill files carry full text.
_ASSISTANT_MAX = 600
_THINKING_MAX = 300
_ARGS_MAX = 220
_RESULT_MAX = 400
_KICKOFF_MAX = 500
# Stream deltas are buffered and flushed as one assistant line when another
# event type arrives (or on completion); capped so a huge reply can't hold memory.
_STREAM_BUFFER_FLUSH_CHARS = 4000
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def live_transcript_root() -> Path:
    """Root directory for live transcripts (profile-safe, never ~/.hermes)."""
    from hermes_constants import get_hermes_dir
    return get_hermes_dir("cache/delegation", "delegation_cache") / "live"


@contextmanager
def _best_effort(what: str):
    """Swallow and debug-log any failure: nothing here may reach the agent loop."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        logger.debug("Live transcript %s failed: %s", what, exc)


def _one_line(text: Any, limit: int) -> str:
    """Collapse to a single line and truncate with an elided-chars note."""
    s = " ".join(str(text or "").split())
    if len(s) > limit:
        s = s[:limit] + f" …(+{len(s) - limit} chars)"
    return s


def _redact(text: str) -> str:
    """Mask credentials (``force=True``: safety boundary, even when the global
    toggle is off); if the redactor is unavailable, withhold rather than leak."""
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True) or ""
    except Exception:  # pragma: no cover - core module; never leak on failure
        return "[line withheld: redaction unavailable]"


def _joined(*parts: str) -> str:
    return " ".join(filter(None, parts))


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class LiveTranscriptWriter:
    """Append-only event log for ONE subagent task. Best-effort: the first write
    failure flips ``_ok`` off and later calls become debug-logged no-ops."""

    def __init__(self, delegation_id: str, task_index: int, goal: str,
                 context: Optional[str] = None, root: Optional[Path] = None):
        self.delegation_id = delegation_id
        self.task_index = task_index
        self._ok = False
        self._lock = threading.Lock()
        self._stream_buf: List[str] = []
        self._stream_len = 0
        self.path: Optional[Path] = None
        with _best_effort(f"init ({delegation_id} task {task_index})"):
            goal_line = _one_line(goal, _KICKOFF_MAX)
            d = (root if root is not None else live_transcript_root()) / delegation_id
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"task-{task_index}.log"
            path.write_text(
                "=== Hermes subagent live transcript ===\n"
                f"delegation: {delegation_id}   task: {task_index}\n"
                f"goal: {_redact(goal_line)}\n"  # header bypasses event(), so redact here too
                f"started: {time.strftime(_TIME_FMT)}\n"
                "(append-only; streams while the subagent runs — tail -f me)\n"
                + "=" * 40 + "\n", encoding="utf-8")
            self.path, self._ok = path, True
            self.event("user", "kickoff: " + goal_line
                       + (f" | context: {_one_line(context, _KICKOFF_MAX)}" if context else ""))

    def event(self, role: str, text: str) -> None:
        """Append one ``HH:MM:SS role | text`` line. Single choke point: every typed
        helper funnels through here so one redaction covers everything."""
        if not self._ok or self.path is None:
            return
        line = f"{time.strftime('%H:%M:%S')} {role:<9}| {_redact(text)}\n"
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:
            self._ok = False
            logger.debug("Live transcript write failed (%s): %s", self.path, exc)

    def _line(self, role: str, text: str, limit: int) -> None:
        if t := _one_line(text, limit):
            self.event(role, t)

    def assistant_text(self, text: str) -> None:
        self._line("assistant", text, _ASSISTANT_MAX)

    def thinking(self, text: str) -> None:
        self._line("think", text, _THINKING_MAX)

    def tool_start(self, name: str, args_preview: Any = None) -> None:
        self.flush_stream()
        self.event("tool", f"-> {name or '?'}({_one_line(args_preview, _ARGS_MAX)})")

    def tool_result(self, name: str, result: Any = None,
                    duration: Any = None, is_error: bool = False) -> None:
        status = "ERROR" if is_error else "ok"
        try:
            dur = "" if duration is None else f" {float(duration):.1f}s"
        except (TypeError, ValueError):
            dur = ""
        self.event("result", f"{name or '?'} {status}{dur}: {_one_line(result, _RESULT_MAX)}")

    def marker(self, text: str) -> None:
        """Lifecycle marker: start / final / error / interrupt / budget."""
        self.flush_stream()
        self.event("final", _one_line(text, _ASSISTANT_MAX))

    def add_stream_delta(self, delta: str) -> None:
        """Buffer streamed assistant reply text; flushed as one line."""
        if not delta or not self._ok:
            return
        self._stream_buf.append(delta)
        self._stream_len += len(delta)
        if self._stream_len >= _STREAM_BUFFER_FLUSH_CHARS:
            self.flush_stream()

    def flush_stream(self) -> None:
        if self._stream_buf:
            text, self._stream_buf, self._stream_len = "".join(self._stream_buf), [], 0
            self.assistant_text(text)

    def _on_complete(self, tool_name, preview, args, kwargs):
        dur = kwargs.get("duration_seconds")
        summary = kwargs.get("summary") or preview
        self.marker(_joined(
            f"status={kwargs.get('status', '?')}",
            f"duration={dur}s" if dur is not None else "",
            f"summary: {_one_line(summary, _RESULT_MAX)}" if summary else ""))

    # Event demux (the tool_progress_callback surface): handler(self, tool_name, preview, args, kwargs).
    _OBSERVERS = {
        "tool.started": lambda s, n, p, a, kw: s.tool_start(str(n or ""), p if p else a),
        "tool.completed": lambda s, n, p, a, kw: s.tool_result(
            str(n or ""), result=kw.get("result"), duration=kw.get("duration"),
            is_error=bool(kw.get("is_error"))),
        # Fired as cb("_thinking", <text>) — text rides in the tool_name slot.
        "_thinking": lambda s, n, p, a, kw: s.thinking(str(n or p or "")),
        # cb("reasoning.available", "_thinking", <text>, None)
        "reasoning.available": lambda s, n, p, a, kw: s.thinking(str(p or "")),
        "subagent.text": lambda s, n, p, a, kw: s.add_stream_delta(str(p or "")),
        "subagent.start": lambda s, n, p, a, kw: s.event("start", _one_line(p, _KICKOFF_MAX)),
        "subagent.complete": _on_complete}

    def observe(self, event_type: Any, tool_name: Any = None, preview: Any = None,
                args: Any = None, **kwargs: Any) -> None:
        """Map a child tool_progress_callback event onto transcript lines.
        Unknown events are ignored. Never raises (event() swallows I/O)."""
        handler = self._OBSERVERS.get(str(event_type or ""))
        if handler is not None:
            handler(self, tool_name, preview, args, kwargs)

    def finalize(self, entry: Dict[str, Any]) -> None:
        """Terminal marker with exit-reason detail subagent.complete lacks."""
        exit_reason = entry.get("exit_reason")
        self.marker(_joined(
            f"end status={entry.get('status', '?')}",
            f"exit_reason={exit_reason}" if exit_reason else "",
            "(iteration budget exhausted)" if exit_reason == "max_iterations" else "",
            f"error: {_one_line(entry['error'], _RESULT_MAX)}" if entry.get("error") else ""))


def wrap_progress_callback(inner_cb, writer: LiveTranscriptWriter):
    """Wrap a child's tool_progress_callback (may be None) so events also land in
    the log; writer failures never propagate. Preserves the ``_flush`` contract."""

    def _cb(event_type, tool_name=None, preview=None, args=None, **kwargs):
        with _best_effort("observe"):
            writer.observe(event_type, tool_name, preview, args, **kwargs)
        if inner_cb is not None:
            inner_cb(event_type, tool_name, preview, args, **kwargs)

    def _flush():
        with _best_effort("flush"):
            writer.flush_stream()
        if callable(getattr(inner_cb, "_flush", None)):
            inner_cb._flush()

    _cb._flush = _flush
    return _cb


def create_live_transcripts(
    task_list: List[Dict[str, Any]], context: Optional[str] = None,
    delegation_id: Optional[str] = None, model: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple[Optional[str], List[Optional[LiveTranscriptWriter]], List[str]]:
    """One pre-headered writer per task + a manifest.json; prunes stale dirs.
    Returns ``(delegation_id, writers, paths)``; on any top-level failure
    ``(None, [None]*n, [])`` so delegation proceeds untouched."""
    n = len(task_list)
    prune_stale_live_dirs()  # best-effort; never raises
    with _best_effort("creation"):
        # Same id shape as async_delegation's so the dir name matches the handle.
        deleg_id = delegation_id or f"deleg_{uuid.uuid4().hex[:8]}"
        made = [LiveTranscriptWriter(deleg_id, i, str(t.get("goal", "")), context=t.get("context") or context)
                for i, t in enumerate(task_list)]
        writers: List[Optional[LiveTranscriptWriter]] = [w if w.path is not None else None for w in made]
        paths: List[str] = [str(w.path) for w in made if w.path is not None]
        if not paths:
            return None, [None] * n, []
        _write_manifest(deleg_id, task_list, paths, model=model, provider=provider)
        return deleg_id, writers, paths
    return None, [None] * n, []


def _manifest_path(delegation_id: str) -> Path:
    return live_transcript_root() / delegation_id / "manifest.json"


def _write_manifest(delegation_id: str, task_list: List[Dict[str, Any]],
                    paths: List[str], model: Optional[str] = None,
                    provider: Optional[str] = None) -> None:
    with _best_effort("manifest write"):
        _dump_json(_manifest_path(delegation_id), {
            "delegation_id": delegation_id, "started": time.strftime(_TIME_FMT),
            "task_count": len(task_list), "model": model, "provider": provider,
            "tasks": [{
                "index": i,
                # Same mounted dir as the .log files, so the goal needs the same redaction.
                "goal": _redact(str(t.get("goal", ""))[:500]),
                "log": paths[i] if i < len(paths) else None,
                "status": "running"} for i, t in enumerate(task_list)]})


def update_manifest_statuses(delegation_id: Optional[str],
                             results: List[Dict[str, Any]]) -> None:
    """Best-effort per-task status update once the batch has aggregated."""
    if not delegation_id:
        return
    with _best_effort("manifest update"):
        mp = _manifest_path(delegation_id)
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        by_index = {r.get("task_index"): r for r in results if isinstance(r, dict)}
        for task in manifest.get("tasks", []):
            r = by_index.get(task.get("index"))
            if r is not None:
                task["status"] = r.get("status", task.get("status"))
                if r.get("exit_reason"):
                    task["exit_reason"] = r["exit_reason"]
        manifest["completed"] = time.strftime(_TIME_FMT)
        _dump_json(mp, manifest)


def prune_stale_live_dirs(max_age_days: int = LIVE_RETENTION_DAYS) -> int:
    """Remove live/<delegation_id> dirs older than the retention window. Best-effort."""
    removed = 0
    with _best_effort("pruning"):
        root = live_transcript_root()
        if not root.is_dir():
            return 0
        cutoff = time.time() - max_age_days * 86400
        for child in root.iterdir():
            try:
                if child.is_dir() and child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    return removed


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def new_live_delegation_id() -> str:
    """Same shape as async_delegation's ids so the dir name matches the handle."""
    return f"deleg_{uuid.uuid4().hex[:8]}"
# ---- END PLUGIN-COMPAT ----
