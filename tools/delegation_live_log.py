"""Live, tail-able transcripts for delegated subagents.

Every ``delegate_task`` dispatch creates one append-only, human-readable log
per child under::

    <hermes_home>/cache/delegation/live/<delegation_id>/task-<n>.log

The files are pre-created with a header at dispatch time (so ``tail -f``
attaches immediately) and then stream one line per child event: assistant
text, thinking, tool calls, tool results, and lifecycle markers. The paths
are returned from ``delegate_task`` so the parent agent (or the user) can
watch a child work instead of waiting blind for the consolidated summary.

Placement under ``cache/delegation`` is deliberate: that directory is
mounted read-only into remote terminal backends (Docker/Modal/SSH) via
``credential_files._CACHE_DIRS``, so the logs are readable from any backend.

Design constraints:

* **Never raise into the agent loop.** Every write is wrapped; the first
  failure disables the writer and degrades to a debug log.
* **Survive child crashes.** Files are opened in append mode per write —
  no long-lived handle to lose, every event is flushed when written.
* **Side-channel only.** Nothing here touches message content, so prompt
  caching is unaffected.
* **No config knobs.** Retention is a module constant (7 days), pruned
  opportunistically on each new dispatch.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Live transcript directories older than this are pruned on new dispatches.
LIVE_RETENTION_DAYS = 7

# Per-line truncation budgets (chars). The .log is a compact operational
# view, not the full-fidelity record — the child's SessionDB transcript and
# the summary spill files carry complete text.
_ASSISTANT_MAX = 600
_THINKING_MAX = 300
_ARGS_MAX = 220
_RESULT_MAX = 400
_KICKOFF_MAX = 500

# Stream deltas are buffered and flushed as one assistant line when another
# event type arrives (or on completion). Cap the buffer so a huge streamed
# reply can't hold memory hostage.
_STREAM_BUFFER_FLUSH_CHARS = 4000


def live_transcript_root() -> Path:
    """Root directory for live transcripts (profile-safe, never ~/.hermes)."""
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("cache/delegation", "delegation_cache") / "live"


def new_live_delegation_id() -> str:
    """Same shape as async_delegation's ids so the dir name matches the handle."""
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _one_line(text: Any, limit: int) -> str:
    """Collapse to a single line and truncate with an elided-chars note."""
    s = str(text or "")
    s = " ".join(s.split())  # collapse newlines/runs of whitespace
    if len(s) > limit:
        omitted = len(s) - limit
        s = s[:limit] + f" …(+{omitted} chars)"
    return s


class LiveTranscriptWriter:
    """Append-only human-readable event log for ONE subagent task.

    All methods are best-effort: the first write failure flips ``_ok`` off
    and subsequent calls become no-ops (debug-logged). Never raises.
    """

    def __init__(self, delegation_id: str, task_index: int, goal: str,
                 context: Optional[str] = None, root: Optional[Path] = None):
        self.delegation_id = delegation_id
        self.task_index = task_index
        self._ok = True
        self._lock = threading.Lock()
        self._stream_buf: List[str] = []
        self._stream_len = 0
        try:
            base = (root if root is not None else live_transcript_root())
            d = base / delegation_id
            d.mkdir(parents=True, exist_ok=True)
            self.path: Optional[Path] = d / f"task-{task_index}.log"
            header = [
                "=== Hermes subagent live transcript ===",
                f"delegation: {delegation_id}   task: {task_index}",
                f"goal: {_one_line(goal, _KICKOFF_MAX)}",
                f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "(append-only; streams while the subagent runs — tail -f me)",
                "=" * 40,
            ]
            self.path.write_text("\n".join(header) + "\n", encoding="utf-8")
            self.event("user", "kickoff: " + _one_line(goal, _KICKOFF_MAX)
                       + (f" | context: {_one_line(context, _KICKOFF_MAX)}" if context else ""))
        except Exception as exc:
            logger.debug("Live transcript init failed (%s task %s): %s",
                         delegation_id, task_index, exc)
            self._ok = False
            self.path = None

    # ── low-level ────────────────────────────────────────────────────────
    def event(self, role: str, text: str) -> None:
        """Append one ``HH:MM:SS role ⟩ text`` line. Flushed per event."""
        if not self._ok or self.path is None:
            return
        line = f"{time.strftime('%H:%M:%S')} {role:<9}| {text}\n"
        try:
            with self._lock:
                # Append mode per write: no held handle, survives child crash,
                # and the close() acts as the flush.
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line)
        except Exception as exc:
            self._ok = False
            logger.debug("Live transcript write failed (%s): %s", self.path, exc)

    # ── typed helpers ────────────────────────────────────────────────────
    def assistant_text(self, text: str) -> None:
        t = _one_line(text, _ASSISTANT_MAX)
        if t:
            self.event("assistant", t)

    def thinking(self, text: str) -> None:
        t = _one_line(text, _THINKING_MAX)
        if t:
            self.event("think", t)

    def tool_start(self, name: str, args_preview: Any = None) -> None:
        self.flush_stream()
        args = _one_line(args_preview, _ARGS_MAX)
        self.event("tool", f"-> {name or '?'}({args})")

    def tool_result(self, name: str, result: Any = None,
                    duration: Any = None, is_error: bool = False) -> None:
        status = "ERROR" if is_error else "ok"
        dur = ""
        try:
            if duration is not None:
                dur = f" {float(duration):.1f}s"
        except (TypeError, ValueError):
            pass
        self.event("result", f"{name or '?'} {status}{dur}: "
                             f"{_one_line(result, _RESULT_MAX)}")

    def marker(self, text: str) -> None:
        """Lifecycle marker: start / final / error / interrupt / budget."""
        self.flush_stream()
        self.event("final", _one_line(text, _ASSISTANT_MAX))

    # ── streamed reply buffering ─────────────────────────────────────────
    def add_stream_delta(self, delta: str) -> None:
        """Buffer streamed assistant reply text; flushed as one line."""
        if not delta or not self._ok:
            return
        self._stream_buf.append(delta)
        self._stream_len += len(delta)
        if self._stream_len >= _STREAM_BUFFER_FLUSH_CHARS:
            self.flush_stream()

    def flush_stream(self) -> None:
        if not self._stream_buf:
            return
        text = "".join(self._stream_buf)
        self._stream_buf = []
        self._stream_len = 0
        self.assistant_text(text)

    # ── event demux (the tool_progress_callback surface) ─────────────────
    def observe(self, event_type: Any, tool_name: Any = None,
                preview: Any = None, args: Any = None, **kwargs: Any) -> None:
        """Map a child tool_progress_callback event onto transcript lines.

        Mirrors the shapes emitted by agent/tool_executor.py,
        agent/conversation_loop.py, and tools/delegate_tool._run_single_child.
        Unknown events are ignored. Never raises (event() swallows I/O).
        """
        et = str(event_type or "")
        if et == "tool.started":
            self.tool_start(str(tool_name or ""), preview if preview else args)
        elif et == "tool.completed":
            self.tool_result(
                str(tool_name or ""),
                result=kwargs.get("result"),
                duration=kwargs.get("duration"),
                is_error=bool(kwargs.get("is_error")),
            )
        elif et == "_thinking":
            # Fired as cb("_thinking", <text>) — the text rides in the
            # tool_name positional slot (see conversation_loop.py).
            self.thinking(str(tool_name or preview or ""))
        elif et == "reasoning.available":
            # cb("reasoning.available", "_thinking", <text>, None)
            self.thinking(str(preview or ""))
        elif et == "subagent.text":
            self.add_stream_delta(str(preview or ""))
        elif et == "subagent.start":
            self.event("start", _one_line(preview, _KICKOFF_MAX))
        elif et == "subagent.complete":
            self.flush_stream()
            status = kwargs.get("status", "?")
            dur = kwargs.get("duration_seconds")
            parts = [f"status={status}"]
            if dur is not None:
                parts.append(f"duration={dur}s")
            summary = kwargs.get("summary") or preview
            if summary:
                parts.append(f"summary: {_one_line(summary, _RESULT_MAX)}")
            self.marker(" ".join(parts))

    def finalize(self, entry: Dict[str, Any]) -> None:
        """Terminal marker from the aggregated result entry.

        Adds exit-reason detail the subagent.complete event doesn't carry
        (budget exhaustion via exit_reason=max_iterations, errors, etc.).
        """
        parts = [f"end status={entry.get('status', '?')}"]
        exit_reason = entry.get("exit_reason")
        if exit_reason:
            parts.append(f"exit_reason={exit_reason}")
        if exit_reason == "max_iterations":
            parts.append("(iteration budget exhausted)")
        if entry.get("error"):
            parts.append(f"error: {_one_line(entry['error'], _RESULT_MAX)}")
        self.marker(" ".join(parts))


def wrap_progress_callback(inner_cb, writer: LiveTranscriptWriter):
    """Wrap a child's tool_progress_callback so events also land in the log.

    ``inner_cb`` may be None (no parent display) — the wrapper still records.
    Writer failures never propagate; inner callback behavior is unchanged
    (its own exceptions are handled by callers exactly as before).
    Preserves the ``_flush`` attribute contract used by _run_single_child.
    """

    def _cb(event_type, tool_name=None, preview=None, args=None, **kwargs):
        try:
            writer.observe(event_type, tool_name, preview, args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — must never hit the agent loop
            logger.debug("Live transcript observe failed: %s", exc)
        if inner_cb is not None:
            inner_cb(event_type, tool_name, preview, args, **kwargs)

    def _flush():
        try:
            writer.flush_stream()
        except Exception:
            pass
        inner_flush = getattr(inner_cb, "_flush", None)
        if callable(inner_flush):
            inner_flush()

    _cb._flush = _flush
    return _cb


# ── dispatch-time helpers ────────────────────────────────────────────────

def create_live_transcripts(
    task_list: List[Dict[str, Any]],
    context: Optional[str] = None,
    delegation_id: Optional[str] = None,
) -> tuple[Optional[str], List[Optional[LiveTranscriptWriter]], List[str]]:
    """Create one pre-headered writer per task + a manifest.json.

    Returns ``(delegation_id, writers, paths)``. On any top-level failure
    returns ``(None, [None]*n, [])`` so delegation proceeds untouched.
    Also opportunistically prunes stale live dirs (retention).
    """
    n = len(task_list)
    try:
        prune_stale_live_dirs()
    except Exception:
        pass
    try:
        deleg_id = delegation_id or new_live_delegation_id()
        writers: List[Optional[LiveTranscriptWriter]] = []
        paths: List[str] = []
        for i, t in enumerate(task_list):
            w = LiveTranscriptWriter(
                deleg_id, i, str(t.get("goal", "")),
                context=t.get("context") or context,
            )
            writers.append(w if w.path is not None else None)
            if w.path is not None:
                paths.append(str(w.path))
        if not paths:
            return None, [None] * n, []
        _write_manifest(deleg_id, task_list, paths)
        return deleg_id, writers, paths
    except Exception as exc:
        logger.debug("Live transcript creation failed: %s", exc)
        return None, [None] * n, []


def _manifest_path(delegation_id: str) -> Path:
    return live_transcript_root() / delegation_id / "manifest.json"


def _write_manifest(delegation_id: str, task_list: List[Dict[str, Any]],
                    paths: List[str]) -> None:
    try:
        manifest = {
            "delegation_id": delegation_id,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_count": len(task_list),
            "tasks": [
                {
                    "index": i,
                    "goal": str(t.get("goal", ""))[:500],
                    "log": paths[i] if i < len(paths) else None,
                    "status": "running",
                }
                for i, t in enumerate(task_list)
            ],
        }
        _manifest_path(delegation_id).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug("Live transcript manifest write failed: %s", exc)


def update_manifest_statuses(delegation_id: Optional[str],
                             results: List[Dict[str, Any]]) -> None:
    """Best-effort per-task status update once the batch has aggregated."""
    if not delegation_id:
        return
    try:
        mp = _manifest_path(delegation_id)
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        by_index = {r.get("task_index"): r for r in results if isinstance(r, dict)}
        for task in manifest.get("tasks", []):
            r = by_index.get(task.get("index"))
            if r is not None:
                task["status"] = r.get("status", task.get("status"))
                if r.get("exit_reason"):
                    task["exit_reason"] = r["exit_reason"]
        manifest["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    except Exception as exc:
        logger.debug("Live transcript manifest update failed: %s", exc)


def prune_stale_live_dirs(max_age_days: int = LIVE_RETENTION_DAYS) -> int:
    """Remove live/<delegation_id> dirs older than the retention window.

    Returns how many were removed. Fully best-effort.
    """
    removed = 0
    try:
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
    except Exception as exc:
        logger.debug("Live transcript pruning failed: %s", exc)
    return removed
