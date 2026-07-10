"""Process-based subagent isolation for delegate_task.

The default delegation path runs child AIAgents on ``ThreadPoolExecutor``
workers inside the parent process.  Every child then shares the parent's
GIL, so a batch of subagents doing CPU-heavy work (SSE parsing, regex,
JSON) starves the parent's event loop — the TUI/desktop WebSocket stalls
for tens of seconds and the desktop app drops the connection (issues
#58576, #57903, #32079).

This module is the opt-in fix (``delegation.process_isolation: true`` in
config.yaml): each batch child runs in its own OS process with its own
GIL, mirroring how OpenClaw spawns subagents as separate Node processes.

Architecture
------------
* ``ChildProcessSpec`` — parent-side handle built by
  ``delegate_tool._build_child_process_spec``.  Carries only *picklable*
  constructor params for the child AIAgent plus parent-side plumbing
  (progress callback, credential lease).  Presents the same duck-typed
  surface the aggregator expects from a child agent (``session_id``,
  ``_delegate_role``, ``interrupt()``).
* ``_child_process_main`` — the ``spawn`` entry point.  Reconstructs the
  AIAgent inside the child process (fresh SQLite connection, fresh httpx
  clients), runs the conversation, and streams progress events back over
  an IPC queue as plain dicts.
* ``run_children_in_processes`` — parent-side poll loop.  Relays queued
  child events into the exact same progress callbacks the thread path
  uses (the TUI/gateway cannot tell the difference), heartbeats parent
  activity from child activity snapshots, propagates interrupts via a
  shared ``multiprocessing.Event``, enforces the optional child timeout
  (processes CAN be hard-killed, unlike threads), and reaps every child
  so no zombies survive the batch.

Known semantic difference vs. the thread path: the cross-agent
file-state registry (``tools.file_state``) is per-process.  The child
checks its own writes against the parent's read snapshot (passed in via
params) and the parent folds the child's writes back into its registry on
completion, so the "subagent modified files you read" reminder and the
parent's stale-write guard both keep working — but two *sibling* children
cannot see each other's in-flight writes while both are still running.
"""

from __future__ import annotations

import logging
import queue as _queue_mod
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Seconds a child gets to unwind after an interrupt / timeout signal
# before the parent hard-terminates its process.
_TERMINATE_GRACE_SECONDS = 10.0
# Seconds to keep waiting for a dead child's result to surface from the
# IPC queue (the feeder thread may still be flushing) before fabricating
# a crash entry.
_DEAD_RESULT_GRACE_SECONDS = 5.0


def _sanitize(obj: Any, _depth: int = 0) -> Any:
    """Return a picklable, plain-data copy of *obj* for IPC transport."""
    if _depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v, _depth + 1) for v in obj]
    return str(obj)


class ChildProcessSpec:
    """Parent-side handle for one process-isolated subagent.

    Duck-types the slice of the child-agent surface that
    ``delegate_task``'s aggregation code touches (``session_id``,
    ``_delegate_role``, ``_subagent_id``, ``interrupt()``) so the generic
    hook/memory/cost plumbing works unchanged in process mode.
    """

    def __init__(
        self,
        *,
        task_index: int,
        goal: str,
        params: Dict[str, Any],
        progress_cb: Optional[Callable] = None,
        pool=None,
        leased_cred_id: Optional[str] = None,
    ) -> None:
        self.task_index = task_index
        self.goal = goal
        self.params = params
        self.progress_cb = progress_cb
        self.pool = pool
        self.leased_cred_id = leased_cred_id

        self.session_id = params.get("session_id") or ""
        self.model = params.get("model")
        self._subagent_id = params.get("subagent_id")
        self._parent_subagent_id = params.get("parent_subagent_id")
        self._delegate_role = params.get("role", "leaf")
        self._delegate_depth = params.get("child_depth", 1)
        self._subagent_goal = goal

        self.process = None
        self.interrupt_event = None  # created once the mp context exists
        self._lock = threading.Lock()
        self._interrupt_requested = False
        self._interrupt_message: Optional[str] = None

    def attach_context(self, ctx) -> None:
        """Create the shared interrupt Event on the spawn context.

        Called by the runner just before ``Process(...)`` is built.  An
        interrupt requested earlier (e.g. background-batch cancellation
        racing process startup) is replayed onto the fresh Event.
        """
        with self._lock:
            self.interrupt_event = ctx.Event()
            if self._interrupt_requested:
                self.interrupt_event.set()

    def interrupt(self, message: Optional[str] = None) -> None:
        """Request the child process stop at its next iteration boundary."""
        with self._lock:
            self._interrupt_requested = True
            self._interrupt_message = message
            if self.interrupt_event is not None:
                try:
                    self.interrupt_event.set()
                except Exception:
                    pass

    def flush_progress(self) -> None:
        cb = self.progress_cb
        if cb is not None and hasattr(cb, "_flush"):
            try:
                cb._flush()
            except Exception as exc:
                logger.debug("Progress callback flush failed: %s", exc)

    def release(self) -> None:
        """Release the credential leased for this child (idempotent)."""
        pool, cred = self.pool, self.leased_cred_id
        self.pool = None
        self.leased_cred_id = None
        if pool is not None and cred is not None:
            try:
                pool.release_lease(cred)
            except Exception as exc:
                logger.debug("Failed to release credential lease: %s", exc)


def _resolve_agent_factory(dotted: Optional[str]):
    """Resolve the AIAgent constructor inside the child process.

    ``dotted`` ("module:attr") exists so tests can substitute a stub agent
    across the spawn boundary, where monkeypatching cannot reach.  Real
    delegations always leave it unset and get run_agent.AIAgent.
    """
    if dotted:
        import importlib

        mod_name, _, attr = dotted.partition(":")
        return getattr(importlib.import_module(mod_name), attr)
    from run_agent import AIAgent

    return AIAgent


def _child_process_main(params, event_queue, result_queue, interrupt_event) -> None:
    """Entry point executed inside the spawned child process.

    Builds the child AIAgent from picklable *params*, runs the delegated
    conversation, relays progress/activity over *event_queue*, and puts
    exactly one result entry (the same shape ``_run_single_child``
    returns) on *result_queue* before exiting.
    """
    task_index = int(params.get("task_index", 0))
    goal = params.get("goal") or ""
    child_start = time.monotonic()
    wall_start = time.time()

    def emit(event_type, tool_name=None, preview=None, args=None, **kwargs) -> None:
        try:
            event_queue.put(
                {
                    "kind": "progress",
                    "task_index": task_index,
                    "event": (
                        event_type,
                        tool_name,
                        _sanitize(preview),
                        _sanitize(args),
                        _sanitize(kwargs),
                    ),
                }
            )
        except Exception:
            pass

    entry: Dict[str, Any] = {
        "task_index": task_index,
        "status": "error",
        "summary": None,
        "error": "Subagent process exited before producing a result.",
        "api_calls": 0,
        "duration_seconds": 0,
        "_child_role": params.get("role"),
    }
    agent = None
    stop = threading.Event()
    try:
        # Non-interactive approval callback: same policy as the thread path
        # (delegation.subagent_auto_approve), installed on this process's
        # main thread since the child owns no TUI/stdin.
        from tools.delegate_tool import (
            _subagent_auto_approve,
            _subagent_auto_deny,
            _summarize_child_run,
        )
        from tools.terminal_tool import set_approval_callback

        set_approval_callback(
            _subagent_auto_approve
            if params.get("auto_approve")
            else _subagent_auto_deny
        )

        session_db = None
        db_path = params.get("session_db_path")
        if db_path:
            try:
                from pathlib import Path

                from hermes_state import SessionDB

                session_db = SessionDB(Path(db_path))
            except Exception as exc:
                logger.warning(
                    "[subagent-%d] could not open session DB %s: %s",
                    task_index,
                    db_path,
                    exc,
                )

        def thinking_cb(text: str) -> None:
            if text:
                emit("_thinking", text)

        factory = _resolve_agent_factory(params.get("agent_factory"))
        # Collect extra params (non-constructor keys) so test stub factories
        # can receive test configuration across the spawn boundary.
        # ONLY pass keys that the real AIAgent constructor doesn't know about
        # — the stub factory accepts **kwargs and reads what it needs, but the
        # real AIAgent will TypeError on unknown kwargs.
        _constructor_keys = {
            "base_url", "api_key", "model", "provider", "api_mode",
            "acp_command", "acp_args", "max_iterations", "max_tokens",
            "reasoning_config", "prefill_messages", "fallback_model",
            "enabled_toolsets", "ephemeral_system_prompt", "session_id",
            "parent_session_id", "providers_allowed", "providers_ignored",
            "providers_order", "provider_sort", "openrouter_min_coding_score",
            "task_index", "task_count", "subagent_id", "parent_subagent_id",
            "child_depth", "parent_turn_id", "session_db_path",
            "parent_reads_snapshot", "parent_session_id", "auto_approve",
            "heartbeat_interval", "agent_factory", "role", "goal",
        }
        _extra = {
            k: v for k, v in params.items()
            if k not in _constructor_keys
        }
        # Always provide these for stub factories that need them.
        # The real AIAgent ignores **_extra kwargs it doesn't accept IF
        # they're not in its signature — but it DOES TypeError on unknown
        # kwargs, so we must only add them when using a stub factory.
        _is_stub = params.get("agent_factory") is not None
        if _is_stub:
            _extra["goal"] = goal
            _extra["role"] = params.get("role", "leaf")
            _extra["subagent_id"] = params.get("subagent_id")
            _extra["parent_subagent_id"] = params.get("parent_subagent_id")
            _extra["_test_config"] = params.get("_test_config")
        agent = factory(
            base_url=params.get("base_url"),
            api_key=params.get("api_key"),
            model=params.get("model") or "",
            provider=params.get("provider"),
            api_mode=params.get("api_mode"),
            acp_command=params.get("acp_command"),
            acp_args=params.get("acp_args"),
            max_iterations=params.get("max_iterations") or 50,
            max_tokens=params.get("max_tokens"),
            reasoning_config=params.get("reasoning_config"),
            prefill_messages=params.get("prefill_messages"),
            fallback_model=params.get("fallback_model"),
            enabled_toolsets=params.get("enabled_toolsets"),
            quiet_mode=True,
            ephemeral_system_prompt=params.get("ephemeral_system_prompt"),
            log_prefix=f"[subagent-{task_index}]",
            platform="subagent",
            skip_context_files=True,
            skip_memory=True,
            clarify_callback=None,
            thinking_callback=thinking_cb,
            session_db=session_db,
            session_id=params.get("session_id"),
            parent_session_id=params.get("parent_session_id"),
            providers_allowed=params.get("providers_allowed"),
            providers_ignored=params.get("providers_ignored"),
            providers_order=params.get("providers_order"),
            provider_sort=params.get("provider_sort"),
            openrouter_min_coding_score=params.get("openrouter_min_coding_score"),
            tool_progress_callback=emit,
            iteration_budget=None,  # fresh budget per subagent
            **_extra,
        )
        agent._delegate_depth = params.get("child_depth", 1)
        agent._delegate_role = params.get("role", "leaf")
        agent._subagent_id = params.get("subagent_id")
        agent._parent_subagent_id = params.get("parent_subagent_id")
        agent._subagent_goal = goal
        agent._parent_turn_id = params.get("parent_turn_id") or ""
        parent_sid = params.get("parent_session_id")
        if parent_sid and getattr(agent, "_session_init_model_config", None) is not None:
            agent._session_init_model_config["_delegate_from"] = parent_sid

        # Watcher thread: propagates the parent's interrupt Event into
        # agent.interrupt() and pushes periodic activity snapshots so the
        # parent's heartbeat/staleness monitor keeps working.
        hb_interval = float(params.get("heartbeat_interval") or 30)
        poll_s = 0.5
        activity_every = max(1, int(hb_interval / poll_s))

        def _watcher() -> None:
            tick = 0
            signaled = False
            while not stop.wait(poll_s):
                if not signaled and interrupt_event.is_set():
                    signaled = True
                    try:
                        agent.interrupt("Interrupted by parent")
                    except Exception:
                        pass
                tick += 1
                if tick % activity_every == 0:
                    try:
                        event_queue.put(
                            {
                                "kind": "activity",
                                "task_index": task_index,
                                "summary": _sanitize(agent.get_activity_summary()),
                            }
                        )
                    except Exception:
                        pass

        threading.Thread(target=_watcher, daemon=True).start()

        emit("subagent.start", preview=goal)

        child_task_id = params.get("subagent_id") or f"subagent-{task_index}"

        def _relay_child_text(delta: str) -> None:
            if delta:
                emit("subagent.text", preview=delta)

        result = agent.run_conversation(
            user_message=goal,
            task_id=child_task_id,
            stream_callback=_relay_child_text,
        )
        stop.set()

        duration = round(time.monotonic() - child_start, 2)
        entry, complete_kwargs = _summarize_child_run(
            agent, result, task_index, duration, child_task_id, wall_start
        )

        # File-state coordination: this process's registry only saw THIS
        # child's tool calls, so check its writes against the parent's read
        # snapshot here and hand the touched paths back for the parent to
        # fold into its own registry.
        try:
            from tools import file_state

            parent_reads = params.get("parent_reads_snapshot") or []
            written_map = file_state.writes_since("", wall_start, parent_reads)
            mod_paths = sorted({p for paths in written_map.values() for p in paths})
            if mod_paths:
                entry["_written_paths"] = mod_paths
                reminder = (
                    "\n\n[NOTE: subagent modified files the parent "
                    "previously read — re-read before editing: "
                    + ", ".join(mod_paths[:8])
                    + (f" (+{len(mod_paths) - 8} more)" if len(mod_paths) > 8 else "")
                    + "]"
                )
                if entry.get("summary"):
                    entry["summary"] = entry["summary"] + reminder
                else:
                    entry["stale_paths"] = mod_paths
        except Exception:
            logger.debug("file_state child-write check failed", exc_info=True)

        emit("subagent.complete", **complete_kwargs)

    except Exception as exc:
        stop.set()
        duration = round(time.monotonic() - child_start, 2)
        logging.exception("[subagent-%d] failed in isolated process", task_index)
        entry = {
            "task_index": task_index,
            "status": "error",
            "summary": None,
            "error": str(exc),
            "api_calls": 0,
            "duration_seconds": duration,
            "_child_role": params.get("role"),
        }
        emit(
            "subagent.complete",
            preview=str(exc),
            status="failed",
            duration_seconds=duration,
            summary=str(exc),
        )
    finally:
        stop.set()
        try:
            result_queue.put(_sanitize(entry))
        except Exception:
            try:
                result_queue.put(
                    {
                        "task_index": task_index,
                        "status": "error",
                        "summary": None,
                        "error": "Subagent result could not be serialised.",
                        "api_calls": 0,
                        "duration_seconds": round(time.monotonic() - child_start, 2),
                        "_child_role": params.get("role"),
                    }
                )
            except Exception:
                pass
        try:
            if agent is not None and hasattr(agent, "close"):
                agent.close()
        except Exception:
            logger.debug("Failed to close child agent in isolated process")


def run_children_in_processes(
    *,
    children: List[Tuple[int, Dict[str, Any], ChildProcessSpec]],
    parent_agent,
    child_timeout: Optional[float],
    on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run a batch of subagents in isolated OS processes and join on them.

    Parent-side counterpart of ``_child_process_main``.  Returns one entry
    per child (same shape as the thread path), sorted by task_index.
    Guarantees no child process outlives the call.
    """
    import multiprocessing as mp

    from tools import file_state
    from tools import delegate_tool as dt

    # "spawn" (not fork): the parent holds live httpx/SSL clients, SQLite
    # connections, and running threads — fork would inherit them in a
    # corrupt state. Spawn gives each child a clean interpreter.
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    result_queue = ctx.Queue()

    wall_start = time.time()
    started_mono = time.monotonic()

    specs: Dict[int, ChildProcessSpec] = {}
    procs: Dict[int, Any] = {}
    entries: Dict[int, Dict[str, Any]] = {}
    dead_since: Dict[int, float] = {}
    timeout_signaled: Dict[int, float] = {}
    child_start_mono: Dict[int, float] = {}  # per-child proc.start() timestamp
    # Per-child heartbeat staleness state (port of _run_single_child's
    # heartbeat loop): [last_iter, last_tool, stale_count, stopped]
    hb_state: Dict[int, list] = {}
    interrupt_signaled_at: Optional[float] = None

    parent_session_id = getattr(parent_agent, "session_id", None)
    parent_turn_id = getattr(parent_agent, "_current_turn_id", "") or ""

    def _fire_start_hook(spec: ChildProcessSpec) -> None:
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook

            _invoke_hook(
                "subagent_start",
                parent_session_id=parent_session_id,
                parent_turn_id=parent_turn_id,
                parent_subagent_id=spec._parent_subagent_id,
                child_session_id=spec.session_id,
                child_subagent_id=spec._subagent_id,
                child_role=spec._delegate_role,
                child_goal=spec.goal,
            )
        except Exception:
            logger.debug("subagent_start hook invocation failed", exc_info=True)

    def _finalize(i: int, entry: Dict[str, Any]) -> None:
        entries[i] = entry
        spec = specs.get(i)
        if spec is None:
            return
        # Fold the child's writes into the parent's file-state registry so
        # the parent's own stale-write guard sees them on later edits.
        for path in entry.pop("_written_paths", []) or []:
            try:
                file_state.note_write(spec._subagent_id or f"subagent-{i}", path)
            except Exception:
                pass
        spec.flush_progress()
        spec.release()
        if spec._subagent_id:
            dt._unregister_subagent(spec._subagent_id)
        proc = procs.get(i)
        if proc is not None:
            try:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
            except Exception:
                pass
        if on_complete is not None:
            try:
                on_complete(entry)
            except Exception:
                logger.debug("on_complete callback failed", exc_info=True)

    def _fabricated(i: int, status: str, error: str) -> Dict[str, Any]:
        spec = specs.get(i)
        return {
            "task_index": i,
            "status": status,
            "summary": None,
            "error": error,
            "exit_reason": status,
            "api_calls": 0,
            "duration_seconds": round(time.monotonic() - started_mono, 2),
            "_child_role": getattr(spec, "_delegate_role", None),
        }

    # ── Launch ───────────────────────────────────────────────────────
    for i, _t, spec in children:
        specs[i] = spec
        hb_state[i] = [0, None, 0, False]
        spec.attach_context(ctx)
        if spec._subagent_id:
            dt._register_subagent(
                {
                    "subagent_id": spec._subagent_id,
                    "parent_id": spec._parent_subagent_id,
                    "depth": max(0, (spec._delegate_depth or 1) - 1),
                    "goal": spec.goal,
                    "model": spec.model if isinstance(spec.model, str) else None,
                    "started_at": time.time(),
                    "status": "running",
                    "tool_count": 0,
                    "agent": spec,
                }
            )
        if spec.progress_cb:
            try:
                spec.progress_cb("subagent.spawn_requested", preview=spec.goal)
            except Exception as exc:
                logger.debug("spawn_requested relay failed: %s", exc)
        proc = ctx.Process(
            target=_child_process_main,
            args=(spec.params, event_queue, result_queue, spec.interrupt_event),
            name=f"hermes-subagent-{i}",
        )
        spec.process = proc
        try:
            proc.start()
        except Exception as exc:
            logger.warning("Failed to start subagent process %d: %s", i, exc)
            _finalize(i, _fabricated(i, "error", f"Failed to start subagent process: {exc}"))
            continue
        child_start_mono[i] = time.monotonic()
        procs[i] = proc
        _fire_start_hook(spec)

    pending = set(procs.keys()) - set(entries.keys())

    def _handle_activity(i: int, summary: Dict[str, Any]) -> None:
        """Port of the thread path's heartbeat: touch parent activity while
        the child makes progress; go quiet once it looks stale so the
        gateway inactivity timeout can fire."""
        st = hb_state.get(i)
        if st is None or st[3]:
            return
        child_tool = summary.get("current_tool")
        child_iter = summary.get("api_call_count", 0) or 0
        child_max = summary.get("max_iterations", 0)
        if child_iter > st[0] or child_tool != st[1]:
            st[0], st[1], st[2] = child_iter, child_tool, 0
        else:
            st[2] += 1
        stale_limit = (
            dt._HEARTBEAT_STALE_CYCLES_IN_TOOL
            if child_tool
            else dt._HEARTBEAT_STALE_CYCLES_IDLE
        )
        if st[2] >= stale_limit:
            st[3] = True
            logger.warning(
                "Subagent %d appears stale (no progress for %d heartbeat "
                "cycles, tool=%s) — stopping heartbeat",
                i,
                st[2],
                child_tool or "<none>",
            )
            return
        if child_tool:
            desc = (
                f"delegate_task: subagent running {child_tool} "
                f"(iteration {child_iter}/{child_max})"
            )
        else:
            child_desc = summary.get("last_activity_desc", "")
            desc = (
                f"delegate_task: subagent {child_desc} "
                f"(iteration {child_iter}/{child_max})"
                if child_desc
                else f"delegate_task: subagent {i} working"
            )
        touch = getattr(parent_agent, "_touch_activity", None)
        if callable(touch):
            try:
                touch(desc)
            except Exception:
                pass

    def _drain_events(block_seconds: float) -> None:
        deadline = time.monotonic() + block_seconds
        first = True
        while True:
            timeout = max(0.0, deadline - time.monotonic()) if first else 0.0
            first = False
            try:
                msg = event_queue.get(timeout=timeout) if timeout else event_queue.get_nowait()
            except (_queue_mod.Empty, OSError, EOFError):
                return
            if not isinstance(msg, dict):
                continue
            i = msg.get("task_index")
            if msg.get("kind") == "activity":
                _handle_activity(i, msg.get("summary") or {})
                continue
            if msg.get("kind") != "progress":
                continue
            spec = specs.get(i)
            cb = spec.progress_cb if spec is not None else None
            if cb is None:
                continue
            try:
                event_type, tool_name, preview, args, kwargs = msg.get("event")
                cb(event_type, tool_name, preview, args, **(kwargs or {}))
            except Exception as exc:
                logger.debug("Child event relay failed: %s", exc)

    def _drain_results() -> None:
        while True:
            try:
                entry = result_queue.get_nowait()
            except (_queue_mod.Empty, OSError, EOFError):
                return
            if not isinstance(entry, dict):
                continue
            i = entry.get("task_index")
            if i in pending:
                pending.discard(i)
                _finalize(i, entry)

    # ── Join loop ────────────────────────────────────────────────────
    while pending:
        now = time.monotonic()

        # Parent interrupted → signal every child once, then give them a
        # grace window to unwind before hard-terminating.
        if (
            interrupt_signaled_at is None
            and getattr(parent_agent, "_interrupt_requested", False) is True
        ):
            interrupt_signaled_at = now
            for i in pending:
                specs[i].interrupt("Parent agent interrupted")

        _drain_events(0.25)
        _drain_results()
        if not pending:
            break
        now = time.monotonic()

        # Per-child wall-clock timeout (delegation.child_timeout_seconds).
        # Measure from each child's own proc.start(), not the batch start,
        # so staggered startup doesn't eat into a child's timeout budget.
        if child_timeout:
            for i in list(pending):
                elapsed = now - child_start_mono.get(i, started_mono)
                if i not in timeout_signaled and elapsed > child_timeout:
                    timeout_signaled[i] = now
                    specs[i].interrupt(f"Timed out after {child_timeout}s")
                elif (
                    i in timeout_signaled
                    and now - timeout_signaled[i] > _TERMINATE_GRACE_SECONDS
                ):
                    proc = procs.get(i)
                    if proc is not None and proc.is_alive():
                        proc.terminate()
                    pending.discard(i)
                    _finalize(
                        i,
                        {
                            **_fabricated(
                                i,
                                "timeout",
                                f"Subagent timed out after {child_timeout}s "
                                f"(isolated process terminated).",
                            ),
                            "exit_reason": "timeout",
                        },
                    )

        # Interrupt grace expired → hard-terminate the stragglers.
        if (
            interrupt_signaled_at is not None
            and now - interrupt_signaled_at > _TERMINATE_GRACE_SECONDS
        ):
            _drain_results()
            for i in list(pending):
                proc = procs.get(i)
                if proc is not None and proc.is_alive():
                    proc.terminate()
                pending.discard(i)
                _finalize(
                    i,
                    _fabricated(
                        i,
                        "interrupted",
                        "Parent agent interrupted — child did not finish in time",
                    ),
                )
            break

        # Reap children that died without delivering a result. The IPC
        # feeder may still be flushing right after exit, so give the
        # result a short grace window before fabricating a crash entry.
        for i in list(pending):
            proc = procs.get(i)
            if proc is None or proc.is_alive():
                dead_since.pop(i, None)
                continue
            if i not in dead_since:
                dead_since[i] = now
                continue
            if now - dead_since[i] > _DEAD_RESULT_GRACE_SECONDS:
                pending.discard(i)
                _finalize(
                    i,
                    _fabricated(
                        i,
                        "error",
                        f"Subagent process exited unexpectedly "
                        f"(exitcode={proc.exitcode}).",
                    ),
                )

    # Trailing events (e.g. subagent.complete relayed just before exit).
    _drain_events(0.25)

    for q in (event_queue, result_queue):
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass

    return [entries[i] for i in sorted(entries.keys())]
