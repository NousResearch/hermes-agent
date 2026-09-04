"""Child progress relay, console formatting and child system-prompt construction for delegate_task."""

from __future__ import annotations

import logging
import enum
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

# Terminal child statuses that mean "the subagent did NOT deliver a usable result". Shared by the CLI spinner echo,
# the gateway failure notice, and the parent-facing failure summary so every surface agrees.
SUBAGENT_FAILURE_STATUSES = frozenset({"failed", "error", "timeout"})

@contextmanager
def _quiet(log_message: Optional[str], *log_args: Any, exc_info: bool = False):
    """Best-effort block: any Exception is swallowed (never reaches the run) and, when ``log_message`` is given,
    logged at debug — the exception fills a trailing unsatisfied ``%s``."""
    try:
        yield
    except Exception as exc:
        if log_message is None:
            return
        if log_message.count("%") > len(log_args):
            log_args = log_args + (exc,)
        logger.debug(log_message, *log_args, exc_info=exc_info)

def _safe_progress(cb: Any, event_type: Any, *args: Any, **kwargs: Any) -> None:
    """Invoke a child progress callback; relay failures never reach the run."""
    if cb:
        with _quiet("Progress callback %s failed: %s", event_type):
            cb(event_type, *args, **kwargs)

def _clean_error_text(error: Any, max_chars: int = 200) -> str:
    """Reduce an error payload (traceback / JSON wall) to one clean line: the exception message (last line of a
    traceback) or the first non-empty line, hard-capped in length."""
    lines = [ln.strip() for ln in str(error or "").strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[-1] if lines[0].startswith("Traceback") else lines[0]
    return line[: max_chars - 3] + "..." if len(line) > max_chars else line

def format_subagent_failure_line(
    goal: Optional[str], status: Optional[str], error: Any = None, duration_seconds: Any = None,
) -> str:
    """One clean, human-readable line describing a failed subagent, rendered
    directly to the user (CLI spinner echo, gateway platform notice), e.g.
    ``⚠️ Subagent failed — "research competitor pricing": Error code: 404 (after 12s)``."""
    goal_label = (goal or "").strip().replace("\n", " ")
    if len(goal_label) > 60:
        goal_label = goal_label[:57] + "..."
    line = f"⚠️ Subagent {'timed out' if status == 'timeout' else 'failed'}"
    if goal_label:
        line += f' — "{goal_label}"'
    err = _clean_error_text(error)
    if err:
        line += f": {err}"
    if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
        line += f" (after {round(duration_seconds)}s)"
    return line


class DelegateEvent(str, enum.Enum):
    """Formal delegation progress event types. The relay normalises incoming legacy strings (``tool.started``,
    ``_thinking``, …) to these via ``_LEGACY_EVENT_MAP``; external consumers (gateway SSE, ACP adapter, CLI) still
    receive the legacy strings during the deprecation window. TASK_SPAWNED / TASK_COMPLETED / TASK_FAILED are
    reserved for future orchestrator lifecycle events, not emitted yet."""

    TASK_SPAWNED = "delegate.task_spawned"
    TASK_PROGRESS = "delegate.task_progress"
    TASK_COMPLETED = "delegate.task_completed"
    TASK_FAILED = "delegate.task_failed"
    TASK_THINKING = "delegate.task_thinking"
    TASK_TOOL_STARTED = "delegate.tool_started"
    TASK_TOOL_COMPLETED = "delegate.tool_completed"

# Legacy child-agent event strings → DelegateEvent.
_LEGACY_EVENT_MAP: Dict[str, DelegateEvent] = {
    "_thinking": DelegateEvent.TASK_THINKING,
    "reasoning.available": DelegateEvent.TASK_THINKING,
    "tool.started": DelegateEvent.TASK_TOOL_STARTED,
    "tool.completed": DelegateEvent.TASK_TOOL_COMPLETED,
    "subagent_progress": DelegateEvent.TASK_PROGRESS,
}

# Event → _ChildProgressRelay method name. Lifecycle strings are emitted by the orchestrator itself (not
# DelegateEvent). Any other DelegateEvent (TASK_TOOL_STARTED and the reserved TASK_* values) takes the tool-started
# path; None means "recognised but ignored".
_LIFECYCLE_EVENTS = frozenset({"subagent.start", "subagent.complete", "subagent.text"})
_EVENT_HANDLERS: Dict[Any, Optional[str]] = {
    "subagent.start": "_on_start",
    "subagent.complete": "_on_complete",
    "subagent.text": "_on_text",
    DelegateEvent.TASK_THINKING: "_on_thinking",
    DelegateEvent.TASK_PROGRESS: "_on_progress",
    DelegateEvent.TASK_TOOL_COMPLETED: None,
}

def _normalize_event(event_type: Any) -> Any:
    """Lifecycle string / DelegateEvent / legacy string / ``delegate.*`` string
    → dispatch key; None for unknown events."""
    if isinstance(event_type, DelegateEvent) or (isinstance(event_type, str) and event_type in _LIFECYCLE_EVENTS):
        return event_type
    try:
        return _LEGACY_EVENT_MAP.get(event_type) or DelegateEvent(event_type)
    except (ValueError, TypeError):
        return None

_CONTEXT_FILES_INTRO = (
    "\nThe workspace's project context files are reproduced below. Their conventions and invariants are binding for "
    "your work in this workspace.\n\n"
)
_COMPLETION_INSTRUCTIONS = (
    "\nComplete this task using the tools available to you. When finished, provide a clear, concise summary of:\n"
    "- What you did\n- What you found or accomplished\n- Any files you created or modified\n- Any issues encountered\n\n"
    "Important workspace rule: Never assume a repository lives at /workspace/... or any other container-style path "
    "unless the task/context explicitly gives that path. If no exact local path is provided, discover it first before "
    "issuing git/workdir-specific commands.\n\n"
    "Keep your final summary tight: lead with outcomes, prefer bullet points over paragraphs, and don't replay your "
    "whole process. Your response is returned to the parent agent as a summary, and overlong summaries crowd out the "
    "parent's context window."
)
_ORCHESTRATOR_BLOCK = (
    "\n## Subagent Spawning (Orchestrator Role)\n"
    "You have access to the `delegate_task` tool and CAN spawn your own subagents to parallelize independent work.\n\n"
    "WHEN to delegate:\n"
    "- The goal decomposes into 2+ independent subtasks that can run in parallel (e.g. research A and B simultaneously).\n"
    "- A subtask is reasoning-heavy and would flood your context with intermediate data.\n\n"
    "WHEN NOT to delegate:\n"
    "- Single-step mechanical work — do it directly.\n"
    "- Trivial tasks you can execute in one or two tool calls.\n"
    "- Re-delegating your entire assigned goal to one worker (that's just pass-through with no value added).\n\n"
    "Coordinate your workers' results and synthesize them before reporting back to your parent. You are responsible "
    "for the final summary, not your workers.\n\n"
)
_LEAF_CHILDREN_NOTE = (
    "Your own children MUST be leaves (cannot delegate further) because they would be at the depth floor — you cannot "
    "pass role='orchestrator' to your own delegate_task calls."
)
_NESTED_CHILDREN_NOTE = (
    "Your own children can themselves be orchestrators or leaves, depending on the `role` you pass to delegate_task. "
    "Default is 'leaf'; pass role='orchestrator' explicitly when a child needs to further decompose its work."
)

def _build_child_system_prompt(
    goal: str, context: Optional[str] = None, *, workspace_path: Optional[str] = None, role: str = "leaf",
    max_spawn_depth: int = 2, child_depth: int = 1,
) -> str:
    """Focused system prompt for a child agent. role='orchestrator' appends a delegation-capability block (modeled on
    OpenClaw's buildSubagentSystemPrompt); its depth note is literal truth grounded in the passed config so the LLM
    can't confabulate nesting."""
    parts = ["You are a focused subagent working on a specific delegated task.", "", f"YOUR TASK:\n{goal}"]
    if context and context.strip():
        parts.append(f"\nCONTEXT:\n{context}")
    if workspace_path and str(workspace_path).strip():
        parts.append(
            "\nWORKSPACE PATH:\n"
            f"{workspace_path}\n"
            "Use this exact path for local repository/workdir operations unless the task explicitly says otherwise."
        )
        # Project context files (AGENTS.md / CLAUDE.md / .cursorrules ...) via the SAME discovery/priority/cap logic
        # as the main agent's prompt: children are built with skip_context_files=True, so without this a subagent
        # works in a repo blind to its conventions. SOUL.md is skipped (identity belongs to the parent).
        # workspace_path comes only from explicit sources (_resolve_workspace_hint, never bare getcwd), so the
        # install-tree-fallback leak doesn't apply. Best-effort.
        _ctx_files = ""
        with _quiet("subagent: workspace context-files load failed", exc_info=True):
            # See #64590.
            from agent.prompt_builder import build_context_files_prompt
            _ctx_files = build_context_files_prompt(cwd=str(workspace_path), skip_soul=True)
        if _ctx_files.strip():
            parts.append(_CONTEXT_FILES_INTRO + _ctx_files.strip())
    parts.append(_COMPLETION_INSTRUCTIONS)
    if role == "orchestrator":
        child_note = _LEAF_CHILDREN_NOTE if child_depth + 1 >= max_spawn_depth else _NESTED_CHILDREN_NOTE
        parts.append(
            _ORCHESTRATOR_BLOCK
            + f"NOTE: You are at depth {child_depth}. The delegation tree is capped at max_spawn_depth={max_spawn_depth}. "
            + child_note
        )
    return "\n".join(parts)

def _resolve_workspace_hint(parent_agent) -> Optional[str]:
    """Best-effort local workspace hint for child prompts: only a concrete
    absolute directory is ever injected (never a fake container path)."""
    candidates = [
        os.getenv("TERMINAL_CWD"), getattr(getattr(parent_agent, "_subdirectory_hints", None), "working_dir", None),
        getattr(parent_agent, "terminal_cwd", None), getattr(parent_agent, "cwd", None),
    ]
    for candidate in filter(None, candidates):
        with _quiet(None):
            text = os.path.abspath(os.path.expanduser(str(candidate)))
            if os.path.isabs(text) and os.path.isdir(text):
                return text
    return None

_BATCH_ORDINALS: Dict[str, int] = {}
_BATCH_ORDINALS_LOCK = threading.Lock()

def format_batch_tag(delegation_id: Optional[str]) -> str:
    """Short human tag for a delegation batch: ``deleg_6a664903`` → ``set 1`` (first batch seen in this process), the
    next distinct id → ``set 2``. Several batches (a parent's fan-out plus a child's nested fan-out, or two
    concurrent tools) print interleaved ``[n/N]`` lines to one console; without a tag ``✓ [3/3]`` and ``✓ [3/9]``
    are indistinguishable, and a raw hex slice is unreadable. Empty string when no id is known so callers can
    concatenate unconditionally."""
    if not isinstance(delegation_id, str) or not delegation_id:
        return ""
    with _BATCH_ORDINALS_LOCK:
        n = _BATCH_ORDINALS.setdefault(delegation_id, len(_BATCH_ORDINALS) + 1)
    return f"set {n}"

def _batch_prefix(delegation_id: Optional[str], task_index: int, task_count: int) -> str:
    """``[set 2 · 3/9] `` for batch children, ``[set 2] `` for a lone child,
    ``[3/9] `` / ``""`` when the batch id is unknown."""
    tag = format_batch_tag(delegation_id)
    if task_count > 1:
        inner = f"{tag} · {task_index + 1}/{task_count}" if tag else f"{task_index + 1}/{task_count}"
        return f"[{inner}] "
    return f"[{tag}] " if tag else ""

def _emit_parent_console(parent_agent, line: str) -> None:
    """Emit a progress line through ``parent_agent._safe_print`` when available so headless stdio hosts (ACP, gateway
    API) can redirect it to stderr; a bare ``print()`` would land on stdout and corrupt JSON-RPC framing."""
    printer = getattr(parent_agent, "_safe_print", None)
    if callable(printer):
        with _quiet(None):
            printer(line)
            return
    print(line)

def _print_completion_line(parent_agent: Any, spinner_ref: Any, line: str, console_line: Optional[str] = None) -> None:
    """Above-spinner line when a spinner exists (console fallback if it raises), else console
    (``console_line`` when given, else the line indented two spaces)."""
    if spinner_ref:
        with _quiet(None):
            spinner_ref.print_above(line)
            return
    _emit_parent_console(parent_agent, f"  {line}" if console_line is None else console_line)

def _short(text: str, n: int) -> str:
    return (text[:n] + "...") if len(text) > n else text


class _ChildProgressRelay:
    """Callable relaying one child's events to the parent display. CLI: prints tree-view lines above the parent's
    delegation spinner. Gateway: batches tool names (``_BATCH_SIZE``) and relays to the parent's progress callback,
    threading the identity kwargs (subagent_id, parent_id, depth, model, toolsets) into every event so the TUI can
    rebuild the live spawn tree and route per-branch controls back by ``subagent_id``."""

    _BATCH_SIZE = 5

    def __init__(
        self, task_index: int, goal: str, spinner: Any, parent_cb: Any, task_count: int,
        subagent_id, parent_id, depth, model, toolsets, session_ref,
    ) -> None:
        self.task_index, self.task_count, self.goal_label = task_index, task_count, (goal or "").strip()
        # session_ref is a SHARED dict filled in later by the caller — keep the identity.
        self.spinner, self.parent_cb, self.session_ref = spinner, parent_cb, session_ref if session_ref is not None else {}
        self.subagent_id, self.parent_id, self.depth, self.model, self.toolsets = (
            subagent_id, parent_id, depth, model, toolsets
        )
        self.batch: List[str] = []
        self.tool_count = 0  # per-subagent running counter

    def _prefix(self) -> str:
        # The batch tag is resolved lazily from session_ref: the relay is built
        # before delegate_task stamps ``_delegation_id`` on the child.
        return _batch_prefix(self.session_ref.get("delegation_id"), self.task_index, self.task_count)

    def _identity_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"task_index": self.task_index, "task_count": self.task_count, "goal": self.goal_label}
        kw.update({k: getattr(self, k) for k in ("subagent_id", "parent_id", "depth", "model") if getattr(self, k) is not None})
        if self.toolsets is not None:
            kw["toolsets"] = list(self.toolsets)
        # child_session_id / delegation_id are filled into the shared ref once
        # the child exists, so every relayed event lets UIs open its session.
        for src, dst in (("session_id", "child_session_id"), ("delegation_id", "delegation_id")):
            if self.session_ref.get(src):
                kw[dst] = str(self.session_ref[src])
        kw["tool_count"] = self.tool_count
        return kw

    def _relay(self, event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
        if self.parent_cb:
            # kwargs override identity (e.g. status, duration_seconds).
            with _quiet("Parent callback failed: %s"):
                self.parent_cb(event_type, tool_name, preview, args, **{**self._identity_kwargs(), **kwargs})

    def _tree_line(self, text: str) -> None:
        """Print one tree-view line above the CLI spinner (no-op without a spinner)."""
        if self.spinner:
            with _quiet("Spinner print_above failed: %s"):
                self.spinner.print_above(f" {self._prefix()}├─ {text}")

    def _flush(self) -> None:
        """Flush remaining batched tool names to the gateway."""
        if self.parent_cb and self.batch:
            self._relay("subagent.progress", preview=f"🔀 {self._prefix()}{', '.join(self.batch)}")
            self.batch.clear()

    # ── Lifecycle events emitted by the orchestrator itself ──
    def _on_start(self, tool_name, preview, args, kwargs):
        if self.goal_label:
            self._tree_line(f"🔀 {_short(self.goal_label, 55)}")
        self._relay("subagent.start", preview=preview or self.goal_label or "", **kwargs)

    def _on_complete(self, tool_name, preview, args, kwargs):
        # Failed child: echo one clean reason line into the CLI tree so the human
        # sees WHY, not just a vanished branch (gateway renders off the relayed event).
        if kwargs.get("status") in SUBAGENT_FAILURE_STATUSES:
            self._tree_line(format_subagent_failure_line(
                self.goal_label, kwargs.get("status"), error=kwargs.get("summary") or preview,
                duration_seconds=kwargs.get("duration_seconds"),
            ))
        self._relay("subagent.complete", preview=preview, **kwargs)

    def _on_text(self, tool_name, preview, args, kwargs):
        # Streamed child reply text, relayed verbatim for gateway watch windows;
        # inert on CLI/TUI (their progress handlers ignore non-tool events).
        self._relay("subagent.text", preview=preview)

    # ── DelegateEvent handlers ──
    def _on_thinking(self, tool_name, preview, args, kwargs):
        text = preview or tool_name or ""
        self._tree_line(f'💭 "{_short(text, 55)}"')
        self._relay("subagent.thinking", preview=text)

    def _on_progress(self, tool_name, preview, args, kwargs):
        # Pre-batched summary from a nested orchestrator's grandchild arrives in the tool_name slot: render distinctly
        # (no tool-emoji lookup) and relay upward without re-batching.
        summary_text = tool_name or preview or ""
        if summary_text:
            self._tree_line(f"🔀 {summary_text}")
        if self.parent_cb:
            with _quiet("Parent callback relay failed: %s"):
                self.parent_cb("subagent_progress", f"{self._prefix()}{summary_text}")

    def _on_tool_started(self, tool_name, preview, args, kwargs):
        self.tool_count += 1
        if self.subagent_id is not None:
            with _active_subagents_lock:
                rec = _active_subagents.get(self.subagent_id)
                if rec is not None:
                    rec["tool_count"] = self.tool_count
                    rec["last_tool"] = tool_name or ""
        if self.spinner:
            from agent.display import get_tool_emoji
            line = f"{get_tool_emoji(tool_name or '')} {tool_name}"
            short = _short(preview, 35) if preview else ""
            self._tree_line(f'{line}  "{short}"' if short else line)
        if self.parent_cb:
            self._relay("subagent.tool", tool_name, preview, args)
            self.batch.append(tool_name or "")
            if len(self.batch) >= self._BATCH_SIZE:
                self._flush()

    def __call__(self, event_type, tool_name: str = None, preview: str = None, args=None, **kwargs):
        key = _normalize_event(event_type)
        method = None if key is None else _EVENT_HANDLERS.get(key, "_on_tool_started")
        if method is not None:
            getattr(self, method)(tool_name, preview, args, kwargs)

def _build_child_progress_callback(
    task_index: int, goal: str, parent_agent, task_count: int = 1, *, subagent_id: Optional[str] = None,
    parent_id: Optional[str] = None, depth: Optional[int] = None, model: Optional[str] = None,
    toolsets: Optional[List[str]] = None, session_ref: Optional[Dict[str, Any]] = None,
) -> Optional[callable]:
    """Relay for one child's events (see ``_ChildProgressRelay``), or None when the parent has neither a spinner nor a
    progress callback — the child then runs with no progress callback at all (zero behavior change)."""
    spinner = getattr(parent_agent, "_delegate_spinner", None)
    parent_cb = getattr(parent_agent, "tool_progress_callback", None)
    if not spinner and not parent_cb:
        return None
    return _ChildProgressRelay(
        task_index, goal, spinner, parent_cb, task_count, subagent_id, parent_id, depth, model, toolsets, session_ref,
    )
