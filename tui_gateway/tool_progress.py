"""Tool lifecycle callbacks (tool.start/complete/progress events), verbose-text capping/redaction, todo-state
projection. Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare."""

from __future__ import annotations

from .method_ctx import bind_module

# Verbose tool text is capped to the Ink render budget (a hair more, so the "[omitted …]" label
# stays informative): unbounded output fed a render-tree blowup that OOM-killed the TUI parent.
# Full output stays in the agent context and the SQLite session, untouched.
# Tool Args/Result text shipped to the TUI for the verbose trail line. The TUI renders only a small
# persisted preview (ui-tui VERBOSE_TRAIL_MAX_CHARS), kept all session and expanded by default — so shipping
# more than that is pure pipe waste AND feeds the Ink render-tree blowup that silently OOM-killed the TUI
# parent (#34095).
_TUI_VERBOSE_TEXT_MAX_CHARS = 1_000
_TUI_VERBOSE_TEXT_MAX_LINES = 16

_TODO_TOOL_NAMES = ("todo_list", "todo")  # legacy alias: pre-rename replays


def _cap_tui_verbose_text(text: str) -> str:
    if len(text) <= _TUI_VERBOSE_TEXT_MAX_CHARS and text.count("\n") < _TUI_VERBOSE_TEXT_MAX_LINES:
        return text
    # Start of the last MAX_LINES lines, then pull forward to the char budget (never mid-line).
    line_start = len(text) - len("\n".join(text.split("\n")[-_TUI_VERBOSE_TEXT_MAX_LINES:]))
    start = max(line_start, len(text) - _TUI_VERBOSE_TEXT_MAX_CHARS)
    if start > line_start:
        next_break = text.find("\n", start)
        if 0 <= next_break < len(text) - 1:
            start = next_break + 1
    tail = text[start:].lstrip()
    omitted_chars = max(0, len(text) - len(tail))
    omitted_lines = text[:start].count("\n")
    omitted = f"{omitted_lines} lines / {omitted_chars} chars" if omitted_lines else f"{omitted_chars} chars"
    return f"[showing verbose tail; omitted {omitted}]\n{tail}"


def _redact_tui_verbose_text(text: str) -> str:
    try:
        from agent.redact import redact_sensitive_text
        redacted = redact_sensitive_text(str(text), force=True)
    except Exception:
        return ""
    return _cap_tui_verbose_text(redacted)


def _verbose_text(render, fallback) -> str:
    """Redacted+capped ``render()``; ``fallback()`` when rendering raises."""
    try:
        raw = render()
    except Exception:
        raw = fallback()
    return _redact_tui_verbose_text(raw)


def _tool_args_text(args: dict) -> str:
    return _verbose_text(lambda: json.dumps(args or {}, indent=2, ensure_ascii=False, default=str), lambda: str(args or {}))


def _tool_result_text(result: object) -> str:
    def render():
        from agent.tool_dispatch_helpers import _multimodal_text_summary
        return _multimodal_text_summary(result)

    return _verbose_text(render, lambda: str(result))


def _fmt_tool_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


def _count_list(obj: object, *path: str) -> int | None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return len(cur) if isinstance(cur, list) else None


# tool name -> (count-from-result, verb, singular, plural) for the tool.complete summary line.
_SUMMARY_COUNTERS = {
    "web_search": (lambda d: _count_list(d, "data", "web"), "Did", "search", "searches"),
    "web_extract": (lambda d: _count_list(d, "results") or _count_list(d, "data", "results"), "Extracted", "page", "pages"),
}


def _tool_summary(name: str, result: str, duration_s: float | None) -> str | None:
    try:
        data = json.loads(result)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return None
    dur = _fmt_tool_duration(duration_s)
    suffix = f" in {dur}" if dur else ""
    warning = str(data.get("fallback_warning") or "").strip()
    if warning:
        return f"{warning}{suffix}"
    entry = _SUMMARY_COUNTERS.get(name)
    n = entry[0](data) if entry else None
    return f"{entry[1]} {n} {entry[2] if n == 1 else entry[3]}{suffix}" if n is not None else None


def _normalize_todo_state(value: object) -> dict | None:
    """Return a client-safe full todo snapshot or ``None`` when malformed."""
    if not isinstance(value, dict) or not isinstance(value.get("todos"), list):
        return None
    try:
        revision = max(0, int(value.get("revision") or 0))
    except (TypeError, ValueError):
        return None
    todos = list(value["todos"])
    # Unused TodoStore snapshot() is {todos: [], revision: 0}: attaching it on resume stamps a client
    # watermark and blocks unversioned tool.start merges. Empty at revision >= 1 is a real clear.
    if not todos and revision == 0:
        return None
    return {"todos": todos, "revision": revision}


def _cache_todo_state(session: dict, state: dict | None) -> None:
    """Keep the newest snapshot on the session (revision-monotonic)."""
    cached = _normalize_todo_state(session.get("todo_state")) if state is not None else None
    if state is not None and (cached is None or state["revision"] >= cached["revision"]):
        session["todo_state"] = state


def _session_todo_state(session: dict) -> dict | None:
    """Return the newest live/cached todo snapshot for a runtime session."""
    cached = _normalize_todo_state(session.get("todo_state"))
    live = None
    snapshot = getattr(getattr(session.get("agent"), "_todo_store", None), "snapshot", None)
    if callable(snapshot):
        try:
            live = _normalize_todo_state(snapshot())
        except Exception:
            logger.debug("failed to read live todo state", exc_info=True)
    if live is not None and (cached is None or live["revision"] >= cached["revision"]):
        cached = live
    if cached is not None:
        session["todo_state"] = cached
    return cached


def _attach_todo_state(payload: dict, session: dict) -> dict:
    """Attach the authoritative todo snapshot to a session response."""
    state = _session_todo_state(session)
    if state is not None:
        payload["todo_state"] = state
    return payload


def _todo_state_from_history(history) -> dict | None:
    """Latest todo snapshot from a loaded transcript, for resume paths that answer before an AIAgent (and
    its live TodoStore) exists: the newest tool result paired with an assistant ``todo`` call IS it."""
    if not isinstance(history, list) or not history:
        return None
    try:
        from tools.todo_tool import MAX_TODO_RESULT_CHARS
        todo_call_ids = {
            call.get("id")
            for msg in history if isinstance(msg, dict)
            for call in msg.get("tool_calls") or []
            if (call.get("function") or {}).get("name") in _TODO_TOOL_NAMES and call.get("id")
        }
        if not todo_call_ids:
            return None
        for msg in reversed(history):
            if not isinstance(msg, dict) or msg.get("role") != "tool" or msg.get("tool_call_id") not in todo_call_ids:
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) > MAX_TODO_RESULT_CHARS or '"todos"' not in content:
                continue
            try:
                return _normalize_todo_state(json.loads(content))
            except Exception:
                continue
        return None
    except Exception:
        logger.debug("failed to derive todo state from history", exc_info=True)
        return None


def _on_tool_start(sid: str, tool_call_id: str, name: str, args: dict):
    session = _sessions.get(sid)
    if session is not None:
        with contextlib.suppress(Exception):
            from agent.display import capture_local_edit_snapshot
            snapshot = capture_local_edit_snapshot(name, args)
            if snapshot is not None:
                session.setdefault("edit_snapshots", {})[tool_call_id] = snapshot
        session.setdefault("tool_started_at", {})[tool_call_id] = time.time()
    if _tool_progress_enabled(sid) or _tool_lifecycle_required_for_ui(name):
        payload: dict[str, object] = {"tool_id": tool_call_id, "name": name, "context": _tool_ctx(name, args)}
        # Full args (not just the 80-char `context` preview) so the desktop's expanded tool row is complete
        # while the tool runs. args.todos may be a partial merge — tool.complete is the truth.
        if args:
            payload["args"] = args
        if _session_verbose(sid) and (args_text := _tool_args_text(args)):
            payload["args_text"] = args_text
        _emit("tool.start", sid, payload)


def _on_tool_complete(sid: str, tool_call_id: str, name: str, args: dict, result: str):
    payload = {"tool_id": tool_call_id, "name": name, "args": args}
    session = _sessions.get(sid)
    snapshot = session.setdefault("edit_snapshots", {}).pop(tool_call_id, None) if session is not None else None
    started_at = session.setdefault("tool_started_at", {}).pop(tool_call_id, None) if session is not None else None
    duration_s = time.time() - started_at if started_at else None
    if duration_s is not None:
        payload["duration_s"] = duration_s
    try:
        payload["result"] = json.loads(result)
    except Exception:
        payload["result"] = result
    summary = _tool_summary(name, result, duration_s)
    if summary:
        payload["summary"] = summary
    if _session_verbose(sid) and (result_text := _tool_result_text(result)):
        payload["result_text"] = result_text
    todo_state = _normalize_todo_state(payload.get("result")) if name in _TODO_TOOL_NAMES else None
    if todo_state is not None:
        payload.update(todo_state)
        if session is not None:
            _cache_todo_state(session, todo_state)
    with contextlib.suppress(Exception):
        from agent.display import render_edit_diff_with_delta
        rendered: list[str] = []
        if render_edit_diff_with_delta(name, result, function_args=args, snapshot=snapshot, print_fn=rendered.append):
            payload["inline_diff"] = "\n".join(rendered)
    if (_tool_progress_enabled(sid) or payload.get("inline_diff") or _tool_lifecycle_required_for_ui(name)
            or name in _TODO_TOOL_NAMES):
        _emit("tool.complete", sid, payload)
    # Task state is application data, not tool-progress chrome: a dedicated full-snapshot event lets
    # every client reconcile without parsing tool args.
    if todo_state is not None:
        _emit("todo.updated", sid, todo_state)


# ── _on_tool_progress dispatch: each handler takes (sid, name, preview, kw) ─────────────────────
# `tool.started` is dropped on purpose: _on_tool_start already emits the authoritative tool.start with
# the stable id and args; an id-less duplicate row makes the desktop live view diverge from history.

def _progress_output_risk(sid, name, preview, kw):
    metadata = kw.get("risk_metadata")
    if isinstance(metadata, dict):
        _emit("tool.output_risk", sid, {
            "tool_id": str(kw.get("tool_call_id") or ""), "name": str(name), "risk": str(metadata.get("risk") or "low"),
            "findings": [str(item) for item in metadata.get("findings", [])], "redacted": bool(metadata.get("redacted", False)),
        })


def _progress_reasoning(sid, name, preview, kw):
    _emit("reasoning.available", sid, {"text": str(preview), **({"verbose": True} if _session_verbose(sid) else {})})


def _progress_moa_reference(sid, name, preview, kw):
    # MoA reference-model output, rendered as a labelled block before the aggregator's response.
    # `name` is the slot label, `preview` the text.
    ref_payload: dict[str, object] = {"label": str(name), "text": str(preview or "")}
    for key, out in (("moa_index", "index"), ("moa_count", "count")):
        if kw.get(key) is not None:
            ref_payload[out] = kw[key]
    _emit("moa.reference", sid, ref_payload)


def _progress_moa_progress(sid, name, preview, kw):
    # Drives the status-bar `MOA: 2/3 refs done`; both counters required for deterministic rendering.
    refs_done, refs_total = kw.get("moa_refs_done"), kw.get("moa_refs_total")
    # Per-reference completion — drives the status-bar progress indicator (`MOA: 2/3 refs done`) requested
    # in issue #59546. Only emitted when both counters are present so the client can render
    # deterministically.
    if refs_done is None or refs_total is None:
        return
    _emit("moa.progress", sid, {"label": str(name or ""), "refs_done": int(refs_done), "refs_total": int(refs_total)})


def _progress_moa_phase(sid, name, preview, kw):
    # Currently only phase="aggregator" fires, once fan-out completes.
    phase = kw.get("moa_phase")
    if not phase:
        return
    phase_payload: dict[str, object] = {"phase": str(phase)}
    for key, out in (("moa_refs_done", "refs_done"), ("moa_refs_total", "refs_total")):
        if kw.get(key) is not None:
            phase_payload[out] = int(kw[key])
    if name:
        phase_payload["aggregator"] = str(name)
    _emit("moa.phase", sid, phase_payload)


def _not_none(v):
    return v is not None


def _str_list(v):
    return [str(x) for x in v]


def _int_or_skip(v):
    """Per-branch token/api rollups tolerate junk from older emitters: unparsable -> field omitted."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# Optional subagent.* payload fields in WIRE ORDER: (source key, present-when, coerce). Identity fields
# are all optional: older emitters omit them and the TUI spawn tree falls back to flat rendering.
# `tool_name`/`text` are fed from the positional name/preview; `output_tail` is a list of dicts.
_SUBAGENT_FIELDS = (
    ("subagent_id", bool, str), ("parent_id", bool, str), ("child_session_id", bool, str),
    ("delegation_id", bool, str), ("depth", _not_none, int), ("model", bool, str), ("tool_count", _not_none, int),
    ("toolsets", bool, _str_list), ("input_tokens", _not_none, _int_or_skip), ("output_tokens", _not_none, _int_or_skip),
    ("reasoning_tokens", _not_none, _int_or_skip), ("api_calls", _not_none, _int_or_skip),
    ("files_read", bool, _str_list), ("files_written", bool, _str_list), ("output_tail", bool, list),
    ("tool_name", bool, str), ("text", bool, str), ("status", bool, str), ("summary", bool, str),
    ("duration_seconds", _not_none, float),
)


def _progress_subagent(sid, name, preview, kw, event_type):
    payload = {"goal": str(kw.get("goal") or ""), "task_count": int(kw.get("task_count") or 1), "task_index": int(kw.get("task_index") or 0)}
    source = {**kw, "tool_name": name, "text": preview}
    for key, present, coerce in _SUBAGENT_FIELDS:
        if present(source.get(key)):
            val = coerce(source[key])
            if val is not None:
                payload[key] = val
    if preview and event_type == "subagent.tool":
        payload["tool_preview"] = str(preview)
        payload["text"] = str(preview)
    # subagent.text is the child's per-token reply, relayed solely to feed a watch window's live mirror
    # (keyed off the child sid); on the parent it's hundreds of ignored frames, so skip it.
    if event_type != "subagent.text":
        _emit(event_type, sid, payload)
    _mirror_subagent_to_child(event_type, payload)


# event_type -> (handler, requires): `requires` names the arg that must be truthy for the row to be
# emitted at all ("name" / "preview" / None).
_PROGRESS_HANDLERS = {
    "tool.output_risk": (_progress_output_risk, "name"), "reasoning.available": (_progress_reasoning, "preview"),
    "moa.reference": (_progress_moa_reference, "name"),
    "moa.aggregating": (lambda sid, name, preview, kw: _emit("moa.aggregating", sid, {"aggregator": str(name or "")}), None),
    "moa.progress": (_progress_moa_progress, None), "moa.phase": (_progress_moa_phase, None),
}


def _on_tool_progress(
    sid: str, event_type: str, name: str | None = None, preview: str | None = None,
    _args: dict | None = None, **_kwargs,
):
    if not _tool_progress_enabled(sid) or (event_type == "tool.started" and name):
        return
    if event_type.startswith("subagent."):
        return _progress_subagent(sid, name, preview, _kwargs, event_type)
    handler, requires = _PROGRESS_HANDLERS.get(event_type, (None, None))
    if handler is not None and (requires is None or {"name": name, "preview": preview}[requires]):
        handler(sid, name, preview, _kwargs)


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
