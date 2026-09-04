"""slash.exec helpers: live-session command output + side-effect mirroring after a worker slash command.

Bodies are rebound onto server.py's globals at install time (see
method_ctx.bind_module), so they reference server.py globals bare.
"""

from __future__ import annotations

import contextlib

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


# ── Live-session slash output ────────────────────────────────────────

# Answered from the live session ONLY when the agent lives on a compute host.
_ISOLATED_SESSION_READ_COMMANDS = frozenset({"context", "tools", "help"})

_NO_AGENT_USAGE = "(._.) No active agent -- send a message first."
_NO_AGENT = "No active agent -- send a message first."


def _format_live_review_output(sid: str, session: Optional[dict], arg: str) -> str:
    """Dispatch /review against the live session's agent. The reviewer runs on the async
    delegation rail; its completion is stamped with the parent's durable session_id, which
    ``_session_owns_notification_event`` matches to drain it back into this chat."""
    if session is None:
        return "Nothing to review yet — send a message first."
    if _session_uses_compute_host(session):
        return "/review runs on the local agent only for now — this session's agent lives on a remote compute host."
    if (agent := session.get("agent")) is None:
        return "Nothing to review yet — send a message first."
    if session.get("running"):
        return "session busy — wait for the current turn to finish, then /review"
    with session.get("history_lock") or contextlib.nullcontext():
        snapshot = list(session.get("history", []))
    snapshot = snapshot or list(getattr(agent, "_session_messages", None) or [])
    try:
        from agent.review_engine import format_dispatch_note, start_review
        result = start_review(agent, snapshot, arg or "")
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"/review failed to start: {exc}"
    return format_dispatch_note(result, arg or "")


def _format_live_usage_output(sid: str, session: dict, arg: str) -> str:
    agent = session.get("agent")
    usage = _session_usage_snapshot(session)
    if agent is None and not usage:
        return _NO_AGENT_USAGE
    if session.get("_metadata_message_count") is not None:
        message_count = int(session.get("_metadata_message_count") or 0)
    else:
        with session["history_lock"]:
            message_count = len(session.get("history", []))

    def n(key: str) -> str:
        return f"{int(usage.get(key) or 0):,}"
    rows = [("Input tokens:", n("input")), ("Output tokens:", n("output"))]
    if int(usage.get("reasoning") or 0):
        rows.append(("Reasoning tokens:", n("reasoning")))
    rows += [("Prompt tokens:", n("prompt")), ("Completion tokens:", n("completion")),
             ("Total tokens:", n("total")), ("API calls:", n("calls"))]
    if usage.get("context_max"):
        pct = int(usage.get("context_percent") or 0)
        rows.append(("Current context:", f"{n('context_used')} / {n('context_max')} ({pct}%)"))
    rows += [("Messages:", f"{message_count:,}"), ("Compressions:", n("compressions"))]
    model = usage.get("model") or _metadata_mirror(session).get("model") or getattr(agent, "model", "") or "(unknown)"
    lines = ["Session Token Usage", "────────────────────────────────────────", f"Model: {model}"]
    return "\n".join(lines + [f"{label:<30}{value}" for label, value in rows])


def _live_session_messages(session: dict) -> Optional[list]:
    """Session-scoped transcript read; None when no db/key or the read fails. Uses
    ``_session_db`` (not ``_get_db()``): a profile session's rows live in its own
    profile's state.db, and through the launch handle this read comes back empty."""
    with _session_db(session) as db:
        if db is not None and session.get("session_key"):
            with contextlib.suppress(Exception):
                return db.get_messages_as_conversation(
                    session["session_key"], include_ancestors=True, include_row_ids=True)
    return None


def _format_live_history_output(sid: str, session: dict, arg: str) -> str:
    with session["history_lock"]:
        history = list(session.get("history", []))
    db_history = _live_session_messages(session)
    messages = _history_to_messages(history if db_history is None else db_history)
    if not messages:
        return "No conversation history yet."
    lines = ["Conversation History", "────────────────────────────────────────"]
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        label = {"user": "You", "assistant": "Hermes"}.get(role, role.title())
        text = str(message.get("text") or message.get("context") or "").strip()
        text = f"{text[:400]}..." if len(text) > 400 else text
        lines.append(f"[{label} #{idx}] {text or '(no text)'}")
    return "\n".join(lines)


def _format_live_prompt_output(sid: str, session: dict, arg: str) -> str:
    agent = session.get("agent")
    mirror = _metadata_mirror(session)
    if agent is None and "system_prompt" not in mirror:
        return _NO_AGENT
    prompt = (
        mirror.get("system_prompt") or getattr(agent, "ephemeral_system_prompt", None)
        or getattr(agent, "_cached_system_prompt", None) or "")
    if not prompt:
        return "Current system prompt is not built yet; send a message first."
    return f"Current system prompt:\n{prompt}"


def _format_live_context_output(sid: str, session: dict, arg: str) -> str:
    from collections import Counter
    try:
        messages = _history_to_messages(_live_session_messages(session) or [])
    except Exception:
        messages = []  # malformed db rows fall back to the live history below
    if not messages:
        with session["history_lock"]:
            messages = _history_to_messages(list(session.get("history", [])))
    usage = _session_usage_snapshot(session)
    mirror = _metadata_mirror(session)
    lines = [f"Conversation: {len(messages)} messages" if messages else "Conversation is empty (no messages yet)."]
    roles = Counter(str(msg.get("role") or "unknown") for msg in messages)
    lines.append("  " + ", ".join(f"{r}: {roles.get(r, 0)}" for r in ("user", "assistant", "tool", "system")))
    if model := mirror.get("model") or usage.get("model") or "":
        lines.append(f"Model: {model}")
    lines.append(f"Provider: {mirror.get('provider') or 'auto'}")
    context_used = int(usage.get("context_used") or usage.get("total") or 0)
    context_max = int(usage.get("context_max") or 0)
    if context_used and context_max:
        lines.append(
            f"Context usage: ~{context_used:,} / {context_max:,} tokens ({(context_used / context_max) * 100:.1f}%)")
    elif context_used:
        lines.append(f"Context usage: ~{context_used:,} tokens")
    if usage.get("compressions"):
        lines.append(f"Compressions: {int(usage.get('compressions') or 0):,}")
    return "\n".join(lines)


def _format_live_tools_output(sid: str, session: dict, arg: str) -> str:
    info = _session_info(session.get("agent"), session)
    groups = info.get("tools") if isinstance(info, dict) else {}
    if not isinstance(groups, dict) or not groups:
        return "No tools available."
    names = sorted({str(n) for g in groups.values() if isinstance(g, list) for n in g})
    if not names:
        return "No tools available."
    return "Available tools ({}):\n{}".format(len(names), "\n".join(f"  {name}" for name in names))


def _format_live_help_output(sid: str, session: dict, arg: str) -> str:
    try:
        from hermes_cli.commands import COMMANDS_BY_CATEGORY
        lines = ["Available commands:", ""]
        for category, commands in COMMANDS_BY_CATEGORY.items():
            lines.append(f"{category}:")
            lines.extend(f"  {cmd:<15} {desc}" for cmd, desc in commands.items())
        return "\n".join(lines)
    except Exception as exc:
        return f"help unavailable: {exc}"


def _format_live_model_output(session: dict) -> str:
    agent = session.get("agent")
    model = getattr(agent, "model", "") if agent is not None else ""
    provider = getattr(agent, "provider", "") if agent is not None else ""
    if not model:
        return "Current model: (unknown)"
    return f"Current model: {model}" + (f" ({provider})" if provider else "")


def _format_live_status_output(sid: str, session: dict, arg: str) -> str:
    response = _methods["session.status"]("status", {"session_id": sid})
    if response.get("error"):
        return str(response["error"].get("message") or "status unavailable")
    return str(response.get("result", {}).get("output") or "")


# name → (reply when there is no session, formatter(sid, session, arg) or a fixed reply).
# A None no-session reply means the formatter handles a missing session itself.
_LIVE_SLASH_OUTPUT = {
    "compress": ("no active session for /compress",
                 lambda sid, session, arg: _mirror_slash_side_effects(sid, session, f"/compress {arg}".strip())),
    "usage": (_NO_AGENT_USAGE, _format_live_usage_output),
    "review": (None, _format_live_review_output),
    "history": ("No conversation history yet.", _format_live_history_output),
    "prompt": (_NO_AGENT, _format_live_prompt_output),
    "status": (None, _format_live_status_output),
    "context": ("Conversation is empty (no messages yet).", _format_live_context_output),
    "tools": ("No tools available.", _format_live_tools_output),
    "help": (None, _format_live_help_output),
    "clear": (None, "Screen clear is terminal-only; desktop/TUI chat left unchanged."),
    "models": (None, "Use /model to view or switch the current model; desktop users can also open the model picker."),
    "rename": (None, "Use /title <name> to rename this session."),
    "effort": (None, "Use /reasoning <effort> to change reasoning effort.")}


def _live_slash_command_output(sid: str, session: Optional[dict], name: str, arg: str) -> Optional[str]:
    """Answer a slash command from the live session instead of the slash worker; None = not ours."""
    name = (name or "").lstrip("/").lower()
    arg = arg or ""
    if name == "model" and not arg.strip():
        return _format_live_model_output(session or {})
    if name in _ISOLATED_SESSION_READ_COMMANDS and not (session is not None and _session_uses_compute_host(session)):
        return None
    entry = _LIVE_SLASH_OUTPUT.get(name)
    if entry is None:
        return None
    no_session_reply, fmt = entry
    if session is None and no_session_reply is not None:
        return no_session_reply
    return fmt(sid, session, arg) if callable(fmt) else fmt


# ── Side-effect mirroring ────────────────────────────────────────────

# Read-then-mutate live agent/session state that a running turn is using; rejected
# while running (parity with session.compress / session.undo and the gateway's
# running-agent /model guard).
_MUTATES_WHILE_RUNNING = frozenset({"model", "personality", "prompt", "compress"})


def _compress_live_with_feedback(sid: str, session: dict, agent, arg: str, *, snapshot_kwargs: bool) -> str:
    """Compress the live session; return the user-facing feedback text (shared by command.dispatch
    /compress and the slash mirror). ``snapshot_kwargs`` forwards the pre-read snapshot to
    ``_compress_session_history``; the raw arg goes through unparsed (the choke point parses
    ``here [N]`` / ``--keep N``). CompressionLockHeld is a clean no-op (skip note returned);
    other errors propagate to the caller, which finalizes the context-engine notification."""
    from agent.conversation_compression import finalize_context_engine_compression_notification
    from agent.manual_compression_feedback import describe_compression_lock_skip, summarize_manual_compression
    from agent.model_metadata import estimate_request_tokens_rough
    with session["history_lock"]:
        before_messages = list(session.get("history", []))
        history_version = int(session.get("history_version", 0))
    sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
    tools = getattr(agent, "tools", None) or None

    def estimate(messages, prompt, tool_defs) -> int:
        return estimate_request_tokens_rough(messages, system_prompt=prompt, tools=tool_defs) if messages else 0
    before_tokens = estimate(before_messages, sys_prompt, tools)
    snapshot = {"approx_tokens": before_tokens, "before_messages": before_messages, "history_version": history_version}
    try:
        if snapshot_kwargs:
            _compress_session_history(session, arg.strip() or None, **snapshot)
        else:
            # The raw argument goes through unparsed: _compress_session_history (the choke point shared by
            # all three manual-compress routes) parses the boundary-aware forms (here [N], up to here,
            # --keep N) and does the partial head/tail split there (#35533).
            _compress_session_history(session, arg)
    except CompressionLockHeld as e:
        return describe_compression_lock_skip(e.holder)
    _sync_session_key_after_compress(sid, session)
    with session["history_lock"]:
        after_messages = list(session.get("history", []))
    after_tokens = estimate(
        after_messages, getattr(agent, "_cached_system_prompt", "") or sys_prompt, getattr(agent, "tools", None) or tools)
    _emit("session.info", sid, _session_info(agent, session))
    fb = summarize_manual_compression(
        before_messages, after_messages, before_tokens, after_tokens,
        compression_state=getattr(agent, "context_compressor", None))
    finalize_context_engine_compression_notification(agent, committed=True)
    return "\n".join(filter(None, [fb["headline"], fb["token_line"], fb.get("note")]))


def _mirror_approvals(sid, session, agent, arg) -> None:
    if arg:  # the worker already persisted approvals.mode; the bare read-only form needs no repaint
        broadcast_session_info()


def _mirror_personality(sid, session, agent, arg) -> None:
    if arg and agent:
        pname, new_prompt = _validate_personality(arg, _load_cfg())
        from hermes_cli.personality import persist_personality  # single owner: no surface drift
        persist_personality(pname)
        _apply_personality_to_session(sid, session, new_prompt, pname)


def _mirror_prompt(sid, session, agent, arg) -> None:
    if agent:
        cfg = _load_cfg()
        agent.ephemeral_system_prompt = _prompt_text((cfg.get("agent") or {}).get("system_prompt", "")) or None
        agent._cached_system_prompt = None


_FAST_TIERS = {"fast": "priority", "on": "priority", "normal": None, "off": None, "auto": "auto", "cold": "cold"}


def _mirror_fast(sid, session, agent, arg) -> None:
    if agent:
        if arg.lower() in _FAST_TIERS:
            agent.service_tier = _FAST_TIERS[arg.lower()]
        _emit("session.info", sid, _session_info(agent, session))


def _mirror_reload_mcp(sid, session, agent, arg) -> str:
    """Live-session side effect for a worker-run ``/reload-mcp``: the worker subprocess
    reloads its OWN throwaway MCP pool, so without this mirror the live session's pool
    (and tool snapshot) never changes. Re-enter the shared ``reload.mcp`` RPC with
    ``confirm=True`` (the worker already ran the interactive gate) so every reload
    path — local pool rebuild, compute-host forwarding, revision-generation bump,
    session tool refresh — stays in lockstep. Previously this called
    ``agent.reload_mcp_tools()``, which no agent ever implemented — a silent no-op
    since the mirror was introduced."""
    if session is not None and session.get("running"):
        return "session busy — /interrupt the current turn before reloading MCP servers"
    response = _methods["reload.mcp"]("reload-mcp-mirror", {"session_id": sid, "confirm": True})
    if error := response.get("error"):
        return f"reload failed: {error.get('message')}"
    return ""


def _mirror_stop(sid, session, agent, arg) -> None:
    from tools.process_registry import process_registry
    process_registry.kill_all()


# name → mirror(sid, session, agent, arg); a falsy return means "no warning".
_SLASH_MIRRORS = {
    "model": lambda sid, session, agent, arg: (
        _apply_model_switch(sid, session, arg).get("warning", "") if arg and agent else ""),
    "approvals": _mirror_approvals, "personality": _mirror_personality, "prompt": _mirror_prompt,
    "compress": lambda sid, session, agent, arg: (
        _compress_live_with_feedback(sid, session, agent, arg, snapshot_kwargs=False) if agent else ""),
    "fast": _mirror_fast,
    "reload-mcp": _mirror_reload_mcp, "stop": _mirror_stop}


def _compute_host_slash(sid: str, session: dict, name: str, command: str) -> tuple[str, str]:
    """Forward a mutating slash command to the session's compute host → ``(status, text)``:
    ``pending`` (compress still running after the wait), ``failed`` (transport error/timeout),
    ``rejected`` (host control.error), ``ok`` (host output; metadata mirror applied). Compress
    waits longer and installs a late-ack adopter so a slow compression still lands here."""
    route_name = f"slash.{name}"
    is_compress = name == "compress"

    def _on_late_ack(late: dict, _sid=sid) -> None:
        _adopt_late_compute_host_compress_ack(_sid, session, late, route_name=route_name)
    try:
        ack = _send_compute_host_control(
            sid, route_name=route_name, command=command, wait=True,
            **({"timeout": _compute_host_compress_wait_seconds(), "on_late_ack": _on_late_ack} if is_compress else {}))
    except queue.Empty:
        if is_compress:
            return "pending", "compression still running in the background; the transcript will refresh when it finishes"
        return "failed", f"compute-host {route_name} failed: timed out"
    except Exception as exc:
        return "failed", f"compute-host {route_name} failed: {exc}"
    if ack.get("type") in {"control.error", "error"}:
        return "rejected", str(ack.get("message") or f"compute-host {route_name} failed")
    _apply_compute_host_metadata_mirror(session, ack)
    return "ok", str(ack.get("output") or "")


def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name, arg, agent = parts[0], (parts[1].strip() if len(parts) > 1 else ""), session.get("agent")
    if name == "compact":  # /compact aliases /compress; the compute-host control forwards the raw alias
        name = "compress"
    if name in _MUTATES_WHILE_RUNNING:
        if _session_uses_compute_host(session):
            return _compute_host_slash(sid, session, name, command)[1]
        if session.get("running"):
            return f"session busy — /interrupt the current turn before running /{name}"
    if (mirror := _SLASH_MIRRORS.get(name)) is None:
        return ""
    try:
        return mirror(sid, session, agent, arg) or ""
    except Exception as e:
        if name == "compress" and agent:
            from agent.conversation_compression import finalize_context_engine_compression_notification
            finalize_context_engine_compression_notification(agent, committed=False)
        return f"live session sync failed: {e}"


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
