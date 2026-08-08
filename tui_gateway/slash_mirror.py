"""Live /slash mirror output helpers (moved verbatim from server.py).

Bodies are byte-identical to their pre-split server.py form; they are
rebound onto server.py's globals at install time (register()).
"""

import types
from typing import Optional


def _format_live_usage_output(session: dict) -> str:
    agent = session.get("agent")
    usage = _session_usage_snapshot(session)
    if agent is None and not usage:
        return "(._.) No active agent -- send a message first."
    if session.get("_metadata_message_count") is not None:
        message_count = int(session.get("_metadata_message_count") or 0)
    else:
        with session["history_lock"]:
            message_count = len(session.get("history", []))
    lines = [
        "Session Token Usage",
        "────────────────────────────────────────",
        f"Model: {usage.get('model') or _metadata_mirror(session).get('model') or getattr(agent, 'model', '') or '(unknown)'}",
        f"Input tokens:                 {int(usage.get('input') or 0):,}",
        f"Output tokens:                {int(usage.get('output') or 0):,}",
    ]
    reasoning = int(usage.get("reasoning") or 0)
    if reasoning:
        lines.append(f"Reasoning tokens:             {reasoning:,}")
    lines.extend(
        [
            f"Prompt tokens:                {int(usage.get('prompt') or 0):,}",
            f"Completion tokens:            {int(usage.get('completion') or 0):,}",
            f"Total tokens:                 {int(usage.get('total') or 0):,}",
            f"API calls:                    {int(usage.get('calls') or 0):,}",
        ]
    )
    if usage.get("context_max"):
        lines.append(
            "Current context:              "
            f"{int(usage.get('context_used') or 0):,} / "
            f"{int(usage.get('context_max') or 0):,} "
            f"({int(usage.get('context_percent') or 0)}%)"
        )
    lines.extend(
        [
            f"Messages:                     {message_count:,}",
            f"Compressions:                 {int(usage.get('compressions') or 0):,}",
        ]
    )
    return "\n".join(lines)


def _format_live_history_output(session: dict) -> str:
    with session["history_lock"]:
        history = list(session.get("history", []))
    db = _get_db()
    if db is not None and session.get("session_key"):
        try:
            history = db.get_messages_as_conversation(
                session["session_key"], include_ancestors=True, include_row_ids=True
            )
        except Exception:
            pass
    messages = _history_to_messages(history)
    if not messages:
        return "No conversation history yet."
    lines = ["Conversation History", "────────────────────────────────────────"]
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        label = "You" if role == "user" else "Hermes" if role == "assistant" else role.title()
        text = str(message.get("text") or message.get("context") or "").strip()
        if len(text) > 400:
            text = f"{text[:400]}..."
        lines.append(f"[{label} #{idx}] {text or '(no text)'}")
    return "\n".join(lines)


def _format_live_prompt_output(session: dict) -> str:
    agent = session.get("agent")
    mirror = _metadata_mirror(session)
    if agent is None and "system_prompt" not in mirror:
        return "No active agent -- send a message first."
    prompt = (
        mirror.get("system_prompt")
        or getattr(agent, "ephemeral_system_prompt", None)
        or getattr(agent, "_cached_system_prompt", None)
        or ""
    )
    if not prompt:
        return "Current system prompt is not built yet; send a message first."
    return f"Current system prompt:\n{prompt}"


def _format_live_context_output(session: dict) -> str:
    messages = []
    db = _get_db()
    if db is not None and session.get("session_key"):
        try:
            messages = _history_to_messages(
                db.get_messages_as_conversation(
                    session["session_key"], include_ancestors=True, include_row_ids=True
                )
            )
        except Exception:
            messages = []
    if not messages:
        with session["history_lock"]:
            messages = _history_to_messages(list(session.get("history", [])))
    usage = _session_usage_snapshot(session)
    mirror = _metadata_mirror(session)
    lines = [
        f"Conversation: {len(messages)} messages" if messages else "Conversation is empty (no messages yet)."
    ]
    roles: dict[str, int] = {}
    for msg in messages:
        role = str(msg.get("role") or "unknown")
        roles[role] = roles.get(role, 0) + 1
    lines.append(
        f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
        f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}"
    )
    model = mirror.get("model") or usage.get("model") or ""
    provider = mirror.get("provider") or "auto"
    if model:
        lines.append(f"Model: {model}")
    lines.append(f"Provider: {provider}")
    context_used = int(usage.get("context_used") or usage.get("total") or 0)
    context_max = int(usage.get("context_max") or 0)
    if context_used:
        if context_max:
            usage_pct = (context_used / context_max) * 100
            lines.append(
                f"Context usage: ~{context_used:,} / {context_max:,} tokens ({usage_pct:.1f}%)"
            )
        else:
            lines.append(f"Context usage: ~{context_used:,} tokens")
    if usage.get("compressions"):
        lines.append(f"Compressions: {int(usage.get('compressions') or 0):,}")
    return "\n".join(lines)


def _format_live_tools_output(session: dict) -> str:
    info = _session_info(session.get("agent"), session)
    groups = info.get("tools") if isinstance(info, dict) else {}
    if not isinstance(groups, dict) or not groups:
        return "No tools available."
    names: list[str] = []
    for group_names in groups.values():
        if isinstance(group_names, list):
            names.extend(str(name) for name in group_names)
    names = sorted(set(names))
    if not names:
        return "No tools available."
    return "Available tools ({}):\n{}".format(
        len(names), "\n".join(f"  {name}" for name in names)
    )


def _format_live_help_output() -> str:
    try:
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        lines = ["Available commands:", ""]
        for category, commands in COMMANDS_BY_CATEGORY.items():
            lines.append(f"{category}:")
            for cmd, desc in commands.items():
                lines.append(f"  {cmd:<15} {desc}")
        return "\n".join(lines)
    except Exception as exc:
        return f"help unavailable: {exc}"


def _format_live_model_output(session: dict) -> str:
    agent = session.get("agent")
    model = getattr(agent, "model", "") if agent is not None else ""
    provider = getattr(agent, "provider", "") if agent is not None else ""
    if model and provider:
        return f"Current model: {model} ({provider})"
    if model:
        return f"Current model: {model}"
    return "Current model: (unknown)"


def _live_slash_command_output(sid: str, session: Optional[dict], name: str, arg: str) -> Optional[str]:
    name = (name or "").lstrip("/").lower()
    arg = arg or ""
    if name == "model" and not arg.strip():
        return _format_live_model_output(session or {})
    if name not in _LIVE_SESSION_DIRECT_COMMANDS:
        if not (
            name in _ISOLATED_SESSION_READ_COMMANDS
            and session is not None
            and _session_uses_compute_host(session)
        ):
            return None

    if name in _ISOLATED_SESSION_READ_COMMANDS and not (
        session is not None and _session_uses_compute_host(session)
    ):
        return None
    if name == "compress":
        if session is None:
            return "no active session for /compress"
        return _mirror_slash_side_effects(sid, session, f"/compress {arg}".strip())
    if name == "usage":
        if session is None:
            return "(._.) No active agent -- send a message first."
        return _format_live_usage_output(session)
    if name == "history":
        if session is None:
            return "No conversation history yet."
        return _format_live_history_output(session)
    if name == "prompt":
        if session is None:
            return "No active agent -- send a message first."
        return _format_live_prompt_output(session)
    if name == "status":
        response = _methods["session.status"]("status", {"session_id": sid})
        if response.get("error"):
            return str(response["error"].get("message") or "status unavailable")
        return str(response.get("result", {}).get("output") or "")
    if name == "context":
        if session is None:
            return "Conversation is empty (no messages yet)."
        return _format_live_context_output(session)
    if name == "tools":
        if session is None:
            return "No tools available."
        return _format_live_tools_output(session)
    if name == "help":
        return _format_live_help_output()
    if name == "clear":
        return "Screen clear is terminal-only; desktop/TUI chat left unchanged."
    if name == "models":
        return "Use /model to view or switch the current model; desktop users can also open the model picker."
    if name == "rename":
        return "Use /title <name> to rename this session."
    if name == "effort":
        return "Use /reasoning <effort> to change reasoning effort."
    return None



def _mirror_slash_side_effects(sid: str, session: dict, command: str) -> str:
    """Apply side effects that must also hit the gateway's live agent."""
    parts = command.lstrip("/").split(None, 1)
    if not parts:
        return ""
    name, arg, agent = (
        parts[0],
        (parts[1].strip() if len(parts) > 1 else ""),
        session.get("agent"),
    )
    if name == "compact":
        # /compact is an alias of /compress in every host. The compute-host
        # slash.compress control forwards the user's raw alias verbatim, so
        # without normalizing here the child mirror silently no-ops — the
        # session never compresses and the deferred context-engine
        # notification wiring below is never exercised for that route.
        name = "compress"

    # Reject agent-mutating commands during an in-flight turn.  These
    # all do read-then-mutate on live agent/session state that the
    # worker thread running agent.run_conversation is using.  Parity
    # with the session.compress / session.undo guards and the gateway
    # runner's running-agent /model guard.
    _MUTATES_WHILE_RUNNING = {"model", "personality", "prompt", "compress"}
    if _session_uses_compute_host(session) and name in _MUTATES_WHILE_RUNNING:
        route_name = f"slash.{name}"
        try:
            ack = _send_compute_host_control(
                sid,
                route_name=route_name,
                command=command,
                wait=True,
            )
        except Exception as exc:
            return f"compute-host {route_name} failed: {exc}"
        if ack.get("type") in {"control.error", "error"}:
            return str(ack.get("message") or f"compute-host {route_name} failed")
        _apply_compute_host_metadata_mirror(session, ack)
        return str(ack.get("output") or "")
    if name in _MUTATES_WHILE_RUNNING and session.get("running"):
        return f"session busy — /interrupt the current turn before running /{name}"

    try:
        if name == "model" and arg and agent:
            result = _apply_model_switch(sid, session, arg)
            return result.get("warning", "")
        elif name == "personality" and arg and agent:
            pname, new_prompt = _validate_personality(arg, _load_cfg())
            _apply_personality_to_session(sid, session, new_prompt, pname)
        elif name == "prompt" and agent:
            cfg = _load_cfg()
            new_prompt = _prompt_text((cfg.get("agent") or {}).get("system_prompt", ""))
            agent.ephemeral_system_prompt = new_prompt or None
            agent._cached_system_prompt = None
        elif name == "compress" and agent:
            # Mirror the session.compress RPC: build a before/after summary so
            # the user gets feedback (#46686). The slash path previously just
            # compressed + emitted session.info and returned "", so the TUI
            # showed no "compressed N → M messages / ~X → ~Y tokens" stats
            # while CLI and gateway both did.
            from agent.manual_compression_feedback import summarize_manual_compression
            from agent.model_metadata import estimate_request_tokens_rough
            from agent.conversation_compression import (
                finalize_context_engine_compression_notification,
            )

            with session["history_lock"]:
                _before_messages = list(session.get("history", []))
            _before_count = len(_before_messages)
            _sys_prompt = getattr(agent, "_cached_system_prompt", "") or ""
            _tools = getattr(agent, "tools", None) or None
            _before_tokens = (
                estimate_request_tokens_rough(
                    _before_messages, system_prompt=_sys_prompt, tools=_tools
                )
                if _before_count
                else 0
            )

            # The raw argument goes through unparsed: _compress_session_history
            # (the choke point shared by all three manual-compress routes)
            # parses the boundary-aware forms (here [N], up to here, --keep N)
            # and does the partial head/tail split there (#35533).
            try:
                _compress_session_history(session, arg)
            except CompressionLockHeld as e:
                from agent.manual_compression_feedback import (
                    describe_compression_lock_skip,
                )
                return describe_compression_lock_skip(e.holder)
            _sync_session_key_after_compress(sid, session)

            with session["history_lock"]:
                _after_messages = list(session.get("history", []))
            _sys_prompt_after = getattr(agent, "_cached_system_prompt", "") or _sys_prompt
            _tools_after = getattr(agent, "tools", None) or _tools
            _after_tokens = (
                estimate_request_tokens_rough(
                    _after_messages, system_prompt=_sys_prompt_after, tools=_tools_after
                )
                if _after_messages
                else 0
            )
            _emit("session.info", sid, _session_info(agent, session))
            _fb = summarize_manual_compression(
                _before_messages,
                _after_messages,
                _before_tokens,
                _after_tokens,
                compression_state=getattr(agent, "context_compressor", None),
            )
            _lines = [_fb["headline"], _fb["token_line"]]
            if _fb.get("note"):
                _lines.append(_fb["note"])
            finalize_context_engine_compression_notification(
                agent,
                committed=True,
            )
            return "\n".join(_lines)
        elif name == "fast" and agent:
            mode = arg.lower()
            if mode in {"fast", "on"}:
                agent.service_tier = "priority"
            elif mode in {"normal", "off"}:
                agent.service_tier = None
            _emit("session.info", sid, _session_info(agent, session))
        elif name == "reload-mcp" and agent and hasattr(agent, "reload_mcp_tools"):
            agent.reload_mcp_tools()
        elif name == "stop":
            from tools.process_registry import process_registry

            process_registry.kill_all()
    except Exception as e:
        if name == "compress" and agent:
            from agent.conversation_compression import (
                finalize_context_engine_compression_notification,
            )

            finalize_context_engine_compression_notification(
                agent,
                committed=False,
            )
        return f"live session sync failed: {e}"
    return ""


_MOVED = (
    _format_live_usage_output,
    _format_live_history_output,
    _format_live_prompt_output,
    _format_live_context_output,
    _format_live_tools_output,
    _format_live_help_output,
    _format_live_model_output,
    _live_slash_command_output,
    _mirror_slash_side_effects,
)


def register(server) -> None:
    """Rebind this module's helpers onto ``server``'s globals."""
    g = vars(server)
    for _fn in _MOVED:
        rebound = types.FunctionType(
            _fn.__code__, g, _fn.__name__, _fn.__defaults__, _fn.__closure__
        )
        rebound.__kwdefaults__ = _fn.__kwdefaults__
        rebound.__doc__ = _fn.__doc__
        rebound.__dict__.update(_fn.__dict__)
        setattr(server, _fn.__name__, rebound)
