"""Session history/message shaping: image-ref messages, content coercion, history->wire messages, in-flight
turn tracking and turn-failure detail. Bodies are rebound onto server.py's globals (method_ctx.bind_module)."""

from __future__ import annotations

from .method_ctx import bind_module


def _active_image_routing_identity(agent: Any) -> tuple[str, str]:
    """Return the live provider/model, falling back before agent startup."""
    from agent.auxiliary_client import _read_main_model, _read_main_provider
    return (getattr(agent, "provider", "") or _read_main_provider(), getattr(agent, "model", "") or _read_main_model())


def _build_image_ref_message(user_text: str, image_paths: list[str]) -> str:
    """Reference attached images by path so the agent analyzes them in-loop with ``vision_analyze``: pre-
    analyzing with the auxiliary vision model blocked submit 60-90s/photo and poisoned auto-titles.

    This used to pre-analyze every image with the auxiliary vision model *before* the turn was dispatched
    (``_enrich_with_attached_images``): serial blocking calls on the submit path — 60-90s per large photo —
    with failures silently swallowed and an interrupt during the window killing the turn with zero API calls
    (#83291). It also prepended the vision description to the first user message, poisoning session
    auto-titles (#82339). The CLI never gates turn dispatch on vision like this, which is why the same
    message was seconds there and minutes on desktop.
    """
    prefix = "\n\n".join(
        f"[The user attached an image: {p.name}]\n[Examine it with the vision_analyze tool using image_url: {p}]"
        for p in map(Path, image_paths) if p.exists()
    )
    text = user_text or ""
    if prefix:
        return f"{prefix}\n\n{text}" if text else prefix
    return text or "What do you see in this image?"


def _build_persist_message_with_image_refs(user_text: str, image_paths: list[str]) -> str:
    """Persisted form of the user's message: ``@image:<path>`` directives (the desktop renders them as
    images); ``_build_image_ref_message``'s ``image_url:`` hint is model-only, never persisted. Caption
    first, directives last: session previews are the first 60 chars of the first user message."""
    from agent.context_references import format_reference_value
    text = user_text or ""
    refs = "\n".join(f"@image:{format_reference_value(p)}" for p in image_paths if Path(p).exists())
    if not refs:
        return text
    return f"{text}\n{refs}" if text else refs


def _build_persist_user_message(user_text: str, image_paths: list[str], run_message: Any) -> Any:
    """Shape the persisted user turn like the model payload: ``_flush_messages_to_session_db`` ignores a
    plain-string override for a list (native-vision) payload, so swap only the text part for the
    ``@image:`` form, keep image parts, drop API-only text parts (barge-in note)."""
    persist_text = _build_persist_message_with_image_refs(user_text, image_paths)
    if not isinstance(run_message, list):
        return persist_text
    image_parts = [p for p in run_message if not (isinstance(p, dict) and p.get("type") == "text")]
    return [{"type": "text", "text": persist_text}, *image_parts]


_HISTORY_TEXT_KINDS = frozenset({"text", "input_text", "output_text"})
_HISTORY_IMAGE_KINDS = frozenset({"image_url", "input_image", "image"})
_HISTORY_AUDIO_KINDS = frozenset({"input_audio", "audio"})


def _history_part_image_url(part: dict) -> str:
    """The URL carried by an image part (``image_url`` dict or str), else ""."""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    return image_url if isinstance(image_url, str) else ""


def _history_dict_text(content: dict, *, image_urls: bool) -> str:
    """Placeholder/text rendering of one structured content dict."""
    kind = content.get("type")
    if kind in _HISTORY_TEXT_KINDS:
        return str(content.get("text") or content.get("content") or "")
    if kind in _HISTORY_IMAGE_KINDS:
        return (_history_part_image_url(content) if image_urls else "") or "[image]"
    if kind in _HISTORY_AUDIO_KINDS:
        return "[audio]"
    if kind:
        return f"[{kind}]"
    if "text" in content:
        return str(content.get("text") or "")
    return "[structured content]"


def _content_display_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(t for t in (_content_display_text(part).strip() for part in content) if t)
    if isinstance(content, dict):
        return _history_dict_text(content, image_urls=False)
    return "" if content is None else str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` (str, parts list, or one structured dict) as a plain string. Image parts
    keep their URL inline so the desktop's ``extractEmbeddedImages`` and the resume payload agree with the
    cached message (else the inline image flashed, then vanished); other shapes become a placeholder."""
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str) or (isinstance(part, dict) and isinstance(part.get("text"), str)):
                chunks.append(part if isinstance(part, str) else part["text"])
            elif isinstance(part, dict) and part.get("type"):
                rendered = _history_dict_text(part, image_urls=True)
                chunks.append(rendered if part["type"] in _HISTORY_TEXT_KINDS else f"\n{rendered}")
        return "".join(chunks)
    if isinstance(content, dict):
        return _history_dict_text(content, image_urls=True)
    return "" if content is None else str(content)


def _history_text_only_part(part: dict) -> bool:
    kind = part.get("type")
    return kind in _HISTORY_TEXT_KINDS or (kind is None and isinstance(part.get("text"), str))


def _is_text_only_busy_payload(content: Any) -> bool:
    """True when a busy submit carries only plain text, not attachments/media."""
    if isinstance(content, list):
        return bool(content) and all(
            isinstance(part, str) or (isinstance(part, dict) and _history_text_only_part(part)) for part in content
        )
    return isinstance(content, (str, int, float)) or (isinstance(content, dict) and _history_text_only_part(content))


def _is_display_hidden_marker(role: str | None, text: str) -> bool:
    """Gateway notices (model-switch, personality) persist as role=user ``[System: …]`` rows so strict providers
    accept them mid-history; they must never render as a user bubble. Filtering in this one projection hides
    them everywhere (raw marker stays in ``session["history"]``) and keeps the desktop's user ordinals stable.

    It also removes the stored marker from the payload the desktop reconciles against, so it can no longer
    shift user-message ordinals and duplicate the optimistic prompt (#67603).
    """
    return role == "user" and text.lstrip().startswith("[System:")


def _skill_scaffold_projection(content_text: str) -> str:
    """The invocation a slash-skill-expanded turn came from, else "" — UIs render ``/work fix the leak``."""
    return describe_skill_invocation(content_text, separator=" ") or ""


def _expand_skill_invocation_for_replay(text: str, task_id: str) -> str:
    """Inverse of :func:`_skill_scaffold_projection`: rewind/regenerate hands back the projected invocation,
    and re-running it verbatim would drop the skill. Unchanged when not resolvable."""
    head, _, arg = (text or "").strip().partition(" ")
    if not head.startswith("/"):
        return text
    try:
        from agent.skill_commands import build_skill_invocation_message, resolve_skill_command_key
        cmd_key = resolve_skill_command_key(head.lstrip("/"))
        return text if cmd_key is None else (build_skill_invocation_message(cmd_key, arg.strip(), task_id=task_id) or text)
    except Exception:  # a skill that no longer resolves must not break the rewind
        logger.debug("skill re-expansion failed for replay", exc_info=True)
        return text


# Opening of the crash-recovery note synthesized by _auto_continue_note; matched (not just built) for
# rows persisted before display typing existed and for the messaging gateway's twin note.
_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn was interrupted mid-run"


def _legacy_display_kind(role: str, text: str) -> str | None:
    """Display type of a synthetic row persisted untyped: new rows are typed at turn start (``persist_user_display_kind``);
    this prefix sniff migrates rows already on disk (a turn killed mid-run never reached the stamp)."""
    return "auto_continue" if role == "user" and text.lstrip().startswith(_AUTO_CONTINUE_NOTE_PREFIX) else None


_HISTORY_REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items")
_HISTORY_ROLES = frozenset({"user", "assistant", "tool", "system"})


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}
    for m in history:
        if not isinstance(m, dict):
            continue
        m = project_compaction_message_for_display(m)
        if m is None:
            continue
        role = m.get("role")
        # display_kind="hidden": model-facing scaffolding the "[System:" sniff does not catch.
        if role not in _HISTORY_ROLES or m.get("display_kind") == "hidden":
            continue
        content_text = _coerce_message_text(m.get("content"))
        if _is_display_hidden_marker(role, content_text):
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn, tc_id = tc.get("function", {}), tc.get("id", "")
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not content_text.strip():
                continue
        if role == "tool":
            tc_name, tc_args = tool_call_args.get(m.get("tool_call_id") or "", (None, None))
            name = tc_name or m.get("tool_name") or "tool"
            args = tc_args or {}
            # `context` is an 80-char preview; ship args so a full-call renderer isn't truncated.
            messages.append({"role": "tool", "name": name, "context": _tool_ctx(name, args), **({"args": args} if args else {})})
            continue
        # A reasoning-only assistant turn is kept so "Thinking…" still shows after resume/reload.
        has_reasoning = role == "assistant" and any(m.get(key) for key in _HISTORY_REASONING_KEYS)
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        # Authoring time (Unix seconds) for display.timestamps; display-only.
        # Display-only: never fed back into model context. See #41531.
        ts = m.get("timestamp")
        if isinstance(ts, (int, float)) and ts > 0:
            msg["timestamp"] = float(ts)
        # Durable row identity (_rows_to_conversation); reactions etc. address persisted messages by it.
        if m.get("_row_id") is not None:
            msg["row_id"] = m["_row_id"]
        # A user turn shows its skill invocation, never the expanded body (rewind re-sends by ordinal).
        invocation = _skill_scaffold_projection(content_text) if role == "user" else ""
        if invocation:
            msg.update(text=invocation, display_kind="skill_invocation")
        if role == "assistant":
            msg.update((key, m[key]) for key in _HISTORY_REASONING_KEYS if m.get(key) is not None)
        # Display-only timeline metadata (model switches, delegation events).
        display_kind = m.get("display_kind") or _legacy_display_kind(role, content_text)
        if display_kind:
            msg["display_kind"] = display_kind
        if m.get("display_metadata"):
            msg["display_metadata"] = m["display_metadata"]
        messages.append(msg)
    return messages


def _coerce_seed_history(value: Any) -> list[dict]:
    history = []
    for item in value if isinstance(value, list) else ():
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant", "system"):
            continue
        content = item.get("text") if item.get("content") is None else item.get("content")
        if isinstance(content, str) and content.strip():
            history.append({"role": item["role"], "content": content})
    return history


def _inflight_text(value: Any) -> str:
    return _content_display_text(value).strip()


def _start_inflight_turn(session: dict, text: Any) -> None:
    now = time.time()
    session["inflight_turn"] = {"assistant": "", "started_at": now, "streaming": True, "updated_at": now, "user": _inflight_text(text)}


def _append_inflight_delta(session: dict, delta: Any) -> None:
    text = "" if delta is None else str(delta)
    if not text:
        return
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "streaming": True, "user": ""}
    turn.update(assistant=f"{turn.get('assistant') or ''}{text}", streaming=True, updated_at=time.time())
    session["inflight_turn"] = turn


def _record_inflight_correction(session: dict, text: Any) -> None:
    """Record an accepted mid-turn correction on the live turn — appended, never written over ``user``,
    so a resuming client can rebuild BOTH bubbles."""
    correction = _inflight_text(text)
    turn = session.get("inflight_turn")
    if not correction or not isinstance(turn, dict):
        return
    # correction_offsets: arrival-order boundary (assistant chars already streamed) so resuming clients
    # place the bubble between the output seen and the output redirected.
    turn = dict(turn)
    turn["corrections"] = [*(turn.get("corrections") or []), correction]
    turn["correction_offsets"] = [*(turn.get("correction_offsets") or []), len(str(turn.get("assistant") or ""))]
    turn["updated_at"] = time.time()
    session["inflight_turn"] = turn


def _clear_inflight_turn(session: dict) -> None:
    session["inflight_turn"] = None


def _fail_inflight_turn(session: dict, error: Any, error_surface: Optional[dict] = None) -> None:
    """Mark the in-flight turn terminal-error but keep it replayable: a failure's terminal frame can be lost on
    WS disconnect and the turn may never have been committed, so the snapshot lets ``session.resume`` replay
    prompt, partial text and error. Lives until the next turn starts or the session closes. Caller holds history_lock."""
    message = str(error) if not isinstance(error, BaseException) else (str(error) or type(error).__name__)
    now = time.time()
    turn = session.get("inflight_turn")
    if not isinstance(turn, dict):
        turn = {"assistant": "", "user": "", "started_at": now}
    turn.update(
        assistant=str(turn.get("assistant") or ""), user=str(turn.get("user") or ""),
        error=message or "turn failed", status="error", recoverable=True,
    )
    if error_surface:  # {layer, code, retryable} so a reconnect renders the same layered error card
        turn["error_surface"] = dict(error_surface)
    else:
        turn.pop("error_surface", None)
    turn.update(streaming=False, updated_at=now)
    session["inflight_turn"] = turn


_TURN_FAILURE_DETAIL_LIMIT = 240
# Shortest prompt run counting as a quote-back: above shared boilerplate, below a quoted sentence.
_TURN_PROMPT_ECHO_WINDOW = 24
# Ceiling on the prompt we shingle (an @-expanded prompt can carry a whole file).
_TURN_PROMPT_ECHO_MAX_PROMPT = 65536


def _strip_prompt_echo(message: str, prompt: Any) -> str:
    """Blank runs of the submitted prompt that ``message`` quotes back: secret redaction is pattern-based and a
    provider 4xx echoing the request carries private prose matching no pattern. Any ``_TURN_PROMPT_ECHO_WINDOW``+
    char run shared with the prompt (or its JSON-escaped form) becomes ``<prompt>``; shingles keep it linear."""
    if not message or not prompt:
        return message
    needle = " ".join(str(prompt).split())[:_TURN_PROMPT_ECHO_MAX_PROMPT]
    window = _TURN_PROMPT_ECHO_WINDOW
    if len(needle) < window or len(message) < window:
        return message
    shingles = {needle[i:i + window] for i in range(len(needle) - window + 1)}
    escaped = json.dumps(needle)[1:-1]
    if escaped != needle:
        shingles.update(escaped[i:i + window] for i in range(len(escaped) - window + 1))
    out: list[str] = []
    i, n = 0, len(message)
    while i <= n - window:
        if message[i:i + window] in shingles:
            j = i + window
            while j < n and message[j - window + 1:j + 1] in shingles:
                j += 1
            out.append("<prompt>")
            i = j
        else:
            out.append(message[i])
            i += 1
    out.append(message[i:])
    return "".join(out)


def _turn_failure_detail(error: Any, reason: Any = None, prompt: Any = None) -> str:
    """Why a turn failed, for the ``tui turn finished`` bookend: ``""`` when nothing to say, else a fragment with
    its own leading space. ``redact_sensitive_text`` removes credentials; ``_strip_prompt_echo`` removes a 4xx
    body quoting ``prompt`` back. This record may gain failure detail, never the user's own content.

    86865 added the bookend to trace compression rotations, so it logs identities and a coarse ``status``
    and deliberately logs no content. 89117 is what the missing cause costs: a report consisting of two
    lines reading ``status=error error_retained=True duration=0.9s`` with no way to tell a provider 4xx from
    a budget wall from a crashed finalizer. The returned-error path -- the one a 0.9 s failure almost always
    takes -- emits no other log line at all; only the exception path prints to stderr, which is why the
    quiet failures are the ones that get filed. See #86865, #89117.
    Content discipline follows #86865's, and it takes two separate steps because it is two separate
    contracts. It does nothing about a 4xx body that quotes the request back, because ordinary private prose
    is not pattern-shaped -- so ``_strip_prompt_echo`` removes that separately, using the submitted
    ``prompt`` itself as the thing to look for.
    """
    reason_text = str(reason or "").strip()
    message = str(error or "").strip()
    if isinstance(error, BaseException):
        message = message or type(error).__name__
    if not message and not reason_text:
        return ""
    try:
        from agent.redact import redact_sensitive_text
        message = redact_sensitive_text(message, force=True)
    except Exception:
        message = "<unredactable>"  # never fail open
    message = " ".join(message.split())
    # After the collapse (same shape both sides), before truncation (a quote must not survive the cut).
    message = _strip_prompt_echo(message, prompt)
    if len(message) > _TURN_FAILURE_DETAIL_LIMIT:
        message = message[:_TURN_FAILURE_DETAIL_LIMIT] + "\u2026"
    out = " failure_reason=%s" % " ".join(reason_text.split()) if reason_text else ""
    return out + (" cause=%r" % message if message else "")


def register(server) -> None:
    """Publish this module's helpers + handlers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))
