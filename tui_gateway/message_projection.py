"""Pure message-display projection helpers (moved verbatim from server.py).

Wave-1 godfile extraction (shard s3, cluster c14): content-text coercion,
busy-payload classification, display-kind inference, and the
history-to-messages projection. Bodies are byte-identical to their pre-split
server.py form. The one shared server global (``_tool_ctx``) is bound at the
bottom of this module; server.py imports this module from its tail, after
``_tool_ctx`` (line 5140) already exists, so the binding cannot cycle.
"""

import json
import logging
from typing import Any

from agent.skill_commands import describe_skill_invocation

logger = logging.getLogger(__name__)


def _content_display_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _content_display_text(part).strip()
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            return "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


def _coerce_message_text(content: Any) -> str:
    """Render ``message['content']`` as a plain string for transport.

    Provider-side, ``content`` may be a string (most common), a list of
    multimodal parts (e.g. ``[{"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {...}}]``), or a single structured
    dict. Calling ``.strip()`` on a list raises ``'list' object has no
    attribute 'strip'`` and breaks session resume entirely.

    Image parts (``image_url``) are preserved by appending the underlying
    URL (data: or http:) into the text. The desktop renderer pulls these
    back out via ``extractEmbeddedImages`` so the user sees the image
    instead of the URL — and it stops the resume payload from disagreeing
    with the cached message (which would otherwise cause the inline image
    to flash, then disappear when the resume payload overwrites the cache).

    Other structured dict shapes (audio, unknown types) fall back to a
    bracketed placeholder so resume doesn't drop the message entirely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
                continue
            kind = part.get("type")
            if kind in {"text", "input_text", "output_text"}:
                t = part.get("text") or part.get("content") or ""
                if t:
                    chunks.append(str(t))
                continue
            if kind in {"image_url", "input_image", "image"}:
                image_url = part.get("image_url")
                url = ""
                if isinstance(image_url, dict):
                    candidate = image_url.get("url")
                    if isinstance(candidate, str):
                        url = candidate
                elif isinstance(image_url, str):
                    url = image_url
                if url:
                    chunks.append(f"\n{url}")
                else:
                    chunks.append("\n[image]")
                continue
            if kind in {"input_audio", "audio"}:
                chunks.append("\n[audio]")
                continue
            if kind:
                chunks.append(f"\n[{kind}]")
        return "".join(chunks)
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in {"text", "input_text", "output_text"}:
            return str(content.get("text") or content.get("content") or "")
        if kind in {"image_url", "input_image", "image"}:
            image_url = content.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                candidate = image_url.get("url")
                if isinstance(candidate, str):
                    url = candidate
            elif isinstance(image_url, str):
                url = image_url
            return url or "[image]"
        if kind in {"input_audio", "audio"}:
            return "[audio]"
        if kind:
            return f"[{kind}]"
        if "text" in content:
            return str(content.get("text") or "")
        return "[structured content]"
    return str(content)


_TEXT_ONLY_BUSY_PART_KINDS = frozenset({"text", "input_text", "output_text"})


def _is_text_only_busy_payload(content: Any) -> bool:
    """True when a busy submit carries only plain text, not attachments/media."""
    if content is None:
        return False
    if isinstance(content, (str, int, float)):
        return True
    if isinstance(content, list):
        if not content:
            return False
        for part in content:
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                return False
            kind = part.get("type")
            if kind in _TEXT_ONLY_BUSY_PART_KINDS:
                continue
            if kind is None and isinstance(part.get("text"), str):
                continue
            return False
        return True
    if isinstance(content, dict):
        kind = content.get("type")
        if kind in _TEXT_ONLY_BUSY_PART_KINDS:
            return True
        return kind is None and isinstance(content.get("text"), str)
    return False


def _is_display_hidden_marker(role: str | None, text: str) -> bool:
    """Gateway bookkeeping notices (model-switch, personality) are persisted as
    role=user ``[System: …]`` rows so strict providers accept them mid-history.
    They are model-facing runtime metadata, not user turns, and must never
    render as a user bubble in ANY client transcript (desktop, TUI, CLI, web).

    Filtering here — the single display projection every surface reads — hides
    them everywhere while the raw marker stays in ``session["history"]`` for the
    model. It also removes the stored marker from the payload the desktop
    reconciles against, so it can no longer shift user-message ordinals and
    duplicate the optimistic prompt (#67603)."""
    return role == "user" and text.lstrip().startswith("[System:")


def _skill_scaffold_projection(content_text: str) -> str:
    """Return the invocation a slash-skill-expanded turn came from, else "".

    A ``/skill`` invocation expands into a model-facing message that embeds the
    whole skill body. That payload belongs to the agent — every UI renders the
    invocation (``/work fix the leak``) instead, so no surface can leak the
    body into a chat bubble.
    """
    return describe_skill_invocation(content_text, separator=" ") or ""


def _expand_skill_invocation_for_replay(text: str, task_id: str) -> str:
    """Re-expand a projected `/skill` invocation before re-running that turn.

    The inverse of :func:`_skill_scaffold_projection`. Because a skill turn is
    displayed as its invocation, a rewind/regenerate hands us back
    ``/work fix the leak`` rather than the body the agent originally saw —
    re-running that verbatim would drop the skill. Re-expanding here keeps the
    body server-side (no client ever holds it) and makes the replayed turn
    identical to the original.

    Returns *text* unchanged when it isn't a resolvable skill invocation.
    """
    head, _, arg = (text or "").strip().partition(" ")
    if not head.startswith("/"):
        return text

    try:
        from agent.skill_commands import (
            build_skill_invocation_message,
            resolve_skill_command_key,
        )

        cmd_key = resolve_skill_command_key(head.lstrip("/"))
        if cmd_key is None:
            return text

        return build_skill_invocation_message(cmd_key, arg.strip(), task_id=task_id) or text
    except Exception:
        # A skill that no longer resolves (renamed, disabled, external dir
        # gone) must not break the rewind — replay the text as typed.
        logger.debug("skill re-expansion failed for replay", exc_info=True)
        return text


# Opening of the crash-recovery note synthesized by _auto_continue_note.
# Matched (not just built) so a row persisted before the display type was
# stamped at turn start still reads as a timeline event, and to recognize the
# messaging gateway's twin note.
_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn was interrupted mid-run"


def _legacy_display_kind(role: str, text: str) -> str | None:
    """Infer the display type of a synthetic row persisted without one.

    Turn-start typing (see ``persist_user_display_kind``) covers everything
    written from here on. Sessions already on disk carry untyped rows — and a
    turn killed mid-run never reached the post-turn stamp at all, which is
    exactly the auto-continue case — so the raw recovery note would paint as a
    user bubble forever. Sniffing the one fixed synthetic prefix is the
    migration for those rows; it is not how new rows get typed.
    """
    if role == "user" and text.lstrip().startswith(_AUTO_CONTINUE_NOTE_PREFIX):
        return "auto_continue"
    return None


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    tool_call_args = {}

    for m in history:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            continue
        # An explicit display_kind="hidden" row is model-facing scaffolding
        # (compaction references, interrupted-turn checkpoints). The string
        # sniff below only catches the "[System:" convention; honor the
        # declared field too, or scaffolding reaches every surface that reads
        # this projection.
        if m.get("display_kind") == "hidden":
            continue
        content_text = _coerce_message_text(m.get("content"))
        if _is_display_hidden_marker(role, content_text):
            continue
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                if tc_id and fn.get("name"):
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_call_args[tc_id] = (fn["name"], args)
            if not content_text.strip():
                continue
        if role == "tool":
            tc_id = m.get("tool_call_id", "")
            tc_info = tool_call_args.get(tc_id) if tc_id else None
            name = (tc_info[0] if tc_info else None) or m.get("tool_name") or "tool"
            args = (tc_info[1] if tc_info else None) or {}
            messages.append(
                {"role": "tool", "name": name, "context": _tool_ctx(name, args)}
            )
            continue
        # An assistant turn may carry only reasoning/thinking content with no
        # visible text (extended-thinking turns, thinking-only recovery
        # responses). Such a turn is persisted with its reasoning fields and is
        # recallable from the transcript, but dropping it here as "empty" makes
        # it vanish from the resumed/reloaded session view while the desktop's
        # reasoning disclosure has nothing to render. Keep it when it carries
        # reasoning so the "Thinking…" block still shows. (#44022)
        reasoning_keys = (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
        )
        has_reasoning = role == "assistant" and any(
            m.get(key) for key in reasoning_keys
        )
        if not content_text.strip() and not has_reasoning:
            continue
        msg = {"role": role, "text": content_text}
        # Durable row identity, stamped by _rows_to_conversation. The renderer's
        # own message ids are ephemeral (timestamp+index derived, and a
        # different shape for live vs rehydrated vs optimistic rows), so
        # anything that addresses a specific persisted message later — message
        # reactions — needs this instead.
        if m.get("_row_id") is not None:
            msg["row_id"] = m["_row_id"]
        if role == "user":
            invocation = _skill_scaffold_projection(content_text)
            if invocation:
                # Show the invocation, never the expanded skill body. The raw
                # payload stays server-side: a rewind/regenerate re-sends the
                # turn by ordinal, so no client needs it.
                msg["text"] = invocation
                msg["display_kind"] = "skill_invocation"
        if role == "assistant":
            for key in reasoning_keys:
                if key in m and m.get(key) is not None:
                    msg[key] = m.get(key)
        # Forward display-only timeline metadata so the TUI can render
        # model switches and delegation completions as events instead of
        # opaque user messages, and hide compaction handoffs entirely.
        display_kind = m.get("display_kind") or _legacy_display_kind(role, content_text)
        if display_kind:
            msg["display_kind"] = display_kind
        if m.get("display_metadata"):
            msg["display_metadata"] = m["display_metadata"]
        messages.append(msg)

    return messages


def _coerce_seed_history(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []

    history = []
    for item in value:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            continue

        content = item.get("content")
        if content is None:
            content = item.get("text")
        if not isinstance(content, str) or not content.strip():
            continue

        history.append({"role": role, "content": content})

    return history


# server.py imports this module from its tail (after ``_tool_ctx`` is defined
# at line 5140), so binding the shared helper here cannot create an import
# cycle. All other names these functions close over are defined above.
from tui_gateway.server import _tool_ctx  # noqa: E402
