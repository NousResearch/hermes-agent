"""Native Hermes model tools for Wisdom Kernel memory."""

from __future__ import annotations

from typing import Any, Callable

from tools.registry import registry, tool_error, tool_result
from wisdom.redaction import redact_for_log
from wisdom.service import (
    WisdomServiceContext,
    accept,
    apply as apply_wisdom,
    applications_payload,
    archive,
    archive_payload,
    capture_payload,
    captures_payload,
    dismiss,
    inbox,
    interpret,
    interpretation_payload,
    original as get_original_capture,
    original_payload,
    related,
    related_payload,
    review,
    review_action_payload,
    review_payload,
    search,
    set_enabled,
    status_payload,
    status_to_dict,
)


WISDOM_TOOL_NAMES = [
    "wisdom_status",
    "wisdom_capture",
    "wisdom_search",
    "wisdom_original",
    "wisdom_interpret",
    "wisdom_apply",
    "wisdom_review",
    "wisdom_related",
    "wisdom_accept",
    "wisdom_dismiss",
    "wisdom_archive",
    "wisdom_inbox",
    "wisdom_set_enabled",
]

_CATEGORIES = ["business", "investing", "health", "life", "inbox"]
_SOURCE_TYPES = [
    "thought",
    "voice",
    "podcast",
    "book",
    "article",
    "meeting",
    "quote",
    "conversation",
    "other",
]
_APPLICATION_TYPES = [
    "task_proposal",
    "reminder_proposal",
    "principle",
    "checklist",
    "client_language",
    "investment_rule",
    "health_experiment",
    "writing_idea",
    "decision_rule",
]


def _always_available() -> bool:
    return True


def _context(args: dict[str, Any], task_id: object | None) -> WisdomServiceContext:
    source_kind = str(args.get("source_kind") or "model_tool").strip() or "model_tool"
    return WisdomServiceContext(
        channel="model_tool",
        source_kind=source_kind,
        session_key=task_id,
        message_ref=None,
    )


def _safe_tool(call: Callable[[], dict[str, Any]]) -> str:
    try:
        return tool_result(call())
    except Exception as exc:
        safe_detail = redact_for_log(f"{type(exc).__name__}: {exc}")
        return tool_error("Wisdom tool failed safely.", detail=safe_detail, ok=False)


def _int_arg(args: dict[str, Any], key: str) -> int:
    value = args.get(key)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def wisdom_status_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    return _safe_tool(lambda: status_payload())


def wisdom_capture_handler(args: dict[str, Any], **kwargs: Any) -> str:
    task_id = kwargs.get("task_id")

    def _call() -> dict[str, Any]:
        return capture_payload(
            str(args.get("text") or ""),
            source_kind=args.get("source_kind"),
            category=args.get("category"),
            source_type=args.get("source_type"),
            context_note=args.get("context_note"),
            context=_context(args, task_id),
        )

    return _safe_tool(_call)


def wisdom_search_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        results = search(
            str(args.get("query") or ""),
            category=args.get("category"),
            limit=args.get("limit"),
        )
        return captures_payload(results, title="Wisdom search")

    return _safe_tool(_call)


def wisdom_original_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    return _safe_tool(lambda: original_payload(_int_arg(args, "capture_id")))


def wisdom_interpret_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        capture_id = _int_arg(args, "capture_id")
        if get_original_capture(capture_id) is None:
            return {"ok": False, "error": f"Capture #{capture_id} was not found.", "capture_id": capture_id}
        return interpretation_payload(interpret(capture_id, create=True), capture_id=capture_id)

    return _safe_tool(_call)


def wisdom_apply_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        capture_id = _int_arg(args, "capture_id")
        if get_original_capture(capture_id) is None:
            return {"ok": False, "error": f"Capture #{capture_id} was not found.", "capture_id": capture_id}
        return applications_payload(
            apply_wisdom(
                capture_id,
                application_type=args.get("application_type"),
                context=args.get("context"),
            ),
            capture_id=capture_id,
        )

    return _safe_tool(_call)


def wisdom_review_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        data = review(
            category=args.get("category"),
            mode=args.get("mode"),
            period=args.get("period"),
            limit=args.get("limit"),
        )
        return review_payload(data)

    return _safe_tool(_call)


def wisdom_related_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        capture_id = _int_arg(args, "capture_id")
        if get_original_capture(capture_id) is None:
            return {"ok": False, "error": f"Capture #{capture_id} was not found.", "capture_id": capture_id}
        return related_payload(related(capture_id, limit=args.get("limit")), capture_id=capture_id)

    return _safe_tool(_call)


def wisdom_accept_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        capture_id = _int_arg(args, "capture_id")
        return review_action_payload("accepted", accept(capture_id), capture_id=capture_id)

    return _safe_tool(_call)


def wisdom_dismiss_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        capture_id = _int_arg(args, "capture_id")
        return review_action_payload("dismissed", dismiss(capture_id), capture_id=capture_id)

    return _safe_tool(_call)


def wisdom_archive_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        capture_id = _int_arg(args, "capture_id")
        return archive_payload(capture_id, archive(capture_id))

    return _safe_tool(_call)


def wisdom_inbox_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        return captures_payload(
            inbox(category=args.get("category"), limit=args.get("limit")),
            title="Wisdom inbox",
        )

    return _safe_tool(_call)


def wisdom_set_enabled_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    def _call() -> dict[str, Any]:
        enabled = bool(args.get("enabled"))
        return {"ok": True, "status": status_to_dict(set_enabled(enabled))}

    return _safe_tool(_call)


WISDOM_STATUS_SCHEMA = {
    "name": "wisdom_status",
    "description": (
        "Check Hermes Wisdom memory state. Use when the user asks whether Wisdom is available, "
        "enabled, ready, or what DB/capture/search status it has. Returns enabled state, capture "
        "mode, DB path, counts, FTS availability, and last capture time."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

WISDOM_CAPTURE_SCHEMA = {
    "name": "wisdom_capture",
    "description": (
        "Durably save a user idea in Hermes Wisdom. Use when the user explicitly asks to remember, "
        "save, capture, note, log, preserve, or record an idea. Trigger phrases include 'remember "
        "this', 'save this thought', 'capture this', 'note this', 'podcast idea', 'book note', "
        "'health note', 'investing thought', and 'business idea'. Do not use for ordinary chat "
        "unless the user clearly requests durable capture. Never claim something is saved unless "
        "this tool succeeds. The text input is preserved as the exact original; secret-like text "
        "is blocked rather than redacted into a changed original."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The exact user wording to preserve as the original capture.",
            },
            "source_kind": {
                "type": "string",
                "description": "Optional source label such as model_tool, text, conversation, podcast, or book.",
            },
            "category": {
                "type": "string",
                "enum": _CATEGORIES,
                "description": "Optional explicit category. Omit when unsure so Wisdom classifies deterministically.",
            },
            "source_type": {
                "type": "string",
                "enum": _SOURCE_TYPES,
                "description": "Optional explicit source type. Omit when unsure.",
            },
            "context_note": {
                "type": "string",
                "description": "Optional short non-secret context note. This does not replace the exact original text.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}

WISDOM_SEARCH_SCHEMA = {
    "name": "wisdom_search",
    "description": (
        "Search saved Hermes Wisdom captures. Use when the user asks to find, search, recall, "
        "retrieve, pull up, or ask what they previously said, thought, or captured about something. "
        "Examples: 'find that idea about peace of mind', 'what have I said about risk', 'show my "
        "thoughts on clients', 'pull up the note where I said...', 'did I capture anything about...'. "
        "Always search Wisdom before answering questions about saved Wisdom ideas. Do not hallucinate "
        "missing results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query in the user's words."},
            "category": {"type": "string", "enum": _CATEGORIES, "description": "Optional category filter."},
            "limit": {"type": "integer", "description": "Maximum number of results, usually 5 or fewer."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

WISDOM_ORIGINAL_SCHEMA = {
    "name": "wisdom_original",
    "description": (
        "Return the exact original text for a saved Wisdom capture. Use when the user asks for exact "
        "wording, original phrasing, verbatim text, 'what exactly did I write', 'show original', or "
        "'what were my exact words'. Do not summarize, rewrite, or clean the result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID."},
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_INTERPRET_SCHEMA = {
    "name": "wisdom_interpret",
    "description": (
        "Interpret a saved Wisdom capture deterministically. Use when the user asks what a saved "
        "capture means, wants a summary, wants the idea clarified, or asks for a lightweight "
        "interpretation. This never overwrites the original."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID."},
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_APPLY_SCHEMA = {
    "name": "wisdom_apply",
    "description": (
        "Create internal application proposals from a saved Wisdom capture. Use when the user asks to "
        "apply, transform, convert, or turn an idea into client language, a checklist, an investment "
        "rule, a health experiment, a principle, writing idea, or decision rule. Examples: 'turn that "
        "into client language', 'make this a checklist', 'make this an investment rule', 'apply this "
        "to x10x'. This creates internal proposals only; it does not create external tasks or reminders."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID."},
            "application_type": {
                "type": "string",
                "enum": _APPLICATION_TYPES,
                "description": "Optional proposal type to return, such as client_language, checklist, or decision_rule.",
            },
            "context": {
                "type": "string",
                "description": "Optional non-secret context for the final response. Durable writes remain deterministic.",
            },
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_REVIEW_SCHEMA = {
    "name": "wisdom_review",
    "description": (
        "Review and prioritize saved Wisdom captures. Use when the user asks 'what should I review', "
        "'review my recent captures', 'what have I captured but not applied', 'show unapplied ideas', "
        "'show high-potential ideas', 'what have I been thinking about business/investing', "
        "or 'what should I do with my "
        "Wisdom notes'. Returns ranked review items with quality indicators, suggested next actions, "
        "and related captures. This is manual only; it does not schedule pings."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": _CATEGORIES, "description": "Optional category filter."},
            "mode": {
                "type": "string",
                "enum": ["needs_review", "unapplied", "high_potential", "all"],
                "description": (
                    "Review mode: needs_review for normal queue, unapplied for captures without proposals, "
                    "high_potential for highest-scoring items, or all for broad inspection."
                ),
            },
            "period": {"type": "string", "description": "Optional human period label such as recent, week, or month."},
            "limit": {"type": "integer", "description": "Maximum number of captures to include."},
        },
        "additionalProperties": False,
    },
}

WISDOM_RELATED_SCHEMA = {
    "name": "wisdom_related",
    "description": (
        "Find deterministic related Wisdom captures for a saved capture. Use when the user asks 'show "
        "related ideas', 'what does this connect to', 'have I said something like this before', "
        "'find related captures for #12', or asks what an idea resembles in prior Wisdom notes. Uses "
        "FTS, category/source matching, keyword overlap, and recency; no embeddings or external calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID to compare from."},
            "limit": {"type": "integer", "description": "Maximum number of related captures, usually 5 or fewer."},
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_ACCEPT_SCHEMA = {
    "name": "wisdom_accept",
    "description": (
        "Mark a Wisdom capture as accepted: worth keeping, reviewing, and compounding. Use when the "
        "user asks to accept, keep, approve, mark as useful, or mark an idea as worth compounding. "
        "This changes review status only; it does not create external actions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID to accept."},
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_DISMISS_SCHEMA = {
    "name": "wisdom_dismiss",
    "description": (
        "Mark a Wisdom capture as dismissed without deleting it. Use when the user says dismiss, not "
        "useful, noise, hide from review, deprioritize, or 'that one is not worth keeping'. The exact "
        "original remains preserved unless separately archived."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID to dismiss from normal review."},
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_ARCHIVE_SCHEMA = {
    "name": "wisdom_archive",
    "description": (
        "Archive a Wisdom capture without deleting it. Use when the user asks to archive, hide, or "
        "remove a saved capture from normal Wisdom surfaces. If the user only says an idea is noise "
        "or not useful, prefer wisdom_dismiss."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capture_id": {"type": "integer", "description": "Wisdom capture ID."},
        },
        "required": ["capture_id"],
        "additionalProperties": False,
    },
}

WISDOM_INBOX_SCHEMA = {
    "name": "wisdom_inbox",
    "description": (
        "List recent non-archived Wisdom captures. Use when the user asks for the Wisdom inbox, recent "
        "raw captures, or saved items that may need review."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": _CATEGORIES, "description": "Optional category filter."},
            "limit": {"type": "integer", "description": "Maximum number of captures to include."},
        },
        "additionalProperties": False,
    },
}

WISDOM_SET_ENABLED_SCHEMA = {
    "name": "wisdom_set_enabled",
    "description": (
        "Turn Hermes Wisdom capture on or off. Use only when the user explicitly asks to enable, disable, "
        "turn on, turn off, pause, or resume Wisdom memory capture. Status/help-style requests should use "
        "wisdom_status instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "description": "True to enable Wisdom; false to disable it."},
        },
        "required": ["enabled"],
        "additionalProperties": False,
    },
}


for _name, _schema, _handler in (
    ("wisdom_status", WISDOM_STATUS_SCHEMA, wisdom_status_handler),
    ("wisdom_capture", WISDOM_CAPTURE_SCHEMA, wisdom_capture_handler),
    ("wisdom_search", WISDOM_SEARCH_SCHEMA, wisdom_search_handler),
    ("wisdom_original", WISDOM_ORIGINAL_SCHEMA, wisdom_original_handler),
    ("wisdom_interpret", WISDOM_INTERPRET_SCHEMA, wisdom_interpret_handler),
    ("wisdom_apply", WISDOM_APPLY_SCHEMA, wisdom_apply_handler),
    ("wisdom_review", WISDOM_REVIEW_SCHEMA, wisdom_review_handler),
    ("wisdom_related", WISDOM_RELATED_SCHEMA, wisdom_related_handler),
    ("wisdom_accept", WISDOM_ACCEPT_SCHEMA, wisdom_accept_handler),
    ("wisdom_dismiss", WISDOM_DISMISS_SCHEMA, wisdom_dismiss_handler),
    ("wisdom_archive", WISDOM_ARCHIVE_SCHEMA, wisdom_archive_handler),
    ("wisdom_inbox", WISDOM_INBOX_SCHEMA, wisdom_inbox_handler),
    ("wisdom_set_enabled", WISDOM_SET_ENABLED_SCHEMA, wisdom_set_enabled_handler),
):
    registry.register(
        name=_name,
        toolset="wisdom",
        schema=_schema,
        handler=_handler,
        check_fn=_always_available,
        emoji="W",
    )
