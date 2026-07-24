#!/usr/bin/env python3
"""
Clarify Form Tool - Batched Clarifying Questions

Presents several clarifying questions in a single interaction: the agent gathers
every field it still needs into ONE tool call, the UI renders them together as a
form, and the user answers each (single- or multi-select, or free text) and
submits once. This replaces firing the single-question ``clarify`` tool
repeatedly for a multi-field flow.

Like ``clarify``, the actual user-interaction logic lives in the platform layer;
this module defines the schema, validation, and a thin dispatcher that delegates
to a platform-provided callback. Only the web dashboard (tui_gateway) wires the
callback today, so on platforms without it the tool degrades to an explanatory
error and the agent falls back to plain ``clarify``.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from tools.clarify_tool import MAX_CHOICES, _flatten_choice
from tools.registry import registry, tool_error


MAX_QUESTIONS = 6


def _normalize_questions(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("questions must be a list.")
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(raw[:MAX_QUESTIONS]):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        choices_raw = item.get("choices")
        choices: Optional[List[str]] = None
        if isinstance(choices_raw, list):
            choices = [s for s in (_flatten_choice(c) for c in choices_raw) if s][:MAX_CHOICES]
            if not choices:
                choices = None
        normalized.append({
            "id": str(item.get("id") or f"q{index + 1}"),
            "question": question,
            "choices": choices,
            "multiSelect": bool(item.get("multiSelect", False)) and choices is not None,
        })
    return normalized


def _normalize_answers(
    questions: List[Dict[str, Any]],
    raw_answers: Any,
) -> List[Dict[str, Any]]:
    by_id = {q["id"]: q for q in questions}
    provided: Dict[str, Any] = {}
    if isinstance(raw_answers, list):
        for entry in raw_answers:
            if isinstance(entry, dict) and entry.get("id") is not None:
                provided[str(entry["id"])] = entry.get("answer")
    responses: List[Dict[str, Any]] = []
    for q in questions:
        answer = provided.get(q["id"])
        if isinstance(answer, list):
            value: Any = [str(a).strip() for a in answer if str(a).strip()]
        else:
            value = str(answer).strip() if answer is not None else ""
        responses.append({"question": q["question"], "answer": value})
    return responses


def clarify_form_tool(
    questions: Optional[List[dict]] = None,
    callback: Optional[Callable] = None,
) -> str:
    try:
        normalized = _normalize_questions(questions)
    except ValueError as exc:
        return tool_error(str(exc))

    if not normalized:
        return tool_error("At least one question with non-empty text is required.")

    if callback is None:
        return json.dumps(
            {"error": "Clarify form tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        raw_answers = callback(normalized)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    if not isinstance(raw_answers, list):
        return json.dumps(
            {"responses": [], "answered": False},
            ensure_ascii=False,
        )

    return json.dumps(
        {"responses": _normalize_answers(normalized, raw_answers), "answered": True},
        ensure_ascii=False,
    )


def check_clarify_form_requirements() -> bool:
    return True


CLARIFY_FORM_SCHEMA = {
    "name": "clarify_form",
    "description": (
        "Ask the user SEVERAL clarifying questions at once, rendered as a single "
        "form the user fills in and submits one time. Use this INSTEAD of calling "
        "`clarify` repeatedly whenever two or more fields are still unresolved "
        "after checking the user's message and earlier turns — e.g. a broadcast "
        "flow missing platform, audience, and schedule.\n\n"
        "Each question carries its own optional `choices` (up to 4 selectable "
        "rows; a 5th 'Other' free-text option is always added by the UI) and an "
        "optional `multiSelect` flag when the user may pick more than one. Omit "
        "`choices` for a free-text question.\n\n"
        "CRITICAL: put each option ONLY in that question's `choices` array — never "
        "enumerate options inside the question text. For a SINGLE field, prefer "
        "the plain `clarify` tool. Do not use either tool for yes/no confirmation "
        "of dangerous commands (the terminal tool handles that)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "maxItems": MAX_QUESTIONS,
                "description": (
                    f"The questions to ask together (up to {MAX_QUESTIONS}). Each "
                    "is its own object; order is preserved in the form."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": (
                                "Optional stable key for this question; auto-assigned "
                                "(q1, q2, …) when omitted."
                            ),
                        },
                        "question": {
                            "type": "string",
                            "description": (
                                "The question text ONLY (e.g. 'Which platform?'). Do "
                                "NOT embed answer options here."
                            ),
                        },
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_CHOICES,
                            "description": (
                                "Selectable options for this question (up to 4). Omit "
                                "entirely for a free-text answer."
                            ),
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": (
                                "Set true when the user may select more than one "
                                "choice for this question. Ignored without `choices`."
                            ),
                        },
                    },
                    "required": ["question"],
                },
            },
        },
        "required": ["questions"],
    },
}


registry.register(
    name="clarify_form",
    toolset="clarify",
    schema=CLARIFY_FORM_SCHEMA,
    handler=lambda args, **kw: clarify_form_tool(
        questions=args.get("questions"),
        callback=kw.get("callback")),
    check_fn=check_clarify_form_requirements,
    emoji="📋",
)
