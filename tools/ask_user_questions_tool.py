#!/usr/bin/env python3
"""Rich mode questions adapted onto Hermes' shared clarification lane.

Plan, UltraPlan and Recon keep the structured ``options`` schema (labels,
descriptions and optional recommendations), while interaction is delegated to
``clarify`` so every surface gets the same batch, multi-select, inline Other,
timeout and cancellation behaviour. The original ``callback(questions)``
contract remains as a compatibility fallback for third-party integrations.

Must survive upstream merges — see skill ``agent-modes``.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error


# Hard caps.  These mirror Claude Code's documented limits and the
# recon/UltraPlan spec ranges:
#   - Plan mode: 2-4 mandatory interview
#   - UltraPlan: 10-15 total, in batches of 3-4
#   - Recon:     exactly 4 upfront
# Tool-level cap of 4 per batch keeps the TUI overlay readable.
MAX_QUESTIONS_PER_CALL = 4
MAX_OPTIONS_PER_QUESTION = 4

# Sentinels used only by the original callback compatibility lane.
# Shared clarify-capable surfaces return inline text and explicit statuses.
OTHER_SENTINEL = "__other__"
SKIPPED_SENTINEL = "__skipped__"
ANSWER_STATUSES = frozenset({"answered", "skipped", "timed_out", "cancelled"})


def _normalise_questions(raw_questions: Any) -> List[Dict[str, Any]]:
    """Validate, trim, and normalise the question list.

    Returns the cleaned list.  Raises ValueError on structural errors so
    the agent gets an actionable error message rather than silent loss.
    """
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(
            "questions must be a non-empty list of question objects"
        )
    if len(raw_questions) > MAX_QUESTIONS_PER_CALL:
        raise ValueError(
            f"too many questions in a single batch: {len(raw_questions)} "
            f"(max {MAX_QUESTIONS_PER_CALL} per ask_user_questions call)"
        )

    cleaned = []
    for i, q in enumerate(raw_questions):
        if not isinstance(q, dict):
            raise ValueError(f"question[{i}] must be an object")
        text = str(q.get("question", "")).strip()
        if not text:
            raise ValueError(f"question[{i}].question is required")
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            raise ValueError(
                f"question[{i}] needs at least 2 options in `options`"
            )
        if len(opts) > MAX_OPTIONS_PER_QUESTION:
            raise ValueError(
                f"question[{i}] has {len(opts)} options (max "
                f"{MAX_OPTIONS_PER_QUESTION})"
            )
        norm_opts = []
        recommended_count = 0
        for j, opt in enumerate(opts):
            if not isinstance(opt, dict):
                raise ValueError(
                    f"question[{i}].options[{j}] must be an object"
                )
            label = str(opt.get("label", "")).strip()
            if not label:
                raise ValueError(
                    f"question[{i}].options[{j}].label is required"
                )
            is_rec = bool(opt.get("recommended", False))
            if is_rec:
                recommended_count += 1
            norm_opts.append({
                "label": label,
                "description": str(opt.get("description", "")).strip() or None,
                "recommended": is_rec,
            })
        if recommended_count > 1:
            raise ValueError(
                f"question[{i}] has {recommended_count} recommended options "
                "(at most one is allowed)"
            )
        header = str(q.get("header", "")).strip()
        if len(header) > 12:
            raise ValueError(f"question[{i}].header exceeds 12 characters")
        cleaned.append({
            "question": text,
            "header": header or None,
            "options": norm_opts,
            "multiSelect": bool(q.get("multiSelect", False)),
        })
    return cleaned


def _coerce_answer(raw: Any) -> Dict[str, Any]:
    """Translate one legacy callback value into an answer + status."""
    if isinstance(raw, list):
        values = [str(value).strip() for value in raw if str(value).strip()]
        return {
            "answer": values,
            "needs_text": False,
            "status": "answered" if values else "skipped",
        }
    label = str(raw or "").strip()
    if not label or label == SKIPPED_SENTINEL:
        return {"answer": "(skipped)", "needs_text": False, "status": "skipped"}
    if label == OTHER_SENTINEL:
        return {
            "answer": "(awaiting text)",
            "needs_text": True,
            "status": "awaiting_text",
        }
    return {"answer": label, "needs_text": False, "status": "answered"}


def _shared_questions(cleaned: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate the rich AUQ schema onto clarify's cross-platform batch lane."""
    from tools.clarify_tool import RECOMMENDED_LABEL

    shared = []
    for index, question in enumerate(cleaned):
        labels = [option["label"] for option in question["options"]]
        choices = [
            f"{option['label']} {RECOMMENDED_LABEL}"
            if option["recommended"]
            else option["label"]
            for option in question["options"]
        ]
        shared.append({
            "qid": f"q{index}",
            "id": None,
            "question": question["question"],
            "choices": choices,
            "choices_offered": labels,
            "multi_select": question["multiSelect"],
            # Rich-capable clients render these; legacy messaging callbacks
            # continue to receive the portable string ``choices`` above.
            "header": question["header"],
            "options": question["options"],
        })
    return shared


def _shared_result(
    cleaned: List[Dict[str, Any]],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Map clarify's shared result onto the AUQ compatibility result shape."""
    responses = list(result.get("responses") or [])
    timed_out = bool(result.get("timed_out"))
    cancelled = bool(result.get("cancelled"))
    answers = []
    for index, question in enumerate(cleaned):
        response = responses[index] if index < len(responses) else {}
        value = response.get("user_response", "")
        status = str(response.get("status") or "")
        if status not in ANSWER_STATUSES:
            if value not in ("", []):
                status = "answered"
            elif timed_out:
                status = "timed_out"
            elif cancelled:
                status = "cancelled"
            else:
                status = "skipped"
        display = value if value not in ("", []) else f"({status.replace('_', ' ')})"
        answers.append({
            "index": index,
            "question": question["question"],
            "answer": display,
            "status": status,
            "needs_text": False,
        })

    payload: Dict[str, Any] = {
        "questions_asked": len(cleaned),
        "answers": answers,
        "responses": responses,
        "needs_followup": [],
    }
    if timed_out:
        payload["timed_out"] = True
    if cancelled:
        payload["cancelled"] = True
    return payload


def _legacy_result(
    cleaned: List[Dict[str, Any]],
    raw_answers: Dict[Any, Any],
) -> Dict[str, Any]:
    """Preserve the original callback contract for third-party callers."""
    answer_rows = []
    responses = []
    needs_followup: List[int] = []
    for index, question in enumerate(cleaned):
        raw = raw_answers.get(index, raw_answers.get(str(index)))
        coerced = _coerce_answer(raw)
        if coerced["needs_text"]:
            needs_followup.append(index)
        answer_rows.append({
            "index": index,
            "question": question["question"],
            "answer": coerced["answer"],
            "status": coerced["status"],
            "needs_text": coerced["needs_text"],
        })
        responses.append({
            "question": question["question"],
            "choices_offered": [option["label"] for option in question["options"]],
            "user_response": "" if coerced["status"] != "answered" else coerced["answer"],
            "status": coerced["status"],
        })
    return {
        "questions_asked": len(cleaned),
        "answers": answer_rows,
        "responses": responses,
        "needs_followup": needs_followup,
    }


def ask_user_questions_tool(
    questions: List[Dict[str, Any]],
    callback: Optional[Callable] = None,
    clarify_callback: Optional[Callable] = None,
) -> str:
    """Ask rich batched questions through the shared clarify interaction lane.

    ``clarify_callback`` is the primary route and supplies multi-select,
    inline free text, desktop/TUI rendering, messaging fallback, timeout and
    cancellation semantics. ``callback`` preserves the original
    ``callback(questions) -> {index: answer}`` contract for integrations that
    have not migrated yet.
    """
    try:
        cleaned = _normalise_questions(questions)
    except ValueError as exc:
        return tool_error(str(exc))

    if clarify_callback is not None:
        try:
            from tools.clarify_tool import run_question_batch

            raw_result = run_question_batch(
                _shared_questions(cleaned), clarify_callback, "Requirements",
            )
            parsed = json.loads(raw_result)
            return json.dumps(_shared_result(cleaned, parsed), ensure_ascii=False)
        except Exception as exc:
            return tool_error(f"Failed to collect user answers: {exc}")

    if callback is None:
        return tool_error(
            "ask_user_questions is unavailable here; use the clarify tool instead."
        )

    try:
        raw_answers = callback(cleaned)
    except Exception as exc:
        return tool_error(f"Failed to collect user answers: {exc}")
    if not isinstance(raw_answers, dict):
        return tool_error("callback must return a dict[index, answer]")
    return json.dumps(_legacy_result(cleaned, raw_answers), ensure_ascii=False)


def check_ask_user_questions_requirements() -> bool:
    """ask_user_questions has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

ASK_USER_QUESTIONS_SCHEMA = {
    "name": "ask_user_questions",
    "description": (
        "Ask the user one or more structured questions with selectable "
        "options, modelled after Claude Code's AskUserQuestion tool. Use "
        "this in plan / UltraPlan / recon modes to gather requirements "
        "before producing a spec, design, or audit. Supports BATCHED "
        "questions in a single call (up to 4 per batch). Mark at most one "
        "option as `recommended: true`; leave every option unmarked when the "
        "trade-off is genuinely open. The UI adds the '(Recommended)' label "
        "automatically. Multi-select and the inline 'Other' free-text row use "
        "the same cross-platform interaction lane as `clarify`, so no second "
        "tool call is required. Results include an explicit status for each "
        "question plus top-level `timed_out` or `cancelled` when applicable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS_PER_CALL,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to present to the user.",
                        },
                        "header": {
                            "type": "string",
                            "maxLength": 12,
                            "description": (
                                "Short label for the question (max 12 chars). "
                                "Renders as a chip at the top of the question panel."
                            ),
                        },
                        "options": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": MAX_OPTIONS_PER_QUESTION,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "type": "string",
                                        "description": "The option text shown to the user.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Optional secondary line explaining the option.",
                                    },
                                    "recommended": {
                                        "type": "boolean",
                                        "default": False,
                                        "description": (
                                            "Mark this option as the recommendation. "
                                            "At most one option per question may be marked; "
                                            "omit it when no option is clearly preferable."
                                        ),
                                    },
                                },
                                "required": ["label"],
                            },
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Allow the user to select multiple options. "
                                "Supported by the shared CLI, TUI, desktop, "
                                "and messaging clarification lane."
                            ),
                        },
                    },
                    "required": ["question", "options"],
                },
                "description": (
                    "List of questions to ask in this batch (1-4). "
                    "Each question is rendered in its own boxed panel with "
                    "numbered, selectable options. Batching keeps the user "
                    "in one focused flow rather than re-prompting per question."
                ),
            },
        },
        "required": ["questions"],
    },
}


# --- Registry ---

registry.register(
    name="ask_user_questions",
    toolset="clarify",
    schema=ASK_USER_QUESTIONS_SCHEMA,
    handler=lambda args, **kw: ask_user_questions_tool(
        questions=args.get("questions", []),
        callback=kw.get("callback"),
        clarify_callback=kw.get("clarify_callback"),
    ),
    check_fn=check_ask_user_questions_requirements,
    emoji="❓",
)
