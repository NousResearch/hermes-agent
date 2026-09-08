"""Clarify tool: structured multiple-choice / open-ended questions to the user.
Schema, validation and a thin dispatcher; the UI lives in a platform-provided
callback (cli.py, gateway/run.py, tui_gateway)."""

import inspect
import json
from typing import Dict, List, Optional, Callable

MAX_CHOICES = 4  # the UI always appends an "Other (type your answer)" row
MAX_QUESTIONS = 5  # independent questions per batch call
# Canonical timeout sentinel. The CLI returns this exact text; the batch loop
# treats it (like ``None``) as "the user walked away" and aborts remaining questions.
TIMEOUT_RESPONSE = ("The user did not provide a response within the time limit. "
                    "Use your best judgement to make the choice and proceed.")
# Applied to the first choice here (not per-surface) so every adapter renders it identically.
RECOMMENDED_LABEL = "(Recommended)"
_UNAVAILABLE = "Clarify tool is not available in this execution context."


def _flatten_choice(c) -> str:
    """Coerce one choice to display text. LLMs sometimes emit dict-shaped choices and ``str(c)``
    would leak the repr onto every surface and back as the answer; unwrap order ``label`` >
    ``description`` > ``text`` > ``title`` (``name``/``value`` excluded: raw component enums,
    not labels). No match -> "" and dropped: no choice beats a garbage label."""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        return next((v.strip() for k in ("label", "description", "text", "title")
                     if isinstance(v := c.get(k), str) and v.strip()), "")
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return "" if c is None else str(c).strip()


def mark_recommended(choices: List[str]) -> List[str]:
    """Suffix the first choice (schema says best-first) with RECOMMENDED_LABEL; idempotent,
    and a lone choice is left untouched (nothing to prefer it over)."""
    first = str(choices[0]).strip() if choices else ""
    if len(choices) < 2 or first != strip_recommended(first):
        return choices
    return [f"{first} {RECOMMENDED_LABEL}"] + list(choices[1:])


def strip_recommended(text: str) -> str:
    """Remove the recommendation label so presentation never leaks into ``user_response``."""
    stripped = str(text).strip()
    if stripped.casefold().endswith(RECOMMENDED_LABEL.casefold()):
        return stripped[: -len(RECOMMENDED_LABEL)].strip()
    return stripped


def _accepts_kwarg(callback, name: str) -> bool:
    """Signature-inspect (never a TypeError retry, which could re-prompt the user) whether
    ``callback`` takes ``name`` or ``**kwargs``; non-introspectable callables are legacy."""
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _invoke_callback(callback, question, choices, multi_select):
    """Invoke the platform callback, passing multi_select if supported."""
    if _accepts_kwarg(callback, "multi_select"):
        return callback(question, choices, multi_select=multi_select)
    return callback(question, choices)


def _json_as(raw: str, kind):
    """``json.loads(raw)`` when it decodes to an instance of ``kind``, else None."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, kind) else None


def _parse_multi_select_response(raw_response) -> List[str]:
    """Parse a list / JSON array / comma-separated reply into stripped non-empty strings."""
    items = raw_response
    if not isinstance(items, list):
        raw = str(items).strip()
        items = _json_as(raw, list) if raw.startswith("[") else None
        if items is None:
            items = raw.split(",")
    return [str(r).strip() for r in items if str(r).strip()]


def _clean_answer(raw, multi: bool):
    """Strip presentation (the label, multi-select JSON) from a locked answer."""
    return [strip_recommended(r) for r in _parse_multi_select_response(raw)] if multi else strip_recommended(raw)


def _clean_choices(choices: list) -> Optional[List[str]]:
    """Flatten, drop empties, cap at MAX_CHOICES; None when nothing survives (open-ended)."""
    cleaned = [s for s in (_flatten_choice(c) for c in choices) if s]
    return cleaned[:MAX_CHOICES] or None


def _is_timeout(raw) -> bool:
    return raw is None or (isinstance(raw, str) and raw.strip() == TIMEOUT_RESPONSE)


# ============================================================================= Batch (multi-question)
# support — issue #18450 =============================================================================
def _normalize_questions(questions) -> tuple:
    """Validate the ``questions`` batch param -> ``(normalized, error)``; an empty list gives
    ``(None, None)`` (fall back to the single-question path). Entries carry ``qid`` (stable
    wire id ``q<index>`` surfaces key answers by; the model's ``id`` is unvalidated text, only
    echoed), ``question``, decorated ``choices``, bare ``choices_offered``, ``multi_select``."""
    if not isinstance(questions, list):
        return None, "questions must be an array of question objects."
    if not questions:
        return None, None
    if len(questions) > MAX_QUESTIONS:
        return None, f"questions supports at most {MAX_QUESTIONS} items."
    normalized = []
    for index, item in enumerate(questions):
        if isinstance(item, str):  # tolerate bare-string items: LLMs sometimes send ["Q1?", "Q2?"]
            item = {"question": item}
        if not isinstance(item, dict):
            return None, f"questions[{index}] must be an object with a 'question'."
        text = str(item.get("question") or "").strip()
        if not text:
            return None, f"questions[{index}].question must be non-empty text."
        choices = item.get("choices")
        if choices is not None:
            if not isinstance(choices, list):
                return None, f"questions[{index}].choices must be a list."
            choices = _clean_choices(choices)
        normalized.append({
            "qid": f"q{index}", "id": str(item.get("id") or "").strip() or None, "question": text,
            "choices": mark_recommended(list(choices)) if choices else None,
            "choices_offered": list(choices) if choices else None,
            "multi_select": bool(item.get("multi_select")) and bool(choices)})
    return normalized, None


def _batch_result(normalized: List[dict], answers: dict, timed_out: bool, cancelled: bool = False) -> str:
    """Assemble batch result JSON with an explicit status per question.

    Answers locked before a timeout or cancellation are preserved. Missing
    answers are labelled ``timed_out`` or ``cancelled`` instead of being
    conflated with a deliberate empty-answer skip. The top-level
    ``timed_out``/``cancelled`` flags (present only when true) tell the agent
    whether blanks are deliberate skips, the user walking away, or a
    cancel-all from a batch-capable surface.
    """
    responses = []
    for entry in normalized:
        present = entry["qid"] in answers
        raw = answers.get(entry["qid"])
        row = {
            **({"id": entry["id"]} if entry["id"] else {}),
            "question": entry["question"], "choices_offered": entry["choices_offered"],
            "user_response": _clean_answer(raw, entry["multi_select"]) if raw else ""}
        if present:
            row["status"] = "answered" if raw else "skipped"
        elif timed_out:
            row["status"] = "timed_out"
        elif cancelled:
            row["status"] = "cancelled"
        else:
            row["status"] = "skipped"
        responses.append(row)
    result: Dict[str, object] = {"responses": responses}
    if timed_out:
        result["timed_out"] = True
    if cancelled:
        result["cancelled"] = True
    return json.dumps(result, ensure_ascii=False)


def run_question_batch(normalized: List[dict], callback, question: str = "") -> str:
    """Dispatch a validated batch to the platform callback.

    Batch-capable callbacks (a ``questions`` kwarg, detected by signature)
    get the whole list once and reply with ``{"answers": {qid: raw}}`` plus
    an optional ``timed_out`` flag — as a dict or a JSON string (the
    tui_gateway ``_block`` bridge can only carry strings).

    Legacy callbacks are looped one question at a time (messaging adapters,
    older plugins). An explicit empty answer is a skip and the loop
    continues; a timeout (``None`` or the ``TIMEOUT_RESPONSE`` sentinel)
    means the user walked away, so the loop aborts instead of pestering
    them with the remaining questions. Answers collected before the abort
    are kept either way.
    """
    if _accepts_kwarg(callback, "questions"):
        raw = callback(question, None, questions=normalized)

        answers: dict = {}
        timed_out = False
        cancelled = False
        if _is_timeout(raw):
            timed_out = True
        elif isinstance(raw, dict):
            answers = dict(raw.get("answers") or {})
            timed_out = bool(raw.get("timed_out"))
            cancelled = bool(raw.get("cancelled"))
        elif isinstance(raw, str) and raw.strip():
            parsed = _json_as(raw, dict)
            if isinstance(parsed, dict):
                answers = dict(parsed.get("answers") or {})
                timed_out = bool(parsed.get("timed_out"))
                cancelled = bool(parsed.get("cancelled"))
            else:
                cancelled = True
        else:
            # A batch-capable surface uses an empty response for cancel-all.
            cancelled = True
        return _batch_result(normalized, answers, timed_out, cancelled)

    answers = {}
    timed_out = False
    for entry in normalized:
        raw = _invoke_callback(callback, entry["question"], entry["choices"], entry["multi_select"])
        if _is_timeout(raw):
            timed_out = True
            break
        answers[entry["qid"]] = raw
    return _batch_result(normalized, answers, timed_out)


def clarify_tool(question: str, choices: Optional[List[str]] = None, multi_select: bool = False,
                 questions: Optional[List[dict]] = None, callback: Optional[Callable] = None) -> str:
    """Ask one question (``question``/``choices``/``multi_select``) or a batch (``questions``
    wins when non-empty). ``callback(question, choices, multi_select=False) -> str`` is
    platform injected (batch-capable ones also take ``questions=``). Returns result JSON.

    Args: question:     The question text to present. choices:      Up to 4 predefined answer choices. When
    omitted the question is purely open-ended. multi_select: When True, the user can select multiple choices
    (checkboxes). The ``user_response`` in the output JSON will be a list of strings instead of a single
    string. Has no effect when ``choices`` is omitted. questions:    Up to 5 independent questions asked as
    one batch (issue #18450). When present (non-empty), the single ``question``/``choices``/``multi_select``
    parameters are ignored and the result JSON is ``{"responses": [...]}`` (plus ``"timed_out": true`` when
    the user stopped answering partway). callback:     Platform-provided function that handles the actual UI
    interaction. Batch-capable platforms additionally accept a ``questions`` keyword and receive the
    normalized list in one call; platforms without it are looped one question at a time. Injected by the
    agent runner (cli.py / gateway).
    """
    if questions is not None:
        normalized, error = _normalize_questions(questions)
        if error:
            return tool_error(error)
        if normalized:
            if callback is None:
                return tool_error(_UNAVAILABLE)
            try:
                return run_question_batch(normalized, callback, str(question or "").strip())
            except Exception as exc:
                return tool_error(f"Failed to get user input: {exc}")
        # Empty questions array → fall through to the single-question path.
    if not question or not question.strip():
        return tool_error("No question provided. Pass questions=[{question: '...', "
                          "choices?: [...], multi_select?: bool}, ...] — a single question "
                          "is a one-entry array.")
    question = question.strip()
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = _clean_choices(choices)
    if callback is None:
        return tool_error(_UNAVAILABLE)
    # The bare list goes back to the agent; the "(Recommended)" label is presentation only.
    shown = mark_recommended(choices) if choices is not None else None
    try:
        raw_response = _invoke_callback(callback, question, shown, multi_select)
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")
    return json.dumps({"question": question, "choices_offered": choices,
                       "user_response": _clean_answer(raw_response, multi_select and choices is not None)},
                      ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user one or more questions when you need a decision, "
        "clarification, or feedback before proceeding. Pass every question "
        f"in `questions` (1-{MAX_QUESTIONS} entries) — a single question is a "
        "one-entry array, and several INDEPENDENT questions belong in ONE "
        "call (one form beats a chain of clarify calls; if one answer would "
        "change another question, ask separately). Per question: "
        f"single-select (up to {MAX_CHOICES} choices — put your recommended "
        "option FIRST, the UI marks it '(Recommended)' and auto-appends an "
        "'Other' free-text row), multi-select (multi_select=true), or "
        "open-ended (omit choices). Options go ONLY in `choices`, never "
        "enumerated inside the question text (choices render as pickable "
        "rows; options written into the question are dead prose the user "
        "can't click). Result: {responses: [...]} in question order (plus "
        "timed_out=true if the user stopped part-way). Prefer deciding "
        "low-stakes questions yourself; don't use this for dangerous-command "
        "confirmation (the terminal tool handles that)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS,
                "description": (
                    "The question(s). Each: question text (options excluded), "
                    "optional choices (recommended first; omit for free-text), "
                    "optional multi_select. Responses come back in question "
                    "order with the question text echoed."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_CHOICES,
                        },
                        "multi_select": {"type": "boolean"},
                    },
                    "required": ["question"],
                },
            },
            # NOTE: the handler also accepts (unadvertised): a per-question
            # `id` (echoed in the matching response — redundant since rows
            # carry the question text and preserve order), and the legacy
            # single-question shape (`question` + `choices` + `multi_select`
            # at top level; a top-level `question` beside `questions` is the
            # batch form's title). One documented way to call.
        },
        "required": ["questions"],
    },
}

# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        multi_select=args.get("multi_select", False),
        questions=args.get("questions"),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
