"""Clarify tool: structured multiple-choice / open-ended questions to the user.
Schema, validation and a thin dispatcher; the UI lives in a platform-provided
callback (cli.py, gateway/run.py, tui_gateway)."""

import inspect
import json
from typing import Any, Dict, List, Optional, Callable, Union

# A choice is a plain label string, or a structured {label, description} object
# (Claude Code style). Descriptions ride the wire for future UI; all matching
# and answers resolve to the label so old surfaces keep working.
Choice = Union[str, Dict[str, str]]

MAX_CHOICES = 4  # the UI always appends an "Other (type your answer)" row
MAX_QUESTIONS = 5  # independent questions per batch call
# Canonical timeout sentinel. The CLI returns this exact text; the batch loop
# treats it (like ``None``) as "the user walked away" and aborts remaining questions.
TIMEOUT_RESPONSE = ("The user did not provide a response within the time limit. "
                    "Use your best judgement to make the choice and proceed.")
# Applied to the first choice here (not per-surface) so every adapter renders it identically.
RECOMMENDED_LABEL = "(Recommended)"
_UNAVAILABLE = "Clarify tool is not available in this execution context."


# Keys LLMs actually emit for a choice label / description. ``name``/``value``
# stay excluded: raw component enums, not labels (a {name, value} dict with no
# label-ish key is dropped, not leaked).
_LABEL_KEYS = ("label", "text", "title", "choice")
_DESC_KEYS = ("description", "detail", "hint")

# Description length cap: labels stay short (desktop drops >200 / newlines);
# descriptions are one-liners, truncated rather than dropped.
MAX_DESCRIPTION_LEN = 500


def choice_label(choice: Choice) -> str:
    """Display/match text of one choice: the label, never the repr."""
    if isinstance(choice, dict):
        return str(choice.get("label") or "").strip()
    return str(choice).strip() if choice is not None else ""


def choice_description(choice: Choice) -> Optional[str]:
    """Description of a structured choice; None for plain strings."""
    if isinstance(choice, dict):
        desc = choice.get("description")
        return str(desc).strip() or None if isinstance(desc, str) else None
    return None


def _normalize_choice(c: Any) -> Optional[Choice]:
    """Coerce one raw choice to a label string or a {label, description} dict.

    Both fields present -> structured (descriptions ride the wire; Phase 2
    renders them). Exactly one present -> that bare string (legacy unwrap).
    No match -> None and dropped: no choice beats a garbage label.
    """
    if isinstance(c, str):
        return c.strip() or None
    if isinstance(c, dict):
        label = next((v.strip() for k in _LABEL_KEYS
                      if isinstance(v := c.get(k), str) and v.strip()), "")
        desc = next((v.strip() for k in _DESC_KEYS
                     if isinstance(v := c.get(k), str) and v.strip()), "")
        if label and desc:
            if len(desc) > MAX_DESCRIPTION_LEN:
                desc = desc[:MAX_DESCRIPTION_LEN].rstrip()
            return {"label": label, "description": desc}
        return label or desc or None
    if isinstance(c, (list, tuple)):
        labels = [choice_label(p) for p in (_normalize_choice(x) for x in c)]
        return " ".join(l for l in labels if l).strip() or None
    return "" if c is None else str(c).strip() or None


def _flatten_choice(c) -> str:
    """Display text of one choice (legacy entry point, now label-only).

    Kept by name: tests and sibling modules import it. Dicts resolve to their
    label so no repr ever reaches a surface or comes back as the answer.
    """
    return choice_label(_normalize_choice(c) or "")


def _with_recommended(choice: Choice) -> Choice:
    """Return ``choice`` with the recommendation suffix on its label (copied)."""
    if isinstance(choice, dict):
        return {"label": f"{choice.get('label', '')} {RECOMMENDED_LABEL}",
                "description": choice.get("description", "")}
    return f"{choice} {RECOMMENDED_LABEL}"


def mark_recommended(choices: List[Choice]) -> List[Choice]:
    """Suffix the first choice (schema says best-first) with RECOMMENDED_LABEL; idempotent,
    and a lone choice is left untouched (nothing to prefer it over)."""
    if len(choices) < 2:
        return choices
    first_label = choice_label(choices[0])
    if first_label == strip_recommended(first_label):
        return [_with_recommended(choices[0])] + list(choices[1:])
    return choices


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


def _answer_text(raw) -> str:
    """One answer value as text: a callback echoing the choice object itself
    (a structured dict) resolves to its label, never the repr."""
    if isinstance(raw, dict):
        return choice_label(raw)
    return str(raw)


def _parse_multi_select_response(raw_response) -> List[str]:
    """Parse a list / JSON array / comma-separated reply into stripped non-empty strings."""
    items = raw_response
    if not isinstance(items, list):
        if isinstance(items, dict):
            return [choice_label(items)] if choice_label(items) else []
        raw = str(items).strip()
        items = _json_as(raw, list) if raw.startswith("[") else None
        if items is None:
            items = raw.split(",")
    return [_answer_text(r).strip() for r in items if _answer_text(r).strip()]


def _bare_choice(choice: Choice) -> Choice:
    """Copy of ``choice`` without the recommendation suffix on its label."""
    if isinstance(choice, dict):
        return {"label": strip_recommended(choice.get("label") or ""),
                "description": choice.get("description", "")}
    return strip_recommended(choice)


def _bare_choices(choices: list) -> List[Choice]:
    """Choices as the agent sees them back (presentation never leaks)."""
    return [_bare_choice(c) for c in choices]


def _clean_answer(raw, multi: bool):
    """Strip presentation (the label, multi-select JSON) from a locked answer."""
    if isinstance(raw, dict):
        raw = choice_label(raw)
    return [strip_recommended(r) for r in _parse_multi_select_response(raw)] if multi else strip_recommended(raw)


def _clean_choices(choices: list) -> Optional[List[Choice]]:
    """Normalize, drop empties, cap at MAX_CHOICES; None when nothing survives (open-ended).

    Structured {label, description} choices survive as dicts so descriptions
    ride the wire; plain labels stay bare strings (legacy payloads unchanged).
    """
    cleaned = [c for c in (_normalize_choice(x) for x in choices) if c and choice_label(c)]
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


def _batch_result(normalized: List[dict], answers: dict, timed_out: bool) -> str:
    """Batch result JSON; unanswered -> "". The top-level ``timed_out`` flag (present only when
    true) tells the agent whether blanks are deliberate skips or the user walking away."""
    responses = []
    for entry in normalized:
        raw = answers.get(entry["qid"])
        responses.append({
            **({"id": entry["id"]} if entry["id"] else {}),
            "question": entry["question"], "choices_offered": entry["choices_offered"],
            "user_response": _clean_answer(raw, entry["multi_select"]) if raw else ""})
    result: Dict[str, object] = {"responses": responses}
    if timed_out:
        result["timed_out"] = True
    return json.dumps(result, ensure_ascii=False)


def _run_batch(normalized: List[dict], callback, question: str) -> str:
    """Dispatch a validated batch. Batch-capable callbacks (``questions`` kwarg) get the
    whole list once and reply ``{"answers": {qid: raw}, "timed_out"?}`` as a dict or JSON
    string (the tui_gateway bridge only carries strings); any other falsy/unparseable reply
    is a cancel-all (mirrors the single-question skip). Legacy callbacks are looped per
    question: an empty answer is a skip, a timeout (``None`` or the sentinel) means the user
    walked away so the loop aborts instead of pestering them; earlier answers are kept."""
    answers: dict = {}
    timed_out = False
    if _accepts_kwarg(callback, "questions"):
        raw = callback(question, None, questions=normalized)
        timed_out = _is_timeout(raw)
        if isinstance(raw, str):
            raw = _json_as(raw, dict)  # the sentinel is not JSON -> None, timed_out stays True
        if isinstance(raw, dict):
            answers = dict(raw.get("answers") or {})
            timed_out = bool(raw.get("timed_out"))
        return _batch_result(normalized, answers, timed_out)
    for entry in normalized:
        raw = _invoke_callback(callback, entry["question"], entry["choices"], entry["multi_select"])
        if _is_timeout(raw):
            timed_out = True
            break
        answers[entry["qid"]] = raw
    return _batch_result(normalized, answers, timed_out)


def clarify_tool(question: str, choices: Optional[List[Any]] = None, multi_select: bool = False,
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
                return _run_batch(normalized, callback, str(question or "").strip())
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
            return tool_error("choices must be a list of strings or {label, description} objects.")
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
        "open-ended (omit choices). A choice is a label string or a "
        "{label, description} object — use the object when the label alone "
        "is ambiguous (the description renders as a subtitle). Options go ONLY in `choices`, never "
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
                    "optional multi_select. A choice is a label string or a "
                    "{label, description} object (description = one-line "
                    "subtitle). Responses come back in question "
                    "order with the question text echoed."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "description": {"type": "string"},
                                        },
                                        "required": ["label"],
                                    },
                                ]
                            },
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
