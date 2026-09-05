#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

Supports both single-select (radio) and multi-select (checkbox) modes via the
``multi_select`` parameter.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import Dict, List, Optional, Callable


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4

# Maximum number of independent questions in one batch clarify call.
MAX_QUESTIONS = 5

# Canonical timeout sentinel returned to the agent when the user never
# answers. The CLI has always returned this exact text; the batch fallback
# loop also recognises it (alongside ``None``) as "the user walked away",
# which aborts the remaining questions instead of pestering one by one.
TIMEOUT_RESPONSE = (
    "The user did not provide a response within the time limit. "
    "Use your best judgement to make the choice and proceed."
)

# Suffix appended to the first choice so the user can see, at a glance, which
# option the agent actually recommends. Applied here rather than per-surface so
# CLI, TUI, desktop, and messaging adapters all render the same label.
RECOMMENDED_LABEL = "(Recommended)"


def _flatten_choice(c) -> str:
    """Coerce a single choice into its user-facing display string.

    The schema declares choices as bare strings, but LLMs sometimes emit
    dict-shaped choices like ``[{"description": "..."}]``. A naive ``str(c)``
    turns the whole dict into its Python repr — ``{'description': '...'}`` —
    which then leaks onto every surface that renders the choice (CLI panel,
    Discord buttons, Telegram numbered list) AND is returned verbatim as the
    user's answer. Normalising here, at the one platform-agnostic entry point,
    fixes the whole class in one place instead of per-adapter.

    Dict unwrap order is the canonical LLM tool-call user-facing keys:
    ``label`` → ``description`` → ``text`` → ``title``. ``name`` and ``value``
    are deliberately excluded — they're component-shaped fields that could
    carry raw enum values or short identifiers, not human-readable labels. A
    dict with none of the canonical keys is dropped (returns ""), since a
    garbage label is worse than no choice at all.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        for key in ("label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return str(c).strip()


def mark_recommended(choices: List[str]) -> List[str]:
    """Label the first choice as the agent's recommendation.

    The schema tells the model to order ``choices`` best-first, so element 0 is
    always the option it would pick itself. Tagging it here — the one
    platform-agnostic entry point — means every surface (CLI panel, TUI,
    desktop card, Telegram buttons) reads the same way without four copies of
    the same string concatenation, and the label can never drift between them.

    Idempotent: a model that writes its own "(recommended)" into the choice is
    left alone rather than getting the suffix twice. A lone choice isn't a
    recommendation — there's nothing to prefer it over — so single-choice lists
    pass through untouched.
    """
    if len(choices) < 2:
        return choices
    first = str(choices[0]).strip()
    if first != strip_recommended(first):
        return choices
    return [f"{first} {RECOMMENDED_LABEL}"] + list(choices[1:])


def strip_recommended(text: str) -> str:
    """Remove the recommendation label from a resolved answer.

    The user picks the decorated string, but the agent asked about the bare
    option — returning "Rebase onto main (Recommended)" as ``user_response``
    would leak presentation into the answer the model reasons about and into
    anything it echoes back.
    """
    stripped = str(text).strip()
    if stripped.casefold().endswith(RECOMMENDED_LABEL.casefold()):
        return stripped[: -len(RECOMMENDED_LABEL)].strip()
    return stripped


def _invoke_callback(callback, question, choices, multi_select):
    """Invoke the platform callback, passing multi_select if supported.

    Uses signature inspection (not a ``TypeError`` retry) to decide whether
    the callback accepts the ``multi_select`` keyword — a retry-on-TypeError
    approach would re-invoke a *compatible* callback that raised TypeError
    internally, potentially prompting the user twice.
    """
    import inspect

    accepts_multi = False
    try:
        sig = inspect.signature(callback)
        params = sig.parameters
        accepts_multi = "multi_select" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        # Builtins / C callables without introspectable signatures:
        # be conservative and use the legacy 2-arg form.
        accepts_multi = False

    if accepts_multi:
        return callback(question, choices, multi_select=multi_select)
    return callback(question, choices)


def _parse_multi_select_response(raw_response) -> List[str]:
    """Parse a multi-select response into a list of cleaned choice strings.

    Handles three forms:
      - Already a list  →  stringify + strip each element
      - JSON array      →  parse and strip
      - Comma-separated →  split, strip, drop empties
    """
    if isinstance(raw_response, list):
        return [str(r).strip() for r in raw_response if str(r).strip()]

    raw = str(raw_response).strip()

    # Try JSON array
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except json.JSONDecodeError:
            pass

    # Fall back to comma-separated
    return [s.strip() for s in raw.split(",") if s.strip()]


# =============================================================================
# Batch (multi-question) support — issue #18450
# =============================================================================

def _normalize_questions(questions) -> tuple:
    """Validate and normalize the ``questions`` batch parameter.

    Returns ``(normalized, error)`` where exactly one is non-None, except the
    empty-list case which returns ``(None, None)`` — an empty array is not an
    error, it just means "no batch here" and the caller falls back to the
    single-question path.

    Each normalized entry carries:
      - ``qid``: stable wire id (``q0``..``qN``, index order). Surfaces key
        their per-question answers by this; a model-supplied ``id`` is NOT
        used on the wire (it's unvalidated text) and only echoed in results.
      - ``id``: the model's optional identifier, or None.
      - ``question``: stripped question text.
      - ``choices``: decorated choice list (recommended label applied), or
        None for open-ended.
      - ``choices_offered``: the bare list as offered, for the result JSON.
      - ``multi_select``: honored only when choices exist.
    """
    if not isinstance(questions, list):
        return None, "questions must be an array of question objects."
    if not questions:
        return None, None
    if len(questions) > MAX_QUESTIONS:
        return None, f"questions supports at most {MAX_QUESTIONS} items."

    normalized = []
    for index, item in enumerate(questions):
        if isinstance(item, str):
            # Tolerate bare-string items: LLMs sometimes send ["Q1?", "Q2?"].
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
            choices = [s for s in (_flatten_choice(c) for c in choices) if s]
            if len(choices) > MAX_CHOICES:
                choices = choices[:MAX_CHOICES]
            if not choices:
                choices = None

        model_id = str(item.get("id") or "").strip() or None

        normalized.append({
            "qid": f"q{index}",
            "id": model_id,
            "question": text,
            "choices": mark_recommended(list(choices)) if choices else None,
            "choices_offered": list(choices) if choices else None,
            "multi_select": bool(item.get("multi_select")) and bool(choices),
        })

    return normalized, None


def _callback_accepts_questions(callback) -> bool:
    """True when the platform callback understands the ``questions`` kwarg.

    Same signature-inspection approach as ``_invoke_callback`` (never a
    TypeError retry — that would re-prompt the user on an internal bug).
    """
    import inspect

    try:
        params = inspect.signature(callback).parameters
        return "questions" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        return False


def _clean_batch_answer(entry: dict, raw) -> object:
    """Strip presentation from one locked answer (label, multi-select JSON)."""
    if entry["multi_select"]:
        return [strip_recommended(r) for r in _parse_multi_select_response(raw)]
    return strip_recommended(raw)


def _batch_result(normalized: List[dict], answers: dict, timed_out: bool) -> str:
    """Assemble the batch result JSON from per-qid answers.

    Unanswered questions surface as empty ``user_response`` — with the
    top-level ``timed_out`` flag (present only when true) telling the agent
    whether those blanks are deliberate skips or the user walking away.
    """
    responses = []
    for entry in normalized:
        row = {}
        if entry["id"]:
            row["id"] = entry["id"]
        row["question"] = entry["question"]
        row["choices_offered"] = entry["choices_offered"]
        raw = answers.get(entry["qid"])
        row["user_response"] = _clean_batch_answer(entry, raw) if raw else ""
        responses.append(row)

    result: Dict[str, object] = {"responses": responses}
    if timed_out:
        result["timed_out"] = True
    return json.dumps(result, ensure_ascii=False)


def _run_batch(normalized: List[dict], callback, question: str) -> str:
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
    if _callback_accepts_questions(callback):
        raw = callback(question, None, questions=normalized)

        answers: dict = {}
        timed_out = False
        if raw is None or (isinstance(raw, str) and raw.strip() == TIMEOUT_RESPONSE):
            timed_out = True
        elif isinstance(raw, dict):
            answers = dict(raw.get("answers") or {})
            timed_out = bool(raw.get("timed_out"))
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                answers = dict(parsed.get("answers") or {})
                timed_out = bool(parsed.get("timed_out"))
        # Any other falsy/unparseable reply is a cancel-all: every answer
        # empty, no timeout flag (mirrors the single-question skip).
        return _batch_result(normalized, answers, timed_out)

    answers = {}
    timed_out = False
    for entry in normalized:
        raw = _invoke_callback(
            callback, entry["question"], entry["choices"], entry["multi_select"],
        )
        if raw is None or (isinstance(raw, str) and raw.strip() == TIMEOUT_RESPONSE):
            timed_out = True
            break
        answers[entry["qid"]] = raw
    return _batch_result(normalized, answers, timed_out)


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    multi_select: bool = False,
    questions: Optional[List[dict]] = None,
    callback: Optional[Callable] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question:     The question text to present.
        choices:      Up to 4 predefined answer choices. When omitted the
                      question is purely open-ended.
        multi_select: When True, the user can select multiple choices
                      (checkboxes).  The ``user_response`` in the output JSON
                      will be a list of strings instead of a single string.
                      Has no effect when ``choices`` is omitted.
        questions:    Up to 5 independent questions asked as one batch
                      (issue #18450). Each item: ``{id?, question, choices?,
                      multi_select?}``. When present (non-empty), the single
                      ``question``/``choices``/``multi_select`` parameters
                      are ignored and the result JSON is ``{"responses":
                      [...]}`` (plus ``"timed_out": true`` when the user
                      stopped answering partway).
        callback:     Platform-provided function that handles the actual UI
                      interaction.  Signature:
                      ``callback(question, choices, multi_select=False) -> str``.
                      Batch-capable platforms additionally accept a
                      ``questions`` keyword and receive the normalized list
                      in one call; platforms without it are looped one
                      question at a time.
                      Injected by the agent runner (cli.py / gateway).

    Returns:
        JSON string with the user's response(s).
    """
    if questions is not None:
        normalized, error = _normalize_questions(questions)
        if error:
            return tool_error(error)
        if normalized:
            if callback is None:
                return tool_error(
                    "Clarify tool is not available in this execution context."
                )
            try:
                return _run_batch(normalized, callback, str(question or "").strip())
            except Exception as exc:
                return tool_error(f"Failed to get user input: {exc}")
        # Empty questions array → fall through to the single-question path.

    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        # LLMs sometimes emit dict-shaped choices (e.g. [{"description": "..."}])
        # instead of bare strings. _flatten_choice unwraps them to their
        # user-facing text here — the single platform-agnostic entry point —
        # so the CLI panel, Discord buttons, and Telegram list all render clean
        # text and the resolved answer is never a raw Python dict repr.
        choices = [s for s in (_flatten_choice(c) for c in choices) if s]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended

    if callback is None:
        return tool_error("Clarify tool is not available in this execution context.")

    # The first choice is the agent's pick (the schema says order best-first),
    # so it reaches every surface carrying the "(Recommended)" label. The bare
    # list is what goes back to the agent — the label is presentation only.
    offered = choices
    if choices is not None:
        choices = mark_recommended(choices)

    try:
        raw_response = _invoke_callback(callback, question, choices, multi_select)
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")

    if multi_select and choices is not None:
        user_response = [strip_recommended(r) for r in _parse_multi_select_response(raw_response)]
    else:
        user_response = strip_recommended(raw_response)

    return json.dumps({
        "question": question,
        "choices_offered": offered,
        "user_response": user_response,
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification, feedback, or a "
        "decision before proceeding. Supports three modes:\n\n"
        "1. **Single-select multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option. List the choice you recommend "
        "FIRST: the UI labels it '(Recommended)' and highlights it by default.\n"
        "2. **Multi-select multiple choice** — set multi_select=true. The user can select "
        "multiple options via checkboxes. user_response will be a list of selected choices.\n"
        "3. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
        "You can also ask SEVERAL questions in ONE call: pass "
        "questions in the `questions` array (each with its own choices/"
        "multi_select, any mix of the three modes). The user answers them all "
        "on a single form, in any order. STRONGLY preferred over a chain of "
        "single-question clarify calls when you need several independent answers.\n"
        "CRITICAL: when you are offering options, put each option ONLY in the "
        "`choices` array — NEVER enumerate the options inside the `question` "
        "text. The UI renders `choices` as selectable rows; options written "
        "into the question string render as dead prose the user can't pick. "
        "Right: question='Which deployment target?', choices=['staging', "
        "'prod']. Wrong: question='Which target? 1) staging 2) prod', choices=[].\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Prefer making a reasonable "
        "default choice yourself when the decision is low-stakes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question itself, and ONLY the question (e.g. 'Which "
                    "deployment target?'). Do NOT embed the answer options here "
                    "— pass them as separate elements in `choices`."
                ),
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each distinct option is its own array element (up to 4). "
                    "ORDER MATTERS: put the option you actually recommend "
                    "FIRST — the UI labels it '(Recommended)' and pre-selects "
                    "it, so a list ordered arbitrarily recommends the wrong "
                    "thing to the user. Do not write '(Recommended)' yourself. "
                    "The UI renders these as pickable rows and auto-appends an "
                    "'Other (type your answer)' option. Omit this parameter "
                    "entirely ONLY for a genuinely open-ended free-text question."
                ),
            },
            "multi_select": {
                "type": "boolean",
                "description": (
                    "When true, the user can select MULTIPLE options (like checkboxes). "
                    "The user_response will be a list of selected choices. "
                    "When false (default), single selection (radio). "
                    "Has no effect when choices is omitted (open-ended question)."
                ),
            },
            "questions": {
                "type": "array",
                "maxItems": MAX_QUESTIONS,
                "description": (
                    "Ask 2-5 INDEPENDENT questions in one call instead of "
                    "several sequential clarify calls — the user answers them "
                    "on one form, in any order. Each item has its own "
                    "question/choices/multi_select (same rules as the "
                    "top-level parameters); optional `id` is echoed back in "
                    "the matching response. When set, the top-level question/"
                    "choices are ignored. put a short batch title in the "
                    "top-level `question` The result is {responses: [...]}, "
                    "with `timed_out: true` added if the user stopped part-way "
                    "(unanswered entries have an empty user_response). Only "
                    "batch questions that are truly independent — if one "
                    "answer would change another question, ask separately."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": (
                                "Optional short identifier echoed in the "
                                "matching response (e.g. 'approach')."
                            ),
                        },
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
        },
        "required": ["question"],
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
