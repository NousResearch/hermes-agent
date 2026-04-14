#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import json
from typing import List, Optional, Callable


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4

MAX_QUESTIONS = 10


def _validate_questions(questions: list) -> Optional[str]:
    if not questions:
        return "questions array must not be empty."
    if len(questions) > MAX_QUESTIONS:
        return f"questions array exceeds maximum of {MAX_QUESTIONS}."
    headers_seen: set = set()
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            return f"questions[{i}] must be an object."
        header = q.get("header")
        if not header or not str(header).strip():
            return f"questions[{i}]: header is required."
        header = str(header).strip()
        if len(header) > 50:
            return f"questions[{i}]: header exceeds 50 characters."
        if header in headers_seen:
            return f"Duplicate header '{header}'."
        headers_seen.add(header)
        question_text = q.get("question")
        if not question_text or not str(question_text).strip():
            return f"questions[{i}]: question text is required."
    return None


def clarify_tool(
    question: str = "",
    choices: Optional[List[str]] = None,
    questions: Optional[List[dict]] = None,
    callback: Optional[Callable] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question: The question text to present.
        choices:  Up to 4 predefined answer choices. When omitted the
                  question is purely open-ended.
        questions: Advanced mode — structured multi-question form. When
                   provided, question and choices are ignored.
        callback: Platform-provided function that handles the actual UI
                  interaction. Signature: callback(question, choices) -> str.
                  Injected by the agent runner (cli.py / gateway).

    Returns:
        JSON string with the user's response.
    """
    # Advanced multi-question mode
    if questions is not None:
        err = _validate_questions(questions)
        if err:
            return tool_error(err)

        if callback is None:
            return json.dumps(
                {"error": "Clarify tool is not available in this execution context."},
                ensure_ascii=False,
            )

        try:
            raw = callback(None, choices=None, questions=questions)
        except Exception as exc:
            return json.dumps(
                {"error": f"Failed to get user input: {exc}"},
                ensure_ascii=False,
            )

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = raw

        return json.dumps({"responses": parsed}, ensure_ascii=False)

    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = [str(c).strip() for c in choices if str(c).strip()]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        user_response = callback(question, choices)
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "user_response": str(user_response).strip(),
        "_note": "The user has made their selection. Continue the conversation based on this choice. Do NOT repeat the question, options, or the user's selection.",
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
        "1. **Multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n"
        "3. **Advanced form** — provide a `questions` array for structured "
        "multi-question forms with options, multi-select, and freeform input.\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "**IMPORTANT — After receiving the result:**\n"
        "Treat the user's response as equivalent to a new user message. "
        "Continue the conversation naturally based on their selection or "
        "answer. Do NOT repeat the question, list of options, or the user's "
        "choice back to them. Instead, proceed with the task, provide "
        "relevant follow-up, or act on their decision directly.\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Prefer making a reasonable "
        "default choice yourself when the decision is low-stakes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to present to the user.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "Up to 4 answer choices. Omit this parameter entirely to "
                    "ask an open-ended question. When provided, the UI "
                    "automatically appends an 'Other (type your answer)' option."
                ),
            },
            "questions": {
                "type": "array",
                "description": (
                    "Advanced mode: array of structured questions. Each renders as "
                    "a form field. When provided, question and choices are ignored."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "string",
                            "maxLength": 50,
                            "description": "Short unique identifier for this question (used as form field key).",
                        },
                        "question": {
                            "type": "string",
                            "maxLength": 200,
                            "description": "The question text displayed to the user.",
                        },
                        "multiSelect": {
                            "type": "boolean",
                            "description": "Allow multiple selections (default false).",
                        },
                        "allowFreeformInput": {
                            "type": "boolean",
                            "description": "Show a text input field for custom answers.",
                        },
                        "options": {
                            "type": "array",
                            "description": "Predefined answer options.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "description": "Option display text."},
                                    "description": {"type": "string", "description": "Optional description."},
                                    "recommended": {"type": "boolean", "description": "Mark as recommended."},
                                },
                                "required": ["label"],
                            },
                        },
                    },
                    "required": ["header", "question"],
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
        questions=args.get("questions"),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
