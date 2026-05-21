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


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    hint_text: Optional[str] = None,
    other_label: Optional[str] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question:    The question text to present.
        choices:     Up to 4 predefined answer choices. When omitted the
                     question is purely open-ended.
        callback:    Platform-provided function that handles the actual UI
                     interaction. Signature: callback(question, choices,
                     hint_text, other_label) -> str.
                     Injected by the agent runner (cli.py / gateway).
        hint_text:   Optional footer instruction shown below the numbered
                     choices (e.g. "回复数字或自行输入" for Chinese). When
                     omitted, the platform uses its default English string.
                     The agent SHOULD populate this in the conversation
                     language so users see fully localised prompts.
        other_label: Optional label for the free-text fallback button/option
                     (e.g. "✏️ その他" for Japanese). When omitted, the
                     platform uses its default English string.

    Returns:
        JSON string with the user's response.
    """
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
        user_response = callback(question, choices, hint_text, other_label)
    except TypeError:
        # Older callback signatures (question, choices) — call without new args
        try:
            user_response = callback(question, choices)
        except Exception as exc:
            return json.dumps(
                {"error": f"Failed to get user input: {exc}"},
                ensure_ascii=False,
            )
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "user_response": str(user_response).strip(),
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
        "decision before proceeding. Supports two modes:\n\n"
        "1. **Multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Prefer making a reasonable "
        "default choice yourself when the decision is low-stakes.\n\n"
        "IMPORTANT — localisation: if the conversation is not in English, set "
        "hint_text and other_label in the conversation language so the entire "
        "prompt feels native to the user (e.g. for Chinese set hint_text to "
        "'请回复数字、选项文字，或直接输入您的答案。' and other_label to '✏️ 其他（自行输入）')."
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
            "hint_text": {
                "type": "string",
                "description": (
                    "Optional. Instruction shown below the numbered choices, "
                    "telling the user how to reply. Write this in the "
                    "conversation language. Example for Chinese: "
                    "'请回复数字、选项文字，或直接输入您的答案。' "
                    "Leave unset to use the platform default (English)."
                ),
            },
            "other_label": {
                "type": "string",
                "description": (
                    "Optional. Label for the free-text fallback button or "
                    "option (the escape hatch when none of the choices fit). "
                    "Write this in the conversation language. "
                    "Example for Japanese: '✏️ その他（自由入力）' "
                    "Leave unset to use the platform default (English)."
                ),
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
        callback=kw.get("callback"),
        hint_text=args.get("hint_text"),
        other_label=args.get("other_label"),
    ),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
