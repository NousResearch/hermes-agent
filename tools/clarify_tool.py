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


def _extract_recent_tool_context(
    messages: list,
    *,
    max_chars: int = 3000,
    max_tool_results: int = 5,
) -> str:
    """Extract recent tool-result context from agent messages for clarify prompts.

    Walks backward through ``messages``, skipping the current assistant turn
    (the one that triggered clarify), and collects tool results from the most
    recent preceding assistant turn that made tool calls.  This ensures the
    user sees the search results, page contents, or other tool output that led
    the agent to ask a clarification question.

    Returns a formatted context block, or empty string when no recent tool
    results exist.
    """
    context_parts: list[str] = []
    found_current_assistant = False

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            if not found_current_assistant:
                # Skip the first assistant-with-tool-calls — that's the turn
                # currently calling clarify, not the source of context.
                found_current_assistant = True
                continue
            else:
                # This is the second assistant-with-tool-calls (the previous
                # turn) — we've collected all its tool results, stop here.
                break

        if role == "tool":
            content = msg.get("content", "")
            if content and isinstance(content, str) and content.strip():
                tool_name = msg.get("name", "")
                label = f"[Tool: {tool_name}]" if tool_name else "[Tool result]"
                context_parts.append(f"{label}\n{content.strip()[:2000]}")

        if len(context_parts) >= max_tool_results:
            break

    if not context_parts:
        return ""

    # Reverse to restore chronological order (most recent work last).
    context_parts.reverse()
    result = "\n\n".join(context_parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n[...truncated...]"

    return result


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    context: Optional[str] = None,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question:   The question text to present.
        choices:    Up to 4 predefined answer choices. When omitted the
                    question is purely open-ended.
        callback:   Platform-provided function that handles the actual UI
                    interaction. Signature: callback(question, choices) -> str.
                    Injected by the agent runner (cli.py / gateway).
        context:    Optional context from recent autonomous work (search
                    results, page contents, tool outputs) to include in the
                    prompt shown to the user so they have the full picture.

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
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        # Prepend context to the question for the user-facing prompt so the
        # user sees the agent's research / tool output that led to the
        # clarification, not just the bare question. The result JSON stores
        # only the original question — context is display-only.
        display_question = question
        if context and context.strip():
            display_question = f"{context.strip()}\n\n{question}"
        user_response = callback(display_question, choices)
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
        "CRITICAL: when you are offering options, put each option ONLY in the "
        "`choices` array — NEVER enumerate the options inside the `question` "
        "text. The UI renders `choices` as selectable rows; options written "
        "into the question string render as dead prose the user can't pick. "
        "Right: question='Which deployment target?', choices=['staging', "
        "'prod']. Wrong: question='Which target? 1) staging 2) prod', choices=[].\n\n"
        "CONTEXT AUTO-INCLUSION: if you have just performed autonomous work "
        "(web searches, page extracts, tool calls) that produced results, "
        "the relevant context (search results, page content, tool output) "
        "is automatically prepended to your question when shown to the user. "
        "You do NOT need to repeat the full context in the question text — "
        "just write a clear, targeted question that makes sense with that "
        "context visible above it.\n\n"
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
                    "— pass them as separate elements in `choices`. Context "
                    "from any preceding autonomous work is automatically "
                    "prepended by the system."
                ),
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each distinct option is its own array element (up to 4). "
                    "The UI renders these as pickable rows and auto-appends an "
                    "'Other (type your answer)' option. Omit this parameter "
                    "entirely ONLY for a genuinely open-ended free-text question."
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
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
