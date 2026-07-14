"""Explicit multi-select user interaction tool.

Unlike ``clarify``, this primitive always requires a finite option list and
returns zero or more selected options as a JSON array. It is exposed only by
platform bundles that provide a suitable interaction path.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from tools.clarify_tool import _flatten_choice
from tools.registry import registry, tool_error


MAX_SELECT_MANY_CHOICES = 20


def select_many_tool(
    question: str,
    choices: Optional[List[str]],
    callback: Optional[Callable] = None,
) -> str:
    """Ask the user to select any number of the offered choices."""
    if not question or not question.strip():
        return tool_error("Question text is required.")
    if not isinstance(choices, list):
        return tool_error("choices must be a list of strings.")

    normalized_choices = [
        value for value in (_flatten_choice(choice) for choice in choices) if value
    ]
    if not normalized_choices:
        return tool_error("At least one non-empty choice is required.")
    if len(normalized_choices) > MAX_SELECT_MANY_CHOICES:
        return tool_error(
            f"select_many supports at most {MAX_SELECT_MANY_CHOICES} choices."
        )
    if callback is None:
        return tool_error("Multi-select is not available in this execution context.")

    question = question.strip()
    try:
        selected = callback(question, normalized_choices)
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")

    if not isinstance(selected, (list, tuple)):
        return tool_error("Multi-select callback returned an invalid response.")

    canonical = set(normalized_choices)
    normalized_selected = []
    for value in selected:
        value = str(value).strip()
        if value not in canonical:
            return tool_error("Multi-select response contained an option that was not offered.")
        if value not in normalized_selected:
            normalized_selected.append(value)

    return json.dumps(
        {
            "question": question,
            "choices_offered": normalized_choices,
            "selected_choices": normalized_selected,
            "cancelled": not normalized_selected,
        },
        ensure_ascii=False,
    )


SELECT_MANY_SCHEMA: Dict[str, Any] = {
    "name": "select_many",
    "description": (
        "Ask the user to select multiple items from a finite list, then wait "
        "for an explicit confirmation. Use this when zero, one, or several "
        "options may be chosen, such as selecting directories to clean. The "
        "result contains `selected_choices` as an array and `cancelled` as a "
        "boolean. For a decision where exactly one option must be chosen, use "
        "`clarify` instead. Put every selectable item in `choices`; do not "
        "duplicate the list in `question`."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question shown above the selectable list.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_SELECT_MANY_CHOICES,
                "description": "The complete list of selectable options.",
            },
        },
        "required": ["question", "choices"],
    },
}


registry.register(
    name="select_many",
    toolset="select_many",
    schema=SELECT_MANY_SCHEMA,
    handler=lambda args, **kw: select_many_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        callback=kw.get("callback"),
    ),
    check_fn=lambda: True,
)
