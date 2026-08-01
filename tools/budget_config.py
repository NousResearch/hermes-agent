"""Configurable budget constants for tool result persistence.

Per-tool resolution: pinned > config overrides > registry > default.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

# Tools whose thresholds must never be overridden.
# Model-visible safety must not depend on an infinite exemption. ``read_file``
# loop prevention now lives in the final result policy, where the current page
# is bounded inline instead of being persisted as another read_file handle.
PINNED_THRESHOLDS: Dict[str, float] = {}

# Defaults matching the current hardcoded values in tool_result_storage.py.
# Kept here as the single source of truth; tool_result_storage.py imports these.
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500

# Universal model-visible tool-result policy. The limit is expressed in units
# of the active TokenCounter; the built-in conservative fallback uses UTF-8
# bytes. Overrides are explicit and local to one call.
RESULT_TOKEN_LIMIT_ARG: str = "result_token_limit"
DEFAULT_RESULT_TOKEN_LIMIT: int = 10_000
MAX_RESULT_TOKEN_LIMIT: int = 32_000


@dataclass(frozen=True)
class ResultTokenBudget:
    """Validated, immutable model-visible budget for one tool call."""

    limit_tokens: int = DEFAULT_RESULT_TOKEN_LIMIT
    override_requested: bool = False


DEFAULT_RESULT_TOKEN_BUDGET = ResultTokenBudget()


def _result_token_limit_property_schema() -> Dict[str, Any]:
    """Build a fresh JSON-schema fragment for the reserved override."""
    return {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_RESULT_TOKEN_LIMIT,
        "default": DEFAULT_RESULT_TOKEN_LIMIT,
        "description": (
            "Optional model-visible result budget for this call only. "
            f"Defaults to {DEFAULT_RESULT_TOKEN_LIMIT}; maximum "
            f"{MAX_RESULT_TOKEN_LIMIT}. Removed before tool execution."
        ),
    }


def extract_result_token_budget(
    arguments: Dict[str, Any],
) -> tuple[Dict[str, Any], ResultTokenBudget, str | None]:
    """Remove and validate the reserved per-call result budget argument.

    A fresh dict is returned so the model's parsed payload is never mutated in
    place. Invalid requests fail closed and still return sanitized business
    arguments, allowing the dispatcher to emit a structured tool error without
    exposing the reserved argument to middleware, hooks, or the handler.
    """
    sanitized = dict(arguments)
    if RESULT_TOKEN_LIMIT_ARG not in sanitized:
        return sanitized, DEFAULT_RESULT_TOKEN_BUDGET, None

    requested = sanitized.pop(RESULT_TOKEN_LIMIT_ARG)
    if isinstance(requested, bool) or not isinstance(requested, int):
        return (
            sanitized,
            DEFAULT_RESULT_TOKEN_BUDGET,
            (
                f"'{RESULT_TOKEN_LIMIT_ARG}' must be an integer between 1 and "
                f"{MAX_RESULT_TOKEN_LIMIT:,}"
            ),
        )
    if not 1 <= requested <= MAX_RESULT_TOKEN_LIMIT:
        return (
            sanitized,
            DEFAULT_RESULT_TOKEN_BUDGET,
            (
                f"'{RESULT_TOKEN_LIMIT_ARG}' must be between 1 and "
                f"{MAX_RESULT_TOKEN_LIMIT:,}"
            ),
        )
    return sanitized, ResultTokenBudget(requested, True), None


def schema_owns_result_token_limit(function_schema: Dict[str, Any] | None) -> bool:
    """Return whether a tool's original schema owns the reserved-looking name."""
    if not isinstance(function_schema, dict):
        return False
    parameters = function_schema.get("parameters")
    if not isinstance(parameters, dict):
        return False
    properties = parameters.get("properties")
    return isinstance(properties, dict) and RESULT_TOKEN_LIMIT_ARG in properties


def extract_result_token_budget_for_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> tuple[Dict[str, Any], ResultTokenBudget, str | None]:
    """Extract Hermes metadata unless the registered tool owns that argument.

    MCP and plugin schemas are third-party contracts. A tool that already
    declares ``result_token_limit`` must receive it unchanged rather than have
    Hermes reinterpret or remove it as model-visible result metadata.
    """
    from tools.registry import registry

    if schema_owns_result_token_limit(registry.get_schema(tool_name)):
        return dict(arguments), DEFAULT_RESULT_TOKEN_BUDGET, None
    return extract_result_token_budget(arguments)


def merge_result_token_budgets(
    outer: ResultTokenBudget,
    inner: ResultTokenBudget,
) -> tuple[ResultTokenBudget, str | None]:
    """Merge direct and Tool Search bridge metadata without ambiguity."""
    if outer.override_requested and inner.override_requested:
        return (
            DEFAULT_RESULT_TOKEN_BUDGET,
            (
                f"'{RESULT_TOKEN_LIMIT_ARG}' may be supplied either on "
                "tool_call or in its underlying arguments, not both"
            ),
        )
    if inner.override_requested:
        return inner, None
    if outer.override_requested:
        return outer, None
    return DEFAULT_RESULT_TOKEN_BUDGET, None


def augment_function_schema_with_result_token_limit(
    function_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Return one stable function-schema copy exposing the call-local override."""
    function = copy.deepcopy(function_schema)
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
        function["parameters"] = parameters
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
        parameters["properties"] = properties
    if RESULT_TOKEN_LIMIT_ARG in properties:
        return function
    properties[RESULT_TOKEN_LIMIT_ARG] = _result_token_limit_property_schema()
    required = parameters.get("required")
    if isinstance(required, list) and RESULT_TOKEN_LIMIT_ARG in required:
        parameters["required"] = [
            name for name in required if name != RESULT_TOKEN_LIMIT_ARG
        ]
    return function


def augment_tool_schemas_with_result_token_limit(
    tool_definitions: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Return stable schema copies exposing the reserved per-call override."""
    augmented: list[Dict[str, Any]] = []
    for definition in tool_definitions:
        item = copy.deepcopy(definition)
        function = item.get("function")
        if not isinstance(function, dict):
            augmented.append(item)
            continue
        item["function"] = augment_function_schema_with_result_token_limit(
            function
        )
        augmented.append(item)
    return augmented


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget constants for the 3-layer tool result persistence system.

    Layer 2 (per-result): resolve_threshold(tool_name) -> threshold in chars.
    Layer 3 (per-turn):   turn_budget -> aggregate char budget across all tool
                          results in a single assistant turn.
    Preview:              preview_size -> inline snippet size after persistence.
    """

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    tool_overrides: Dict[str, int] = field(default_factory=dict)

    def resolve_threshold(self, tool_name: str) -> int | float:
        """Resolve the persistence threshold for a tool.

        Priority: pinned -> tool_overrides -> registry per-tool -> default.

        The registry per-tool value is capped at ``default_result_size`` so a
        context-scaled budget (small model) actually constrains tools that
        register a large fixed ``max_result_size_chars`` (web/terminal/x_search
        all register 100K). For the default budget this is a no-op because both
        equal 100K; for a scaled-down budget it prevents a per-tool registry
        value from re-inflating the cap past the model's window (#23767).
        """
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        if tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]
        from tools.registry import registry
        registry_value = registry.get_max_result_size(tool_name, default=self.default_result_size)
        if registry_value == float("inf"):
            return registry_value
        return min(registry_value, self.default_result_size)


# Default config -- matches current hardcoded behavior exactly.
DEFAULT_BUDGET = BudgetConfig()


# Token<->char conversion used when scaling the budget to a model's context
# window. Deliberately conservative (a smaller divisor = more chars per token =
# a larger char budget) would UNDER-protect small models, so we use the same
# rough 4-chars-per-token ratio the estimator uses (agent/model_metadata.py).
_CHARS_PER_TOKEN: int = 4

# Fraction of a model's context window we allow a SINGLE tool result to occupy
# before persisting/truncating it, and the fraction the WHOLE turn's tool
# output may occupy. Tool output is not the only thing in the window (system
# prompt, tool schemas, conversation history, the model's own reply all
# compete), so these stay well under 1.0.
_PER_RESULT_WINDOW_FRACTION: float = 0.15
_PER_TURN_WINDOW_FRACTION: float = 0.30

# Floor so even a tiny-but-admitted model still gets a usable preview/result
# rather than a 0-char budget.
_MIN_RESULT_SIZE_CHARS: int = 8_000
_MIN_TURN_BUDGET_CHARS: int = 16_000


def budget_for_context_window(context_length: int | None) -> BudgetConfig:
    """Return a BudgetConfig scaled to the active model's context window.

    The fixed defaults (100K result / 200K turn chars) are correct for large
    (200K+ token) models but blind to small ones: on a 65K-token model a single
    tool result persisted at the 100K-char threshold, or a 200K-char turn
    budget (~50K tokens), can by itself approach or exceed the whole window and
    force an oversized request (#23767).

    Scaling keeps large models byte-identical to today (the proportional value
    is clamped to the existing defaults as a CAP) while shrinking the budget for
    small models proportionally to their window, floored so a usable preview
    always survives.
    """
    if not context_length or context_length <= 0:
        return DEFAULT_BUDGET

    window_chars = context_length * _CHARS_PER_TOKEN
    per_result = int(window_chars * _PER_RESULT_WINDOW_FRACTION)
    per_turn = int(window_chars * _PER_TURN_WINDOW_FRACTION)

    # Clamp: never exceed the historical defaults (so large models are
    # unchanged), never drop below the floor (so tiny models stay usable).
    per_result = max(_MIN_RESULT_SIZE_CHARS, min(per_result, DEFAULT_RESULT_SIZE_CHARS))
    per_turn = max(_MIN_TURN_BUDGET_CHARS, min(per_turn, DEFAULT_TURN_BUDGET_CHARS))

    return BudgetConfig(
        default_result_size=per_result,
        turn_budget=per_turn,
        preview_size=DEFAULT_PREVIEW_SIZE_CHARS,
    )
