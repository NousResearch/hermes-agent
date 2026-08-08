"""Configurable budget constants for tool result persistence.

Per-tool resolution: pinned > config overrides > registry > default.
"""

from dataclasses import dataclass, field
from typing import Dict

# Tools whose thresholds must never be overridden.
# read_file=inf prevents infinite persist->read->persist loops.
PINNED_THRESHOLDS: Dict[str, float] = {
    "read_file": float("inf"),
}

# Defaults matching the current hardcoded values in tool_result_storage.py.
# Kept here as the single source of truth; tool_result_storage.py imports these.
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500


# --- CONFIG-DRIVEN OVERRIDES ---------------------------------------------------
# The per-result cap (100K chars) silently truncates large MCP/tool results (e.g.
# a code-search server returning a 200K-char snippet), and there is no config lever
# for it: `tool_output.max_bytes` only bounds terminal output, not generic results.
# This reads an OPTIONAL `tool_budget:` block from the active config.yaml so the
# caps are tunable, and lets specific tools be pinned to no cap. Fully guarded:
# with no block (or on any error) every constant is byte-identical to before.
#   tool_budget:
#     default_result_size_chars: 500000   # per tool result ("inf" = unlimited)
#     turn_budget_chars: 2000000          # aggregate per assistant turn
#     preview_size_chars: 8000            # inline preview before the truncation note
#     per_result_window_fraction: 0.5     # of ctx window a single result may take
#     per_turn_window_fraction: 1.0       # of ctx window the whole turn may take
#     unlimited_tools: [get_code_snippet, search_code, get_architecture, trace_path]
def _load_budget_overrides() -> Dict:
    """Return the `tool_budget:` mapping from the active config, or {} if absent.

    Reads a single config file — ``$HERMES_HOME/config.yaml`` (falling back to
    ``~/.hermes/config.yaml`` only when HERMES_HOME is unset). Fully guarded: any
    error returns {}. Respecting HERMES_HOME keeps this hermetically testable —
    a test points HERMES_HOME at a tmp dir and gets exactly that config.
    """
    import os
    try:
        import yaml  # type: ignore
        home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
        path = os.path.join(home, "config.yaml")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            block = data.get("tool_budget")
            if isinstance(block, dict):
                return block
    except Exception:
        pass
    return {}


_BUDGET_OVERRIDES: Dict = _load_budget_overrides()


def _ov_num(key: str, default):
    v = _BUDGET_OVERRIDES.get(key)
    if v is None:
        return default
    if isinstance(v, str) and v.strip().lower() in ("inf", "unlimited", "none"):
        return float("inf")
    try:
        return type(default)(v)
    except Exception:
        return default


DEFAULT_RESULT_SIZE_CHARS = _ov_num("default_result_size_chars", DEFAULT_RESULT_SIZE_CHARS)
DEFAULT_TURN_BUDGET_CHARS = _ov_num("turn_budget_chars", DEFAULT_TURN_BUDGET_CHARS)
DEFAULT_PREVIEW_SIZE_CHARS = _ov_num("preview_size_chars", DEFAULT_PREVIEW_SIZE_CHARS)
for _tool in (_BUDGET_OVERRIDES.get("unlimited_tools") or []):
    if isinstance(_tool, str):
        PINNED_THRESHOLDS[_tool] = float("inf")
# -------------------------------------------------------------------------------


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
# Config-driven (tool_budget.per_result_window_fraction / per_turn_window_fraction):
_PER_RESULT_WINDOW_FRACTION = _ov_num("per_result_window_fraction", _PER_RESULT_WINDOW_FRACTION)
_PER_TURN_WINDOW_FRACTION = _ov_num("per_turn_window_fraction", _PER_TURN_WINDOW_FRACTION)

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
