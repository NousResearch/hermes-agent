"""Configurable budget constants for tool result persistence.
Per-tool resolution: pinned > config overrides > registry > default."""

from dataclasses import dataclass, field
from typing import Dict

# Never overridden; read_file=inf prevents infinite persist->read->persist loops.
PINNED_THRESHOLDS: Dict[str, float] = {"read_file": float("inf")}

# Single source of truth for the defaults; tool_result_storage.py imports these.
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500

# Tighter per-result default for ``mcp_`` tools: MCP servers routinely return
# un-paginated 20-50K payloads that sail under the generic 100K threshold; spillover
# keeps the full payload on disk. Config: ``tool_output.mcp_result_size_chars``.
DEFAULT_MCP_RESULT_SIZE_CHARS: int = 50_000
# Same prefix the untrusted-content wrapper keys on (agent/tool_dispatch_helpers.py).
MCP_TOOL_PREFIX: str = "mcp_"


def _configured_budget_block() -> dict:
    """Return the spillover config block, preferring the recognized ``tool_output`` key.

    ``tool_budget`` was the short-lived original spelling (shared with #80508). Keep it
    as a read-only compatibility fallback so existing installations do not silently
    lose their MCP threshold after upgrading. Goes through ``load_config_readonly``
    (the sanctioned path; raw config.yaml parsing outside owner modules is test-guarded).

    ``tool_output`` ships in DEFAULT_CONFIG, so the loader's deep-merge means the
    primary block always exists with default values — a whole-block "primary else
    legacy" check would never fall back. Fall back PER KEY: a merged-in default
    must not shadow an explicit legacy setting.
    """
    try:
        from hermes_cli.config import load_config_readonly
        data = load_config_readonly()
        if not isinstance(data, dict):
            return {}
        primary = data.get("tool_output")
        primary = primary if isinstance(primary, dict) else {}
        legacy = data.get("tool_budget")
        legacy = legacy if isinstance(legacy, dict) else {}
        if not legacy:
            return primary
        merged = dict(primary)
        if (merged.get("mcp_result_size_chars") == DEFAULT_MCP_RESULT_SIZE_CHARS
                and "mcp_result_size_chars" in legacy):
            merged["mcp_result_size_chars"] = legacy["mcp_result_size_chars"]
        if not merged.get("tool_overrides") and "tool_overrides" in legacy:
            merged["tool_overrides"] = legacy["tool_overrides"]
        return merged
    except Exception:
        return {}


def _configured_mcp_result_size() -> int:
    """Read ``tool_output.mcp_result_size_chars`` from active config."""
    try:
        raw = _configured_budget_block().get("mcp_result_size_chars")
        if raw is not None and not isinstance(raw, bool):
            value = int(raw)
            if value > 0:
                return value
    except Exception:
        pass
    return DEFAULT_MCP_RESULT_SIZE_CHARS


def _configured_tool_overrides() -> Dict[str, int]:
    """Read positive per-tool spill thresholds from ``tool_output``.

    Tool handlers and providers do not always keep their documented result
    shape bounded. A named override lets an operator spill a known-chatty tool
    before the generic 100K threshold without shrinking every tool or the
    model's context window. Invalid entries are ignored fail-closed.
    """
    try:
        raw = _configured_budget_block().get("tool_overrides")
        if not isinstance(raw, dict):
            return {}
        overrides: Dict[str, int] = {}
        for name, threshold in raw.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(threshold, bool):
                continue
            try:
                value = int(threshold)
            except (TypeError, ValueError):
                continue
            if value > 0:
                overrides[name.strip()] = value
        return overrides
    except Exception:
        return {}


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget constants: per-result threshold (``resolve_threshold``),
    per-turn aggregate (``turn_budget``) and inline snippet size (``preview_size``)."""

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    mcp_result_size: int = DEFAULT_MCP_RESULT_SIZE_CHARS
    tool_overrides: Dict[str, int] = field(default_factory=dict)

    def resolve_threshold(self, tool_name: str) -> int | float:
        """Priority: pinned -> tool_overrides -> mcp_ prefix -> registry per-tool -> default.
        MCP tools get ``mcp_result_size`` (no registry entry). MCP and registry values
        are capped at ``default_result_size`` so a context-scaled budget for a small
        model still constrains tools registering a fixed 100K ``max_result_size_chars``.

        For the default budget this is a no-op because both equal 100K; for a scaled-down budget it prevents
        a per-tool registry value from re-inflating the cap past the model's window (#23767).
        """
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        if tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]
        if tool_name.startswith(MCP_TOOL_PREFIX):
            return min(self.mcp_result_size, self.default_result_size)
        from tools.registry import registry
        registry_value = registry.get_max_result_size(tool_name, default=self.default_result_size)
        if registry_value == float("inf"):
            return registry_value
        return min(registry_value, self.default_result_size)


# Default config -- matches the historical hardcoded behavior exactly.
DEFAULT_BUDGET = BudgetConfig()

# Same rough 4-chars-per-token the estimator uses (agent/model_metadata.py);
# a smaller divisor would UNDER-protect small models.
_CHARS_PER_TOKEN: int = 4
# Window fraction ONE result / the WHOLE turn's tool output may occupy — well
# under 1.0 since system prompt, schemas, history and the reply all compete.
_PER_RESULT_WINDOW_FRACTION: float = 0.15
_PER_TURN_WINDOW_FRACTION: float = 0.30
# Floors so a tiny model still gets a usable result, never a 0-char budget.
_MIN_RESULT_SIZE_CHARS: int = 8_000
_MIN_TURN_BUDGET_CHARS: int = 16_000


def budget_for_context_window(context_length: int | None) -> BudgetConfig:
    """Return a BudgetConfig scaled to the model's context window: the fixed
    defaults suit 200K+ models but on 65K one result/turn can fill the window.
    The proportional value is clamped to the defaults as a CAP (large models
    stay byte-identical) and floored so a usable preview always survives.

    The fixed defaults (100K result / 200K turn chars) are correct for large (200K+ token) models but blind
    to small ones: on a 65K-token model a single tool result persisted at the 100K-char threshold, or a
    200K-char turn budget (~50K tokens), can by itself approach or exceed the whole window and force an
    oversized request (#23767).
    """
    mcp_result_size = _configured_mcp_result_size()
    tool_overrides = _configured_tool_overrides()
    if not context_length or context_length <= 0:
        if (
            mcp_result_size == DEFAULT_MCP_RESULT_SIZE_CHARS
            and not tool_overrides
        ):
            return DEFAULT_BUDGET
        return BudgetConfig(
            mcp_result_size=mcp_result_size,
            tool_overrides=tool_overrides,
        )
    window_chars = context_length * _CHARS_PER_TOKEN
    per_result = max(_MIN_RESULT_SIZE_CHARS, min(int(window_chars * _PER_RESULT_WINDOW_FRACTION), DEFAULT_RESULT_SIZE_CHARS))
    return BudgetConfig(
        default_result_size=per_result,
        turn_budget=max(_MIN_TURN_BUDGET_CHARS, min(int(window_chars * _PER_TURN_WINDOW_FRACTION), DEFAULT_TURN_BUDGET_CHARS)),
        preview_size=DEFAULT_PREVIEW_SIZE_CHARS,
        mcp_result_size=mcp_result_size,
        tool_overrides={
            name: min(threshold, per_result)
            for name, threshold in tool_overrides.items()
        },
    )
