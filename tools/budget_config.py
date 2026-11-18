"""Configurable budget constants for tool result persistence.
Per-tool resolution: pinned > config overrides > registry > default."""

import logging
from dataclasses import dataclass, field
from typing import Dict

from agent.model_metadata import CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

# Never overridden; read_file=inf prevents infinite persist->read->persist loops.
PINNED_THRESHOLDS: dict[str, float] = {"read_file": float("inf")}

# Single source of truth for the defaults; tool_result_storage.py imports these.
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500

# Tighter per-result default for ``mcp_`` tools: MCP servers routinely return
# un-paginated 20-50K payloads that sail under the generic 100K threshold; spillover
# keeps the full payload on disk. Config: ``tool_budget.mcp_result_size_chars``.
DEFAULT_MCP_RESULT_SIZE_CHARS: int = 50_000
# Same prefix the untrusted-content wrapper keys on (agent/tool_dispatch_helpers.py).
MCP_TOOL_PREFIX: str = "mcp_"


def _positive_int(raw) -> int | None:
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _configured_budget() -> dict:
    """The user's ``tool_budget:`` caps, read per build via ``load_config_readonly`` (the sanctioned
    path; raw config.yaml parsing outside owner modules is test-guarded). Bad values are dropped,
    so a typo keeps the built-in default instead of disabling spillover.

    ``{mcp_result_size, default_result_size, turn_budget}`` (each optional) plus ``tool_overrides``:
    ``unlimited_tools`` names map to ``inf`` and ``mcp_servers.<name>.max_result_chars`` maps onto
    every tool that server registered (provenance from registration, never the ``mcp_<server>_``
    name shape, which two servers can share).
    """
    out: dict = {"tool_overrides": {}}
    try:
        from hermes_cli.config import load_config_readonly
        data = load_config_readonly()
    except Exception:  # health: allow BLE001 -- a broken config must never block a tool result
        logger.debug("tool_budget: config unreadable, using built-in caps", exc_info=True)
        return out
    if not isinstance(data, dict):
        return out
    block = data.get("tool_budget")
    if isinstance(block, dict):
        for key, field_name in (("mcp_result_size_chars", "mcp_result_size"),
                                ("default_result_size_chars", "default_result_size"),
                                ("turn_budget_chars", "turn_budget")):
            if (value := _positive_int(block.get(key))) is not None:
                out[field_name] = value
        unlimited = block.get("unlimited_tools")
        for name in unlimited if isinstance(unlimited, list) else ():
            if isinstance(name, str) and name.strip():
                out["tool_overrides"][name.strip()] = float("inf")
    servers = data.get("mcp_servers")
    limits = {name: limit for name, cfg in (servers.items() if isinstance(servers, dict) else ())
              if isinstance(cfg, dict) and (limit := _positive_int(cfg.get("max_result_chars"))) is not None}
    if limits:
        from tools.mcp_tool import _mcp_tool_server_names
        for tool, server in dict(_mcp_tool_server_names).items():
            if server in limits:
                out["tool_overrides"].setdefault(tool, limits[server])
    return out


def _configured_mcp_result_size() -> int:
    """``tool_budget.mcp_result_size_chars``, else the built-in default."""
    return _configured_budget().get("mcp_result_size", DEFAULT_MCP_RESULT_SIZE_CHARS)


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget constants: per-result threshold (``resolve_threshold``),
    per-turn aggregate (``turn_budget``) and inline snippet size (``preview_size``)."""

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    mcp_result_size: int = DEFAULT_MCP_RESULT_SIZE_CHARS
    tool_overrides: dict[str, int] = field(default_factory=dict)

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

# Same rough chars-per-token the estimator uses; a smaller divisor would UNDER-protect small models.
_CHARS_PER_TOKEN: int = CHARS_PER_TOKEN
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
    cfg = _configured_budget()
    mcp_result_size = cfg.get("mcp_result_size", DEFAULT_MCP_RESULT_SIZE_CHARS)
    result_cap = cfg.get("default_result_size", DEFAULT_RESULT_SIZE_CHARS)
    turn_cap = cfg.get("turn_budget", DEFAULT_TURN_BUDGET_CHARS)
    overrides = cfg["tool_overrides"]
    if not context_length or context_length <= 0:
        if len(cfg) == 1 and not overrides:
            return DEFAULT_BUDGET
        return BudgetConfig(default_result_size=result_cap, turn_budget=turn_cap,
                            mcp_result_size=mcp_result_size, tool_overrides=overrides)
    # A raised cap still scales down for a small window: the user moves the ceiling, not the floor
    # under the model's context.
    window_chars = context_length * _CHARS_PER_TOKEN
    window_result = max(_MIN_RESULT_SIZE_CHARS, int(window_chars * _PER_RESULT_WINDOW_FRACTION))
    # A per-server limit is bounded by the window too; only an explicit unlimited_tools entry is not.
    overrides = {name: v if v == float("inf") else min(v, window_result) for name, v in overrides.items()}
    return BudgetConfig(
        default_result_size=max(_MIN_RESULT_SIZE_CHARS, min(int(window_chars * _PER_RESULT_WINDOW_FRACTION), result_cap)),
        turn_budget=max(_MIN_TURN_BUDGET_CHARS, min(int(window_chars * _PER_TURN_WINDOW_FRACTION), turn_cap)),
        preview_size=DEFAULT_PREVIEW_SIZE_CHARS,
        mcp_result_size=mcp_result_size,
        tool_overrides=overrides,
    )
