"""Progressive tool disclosure ("tool search") for Hermes Agent.

When enabled, MCP, non-core plugin tools, and selected verbose Hermes built-ins
are replaced in the model-visible tools array by three bridge tools —
``tool_search``, ``tool_describe``, ``tool_call`` — and surfaced on demand.

Design constraints this module is built around (see ``openclaw-tool-search-report``
for the full rationale):

* Only an explicit allowlist of verbose core tools may defer. Cheap
  foundational tools stay direct, and unknown tools are never silently dropped.
* The threshold gate runs every assembly: when deferrable tools are below the
  configured auto threshold (context percentage capped by a fixed token
  ceiling), tool search is a no-op and the tools array passes through
  unchanged.
* The catalog is stateless across turns and tools-array assemblies. It is
  rebuilt from the current tool-defs list every time. This is the lesson
  from OpenClaw's cron regression (openclaw/openclaw#84141): a session-keyed
  catalog that drifts out of sync with the live tool registry produces
  silent tool dropouts.
* Bridge tools route through ``model_tools.handle_function_call`` exactly
  like a direct call, so guardrails, plugin pre/post hooks, approval flows,
  and tool-result truncation all fire identically.
* Display and trajectory unwrap is implemented here so the user (CLI activity
  feed, gateway, saved trajectories) always sees the underlying tool, not
  the bridge.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("tools.tool_search")


# Bridge tool names. These names are reserved and may not collide with a
# user/plugin/MCP tool — registration of any tool with these names is
# rejected by the registry's existing override-protection logic.
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"

BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})

# When estimating tokens from char count without a real tokenizer, this is
# the cheap rule of thumb that's stable across providers. Roughly 4 chars
# per token for English+JSON. Underestimating leads to false negatives
# (tool search not activated when it should); overestimating leads to false
# positives (activated when not needed). 4.0 errs slightly toward
# underestimating, which is the safer default.
CHARS_PER_TOKEN = 4.0


# Core tools whose schemas are large enough to justify progressive disclosure.
# The always-direct core remains the substrate a model commonly needs without a
# discovery hop: web search/extract, basic file reads/writes/search, process
# inspection, todo, and GUI terminal-tab helpers. Everything here is still in
# the session's scope; the bridge only moves its verbose schema behind
# tool_search/tool_describe/tool_call.
DEFAULT_DEFERRABLE_CORE_TOOLS = frozenset({
    "terminal",
    "patch",
    "vision_analyze",
    "image_generate",
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_get_images",
    "browser_vision",
    "browser_console",
    "browser_cdp",
    "browser_dialog",
    "text_to_speech",
    "memory",
    "session_search",
    "execute_code",
    "delegate_task",
    "cronjob",
    "ha_list_entities",
    "ha_get_state",
    "ha_list_services",
    "ha_call_service",
    "kanban_show",
    "kanban_list",
    "kanban_complete",
    "kanban_block",
    "kanban_heartbeat",
    "kanban_comment",
    "kanban_create",
    "kanban_link",
    "kanban_unblock",
    "computer_use",
})


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off"
    threshold_pct: float  # 0..100 — only used when enabled == "auto"
    auto_token_threshold: int
    search_default_limit: int
    max_search_limit: int
    defer_core_tools: bool
    defer_toolsets: tuple  # toolset names whose tools should be deferred

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolSearchConfig":
        """Build a config from a raw dict / bool / None.

        Accepts the legacy bool shape (``tools.tool_search: true``) and the
        dict shape (``tools.tool_search: {enabled: auto, ...}``). Validates
        and clamps every numeric field; unknown values fall back to safe
        defaults rather than raising, so a typo in user config does not
        break the agent.
        """
        if raw is True:
            return cls(enabled="auto", threshold_pct=10.0,
                       auto_token_threshold=8_000,
                       search_default_limit=5, max_search_limit=20,
                       defer_core_tools=False, defer_toolsets=())
        if raw is False:
            return cls(enabled="off", threshold_pct=10.0,
                       auto_token_threshold=8_000,
                       search_default_limit=5, max_search_limit=20,
                       defer_core_tools=False, defer_toolsets=())
        if not isinstance(raw, dict):
            return cls(enabled="auto", threshold_pct=10.0,
                       auto_token_threshold=8_000,
                       search_default_limit=5, max_search_limit=20,
                       defer_core_tools=False, defer_toolsets=())

        enabled_raw = str(raw.get("enabled", "auto")).strip().lower()
        if enabled_raw in ("true", "1", "yes"):
            enabled = "on"
        elif enabled_raw in ("false", "0", "no"):
            enabled = "off"
        elif enabled_raw in ("auto", "on", "off"):
            enabled = enabled_raw
        else:
            enabled = "auto"

        threshold_pct = _safe_float(raw.get("threshold_pct"), 10.0)
        threshold_pct = max(0.0, min(100.0, threshold_pct))
        auto_token_threshold = max(1_000, min(
            100_000,
            _safe_int(raw.get("auto_token_threshold"), 8_000),
        ))

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 20)))
        search_default_limit = max(1, min(max_search_limit,
                                          _safe_int(raw.get("search_default_limit"), 5)))
        defer_core_tools = _safe_bool(raw.get("defer_core_tools"), False)

        # Parse defer_toolsets: a list of toolset names whose tools should defer.
        _raw_defer_ts = raw.get("defer_toolsets")
        if isinstance(_raw_defer_ts, (list, tuple)):
            defer_toolsets = tuple(str(s).strip() for s in _raw_defer_ts if str(s).strip())
        else:
            defer_toolsets = ()

        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            auto_token_threshold=auto_token_threshold,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            defer_core_tools=defer_core_tools,
            defer_toolsets=defer_toolsets,
        )


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return fallback


def load_config() -> ToolSearchConfig:
    """Load tool-search config from the user config file."""
    try:
        from hermes_cli.config import load_config as _load
        cfg = _load() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        if not isinstance(tools_cfg, dict):
            tools_cfg = {}
        return ToolSearchConfig.from_raw(tools_cfg.get("tool_search"))
    except Exception as e:
        logger.debug("Failed to load tool-search config: %s", e)
        return ToolSearchConfig.from_raw(None)


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------


def _core_tool_names() -> frozenset[str]:
    """Return the set of tool names that must NEVER be deferred.

    Imported lazily because ``toolsets`` imports from ``tools.registry``
    and we don't want a hard cycle.
    """
    try:
        from toolsets import _HERMES_CORE_TOOLS
        return frozenset(_HERMES_CORE_TOOLS)
    except Exception:
        return frozenset()


def is_deferrable_tool_name(name: str, config: Optional[ToolSearchConfig] = None) -> bool:
    """Return True if a tool with this name is *eligible* for deferral.

    A tool is deferrable iff it is registered with an MCP toolset prefix, is
    not in ``_HERMES_CORE_TOOLS``, or is an explicitly allowed verbose core
    tool and core deferral is enabled. Core tools outside the allowlist are
    always direct, even when their schema is large.
    """
    if name in BRIDGE_TOOL_NAMES:
        return False
    if config is None:
        config = load_config()
    if name in _core_tool_names():
        # Check toolset-based deferral first (per-toolset config)
        if config.defer_toolsets:
            try:
                from tools.registry import registry
                entry = registry.get_entry(name)
                if entry is not None and entry.toolset in config.defer_toolsets:
                    return True
            except Exception:
                pass
        # Fallback to per-tool allowlist when defer_core_tools is enabled
        return bool(config.defer_core_tools and name in DEFAULT_DEFERRABLE_CORE_TOOLS)
    # Check registry toolset for MCP prefix.
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return False
        if entry.toolset.startswith("mcp-"):
            return True
        # Non-MCP, non-core → plugin tool, eligible.
        return True
    except Exception:
        return False


def classify_tools(
    tool_defs: List[Dict[str, Any]],
    config: Optional[ToolSearchConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable).

    ``visible`` retains every tool that must stay in the model-facing array:
    every core tool, plus any tool we can't classify. ``deferrable`` is the
    candidate set for catalog entry.
    """
    visible: List[Dict[str, Any]] = []
    deferrable: List[Dict[str, Any]] = []
    if config is None:
        config = load_config()
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if name in BRIDGE_TOOL_NAMES:
            # Should never happen — bridge tools are added after classification —
            # but be defensive.
            continue
        if is_deferrable_tool_name(name, config=config):
            deferrable.append(td)
        else:
            visible.append(td)
    return visible, deferrable


# ---------------------------------------------------------------------------
# Token estimation and threshold gate
# ---------------------------------------------------------------------------


def estimate_tokens_from_schemas(tool_defs: Iterable[Dict[str, Any]]) -> int:
    """Estimate the token cost of a tool-defs list via the chars/4 rule.

    Cheap and stable across providers. The number doesn't need to be exact —
    it gates the activate/skip decision, and a typical 200K context with a
    10% threshold means the decision flips around 20K tokens of schema.
    Order-of-magnitude precision is fine.
    """
    total_chars = 0
    for td in tool_defs:
        try:
            total_chars += len(json.dumps(td, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            total_chars += len(str(td))
    return int(math.ceil(total_chars / CHARS_PER_TOKEN))


def should_activate(
    config: ToolSearchConfig,
    deferrable_tokens: int,
    context_length: Optional[int],
) -> bool:
    """Decide whether tool search should activate for the current assembly.

    ``"off"`` skips unconditionally. ``"on"`` activates unconditionally
    (as long as there is at least one deferrable tool — there's no point
    swapping a no-op). ``"auto"`` activates when the deferrable schemas
    would consume ``threshold_pct`` of context or more.
    """
    if config.enabled == "off":
        return False
    if deferrable_tokens <= 0:
        return False
    if config.enabled == "on":
        return True
    # auto
    threshold_tokens = auto_threshold_tokens(config, context_length)
    return deferrable_tokens >= threshold_tokens


def auto_threshold_tokens(config: ToolSearchConfig, context_length: Optional[int]) -> int:
    """Return the token threshold for ``auto`` activation.

    ``auto_token_threshold`` is the primary knob — a fixed absolute token
    budget that applies regardless of context window size. This is correct
    because the cost of a bloated tools array is roughly context-independent:
    it's paid every turn regardless of window size, and it poisons the prompt
    cache prefix identically.

    ``threshold_pct`` is a secondary clamp for short-context models where
    the percentage threshold would be *lower* than the absolute threshold
    (e.g. a 16K-context model with threshold_pct=10 = 1,600 tokens, which
    is below the 8K default). For large-context models (1M tokens), the
    absolute threshold dominates and the percentage is irrelevant.

    When ``threshold_pct == 0``, gating is purely token-threshold-based.
    """
    if not context_length or context_length <= 0:
        return config.auto_token_threshold
    if config.threshold_pct <= 0:
        # Pure token-threshold mode — percentage gate disabled.
        return config.auto_token_threshold
    pct_threshold = int(context_length * (config.threshold_pct / 100.0))
    # Take the MIN: short-context models get the lower (percentage) gate,
    # large-context models get capped by the absolute threshold.
    return max(1, min(pct_threshold, config.auto_token_threshold))


# ---------------------------------------------------------------------------
# Catalog + BM25 retrieval
# ---------------------------------------------------------------------------


@dataclass
class CatalogEntry:
    """One deferrable tool, in a form the bridge tools can search and serve."""

    name: str
    description: str
    schema: Dict[str, Any]  # The full {"type":"function", "function": {...}} entry.
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # Toolset name, e.g. "mcp-github" or "kanban"

    # Pre-tokenized fields for BM25.
    _tokens: List[str] = field(default_factory=list)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _entry_search_text(td: Dict[str, Any]) -> str:
    """Build the search-text blob for a deferrable tool.

    Includes the tool name (with underscores broken into words so BM25 can
    match against query terms), the description, and the names of the
    top-level parameters. Schema bodies are deliberately excluded —
    indexing them adds noise without improving recall in our measurement.
    """
    fn = td.get("function") or {}
    name = fn.get("name", "")
    desc = fn.get("description", "") or ""
    params = ((fn.get("parameters") or {}).get("properties") or {})
    param_names = " ".join(params.keys())
    # Break snake_case and dotted names into words for BM25.
    name_words = name.replace("_", " ").replace(".", " ").replace("-", " ").replace(":", " ")
    return f"{name_words} {desc} {param_names}"


def _classify_source(name: str) -> Tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    if name in _core_tool_names():
        try:
            from tools.registry import registry
            entry = registry.get_entry(name)
            return ("core", entry.toolset if entry is not None else "core")
        except Exception:
            return ("core", "core")
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return ("other", "")
        if entry.toolset.startswith("mcp-"):
            return ("mcp", entry.toolset)
        return ("plugin", entry.toolset)
    except Exception:
        return ("other", "")


def build_catalog(tool_defs: List[Dict[str, Any]]) -> List[CatalogEntry]:
    """Build the deferred-tool catalog from a tool-defs list.

    Caller is expected to pass only the deferrable subset (``classify_tools``
    returns it as the second element).
    """
    catalog: List[CatalogEntry] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        desc = fn.get("description", "") or ""
        source, source_name = _classify_source(name)
        entry = CatalogEntry(
            name=name,
            description=desc,
            schema=td,
            source=source,
            source_name=source_name,
            _tokens=_tokenize(_entry_search_text(td)),
        )
        catalog.append(entry)
    return catalog


def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                doc_lengths: List[int], avg_dl: float,
                doc_freq: Dict[str, int], n_docs: int,
                k1: float = 1.5, b: float = 0.75) -> float:
    """Standard BM25 score for one query against one document.

    Inlined small implementation rather than adding a dependency. Performance
    is fine — the catalog is bounded by N (tools) typically < 500, and we
    score against the in-memory tokens list.
    """
    if not doc_tokens:
        return 0.0
    score = 0.0
    dl = len(doc_tokens)
    # Pre-count tokens in the doc.
    doc_tf: Dict[str, int] = {}
    for t in doc_tokens:
        doc_tf[t] = doc_tf.get(t, 0) + 1
    for q in query_tokens:
        df = doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = doc_tf.get(q, 0)
        if tf == 0:
            continue
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
        score += idf * norm
    return score


def search_catalog(catalog: List[CatalogEntry], query: str, limit: int = 5) -> List[CatalogEntry]:
    """Return the top-``limit`` catalog entries for ``query`` by BM25.

    Falls back to a stable name-substring match when BM25 yields no hits
    above zero. That ensures a query like ``"github"`` against a catalog
    where every tool is named ``github_*`` still returns results — BM25
    can underperform when query and document share only one token that
    appears in every document (zero IDF).
    """
    if not catalog or limit <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Precompute doc statistics.
    doc_lengths = [len(e._tokens) for e in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq: Dict[str, int] = {}
    for e in catalog:
        seen = set(e._tokens)
        for t in seen:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    n_docs = len(catalog)

    scored: List[Tuple[float, CatalogEntry]] = []
    for entry in catalog:
        s = _bm25_score(query_tokens, entry._tokens, doc_lengths, avg_dl,
                        doc_freq, n_docs)
        if s > 0:
            scored.append((s, entry))

    if not scored:
        # Substring fallback against the original tool name.
        ql = query.lower()
        for entry in catalog:
            if ql in entry.name.lower():
                scored.append((0.1, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


# ---------------------------------------------------------------------------
# Bridge tool schemas
# ---------------------------------------------------------------------------


def bridge_tool_schemas(deferred_count: int) -> List[Dict[str, Any]]:
    """Build the bridge tool schemas to inject in place of deferred tools.

    The schemas are intentionally short — every byte added here is a byte
    the user pays on every turn. Descriptions are tuned to be unambiguous
    about the call sequence the model should follow.
    """
    desc_search = (
        f"Search {deferred_count} additional tools that are loaded on demand. "
        "Returns up to ``limit`` matches with name and description. Follow "
        f"with `{TOOL_DESCRIBE_NAME}` to load a tool's full parameter schema, "
        f"then `{TOOL_CALL_NAME}` to invoke it. Tools listed at the top of this "
        "system prompt are already available and do not need to be searched."
    )
    desc_describe = (
        f"Load the full JSON schema for one tool returned by `{TOOL_SEARCH_NAME}`. "
        f"Required before `{TOOL_CALL_NAME}` if the tool's parameters are unknown."
    )
    desc_call = (
        "Invoke a deferred tool by name with the given arguments. Argument shape "
        f"matches the tool's schema (see `{TOOL_DESCRIBE_NAME}`). Policy, hooks, "
        "and approvals run exactly as for any directly-listed tool."
    )

    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_NAME,
                "description": desc_search,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords describing the capability you need (e.g. 'create github issue').",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Default 5.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_DESCRIBE_NAME,
                "description": desc_describe,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name (as returned by tool_search).",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_CALL_NAME,
                "description": desc_call,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name to invoke.",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the tool, matching its schema.",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Public entry point: assemble tool-defs with optional tool search
# ---------------------------------------------------------------------------


@dataclass
class AssemblyResult:
    """Outcome of one assembly. Useful for tests and observability."""

    tool_defs: List[Dict[str, Any]]
    activated: bool
    deferred_count: int = 0
    deferred_tokens: int = 0
    threshold_tokens: int = 0


def assemble_tool_defs(
    tool_defs: List[Dict[str, Any]],
    *,
    context_length: Optional[int] = None,
    config: Optional[ToolSearchConfig] = None,
    hot_tools: Optional[frozenset] = None,
) -> AssemblyResult:
    """Return the tool-defs list the model should actually see.

    When tool search is inactive (off, no deferrable tools, or below
    threshold), this is a passthrough. When active, MCP and plugin tools
    are stripped from the visible list and replaced with the three bridge
    tools. Core tools are *never* deferred regardless of config.

    Hot-tools promotion: tools in ``hot_tools`` are re-inflated into the
    direct array even if they would otherwise be deferred. This set should
    only grow within a session (never shrink) to preserve prompt cache
    stability — see system_prompt.py module docstring.

    Idempotent: calling with bridge tools already in the input is a no-op
    (they classify as non-core/non-deferrable but their names are reserved,
    so they are filtered out of the deferrable set).
    """
    if config is None:
        config = load_config()

    # Defensive: strip any bridge tools that may already be in the list
    # (e.g. someone called assemble twice).
    incoming = [td for td in tool_defs
                if (td.get("function") or {}).get("name") not in BRIDGE_TOOL_NAMES]

    visible, deferrable = classify_tools(incoming, config=config)
    if not deferrable:
        return AssemblyResult(tool_defs=incoming, activated=False)

    hidden_names: set[str] = set()
    hidden_defs: List[Dict[str, Any]] = []
    _hot = hot_tools or frozenset()
    for td in deferrable:
        name = (td.get("function") or {}).get("name", "")
        source, _ = _classify_source(name)
        # Keep plugin tools directly visible when tool_search activates.
        if source == "plugin":
            continue
        # Hot-tools promotion: keep recently-used tools direct.
        if name in _hot:
            continue
        hidden_names.add(name)
        hidden_defs.append(td)

    if not hidden_defs:
        return AssemblyResult(tool_defs=incoming, activated=False)

    deferrable_tokens = estimate_tokens_from_schemas(hidden_defs)
    if not should_activate(config, deferrable_tokens, context_length):
        return AssemblyResult(
            tool_defs=incoming,
            activated=False,
            deferred_count=len(hidden_defs),
            deferred_tokens=deferrable_tokens,
            threshold_tokens=auto_threshold_tokens(config, context_length),
        )

    bridge = bridge_tool_schemas(len(deferrable))
    result = [
        td for td in incoming
        if (td.get("function") or {}).get("name") not in hidden_names
    ] + bridge
    threshold_tokens = auto_threshold_tokens(config, context_length)

    logger.info(
        "tool_search activated: %d core/visible tools kept, %d deferred (~%d tokens, threshold ~%d)",
        len(result) - len(bridge), len(hidden_defs), deferrable_tokens, threshold_tokens,
    )

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(hidden_defs),
        deferred_tokens=deferrable_tokens,
        threshold_tokens=threshold_tokens,
    )


# ---------------------------------------------------------------------------
# Bridge tool dispatch
# ---------------------------------------------------------------------------
def is_bridge_tool(name: str) -> bool:
    return name in BRIDGE_TOOL_NAMES


def _format_search_hit(entry: CatalogEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "source": entry.source,
        "source_name": entry.source_name,
        # Cap description so a chatty MCP server doesn't blow up the result.
        "description": (entry.description or "")[:400],
    }


def dispatch_tool_search(args: Dict[str, Any],
                         *,
                         current_tool_defs: List[Dict[str, Any]],
                         config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_search`` bridge tool. Returns a JSON string."""
    if config is None:
        config = load_config()
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    raw_limit = args.get("limit")
    if raw_limit is None:
        limit = config.search_default_limit
    else:
        limit = max(1, min(config.max_search_limit, _safe_int(raw_limit, config.search_default_limit)))

    _, deferrable = classify_tools(current_tool_defs, config=config)
    catalog = build_catalog(deferrable)
    hits = search_catalog(catalog, query, limit=limit)
    return json.dumps({
        "query": query,
        "total_available": len(catalog),
        "matches": [_format_search_hit(h) for h in hits],
    }, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]],
                           config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns a JSON string."""
    name = str(args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name is required"}, ensure_ascii=False)
    if config is None:
        config = load_config()
    if not is_deferrable_tool_name(name, config=config):
        return json.dumps({
            "error": (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search."
            ),
        }, ensure_ascii=False)
    _, deferrable = classify_tools(current_tool_defs, config=config)
    for td in deferrable:
        fn = td.get("function") or {}
        if fn.get("name") == name:
            return json.dumps({
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }, ensure_ascii=False)
    return json.dumps({
        "error": f"'{name}' is not currently available. Re-run tool_search to refresh.",
    }, ensure_ascii=False)


def scoped_deferrable_names(
    tool_defs: List[Dict[str, Any]],
    config: Optional[ToolSearchConfig] = None,
) -> frozenset[str]:
    """Return the set of deferrable tool names present in ``tool_defs``.

    ``tool_defs`` is expected to be the *pre-assembly* tool list for the
    current session's toolset scope (i.e. what
    ``get_tool_definitions(skip_tool_search_assembly=True)`` returns for the
    session's enabled/disabled toolsets). The resulting set is the universe of
    tools the session may legitimately reach through ``tool_call``. Used as a
    scoping gate by both the ``model_tools`` bridge dispatch and the
    ``tool_executor`` unwrap so a restricted-toolset session can never invoke
    an out-of-scope tool via the bridge.
    """
    names: set[str] = set()
    if config is None:
        config = load_config()
    for td in tool_defs:
        name = (td.get("function") or {}).get("name", "")
        if name and is_deferrable_tool_name(name, config=config):
            names.add(name)
    return frozenset(names)


def resolve_underlying_call(
    args: Dict[str, Any],
    config: Optional[ToolSearchConfig] = None,
) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Parse a ``tool_call`` invocation into (underlying_name, args, error_msg).

    Used by:
    * the dispatcher in ``model_tools.handle_function_call``,
    * the display layer (so the activity feed shows the underlying tool),
    * the trajectory recorder.

    On parse error, returns ``(None, {}, error_message)``.
    """
    name = str(args.get("name") or "").strip()
    if not name:
        return None, {}, "tool_call requires a 'name' argument"
    if name in BRIDGE_TOOL_NAMES:
        return None, {}, f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
    raw_args = args.get("arguments")
    if raw_args is None:
        raw_args = {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return None, {}, f"tool_call 'arguments' is not valid JSON: {e}"
    if not isinstance(raw_args, dict):
        return None, {}, "tool_call 'arguments' must be an object"
    if config is None:
        config = load_config()
    if not is_deferrable_tool_name(name, config=config):
        return None, {}, (
            f"'{name}' is not a deferrable tool. If it appears in the model-facing tools "
            "list already, call it directly instead of via tool_call."
        )
    return name, raw_args, None


__all__ = [
    "TOOL_SEARCH_NAME",
    "TOOL_DESCRIBE_NAME",
    "TOOL_CALL_NAME",
    "BRIDGE_TOOL_NAMES",
    "ToolSearchConfig",
    "CatalogEntry",
    "AssemblyResult",
    "load_config",
    "is_deferrable_tool_name",
    "classify_tools",
    "estimate_tokens_from_schemas",
    "should_activate",
    "auto_threshold_tokens",
    "DEFAULT_DEFERRABLE_CORE_TOOLS",
    "build_catalog",
    "search_catalog",
    "bridge_tool_schemas",
    "assemble_tool_defs",
    "is_bridge_tool",
    "dispatch_tool_search",
    "dispatch_tool_describe",
    "resolve_underlying_call",
    "scoped_deferrable_names",
]
