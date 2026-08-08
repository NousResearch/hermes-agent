"""Progressive tool disclosure ("tool search") for Hermes Agent.

When enabled, MCP and non-core plugin tools are replaced in the model-visible
tools array by three bridge tools — ``tool_search``, ``tool_describe``,
``tool_call`` — and surfaced on demand. Core Hermes tools never defer.

Design constraints this module is built around:

* Core tools defined in ``toolsets._HERMES_CORE_TOOLS`` are *never* deferred.
  Always-load means always-load. No exceptions.
* The three bridge schemas are byte-stable and present whenever Tool Search is
  enabled, including when the current catalog is empty. MCP/plugin additions,
  removals, and schema edits therefore never mutate the model-facing ``tools=``
  prefix or invalidate its provider prompt cache.
* The catalog is resolved from the current session-scoped live registry on
  every search/describe/call. There is no catalog embedded in the bridge
  description and no session-keyed catalog that can drift out of sync. A
  compact fingerprinted manifest rides on the first real user turn and again
  after changes, preserving both capability discoverability and the cached
  ``tools=`` prefix.
* Bridge tools route through ``model_tools.handle_function_call`` exactly
  like a direct call, so guardrails, plugin pre/post hooks, approval flows,
  and tool-result truncation all fire identically.
* Display and trajectory unwrap is implemented here so the user (CLI activity
  feed, gateway, saved trajectories) always sees the underlying tool, not
  the bridge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.registry import tool_error

logger = logging.getLogger("tools.tool_search")


# Bridge tool names. These names are reserved and may not collide with a
# user/plugin/MCP tool — registration of any tool with these names is
# rejected by the registry's existing override-protection logic.
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"

BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})

# When reporting the schema tokens saved by progressive disclosure, use the
# cheap rule of thumb that's stable across providers: roughly 4 chars per
# token for English+JSON. This is observability only; it does not decide
# whether the stable bridge is present.
CHARS_PER_TOKEN = 4.0


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off"
    search_default_limit: int
    max_search_limit: int
    # The catalog manifest is delivered on a real user turn, never embedded in
    # the bridge schema. This preserves the listing-mode discoverability win
    # without making ``tools=`` dynamic.
    listing: str = "auto"  # "auto" | "on" | "off"
    threshold_pct: float = 5.0
    # Absolute cap on the user-turn catalog snapshot. Effective budget is the
    # smaller of this value and ``threshold_pct`` of the context window.
    listing_max_tokens: int = 4000

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
            return cls(enabled="auto", search_default_limit=5, max_search_limit=20)
        if raw is False:
            return cls(enabled="off", search_default_limit=5, max_search_limit=20)
        if not isinstance(raw, dict):
            return cls(enabled="auto", search_default_limit=5, max_search_limit=20)

        enabled_raw = str(raw.get("enabled", "auto")).strip().lower()
        if enabled_raw in ("true", "1", "yes"):
            enabled = "on"
        elif enabled_raw in ("false", "0", "no"):
            enabled = "off"
        elif enabled_raw in ("auto", "on", "off"):
            enabled = enabled_raw
        else:
            enabled = "auto"

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 20)))
        search_default_limit = max(1, min(max_search_limit,
                                          _safe_int(raw.get("search_default_limit"), 5)))

        listing_raw = str(raw.get("listing", "auto")).strip().lower()
        if listing_raw in ("true", "1", "yes"):
            listing = "on"
        elif listing_raw in ("false", "0", "no"):
            listing = "off"
        elif listing_raw in ("auto", "on", "off"):
            listing = listing_raw
        else:
            listing = "auto"
        threshold_pct = max(
            0.0, min(100.0, _safe_float(raw.get("threshold_pct"), 5.0))
        )
        listing_max_tokens = max(
            200, min(60000, _safe_int(raw.get("listing_max_tokens"), 4000))
        )

        return cls(
            enabled=enabled,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            listing=listing,
            threshold_pct=threshold_pct,
            listing_max_tokens=listing_max_tokens,
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


def is_enabled_in_config(config: Any) -> bool:
    """Return whether a full Hermes config selects the stable bridge mode."""
    tools_cfg = config.get("tools") if isinstance(config, dict) else None
    raw = tools_cfg.get("tool_search") if isinstance(tools_cfg, dict) else None
    return ToolSearchConfig.from_raw(raw).enabled != "off"


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


def is_deferrable_tool_name(name: str) -> bool:
    """Return True if a tool with this name is *eligible* for deferral.

    A tool is deferrable iff it is registered with an MCP toolset prefix
    OR it is not in ``_HERMES_CORE_TOOLS``. Core tools are never deferred
    even when their toolset is technically plugin-provided (this protects
    against accidental shadowing).
    """
    if name in BRIDGE_TOOL_NAMES:
        return False
    if name in _core_tool_names():
        return False
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


def classify_tools(tool_defs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable).

    ``visible`` retains every tool that must stay in the model-facing array:
    every core tool, plus any tool we can't classify. ``deferrable`` is the
    candidate set for catalog entry.
    """
    visible: List[Dict[str, Any]] = []
    deferrable: List[Dict[str, Any]] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if name in BRIDGE_TOOL_NAMES:
            # Should never happen — bridge tools are added after classification —
            # but be defensive.
            continue
        if is_deferrable_tool_name(name):
            deferrable.append(td)
        else:
            visible.append(td)
    return visible, deferrable


# ---------------------------------------------------------------------------
# Token estimation and activation
# ---------------------------------------------------------------------------


def estimate_tokens_from_schemas(tool_defs: Iterable[Dict[str, Any]]) -> int:
    """Estimate the token cost of a tool-defs list via the chars/4 rule.

    Cheap and stable across providers. This value is used only for logs and
    assembly metadata, so order-of-magnitude precision is sufficient.
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

    ``"off"`` skips unconditionally. ``"on"`` and ``"auto"`` keep the stable
    bridge present even when the live catalog is empty. This avoids a cache
    boundary when the first MCP/plugin tool arrives or the last one disappears.
    The unused numeric arguments remain for compatibility with existing callers.
    """
    del deferrable_tokens, context_length
    return config.enabled != "off"


def listing_token_budget(
    config: ToolSearchConfig,
    context_length: Optional[int],
) -> int:
    """Return the budget for the user-turn catalog manifest.

    The manifest uses the former tiered-listing budget, but its bytes now live
    at the append-only edge of the conversation instead of inside ``tools=``.
    """
    if context_length and context_length > 0:
        pct_leg = int(context_length * (config.threshold_pct / 100.0))
    else:
        pct_leg = 10_000
    return max(0, min(config.listing_max_tokens, pct_leg))


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
# User-turn catalog manifest
# ---------------------------------------------------------------------------


_SENTENCE_END_RE = re.compile(r"[.!?\n]")
_CATALOG_SNAPSHOT_RE = re.compile(
    r"\[HERMES TOOL CATALOG SNAPSHOT id=([0-9a-f]{16})\b"
)


@dataclass(frozen=True)
class CatalogSnapshot:
    """A deterministic, session-scoped catalog snapshot for one user turn."""

    snapshot_id: str
    count: int
    pending_mcp_servers: Tuple[str, ...]
    listing_form: str
    notice: str


def _short_desc(description: str, max_chars: int = 60) -> str:
    """Return the first sentence of a tool description, clipped tersely."""
    text = " ".join((description or "").split())
    if not text:
        return ""
    match = _SENTENCE_END_RE.search(text)
    if match:
        text = text[:match.start() + (1 if text[match.start()] == "." else 0)]
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(",;: ") + "…"


def _listing_group_label(source_name: str) -> str:
    label = source_name or "other"
    return label[4:] if label.startswith("mcp-") else label


def build_catalog_listing(
    deferrable: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
) -> Optional[str]:
    """Render a skills-style manifest of the deferred catalog.

    One line per tool — ``name: short description`` — grouped under a
    heading per source (MCP server / plugin toolset), exactly like the
    bundled-skills listing in the system prompt:

        github tools: (44)
        - create_issue: Open a new issue in a GitHub repository.
        - merge_pull_request: Merge an open pull request.
        ...

    Ordering is deterministic (groups and tools sorted by name) so the
    rendered block is byte-stable across assemblies of the same catalog —
    this keeps the request prefix cacheable across turns.

    Token-budget fallbacks (cheap chars/4 estimate, same rule as the
    activation gate):
      1. full listing (names + short descriptions)
      2. names-only listing, still grouped
      3. server-level summary — one line per MCP server / plugin toolset
         (name + tool count), so the model always knows WHICH domains are
         reachable through the bridge even when per-tool names don't fit
      4. ``None`` — only when the summary itself exceeds the budget
    """
    text, _form = build_catalog_listing_with_form(deferrable, max_tokens=max_tokens)
    return text


def build_catalog_listing_with_form(
    deferrable: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
) -> Tuple[Optional[str], str]:
    """Render a deterministic, budgeted catalog manifest.

    The fallback order is full descriptions, names only, then per-server
    summaries. Degradation is per server so one huge MCP surface cannot hide
    the names of a small co-attached server.
    """
    if not deferrable:
        return None, "none"

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for tool_def in deferrable:
        fn = tool_def.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        label = _listing_group_label(
            source_name if source != "other" else "other"
        )
        groups.setdefault(label, []).append(
            (name, _short_desc(fn.get("description", "")))
        )

    if not groups:
        return None, "none"

    def render_group(label: str, mode: str) -> str:
        tools = sorted(groups[label])
        if mode == "summary":
            return (
                f"{label} ({len(tools)} tools — names not listed; "
                f"discover via `{TOOL_SEARCH_NAME}`)"
            )
        lines = [f"{label} tools ({len(tools)}):"]
        if mode == "full":
            lines.extend(
                f"- {name}: {desc}" if desc else f"- {name}"
                for name, desc in tools
            )
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    def assemble(modes: Dict[str, str]) -> str:
        return "\n".join(
            render_group(label, modes[label]) for label in sorted(groups)
        )

    def fits(text: str) -> bool:
        return math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens

    modes = {label: "full" for label in groups}
    rendered = assemble(modes)
    if fits(rendered):
        return rendered, "full"

    modes = {label: "names" for label in groups}
    rendered = assemble(modes)
    if fits(rendered):
        return rendered, "names"

    by_size = sorted(
        groups,
        key=lambda label: (-len(render_group(label, "names")), label),
    )
    for label in by_size:
        modes[label] = "summary"
        rendered = assemble(modes)
        if fits(rendered):
            form = (
                "groups"
                if all(mode == "summary" for mode in modes.values())
                else "mixed"
            )
            return rendered, form

    return None, "none"


def _catalog_snapshot_id(
    deferrable: List[Dict[str, Any]],
    listing: Optional[str],
    listing_form: str,
    pending_mcp_servers: Tuple[str, ...],
) -> str:
    """Fingerprint full schemas and their rendered manifest representation."""
    ordered = sorted(
        deferrable,
        key=lambda tool_def: (tool_def.get("function") or {}).get("name", ""),
    )
    try:
        payload = json.dumps(
            {
                "schemas": ordered,
                "listing": listing,
                "listing_form": listing_form,
                "pending_mcp_servers": pending_mcp_servers,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        payload = repr((ordered, listing, listing_form, pending_mcp_servers))
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def build_catalog_snapshot(
    current_tool_defs: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
    pending_mcp_servers: Iterable[str] = (),
) -> CatalogSnapshot:
    """Build the API-only catalog snapshot attached to a real user turn.

    Full schemas remain behind ``tool_describe``. The compact name/description
    manifest restores capability discoverability while keeping all dynamic
    catalog bytes out of the provider's model-facing ``tools=`` prefix.
    """
    _, deferrable = classify_tools(current_tool_defs)
    pending_servers = tuple(
        sorted(
            {
                str(server_name).strip()
                for server_name in pending_mcp_servers
                if str(server_name).strip()
            }
        )
    )
    # Reserve room for the framing and instructions around the manifest.
    listing, listing_form = build_catalog_listing_with_form(
        deferrable,
        max_tokens=max(0, max_tokens - 220),
    )
    snapshot_id = _catalog_snapshot_id(
        deferrable,
        listing,
        listing_form,
        pending_servers,
    )

    lines = [
        f"[HERMES TOOL CATALOG SNAPSHOT id={snapshot_id} "
        f"count={len(deferrable)} pending_servers={len(pending_servers)}",
        "Runtime metadata; this is not part of the user's request. This snapshot "
        "supersedes every earlier tool-catalog snapshot in the conversation.",
    ]
    if not deferrable:
        lines.append(
            "No deferred MCP/plugin tools are currently ready. The stable bridge "
            "remains present so later catalog changes do not alter tools=."
        )
    elif listing is None:
        lines.append(
            f"{len(deferrable)} deferred tools are available, but their manifest "
            f"does not fit the listing budget. Use `{TOOL_SEARCH_NAME}` before "
            "deciding that a capability is unavailable."
        )
    else:
        lines.append(
            "The deferred capabilities below are available. When an exact tool "
            f"name is visible, load its current schema with `{TOOL_DESCRIBE_NAME}`; "
            f"use `{TOOL_SEARCH_NAME}` for summarized servers or uncertain matches."
        )
        if listing_form == "mixed":
            lines.append(
                "Servers marked 'names not listed' also have available tools; "
                "search them before substituting terminal/browser or declaring "
                "the capability unavailable."
            )
        lines.append(listing)
    if pending_servers:
        rendered_servers = ", ".join(f"`{name.replace('`', '')}`" for name in pending_servers)
        lines.append(
            f"MCP servers still initializing ({len(pending_servers)}): "
            f"{rendered_servers}. Their tools are not included in the ready "
            "catalog yet. The live catalog may expand during this turn; use or "
            f"re-run `{TOOL_SEARCH_NAME}` before concluding that one of these "
            "capabilities is unavailable."
        )
    lines.append("END HERMES TOOL CATALOG SNAPSHOT]")
    return CatalogSnapshot(
        snapshot_id=snapshot_id,
        count=len(deferrable),
        pending_mcp_servers=pending_servers,
        listing_form=listing_form,
        notice="\n".join(lines),
    )


def catalog_snapshot_id_from_text(content: Any) -> Optional[str]:
    """Extract a Hermes-emitted snapshot id from a prior API message copy."""
    if isinstance(content, list):
        for part in reversed(content):
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            snapshot_id = catalog_snapshot_id_from_text(part.get("text"))
            if snapshot_id:
                return snapshot_id
        return None
    if not isinstance(content, str):
        return None
    match = _CATALOG_SNAPSHOT_RE.search(content)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Bridge tool schemas
# ---------------------------------------------------------------------------


def bridge_tool_schemas() -> List[Dict[str, Any]]:
    """Build the bridge tool schemas to inject in place of deferred tools.

    The schemas are intentionally short — every byte added here is a byte
    the user pays on every turn. Descriptions are tuned to be unambiguous
    about the call sequence the model should follow.

    They deliberately contain no catalog count, server list, tool name, or
    dynamic description. Discovery happens against the live registry when the
    model calls the bridge, so this return value must remain byte-identical as
    MCP/plugin tools change during a conversation.
    """
    desc_search = (
        "Search MCP and plugin tools that are discovered from the live catalog "
        "on demand. Returns up to ``limit`` matches with name and description. Follow "
        f"with `{TOOL_DESCRIBE_NAME}` to load a tool's full parameter schema, "
        f"then `{TOOL_CALL_NAME}` to invoke it. Tools exposed alongside this "
        "bridge are already available and do not need to be searched. "
        "Re-run this search after a tool-catalog update."
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
    # Kept for compatibility with telemetry/tests from the tiered-listing
    # implementation. Stable live discovery is always the search-only tier.
    tier: int = 0
    listing_form: str = "none"


def assemble_tool_defs(
    tool_defs: List[Dict[str, Any]],
    *,
    context_length: Optional[int] = None,
    config: Optional[ToolSearchConfig] = None,
) -> AssemblyResult:
    """Return the tool-defs list the model should actually see.

    When Tool Search is off, this is a passthrough. Otherwise MCP and plugin
    tools are stripped from the visible list and replaced with the same three
    byte-stable bridge tools. The bridge remains present for an empty catalog,
    preventing a 0→N or N→0 MCP change from mutating the request prefix. Core
    tools are *never* deferred regardless of config.

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

    visible, deferrable = classify_tools(incoming)
    deferrable_tokens = estimate_tokens_from_schemas(deferrable)
    if not should_activate(config, deferrable_tokens, context_length):
        return AssemblyResult(
            tool_defs=incoming,
            activated=False,
            deferred_count=len(deferrable),
            deferred_tokens=deferrable_tokens,
            tier=0,
        )

    result = visible + bridge_tool_schemas()

    logger.info(
        "tool_search active with stable bridge: %d core/visible tools kept, "
        "%d deferred (~%d tokens)",
        len(visible), len(deferrable), deferrable_tokens,
    )

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(deferrable),
        deferred_tokens=deferrable_tokens,
        tier=2,
        listing_form="none",
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


def _available_source_summary(catalog: List[CatalogEntry]) -> List[Dict[str, Any]]:
    """Return a compact, deterministic summary of connected deferred sources.

    Included only when search returns no matches. This gives the model enough
    evidence to retry with a source/action query instead of treating a lexical
    miss as proof that the capability is unavailable, without adding anything
    to the fixed per-turn prompt.
    """
    counts: Dict[str, int] = {}
    for entry in catalog:
        # _listing_group_label already falls back to "other" for empty
        # source names, matching the listing path's grouping.
        label = _listing_group_label(entry.source_name)
        counts[label] = counts.get(label, 0) + 1
    return [
        {"name": name, "tool_count": counts[name]}
        for name in sorted(counts)
    ]


def dispatch_tool_search(args: Dict[str, Any],
                         *,
                         current_tool_defs: List[Dict[str, Any]],
                         config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_search`` bridge tool. Returns a JSON string."""
    if config is None:
        config = load_config()
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")

    raw_limit = args.get("limit")
    if raw_limit is None:
        limit = config.search_default_limit
    else:
        limit = max(1, min(config.max_search_limit, _safe_int(raw_limit, config.search_default_limit)))

    _, deferrable = classify_tools(current_tool_defs)
    catalog = build_catalog(deferrable)
    hits = search_catalog(catalog, query, limit=limit)
    result: Dict[str, Any] = {
        "query": query,
        "total_available": len(catalog),
        "matches": [_format_search_hit(h) for h in hits],
    }
    if not hits and catalog:
        result["available_sources"] = _available_source_summary(catalog)
        result["hint"] = (
            "No lexical match was found, but the sources above are connected "
            "and their tools remain available. Retry tool_search with the "
            "service name plus a concrete action or object before concluding "
            "the capability is unavailable."
        )
    return json.dumps(result, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]]) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns a JSON string."""
    name = str(args.get("name") or "").strip()
    if not name:
        return tool_error("name is required")
    if not is_deferrable_tool_name(name):
        return tool_error(
            f"'{name}' is not a deferrable tool. If you see it in the tools list "
            "already, call it directly; otherwise check the spelling against tool_search."
        )
    _, deferrable = classify_tools(current_tool_defs)
    for td in deferrable:
        fn = td.get("function") or {}
        if fn.get("name") == name:
            return json.dumps({
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }, ensure_ascii=False)
    return tool_error(
        f"'{name}' is not currently available. Re-run tool_search to refresh."
    )


def scoped_deferrable_names(tool_defs: List[Dict[str, Any]]) -> frozenset[str]:
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
    for td in tool_defs:
        name = (td.get("function") or {}).get("name", "")
        if name and is_deferrable_tool_name(name):
            names.add(name)
    return frozenset(names)


def validate_deferred_call_args(name: str, args: Dict[str, Any]) -> Optional[str]:
    """Probe-validate ``tool_call`` arguments against the deferred tool's schema.

    A deferred tool's parameter schema is invisible to the model until it
    calls ``tool_describe`` — so models routinely invoke deferred tools
    "blind" by name alone, omitting required arguments. Dispatching such a
    call produces an opaque downstream failure (``KeyError: 'document_id'``)
    that tells the model nothing about what the tool expects, and cheap
    models loop on it until the iteration budget dies.

    Port of the describe-first probe-validation fix from nearai/ironclaw#5149:
    when required arguments are missing, return the tool's parameter schema
    instead of dispatching blind — the model repairs the call in one
    round-trip. Valid calls (and any call we can't confidently validate)
    dispatch untouched, so this can never block a legitimate invocation.

    Only *key absence* of schema-``required`` fields counts as invalid.
    No type checking, no null rejection — nullable/typed edge cases are the
    tool's own business, and ``coerce_tool_args`` already handles type repair
    downstream. Returns a JSON error string when invalid, ``None`` when the
    call should dispatch.
    """
    try:
        from tools.registry import registry as _registry
        schema = _registry.get_schema(name)
        if not isinstance(schema, dict):
            return None
        fn = schema.get("function") if schema.get("type") == "function" else schema
        if not isinstance(fn, dict):
            return None
        params = fn.get("parameters")
        if not isinstance(params, dict):
            return None
        required = params.get("required")
        if not isinstance(required, list) or not required:
            return None
        missing = [r for r in required if isinstance(r, str) and r not in args]
        if not missing:
            return None
        return tool_error(
            f"tool_call to '{name}' is missing required argument(s): "
            f"{', '.join(missing)}. The tool was NOT invoked.",
            parameters=params,
            hint=(
                "Retry tool_call with 'arguments' matching the parameters "
                "schema above."
            ),
        )
    except Exception:  # pragma: no cover — never block dispatch on validator bugs
        logger.debug("validate_deferred_call_args failed for %s", name, exc_info=True)
        return None


def resolve_underlying_call(args: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
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
    if not is_deferrable_tool_name(name):
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
    "CatalogSnapshot",
    "AssemblyResult",
    "load_config",
    "is_enabled_in_config",
    "is_deferrable_tool_name",
    "classify_tools",
    "estimate_tokens_from_schemas",
    "should_activate",
    "listing_token_budget",
    "build_catalog",
    "search_catalog",
    "build_catalog_listing",
    "build_catalog_listing_with_form",
    "build_catalog_snapshot",
    "catalog_snapshot_id_from_text",
    "bridge_tool_schemas",
    "assemble_tool_defs",
    "is_bridge_tool",
    "dispatch_tool_search",
    "dispatch_tool_describe",
    "resolve_underlying_call",
    "scoped_deferrable_names",
    "validate_deferred_call_args",
]
