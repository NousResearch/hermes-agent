"""Progressive tool disclosure ("tool search") for Hermes Agent.

When enabled, MCP and non-core plugin tools are replaced in the model-visible
tools array by three bridge tools — ``tool_search``, ``tool_describe``,
``tool_call`` — and surfaced on demand. Core Hermes tools never defer.

Design constraints this module is built around (see ``openclaw-tool-search-report``
for the full rationale):

* Core tools defined in ``toolsets._HERMES_CORE_TOOLS`` are *never* deferred.
  Always-load means always-load. No exceptions.
* Session-gated GUI toolsets (``desktop_ui``, ``project``) are also never
  deferred. They stay off the core list so CLI and messaging never pay for
  their schemas, but once a session enables them they stay in the
  model-facing array. Tool Search is for MCP/plugin catalog bloat, not for
  hiding the tools that define this session's surface.
* Tiered disclosure (July 2026 plan): the moment ANY deferrable (MCP/plugin)
  tools are present, they hide behind the bridge. What scales with catalog
  size is the *listing*, not the activation decision:
    - Tier 0 — no MCP/plugin tools: pure passthrough, everything eager.
    - Tier 1 — deferred tools whose catalog listing fits the listing budget
      (``min(threshold_pct`` of context — default 5% — ``, listing_max_tokens)``):
      bridge + skills-style listing (name + short description per tool),
      degrading to a names-only listing when the full form is over budget.
    - Tier 2 — per-tool listing over budget even names-only (e.g.
      Cloudflare's flat API surface, ~3,300 tools whose names alone are
      ~32K tokens): bare bridge + a one-line-per-server summary (server
      name + tool count) so the model still knows WHICH domains are
      reachable; individual tools are discoverable only via ``tool_search``.
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

import copy
import functools
import json
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import snowballstemmer

from tools.registry import tool_error

logger = logging.getLogger("tools.tool_search")

_SCHEMA_LITERAL_KEYS = frozenset({"const", "default", "enum", "example", "examples"})


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

# Bound the work one tool_search bridge call can request.
_MAX_QUERIES_PER_CALL = 10
# Bound the work one tool_describe bridge call can request.
_MAX_DESCRIBE_NAMES_PER_CALL = 10


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off" — "auto" is an alias of "on" today
    # Listing budget as a percentage of the model's context window. Under
    # tiered disclosure this no longer gates *activation* (any deferrable
    # tool activates the bridge) — it bounds how much context the embedded
    # catalog listing may consume before disclosure degrades:
    # full listing -> names-only -> bare bridge (tier 2).
    threshold_pct: float  # 0..100
    search_default_limit: int
    max_search_limit: int
    # Catalog listing ("skills-style" progressive disclosure): when active,
    # a grouped name + short-description manifest of every deferred tool is
    # embedded in the tool_search bridge description, so capabilities stay
    # DISCOVERABLE (like the skills listing in the system prompt) while full
    # schemas stay deferred.  "auto" = include when it fits the listing
    # budget (falls back to names-only, then to none = bare bridge);
    # "on" = same rendering, explicit intent; "off" = always bare bridge.
    listing: str = "auto"  # "auto" | "on" | "off"
    # Absolute cap on the embedded listing, regardless of context size.
    # Effective budget = min(listing_max_tokens, threshold_pct% of context).
    listing_max_tokens: int = 4000
    # Core/GUI tool names deferred behind the bridge. None = use the curated
    # default (_DEFAULT_DEFERRED_TOOLS); an explicit list from config
    # replaces the default wholesale ([] = defer no core tools — legacy).
    defer_tools: Optional[frozenset] = None

    @property
    def effective_defer_tools(self) -> frozenset:
        return _DEFAULT_DEFERRED_TOOLS if self.defer_tools is None else self.defer_tools

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
            return cls(enabled="auto", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=25)
        if raw is False:
            return cls(enabled="off", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=25)
        if not isinstance(raw, dict):
            return cls(enabled="auto", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=25)

        enabled_raw = str(raw.get("enabled", "auto")).strip().lower()
        if enabled_raw in ("true", "1", "yes"):
            enabled = "on"
        elif enabled_raw in ("false", "0", "no"):
            enabled = "off"
        elif enabled_raw in ("auto", "on", "off"):
            enabled = enabled_raw
        else:
            enabled = "auto"

        threshold_pct = _safe_float(raw.get("threshold_pct"), 5.0)
        threshold_pct = max(0.0, min(100.0, threshold_pct))

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 25)))
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
        listing_max_tokens = max(200, min(60000, _safe_int(raw.get("listing_max_tokens"), 4000)))

        defer_raw = raw.get("defer")
        if isinstance(defer_raw, (list, tuple, set)):
            defer_tools = frozenset(
                str(n).strip() for n in defer_raw if str(n).strip()
            )
        else:
            defer_tools = None  # curated default

        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            listing=listing,
            listing_max_tokens=listing_max_tokens,
            defer_tools=defer_tools,
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


def load_config_readonly() -> ToolSearchConfig:
    """Load tool-search config without copying the cached full config."""
    try:
        from hermes_cli.config import load_config_readonly as _load
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


# Session-gated GUI toolsets. Off ``_HERMES_CORE_TOOLS`` so non-GUI clients
# never pay their schema; once a session enables them they stay direct
# UNLESS the deferral list (below) names them.
_DIRECT_SURFACE_TOOLSETS = frozenset({"desktop_ui", "project"})

# Core-tool deferral (2026-08, maintainer-directed): the curated set of
# event-triggered tools that hide behind the bridge BY DEFAULT. These are
# tools a session reaches for when something specific happens (user asks
# for a tour / a cron job / a screenshot / a clarification), not tools in
# the every-turn working set — so a catalog stub is enough to find them.
# Config override: ``tools.tool_search.defer`` (list of tool names);
# ``[]`` restores the legacy everything-eager behavior, any other list
# replaces this default wholesale. Names here are POST-rename.
#
# ``clarify`` was in the original curated set but was pulled back to eager
# after the maintainer A/B (PR #97979, 288 runs × 3 model tiers): with the
# schema visible models used structured clarify 18/18 on ambiguous tasks;
# deferred, usage collapsed to 7/18 (gpt-terra 0/6) — models fell back to
# plain-text questions, losing the structured-choice UX and costing an
# extra user round-trip. The ask-the-user affordance has to be ambient to
# fire; a catalog stub is not enough. (~250 tok to keep it eager.)
_DEFAULT_DEFERRED_TOOLS = frozenset({
    "computer_use", "session_search", "image_generate",
    "todo_list", "process_manage", "cronjob_manage",
    # Desktop GUI surface (desktop_ui + project toolsets)
    "drive_preview", "gui_tour", "desktop_preview", "annotate_preview",
    "show_tip", "setup_mcp", "desktop_project", "close_terminal",
    "apply_layout", "read_terminal", "read_window_below", "focus_pane",
})


def is_deferrable_tool_name(name: str, defer_tools: Optional[frozenset] = None) -> bool:
    """Return True if a tool with this name is *eligible* for deferral.

    A tool is deferrable iff:
    * it is named in ``defer_tools`` (the maintainer-curated core-deferral
      set, or the user's ``tools.tool_search.defer`` override) — this is
      the 2026-08 revision of the old "core never defers" rule: core tools
      in the WORKING set (terminal, files, memory, ...) still never defer,
      but the curated event-triggered set (computer_use, the GUI
      surface, ...) hides behind the bridge by default; OR
    * it is registered with an MCP toolset prefix; OR
    * it is neither in ``_HERMES_CORE_TOOLS`` nor a session-gated GUI
      surface toolset (plugin tools).
    """
    if name in BRIDGE_TOOL_NAMES:
        return False
    if defer_tools is not None and name in defer_tools:
        return True
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
        if entry.toolset in _DIRECT_SURFACE_TOOLSETS:
            return False
        # Non-MCP, non-core → plugin tool, eligible.
        return True
    except Exception:
        return False


def _describe_classification(
    name: str,
    defer_tools: Optional[frozenset] = None,
) -> Literal["available", "not_found", "not_deferrable"]:
    """Classify a describe name without treating unknown names as errors."""
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
    except Exception:
        return "not_found"
    if entry is None:
        return "not_found"
    if defer_tools is not None and name in defer_tools:
        return "available"
    if (
        name in BRIDGE_TOOL_NAMES
        or name in _core_tool_names()
        or entry.toolset in _DIRECT_SURFACE_TOOLSETS
    ):
        return "not_deferrable"
    return "available"


def classify_tools(
    tool_defs: List[Dict[str, Any]],
    defer_tools: Optional[frozenset] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable).

    ``visible`` retains every tool that must stay in the model-facing array.
    ``deferrable`` is the candidate set for catalog entry — MCP/plugin tools
    plus any core/GUI tool named in ``defer_tools``.
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
        if is_deferrable_tool_name(name, defer_tools):
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

    ``"off"`` skips unconditionally. ``"on"`` and ``"auto"`` activate whenever
    at least one deferrable tool exists (there's no point swapping a no-op).

    ``"auto"`` is an ALIAS of ``"on"`` under tiered disclosure — it is kept
    as the shipped default so that a future budget-gated mode ("inline the
    schemas when they fit, defer only when they don't") can change ``auto``'s
    behavior without breaking users who explicitly pinned ``on`` or ``off``.
    Do not add behavior that distinguishes them without that design; see the
    config reference for the user-facing statement of this contract.

    Tiered-disclosure semantics (July 2026): the presence of ANY MCP/plugin
    tool activates the bridge — schemas always defer. What the threshold now
    controls is the *listing budget* (see :func:`listing_token_budget`), not
    activation. ``context_length`` is retained in the signature for
    backward compatibility with existing callers.
    """
    if config.enabled == "off":
        return False
    if deferrable_tokens <= 0:
        return False
    return True


def listing_token_budget(
    config: ToolSearchConfig,
    context_length: Optional[int],
) -> int:
    """Effective token budget for the embedded catalog listing.

    ``min(listing_max_tokens, threshold_pct% of context)``. Without a known
    context size, the percentage leg falls back to a fixed 10K cutoff
    (5% of a typical 200K window).
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


# ASCII alphanumeric runs, plus individual CJK characters.  CJK languages
# (Chinese, Japanese, Korean) don't use spaces between words, so each
# character is its own token — the standard lightweight approach for BM25
# without pulling in a segmentation dependency (#78985).
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+"
    r"|[\u4e00-\u9fff\u3400-\u4dbf]"   # CJK Unified Ideographs (+ Ext A)
    r"|[\u3040-\u309f\u30a0-\u30ff]"   # Hiragana + Katakana
    r"|[\uac00-\ud7af]"                # Hangul Syllables
)

# Snowball stemmer instances keep mutable parsing state, so they are not
# safe to share across threads — and bridge dispatch can run on parallel
# tool-call threads. One stemmer per thread, created lazily.
_thread_local = threading.local()


def _stemmer() -> Any:
    st = getattr(_thread_local, "stemmer", None)
    if st is None:
        st = snowballstemmer.stemmer("english")
        _thread_local.stemmer = st
    return st


@functools.lru_cache(maxsize=16384)
def _stem(token: str) -> str:
    """Stem one token, memoized across stateless catalog rebuilds."""
    return _stemmer().stemWord(token)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, Snowball-stemmed (English).

    Stemming is applied here so it hits BOTH the index path
    (:func:`build_catalog` via :func:`_entry_search_text`) and the query
    path (:func:`search_catalog`) identically — a query for "issues"
    matches a tool named ``create_issue``.
    """
    if not text:
        return []
    return [_stem(token.lower()) for token in _TOKEN_RE.findall(text)]


def _entry_search_text(td: Dict[str, Any], source_label: str = "") -> str:
    """Build the search-text blob for a deferrable tool.

    Includes the tool name (with underscores broken into words so BM25 can
    match against query terms), the source label (the MCP server / plugin
    toolset the tool belongs to, e.g. ``linear`` for toolset ``mcp-linear``),
    the description, and the names of the top-level parameters. Schema
    bodies are deliberately excluded — indexing them adds noise without
    improving recall in our measurement.

    The ``mcp__`` name prefix is stripped before splitting: ``mcp`` appears
    in every native MCP tool document, so its IDF collapses to near zero —
    it is dead weight in every document and useless as a query term.
    Indexing the source label is what makes a service-name query ("linear")
    reach a tool whose NAME does not carry the service (a plugin tool named
    ``create_issue``, or any catalog whose naming omits the vendor).
    """
    fn = td.get("function") or {}
    name = fn.get("name", "")
    if name.startswith("mcp__"):
        name = name[len("mcp__"):]
    desc = fn.get("description", "") or ""
    params = ((fn.get("parameters") or {}).get("properties") or {})
    param_names = " ".join(params.keys())
    # Break snake_case and dotted names into words for BM25.
    name_words = name.replace("_", " ").replace(".", " ").replace("-", " ").replace(":", " ")
    extra = source_label if source_label and source_label not in name_words.split() else ""
    return f"{name_words} {extra} {desc} {param_names}"


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
        # Index the human-facing group label ("linear", not "mcp-linear") so
        # a service-name query matches tools from that source even when the
        # tool's own name omits the service.
        source_label = _listing_group_label(source_name) if source_name else ""
        entry = CatalogEntry(
            name=name,
            description=desc,
            schema=td,
            source=source,
            source_name=source_name,
            _tokens=_tokenize(_entry_search_text(td, source_label)),
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


_CorpusStats = Tuple[List[int], float, Dict[str, int], int]


def _corpus_stats(catalog: List[CatalogEntry]) -> _CorpusStats:
    """Compute the BM25 statistics shared by every query over a catalog."""
    doc_lengths = [len(entry._tokens) for entry in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq: Dict[str, int] = {}
    for entry in catalog:
        for token in set(entry._tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return doc_lengths, avg_dl, doc_freq, len(catalog)


def search_catalog(
    catalog: List[CatalogEntry],
    query: str,
    limit: int = 5,
    *,
    corpus_stats: Optional[_CorpusStats] = None,
) -> List[CatalogEntry]:
    """Return the top-``limit`` catalog entries for ``query`` by BM25.

    Falls back to a stable name-substring match when every query token
    misses every document — e.g. the query ``"hub"`` against ``github_*``
    tools ("hub" is a substring of the name but never a token, so BM25
    scores nothing). The IDF variant used here,
    ``log(1 + (N - df + 0.5) / (df + 0.5))``, is strictly positive even
    when a term appears in every document, so the fallback only runs when
    no query token appears in any document.
    """
    if not catalog or limit <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    if corpus_stats is None:
        corpus_stats = _corpus_stats(catalog)
    doc_lengths, avg_dl, doc_freq, n_docs = corpus_stats

    scored: List[Tuple[float, CatalogEntry]] = []
    exact_name = query.strip().lower()
    for entry in catalog:
        if entry.name.lower() == exact_name:
            scored.append((float("inf"), entry))
            continue
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


# A sentence ends at ., !, or ? followed by whitespace or end-of-string, but
# not at the end of a common dotted abbreviation.
_SENTENCE_END_RE = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?](?=\s|$)")


def _short_desc(description: str, max_chars: int = 60) -> str:
    """First sentence of a tool description, clipped to ``max_chars``.

    A terminator must be followed by whitespace or end-of-string; ``e.g.``,
    ``i.e.``, and ``etc.`` do not end a sentence. Whitespace normalization and
    the unbounded regex search both remain linear-time on hostile input.
    """
    text = " ".join((description or "").split())
    if not text:
        return ""
    m = _SENTENCE_END_RE.search(text)
    if m:
        text = text[:m.end()]
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(",;: ") + "…"


def _listing_group_label(source_name: str) -> str:
    """Human-facing group heading for a toolset, e.g. ``mcp-github`` -> ``github``."""
    label = source_name or "other"
    if label.startswith("mcp-"):
        label = label[4:]
    return label


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
    """Like :func:`build_catalog_listing` but also reports the form used.

    Returns ``(text, form)`` where ``form`` is ``"full"`` (names + short
    descriptions), ``"names"`` (names-only fallback), ``"mixed"`` (per-server
    degradation: small servers keep per-tool lines, oversized servers
    collapse to a name + tool-count summary line), ``"groups"`` (every
    server summarized), or ``"none"`` (over budget in every form).

    Degradation is PER SERVER, not global: one huge server (Cloudflare's
    3,320 flat tools) must not cost a small co-attached server (Linear's 24)
    its listing. Greedy fit, smallest rendered group first, is deterministic
    for a given catalog — byte-stable across assemblies, cache-safe.
    """
    if not deferrable:
        return None, "none"

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for td in deferrable:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        label = _listing_group_label(source_name if source != "other" else "other")
        groups.setdefault(label, []).append((name, _short_desc(fn.get("description", ""))))

    if not groups:
        return None, "none"

    def render_group(label: str, mode: str) -> str:
        """Render one server's block. mode: 'full' | 'names' | 'summary'."""
        tools = sorted(groups[label])
        if mode == "summary":
            return (f"{label} ({len(tools)} tools — names not listed; "
                    f"discover via `{TOOL_SEARCH_NAME}`)")
        lines = [f"{label} tools ({len(tools)}):"]
        if mode == "full":
            for name, desc in tools:
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    header = ("Deferred tool catalog (call schemas via "
              f"`{TOOL_DESCRIBE_NAME}`, invoke via `{TOOL_CALL_NAME}`):")

    def assemble(modes: Dict[str, str]) -> str:
        return "\n".join([header] + [render_group(lbl, modes[lbl])
                                     for lbl in sorted(groups)])

    def fits(text: str) -> bool:
        return math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens

    # 1. Everything full.
    modes = {lbl: "full" for lbl in groups}
    if fits(assemble(modes)):
        return assemble(modes), "full"

    # 2. Everything names-only.
    modes = {lbl: "names" for lbl in groups}
    if fits(assemble(modes)):
        return assemble(modes), "names"

    # 3. Per-server degradation: collapse the LARGEST rendered groups to
    #    summary lines first, keeping per-tool names for small servers.
    #    Deterministic: size then label. One oversized server (Cloudflare)
    #    must not cost a small co-attached server (Linear) its listing.
    by_size = sorted(groups, key=lambda lbl: (-len(render_group(lbl, "names")), lbl))
    for lbl in by_size:
        modes[lbl] = "summary"
        if fits(assemble(modes)):
            form = "groups" if all(m == "summary" for m in modes.values()) else "mixed"
            return assemble(modes), form

    # 4. Even the all-summary form is over budget.
    return None, "none"


def bridge_tool_schemas(
    deferred_count: int,
    listing: Optional[str] = None,
    listing_form: str = "",
) -> List[Dict[str, Any]]:
    """Build the bridge tool schemas to inject in place of deferred tools.

    The schemas are intentionally short — every byte added here is a byte
    the user pays on every turn. Descriptions are tuned to be unambiguous
    about the call sequence the model should follow.

    When ``listing`` is provided (see :func:`build_catalog_listing`), it is
    embedded in the ``tool_search`` description so every deferred capability
    stays *visible* by name — the skills-listing pattern — closing the
    "model doesn't know what it doesn't know" gap while full parameter
    schemas remain deferred. ``listing_form`` selects the framing: per-tool
    forms ("full"/"names") tell the model it may skip the search when it
    sees the exact name; the server-summary form ("groups") tells it which
    DOMAINS are reachable and that search is mandatory for tool discovery.
    """
    desc_search = (
        f"Search {deferred_count} additional tools that are loaded on demand. "
        "Takes a list of queries searched in parallel against the same "
        "catalog; send one query per distinct capability you need. Returns "
        "matching tool names grouped per query plus a shared map with each "
        "tool's description. Follow with "
        f"`{TOOL_DESCRIBE_NAME}` to load full parameter schemas, "
        f"then `{TOOL_CALL_NAME}` to invoke. Tools listed at the top of this "
        "system prompt are already available and do not need to be searched."
    )
    if listing and listing_form == "groups":
        desc_search += (
            "\n\nThe servers below are connected and their tools ARE available "
            "through this bridge. For any request in these domains, search "
            "here FIRST — do not claim the capability is unavailable and do "
            "not substitute a generic tool (terminal/browser) without "
            "searching.\n\n" + listing
        )
    elif listing:
        desc_search += (
            "\n\nEvery deferred capability is listed below. If a tool name "
            "appears here, do NOT claim it is unavailable — load it with "
            f"`{TOOL_DESCRIBE_NAME}` (skip `{TOOL_SEARCH_NAME}` when you "
            "already see the exact name)."
        )
        if listing_form == "mixed":
            desc_search += (
                " For servers marked 'names not listed', the tools exist "
                f"too — find them with `{TOOL_SEARCH_NAME}` before "
                "concluding anything is missing."
            )
        desc_search += "\n\n" + listing
    desc_describe = (
        f"Load the full JSON schemas for tools returned by `{TOOL_SEARCH_NAME}`. "
        f"Required before `{TOOL_CALL_NAME}` if a tool's parameters are unknown. "
        "Batch every schema you need into one call."
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
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries, each a few keywords describing one capability (e.g. ['create github issue', 'send slack message']). Searched in parallel; results come back grouped per query. A single string is accepted and treated as one query.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of matches per query. Defaults to 5 and is clamped to the configured maximum (25 by default).",
                        },
                    },
                    "required": ["queries"],
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
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact tool names (as returned by tool_search). A single string is accepted and treated as one name.",
                        },
                    },
                    "required": ["names"],
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
    # Disclosure tier actually applied:
    #   0 = passthrough (no deferrable tools, or tool_search off)
    #   1 = bridge + catalog listing (full or names-only)
    #   2 = bare bridge — catalog too large for any listing form
    tier: int = 0
    listing_form: str = "none"  # "full" | "names" | "none"


def assemble_tool_defs(
    tool_defs: List[Dict[str, Any]],
    *,
    context_length: Optional[int] = None,
    config: Optional[ToolSearchConfig] = None,
) -> AssemblyResult:
    """Return the tool-defs list the model should actually see.

    When tool search is inactive (off, no deferrable tools, or below
    threshold), this is a passthrough. When active, MCP and plugin tools
    are stripped from the visible list and replaced with the three bridge
    tools. Core tools are *never* deferred regardless of config.

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

    visible, deferrable = classify_tools(incoming, config.effective_defer_tools)
    if not deferrable:
        return AssemblyResult(tool_defs=incoming, activated=False)

    deferrable_tokens = estimate_tokens_from_schemas(deferrable)
    if not should_activate(config, deferrable_tokens, context_length):
        return AssemblyResult(
            tool_defs=incoming,
            activated=False,
            deferred_count=len(deferrable),
            deferred_tokens=deferrable_tokens,
            threshold_tokens=int((context_length or 0) * (config.threshold_pct / 100.0)),
            tier=0,
        )

    listing = None
    listing_form = "none"
    listing_budget = listing_token_budget(config, context_length)
    if config.listing != "off":
        listing, listing_form = build_catalog_listing_with_form(
            deferrable, max_tokens=listing_budget)
    bridge = bridge_tool_schemas(len(deferrable), listing=listing,
                                 listing_form=listing_form)
    result = visible + bridge
    # Tier 1 = per-tool listing for at least part of the catalog (full,
    # names, or mixed). Tier 2 = search-only discovery; the server-level
    # "groups" summary keeps domains visible but individual tools are only
    # reachable via tool_search.
    tier = 1 if listing_form in ("full", "names", "mixed") else 2

    logger.info(
        "tool_search activated (tier %d): %d core/visible tools kept, %d deferred "
        "(~%d tokens), listing %s (budget ~%d tokens)",
        tier, len(visible), len(deferrable), deferrable_tokens,
        listing_form, listing_budget,
    )

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(deferrable),
        deferred_tokens=deferrable_tokens,
        threshold_tokens=listing_budget,
        tier=tier,
        listing_form=listing_form,
    )


# ---------------------------------------------------------------------------
# Bridge tool dispatch
# ---------------------------------------------------------------------------


def is_bridge_tool(name: str) -> bool:
    return name in BRIDGE_TOOL_NAMES


def _shared_tool_record(entry: CatalogEntry) -> Dict[str, Any]:
    """One record for the response's shared ``tools`` map.

    Held once per tool no matter how many query groups matched it — the
    per-query groups carry names only. ``required`` lists the schema's
    required parameter names so the model can attempt a call without a
    ``tool_describe`` round-trip when the required surface is trivial.
    """
    schema = entry.schema if isinstance(entry.schema, dict) else {}
    fn = schema.get("function")
    if not isinstance(fn, dict):
        fn = {}
    params = fn.get("parameters")
    if not isinstance(params, dict):
        params = {}
    required = params.get("required")
    if not isinstance(required, list):
        required = []
    return {
        "source": entry.source,
        "source_name": entry.source_name,
        # Cap description so a chatty MCP server doesn't blow up the result.
        "description": (entry.description or "")[:400],
        "required": [r[:64] for r in required if isinstance(r, str)][:32],
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
    """Execute the ``tool_search`` bridge tool. Returns a JSON string.

    Accepts ``queries: [str, ...]`` — each query is searched independently
    against the same catalog. The response groups matching tool NAMES per
    query and carries each matched tool's record exactly once in a shared
    ``tools`` map::

        {
          "queries": ["...", "..."],
          "total_available": 215,
          "results": [{"query": "...", "matches": ["<tool name>", ...]}, ...],
          "tools": {"<tool name>": {"source": ..., "source_name": ...,
                                     "description": ..., "required": [...]}}
        }

    ``limit`` applies PER QUERY. Each query group that returns no matches gets
    an ``available_sources`` + ``hint`` block so a lexical miss is not mistaken
    for a missing capability.
    """
    if config is None:
        config = load_config()

    raw_queries = args.get("queries")
    if isinstance(raw_queries, str):
        # A bare string is an understandable model slip; treat as one query.
        raw_queries = [raw_queries]
    if not isinstance(raw_queries, list):
        return tool_error("queries is required and must be an array of strings")
    queries = [str(q).strip() for q in raw_queries if str(q or "").strip()]
    if not queries:
        return tool_error("queries is required and must contain at least one non-empty string")
    if len(queries) > _MAX_QUERIES_PER_CALL:
        return tool_error(
            f"too many queries: {len(queries)} > max {_MAX_QUERIES_PER_CALL}. "
            "Retry with fewer, more targeted queries."
        )

    raw_limit = args.get("limit")
    if raw_limit is None:
        limit = config.search_default_limit
    else:
        limit = max(1, min(config.max_search_limit, _safe_int(raw_limit, config.search_default_limit)))

    _, deferrable = classify_tools(
        current_tool_defs, load_config_readonly().effective_defer_tools
    )
    catalog = build_catalog(deferrable)

    results: List[Dict[str, Any]] = []
    tools_map: Dict[str, Dict[str, Any]] = {}
    corpus_stats = _corpus_stats(catalog)
    available_sources = _available_source_summary(catalog) if catalog else []
    for query in queries:
        hits = search_catalog(catalog, query, limit=limit, corpus_stats=corpus_stats)
        for h in hits:
            if h.name not in tools_map:
                tools_map[h.name] = _shared_tool_record(h)
        group: Dict[str, Any] = {"query": query, "matches": [h.name for h in hits]}
        if not hits and catalog:
            group["available_sources"] = available_sources
            group["hint"] = (
                "This query returned no lexical matches, but the sources above "
                "are connected and their tools remain available. Retry "
                "tool_search with the service name plus a concrete action or "
                "object before concluding the capability is unavailable."
            )
        results.append(group)

    result: Dict[str, Any] = {
        "queries": queries,
        "total_available": len(catalog),
        "results": results,
        "tools": tools_map,
    }
    return json.dumps(result, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]],
                           config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns a JSON string.

    Accepts ``names: [str, ...]`` and returns a map keyed by tool name::

        {
          "tools": {"<name>": {"description": ..., "parameters": {...}}, ...},
          "not_found": ["<name>", ...],   # only when some names missed
          "errors": {"<name>": "..."}     # only for non-deferrable names
        }

    Unknown/unregistered names and registered deferrable names absent from the
    current assembly land in ``not_found`` instead of failing the whole call.
    Registered non-deferrable names keep their per-name message in ``errors``.
    Duplicates are deduped silently.
    """
    if config is None:
        config = load_config_readonly()

    raw_names = args.get("names")
    if isinstance(raw_names, str):
        # A bare string is an understandable model slip; treat as one name.
        raw_names = [raw_names]
    if not isinstance(raw_names, list):
        return tool_error("names is required and must be an array of strings")
    names: List[str] = []
    for n in raw_names:
        n = str(n or "").strip()
        if n and n not in names:
            names.append(n)
    if not names:
        return tool_error("names is required and must contain at least one non-empty string")
    if len(names) > _MAX_DESCRIBE_NAMES_PER_CALL:
        return tool_error(
            f"too many names: {len(names)} > max {_MAX_DESCRIBE_NAMES_PER_CALL}. "
            "Retry with fewer names per call."
        )

    _, deferrable = classify_tools(
        current_tool_defs, load_config_readonly().effective_defer_tools
    )
    by_name: Dict[str, Dict[str, Any]] = {}
    for td in deferrable:
        fn = td.get("function") or {}
        if fn.get("name"):
            by_name[fn["name"]] = fn

    tools: Dict[str, Dict[str, Any]] = {}
    not_found: List[str] = []
    errors: Dict[str, str] = {}
    for name in names:
        fn = by_name.get(name)
        if fn is not None:
            tools[name] = {
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        elif _describe_classification(
            name, load_config_readonly().effective_defer_tools
        ) == "not_deferrable":
            errors[name] = (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search."
            )
        else:
            not_found.append(name)

    result: Dict[str, Any] = {"tools": tools}
    if not_found:
        result["not_found"] = not_found
        result["hint"] = "Names in not_found are not currently available. Re-run tool_search to refresh."
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


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
    defer_tools = load_config_readonly().effective_defer_tools
    for td in tool_defs:
        name = (td.get("function") or {}).get("name", "")
        if name and is_deferrable_tool_name(name, defer_tools):
            names.add(name)
    return frozenset(names)


def _schema_for_local_validation(node: Any) -> Any:
    """Return a JSON-Schema-compatible copy that honors ``nullable: true``.

    Some MCP/plugin schemas use OpenAPI's ``nullable`` extension instead of a
    JSON Schema null union.  Hermes' normal coercion path accepts that shape;
    mirror it here so local validation never rejects a value dispatch would
    intentionally accept.
    """
    if isinstance(node, list):
        return [_schema_for_local_validation(item) for item in node]
    if not isinstance(node, dict):
        return node

    normalized = {}
    for key, value in node.items():
        if key == "nullable":
            continue
        # These keywords contain instance data, not nested schemas. An enum
        # value such as {"nullable": true} must remain byte-for-byte data.
        normalized[key] = (
            copy.deepcopy(value)
            if key in _SCHEMA_LITERAL_KEYS
            else _schema_for_local_validation(value)
        )
    if node.get("nullable") is not True:
        return normalized

    schema_type = normalized.get("type")
    if isinstance(schema_type, str):
        if schema_type != "null":
            normalized["type"] = [schema_type, "null"]
        return normalized
    if isinstance(schema_type, list):
        if "null" not in schema_type:
            normalized["type"] = [*schema_type, "null"]
        return normalized

    # ``nullable`` alongside a $ref/combinator has no ``type`` to extend.
    # Wrap the original constraint so local references keep resolving from the
    # parameters schema's root while null remains an explicit alternative.
    return {"anyOf": [normalized, {"type": "null"}]}


def _schema_has_external_ref(node: Any) -> bool:
    """Return whether *node* contains a non-local ``$ref``.

    Local validation must never turn a tool call into an implicit network
    fetch.  Schemas with remote/file references remain the underlying tool's
    responsibility and therefore follow the existing fail-open contract.
    """
    if isinstance(node, list):
        return any(_schema_has_external_ref(item) for item in node)
    if not isinstance(node, dict):
        return False
    ref = node.get("$ref")
    if isinstance(ref, str) and not ref.startswith("#"):
        return True
    return any(
        _schema_has_external_ref(value)
        for key, value in node.items()
        if key not in _SCHEMA_LITERAL_KEYS
    )


def _validation_path(error: Any) -> str:
    """Format a jsonschema error path as a compact argument path."""
    path = "arguments"
    for part in getattr(error, "absolute_path", ()):
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            path += f".{part}"
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path


def validate_deferred_call_args(name: str, args: Dict[str, Any]) -> Optional[str]:
    """Validate ``tool_call`` arguments against the deferred tool's schema.

    A deferred tool's parameter schema is invisible to the model until it
    calls ``tool_describe`` — so models routinely invoke deferred tools
    "blind" by name alone, omitting required arguments. Dispatching such a
    call produces an opaque downstream failure (``KeyError: 'document_id'``)
    that tells the model nothing about what the tool expects, and cheap
    models loop on it until the iteration budget dies.

    Keep the original describe-first required-field probe from
    nearai/ironclaw#5149, then run the same schema-guided coercion used by
    normal dispatch and validate the repaired copy.  This restores the
    concrete-schema checks that the provider cannot perform through the
    generic ``arguments: object`` bridge.

    Missing/malformed schemas, unavailable validators, and external references
    fail open so validation cannot make a previously callable tool unavailable.
    Returns a JSON error string when invalid, ``None`` when the call should
    dispatch through the existing middleware/hook/approval pipeline.
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
        if isinstance(required, list) and required:
            missing = [r for r in required if isinstance(r, str) and r not in args]
            if missing:
                return tool_error(
                    f"tool_call to '{name}' is missing required argument(s): "
                    f"{', '.join(missing)}. The tool was NOT invoked.",
                    path="arguments",
                    constraint="required",
                    parameters=params,
                    hint=(
                        "Retry tool_call with 'arguments' matching the parameters "
                        "schema above."
                    ),
                )

        validation_schema = _schema_for_local_validation(params)
        if _schema_has_external_ref(validation_schema):
            logger.debug(
                "Skipping local deferred-argument validation for %s: external $ref",
                name,
            )
            return None

        # Validate the same repaired shape normal dispatch will receive. Work on
        # a copy because coerce_tool_args may normalize values in place; actual
        # dispatch performs the canonical coercion again after this probe.
        candidate_args = dict(args)
        try:
            from model_tools import coerce_tool_args
            candidate_args = coerce_tool_args(name, candidate_args)
        except Exception:
            logger.debug("Deferred-argument coercion failed for %s", name, exc_info=True)
            candidate_args = dict(args)

        try:
            from jsonschema.exceptions import best_match
            from jsonschema.validators import validator_for
        except ImportError:
            logger.debug(
                "jsonschema unavailable; keeping required-only validation for %s",
                name,
            )
            return None

        validator_cls = validator_for(validation_schema)
        validator_cls.check_schema(validation_schema)
        validation_error = best_match(
            validator_cls(validation_schema).iter_errors(candidate_args)
        )
        if validation_error is None:
            return None

        path = _validation_path(validation_error)
        constraint = str(getattr(validation_error, "validator", None) or "schema")
        detail = re.sub(r"\s+", " ", str(validation_error.message)).strip()
        if len(detail) > 600:
            detail = detail[:597] + "..."
        return tool_error(
            f"tool_call to '{name}' failed argument validation at {path} "
            f"({constraint}): {detail}. The tool was NOT invoked.",
            path=path,
            constraint=constraint,
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
    if not is_deferrable_tool_name(name, load_config_readonly().effective_defer_tools):
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
    "build_catalog",
    "build_catalog_listing",
    "build_catalog_listing_with_form",
    "listing_token_budget",
    "search_catalog",
    "bridge_tool_schemas",
    "assemble_tool_defs",
    "is_bridge_tool",
    "dispatch_tool_search",
    "dispatch_tool_describe",
    "resolve_underlying_call",
    "scoped_deferrable_names",
    "validate_deferred_call_args",
]
