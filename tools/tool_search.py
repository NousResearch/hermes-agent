"""Progressive tool disclosure ("tool search") for Hermes Agent.

When enabled, MCP and non-core plugin tools are replaced in the model-visible
tools array by three bridge tools — ``tool_search``, ``tool_describe``,
``tool_call`` — and surfaced on demand. Core Hermes tools never defer.

Design constraints this module is built around (see ``openclaw-tool-search-report``
for the full rationale):

* Core tools defined in ``toolsets._HERMES_CORE_TOOLS`` are *never* deferred.
  Always-load means always-load. No exceptions.
* The threshold gate runs every assembly: when deferrable tools would consume
  less than ``threshold_pct`` of the model's context window (default 10%),
  tool search is a no-op and the tools array passes through unchanged.
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

Optional enhancement (fulfils GitHub issue NousResearch/hermes-agent#13332
"Hybrid Tool Pre-Selection"):

* **Embedding reranker** (opt-in, needs an OpenAI-compatible embeddings
  endpoint) — reorders BM25 candidates by semantic similarity. Two modes:
  ``rerank`` (pure cosine, highest accuracy) and ``rrf`` (Reciprocal Rank
  Fusion of BM25 + embedding ranks, zero-regression safe mode). Enabled via
  ``tools.tool_search.reranker.enabled``. Default OFF; gracefully falls back
  to BM25 on any endpoint failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("tools.tool_search")

# Module-level lock guards the reranker singleton and the module-level
# embedding cache against torn writes under concurrent gateway requests.
# Fix HIGH-1: thread-safety for _RERANKER_SINGLETON and _EMBED_CACHE writes.
_GLOBAL_LOCK: threading.Lock = threading.Lock()

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


# ---------------------------------------------------------------------------
# Reranker config sub-dataclass (Enhancement B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for the optional embedding reranker.

    Benchmarked results (194 tools / 98 labeled queries, nomic-embed-text-v2-moe,
    fulfils NousResearch/hermes-agent#13332):
      mode=rerank, full retrieve: R@5 = 0.810 (+0.176 vs BM25 baseline 0.634)
      mode=rrf,    k=10:          R@5 = 0.770 (+0.136), zero regressions

    Critical finding: nomic-embed-text-v2-moe REQUIRES task prefixes
    ("search_query:" / "search_document:"). Without them R@5 drops by ~0.194.
    The prefix defaults below reflect this; set to "" for models that do not
    want prefixes.
    """

    enabled: bool
    endpoint: str           # OpenAI-compatible /v1/embeddings URL
    model: str
    mode: str               # "rerank" | "rrf"
    rrf_k: int              # RRF k parameter (default 10; k=10 beat k=60 in bench)
    top_k: int              # how many results to return (matches config.search_default_limit)
    query_prefix: str       # prepended to query text before embedding
    doc_prefix: str         # prepended to each tool's embed text
    api_key: str            # bearer token (empty = no auth header)
    timeout: float          # HTTP timeout in seconds

    @classmethod
    def from_raw(cls, raw: Any) -> "RerankerConfig":
        """Parse reranker config from the tools.tool_search.reranker dict.

        Why: Provides a typed, validated config with safe defaults. Unknown
        keys and malformed values fall back gracefully so a typo in user
        config never breaks tool discovery.
        What: Accepts dict with enabled/endpoint/model/mode/rrf_k/top_k/
              query_prefix/doc_prefix/api_key/timeout keys.
        Test: from_raw({"enabled": true, "endpoint": "http://x"}) produces
              enabled=True with nomic prefixes; from_raw(None) gives
              enabled=False; from_raw({"mode": "bad"}) falls back to "rerank".
        """
        if not isinstance(raw, dict):
            return cls(
                enabled=False, endpoint="", model="nomic-embed-text-v2-moe",
                mode="rerank", rrf_k=10, top_k=5,
                query_prefix="search_query: ", doc_prefix="search_document: ",
                api_key="", timeout=5.0,
            )
        enabled = bool(raw.get("enabled", False))
        endpoint = str(raw.get("endpoint") or "").strip()
        model = str(raw.get("model") or "nomic-embed-text-v2-moe").strip()
        mode_raw = str(raw.get("mode") or "rerank").strip().lower()
        mode = mode_raw if mode_raw in ("rerank", "rrf") else "rerank"
        rrf_k = max(1, _safe_int(raw.get("rrf_k"), 10))
        top_k = max(1, min(50, _safe_int(raw.get("top_k"), 5)))
        # Task prefixes — default to nomic requirements; set to "" to disable.
        query_prefix = str(raw.get("query_prefix", "search_query: "))
        doc_prefix = str(raw.get("doc_prefix", "search_document: "))
        api_key = str(raw.get("api_key") or "").strip()
        timeout = max(0.1, _safe_float(raw.get("timeout"), 5.0))
        return cls(
            enabled=enabled, endpoint=endpoint, model=model, mode=mode,
            rrf_k=rrf_k, top_k=top_k, query_prefix=query_prefix,
            doc_prefix=doc_prefix, api_key=api_key, timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off"
    threshold_pct: float  # 0..100 — only used when enabled == "auto"
    search_default_limit: int
    max_search_limit: int
    reranker: RerankerConfig

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolSearchConfig":
        """Build a config from a raw dict / bool / None.

        Accepts the legacy bool shape (``tools.tool_search: true``) and the
        dict shape (``tools.tool_search: {enabled: auto, ...}``). Validates
        and clamps every numeric field; unknown values fall back to safe
        defaults rather than raising, so a typo in user config does not
        break the agent.

        Reranker defaults OFF (requires an external endpoint). Legacy
        bool/None callers pick up reranker-off automatically.
        """
        if raw is True:
            return cls(
                enabled="auto", threshold_pct=10.0,
                search_default_limit=5, max_search_limit=20,
                reranker=RerankerConfig.from_raw(None),
            )
        if raw is False:
            return cls(
                enabled="off", threshold_pct=10.0,
                search_default_limit=5, max_search_limit=20,
                reranker=RerankerConfig.from_raw(None),
            )
        if not isinstance(raw, dict):
            return cls(
                enabled="auto", threshold_pct=10.0,
                search_default_limit=5, max_search_limit=20,
                reranker=RerankerConfig.from_raw(None),
            )

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

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 20)))
        search_default_limit = max(1, min(max_search_limit,
                                          _safe_int(raw.get("search_default_limit"), 5)))

        reranker = RerankerConfig.from_raw(raw.get("reranker"))

        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            reranker=reranker,
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
    if not context_length or context_length <= 0:
        # Without a known context size, fall back to a fixed 20K-token cutoff
        # — the cliff above which Anthropic and OpenAI both saw quality drops.
        return deferrable_tokens >= 20_000
    threshold_tokens = int(context_length * (config.threshold_pct / 100.0))
    return deferrable_tokens >= threshold_tokens


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

    # Text used for embedding (populated lazily by the reranker).
    _embed_text: str = field(default="")


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


def _entry_embed_text(td: Dict[str, Any]) -> str:
    """Build the embedding text for a tool (concise: name + description).

    Why: Concise text avoids diluting embedding signal with parameter noise.
         The name alone provides the primary discriminator; the description
         provides semantic context.
    What: Returns "name: description" without parameter text.
    Test: For a tool with name="web_search" and desc="Search the web",
          returns "web_search: Search the web".
    """
    fn = td.get("function") or {}
    name = fn.get("name", "")
    desc = fn.get("description", "") or ""
    return f"{name}: {desc}"


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
            _embed_text=_entry_embed_text(td),
        )
        catalog.append(entry)
    return catalog


def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                avg_dl: float,
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


def _bm25_stats(
    catalog: List[CatalogEntry],
) -> Tuple[float, Dict[str, int], int]:
    """Compute shared BM25 corpus statistics for a catalog.

    Why: avg_dl, doc_freq, and n_docs are corpus-level constants reused
    for every document score. Pre-computing them avoids O(n^2) recomputation
    when scoring the same catalog under multiple query token sets.
    What: Returns (avg_dl, doc_freq, n_docs).
    Test: A catalog of one doc with 4 tokens has avg_dl=4, n_docs=1.
    """
    token_lengths = [len(e._tokens) for e in catalog]
    avg_dl = sum(token_lengths) / max(len(token_lengths), 1)
    doc_freq: Dict[str, int] = {}
    for e in catalog:
        seen = set(e._tokens)
        for t in seen:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return avg_dl, doc_freq, len(catalog)


def _bm25_ranked(
    catalog: List[CatalogEntry],
    query_tokens: List[str],
    raw_query: str = "",
) -> List[Tuple[float, CatalogEntry]]:
    """Score all catalog entries with BM25 and return full sorted list.

    Why: Separating scoring from slicing lets the reranker receive the full
    BM25 ranking to fuse with embedding ranking (RRF or pure rerank).
    What: Returns all entries sorted descending by BM25 score; falls back
    to name-substring matching when all BM25 scores are zero.
    Test: Query "github" against a catalog with a github_* tool should
          put that tool at rank 1.

    Substring fallback checks the original lowercased query string against
    each tool name — identical to the upstream main behaviour.
    """
    avg_dl, doc_freq, n_docs = _bm25_stats(catalog)

    scored: List[Tuple[float, CatalogEntry]] = []
    for entry in catalog:
        s = _bm25_score(query_tokens, entry._tokens, avg_dl, doc_freq, n_docs)
        scored.append((s, entry))

    has_hits = any(s > 0 for s, _ in scored)
    if not has_hits:
        # Substring fallback against the original tool name.
        # Use the raw lowercased query (not re-joined tokens) to match
        # upstream main behaviour exactly.
        ql = raw_query.lower()
        scored = [
            (0.1 if ql in e.name.lower() else 0.0, e)
            for _, e in scored
        ]

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def search_catalog(
    catalog: List[CatalogEntry],
    query: str,
    limit: int = 5,
    config: Optional[ToolSearchConfig] = None,
    reranker: Optional["EmbeddingReranker"] = None,
) -> List[CatalogEntry]:
    """Return the top-``limit`` catalog entries for ``query`` by BM25.

    Falls back to a stable name-substring match when BM25 yields no hits
    above zero. That ensures a query like ``"github"`` against a catalog
    where every tool is named ``github_*`` still returns results — BM25
    can underperform when query and document share only one token that
    appears in every document (zero IDF).

    When a ``reranker`` is provided (and ``config.reranker.enabled`` is
    True), the full BM25 ranking is passed to the reranker which re-orders
    or fuses it with embedding similarity before slicing to ``limit``
    (Enhancement B). Any reranker failure falls back to BM25 silently.

    When reranker is None (the default), this function is byte-for-byte
    equivalent to the upstream main BM25 path.
    """
    if not catalog or limit <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    all_scored = _bm25_ranked(catalog, query_tokens, raw_query=query)

    # Enhancement B: embedding reranker.
    if reranker is not None:
        try:
            reranked = reranker.rerank(query, all_scored, limit, config)
            return reranked
        except Exception as exc:
            logger.debug(
                "tool_search reranker failed, falling back to BM25: %s", exc
            )
            # Fall through to BM25 slice below.

    # Filter out zero-score entries to preserve the original BM25 contract:
    # "no hits above zero → empty result" (substring fallback already gives
    # a non-zero score to substring matches, so this filter is safe).
    return [e for s, e in all_scored[:limit] if s > 0]


# ---------------------------------------------------------------------------
# Embedding reranker (Enhancement B)
# Issue #13332: Hybrid Tool Pre-Selection
#
# Proven results on 194 tools / 98 labeled queries
# (see /tmp/tool-rerank-poc/BENCHMARK_WRITEUP.md and bench_tiers.py):
#
#   BM25 baseline R@5:          0.634
#   mode=rerank (pure cosine):  0.810  (+0.176)  — highest accuracy
#   mode=rrf k=10:              0.770  (+0.136)  — zero regressions
#
# Critical finding: nomic-embed-text-v2-moe REQUIRES task prefixes.
# Without "search_query:"/"search_document:" R@5 drops by 0.194.
# Default prefixes match nomic requirements; set to "" for other models.
#
# Design: retrieve the FULL catalog (not a narrow BM25 shortlist) because
# tool embeddings are cached and per-query embed cost is ~N-independent.
# Narrow retrieval silently discards gains (R@5 0.691 at N=10 vs 0.810
# at FULL — see bench_tiers.py Table 3.1).
# ---------------------------------------------------------------------------


def _cosine(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity (no numpy).

    Why: Embedding reranker needs cosine without adding numpy as a dep.
    What: dot(a,b) / (|a| * |b|); returns 0.0 for zero-norm vectors.
    Test: _cosine([1,0],[0,1]) == 0.0; _cosine([1,1],[1,1]) ≈ 1.0.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _rrf_fuse(
    bm25_ranked: List[Tuple[float, CatalogEntry]],
    embed_ranked: List[Tuple[float, CatalogEntry]],
    k: int,
    top_n: int,
) -> List[CatalogEntry]:
    """Reciprocal Rank Fusion of BM25 and embedding ranked lists.

    Why: RRF combines two ranked lists without needing score normalisation.
    score(doc) = 1/(k + rank_bm25) + 1/(k + rank_embed).
    What: Returns top_n entries by fused score.
    Test: A tool that is rank-1 in both lists should outscore anything
          that is rank-2 in both lists.

    k=10 beat the textbook k=60 in benchmarks (R@5 0.770 vs 0.748) with
    zero regressions — lower k boosts top-rank items more aggressively
    which suits tool selection where the correct tool usually has strong
    lexical OR semantic signal.
    """
    scores: Dict[str, float] = {}
    name_to_entry: Dict[str, CatalogEntry] = {}

    for rank_idx, (_, entry) in enumerate(bm25_ranked, start=1):
        scores[entry.name] = scores.get(entry.name, 0.0) + 1.0 / (k + rank_idx)
        name_to_entry[entry.name] = entry

    for rank_idx, (_, entry) in enumerate(embed_ranked, start=1):
        scores[entry.name] = scores.get(entry.name, 0.0) + 1.0 / (k + rank_idx)
        name_to_entry[entry.name] = entry

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name_to_entry[name] for name, _ in fused[:top_n]]


class EmbeddingReranker:
    """Reranks BM25 candidates using embeddings from an OpenAI-compatible endpoint.

    Why: Dense retrieval catches semantic queries BM25 misses (e.g. "spin up
    a VM" → mcp_proxmox_create_vm when the tool text says "virtual machine").
    What: Embeds the query + all tool texts, then reorders by cosine similarity
    (mode=rerank) or fuses with BM25 ranks via RRF (mode=rrf).

    Tool embeddings are cached by md5(model + embed_text) and are only
    recomputed when the catalog changes. Per-query cost is one embed call.

    Graceful fallback: any exception propagates to search_catalog which
    catches it and returns the BM25 result unchanged.

    Test: Instantiate with a mock urlopen that returns a fixed vector;
          assert the returned order reflects embedding scores, not BM25 scores.
    """

    def __init__(self, cfg: RerankerConfig) -> None:
        self._cfg = cfg
        # Cache: md5(model + text) → embedding vector
        self._cache: Dict[str, List[float]] = {}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts via the configured endpoint.

        Why: Centralises the HTTP call so mock tests only need to patch one
        method.
        What: POSTs to /v1/embeddings with model + input; returns ordered
        embedding vectors. Applies query_prefix / doc_prefix at call site.
        Test: Mock urllib.request.urlopen; assert model + input keys present
              in the POST body, and that returned vectors match fixture data.
        """
        cfg = self._cfg
        payload = json.dumps({"model": cfg.model, "input": texts}).encode("utf-8")
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        req = urllib.request.Request(
            cfg.endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:  # noqa: S310
            data = json.loads(resp.read())
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def _embed_with_cache(self, texts: List[str]) -> List[List[float]]:
        """Embed texts, serving cached vectors for already-seen texts.

        Why: Tool embeddings are stable across queries; caching eliminates
        repeated embed calls for the same tool text.
        What: md5(model + text) is the cache key; only uncached texts hit
              the endpoint. Cache reads/writes are guarded by _GLOBAL_LOCK;
              the slow HTTP call runs OUTSIDE the lock so concurrent threads
              are not serialised behind network I/O.
        Test: Call twice with same texts; assert endpoint called only once
              (check _embed call count via monkeypatching).
        """
        cfg = self._cfg
        keys = [hashlib.md5((cfg.model + t).encode("utf-8"), usedforsecurity=False).hexdigest()
                for t in texts]

        # Fast path (lock-free read): skip the lock entirely when all keys are
        # already cached — the common case on the hot path.
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            # Step 1: lock briefly to compute which keys are still absent.
            # Another thread may have populated some while we waited, so we
            # re-check inside the lock (double-checked locking pattern).
            with _GLOBAL_LOCK:
                still_missing_idx = [i for i in missing_idx if keys[i] not in self._cache]
                still_missing_texts = [texts[i] for i in still_missing_idx]

            # Step 2: call _embed OUTSIDE the lock — this is the slow network
            # call (~5 s timeout) and must not block other concurrent threads.
            # Two threads may occasionally fetch the same key concurrently; that
            # is the accepted trade-off vs. serialising all embedding I/O behind
            # the lock (same text → same vector, last write wins, no corruption).
            if still_missing_texts:
                new_vecs = self._embed(still_missing_texts)
                # Fix MEDIUM: guard against short responses from the endpoint.
                # An opaque KeyError on cache read is harder to diagnose than
                # an explicit ValueError here; search_catalog's except clause
                # routes both to the BM25 fallback.
                if len(new_vecs) != len(still_missing_texts):
                    raise ValueError(
                        f"embedding endpoint returned {len(new_vecs)} vectors "
                        f"for {len(still_missing_texts)} texts"
                    )

                # Step 3: lock briefly again to write new vectors into the cache.
                fetched: Dict[str, List[float]] = {
                    keys[i]: vec for i, vec in zip(still_missing_idx, new_vecs)
                }
                with _GLOBAL_LOCK:
                    self._cache.update(fetched)

        return [self._cache[k] for k in keys]

    def rerank(
        self,
        query: str,
        bm25_all: List[Tuple[float, CatalogEntry]],
        limit: int,
        config: Optional[ToolSearchConfig],  # noqa: ARG002
    ) -> List[CatalogEntry]:
        """Rerank the full BM25 result list using embedding similarity.

        Why: The full-catalog retrieve is nearly free when embeddings are
        cached; narrow retrieval silently discards semantic gains.
        What: Embeds the query and all tool texts (with task prefixes),
              then applies the configured mode (rerank or rrf).
        Test: Provide a catalog where a tool semantically matches the query
              but has zero BM25 score; assert it appears in the top result
              after reranking.
        """
        cfg = self._cfg
        # Fix CRITICAL 2: honour the caller's limit exactly.
        # search_default_limit is already applied by dispatch_tool_search before
        # calling search_catalog; returning more than limit here violates the
        # search_catalog(limit=N) contract and over-returns to the model.
        top_k = limit

        # Embed all tool texts (cache handles repeated calls efficiently).
        doc_texts = [f"{cfg.doc_prefix}{e._embed_text}" for _, e in bm25_all]
        doc_vecs = self._embed_with_cache(doc_texts)

        # Embed query (cached by md5 key; repeated identical queries are served from cache).
        q_text = f"{cfg.query_prefix}{query}"
        (q_vec,) = self._embed_with_cache([q_text])

        # Fix HIGH-2: dimension-mismatch guard.
        # If the endpoint returns vectors of mismatched or zero length (e.g.
        # model swapped mid-session, partial/truncated response), do NOT
        # silently zip()-truncate — return None so the caller falls back to
        # BM25. We raise ValueError here; search_catalog catches all Exception.
        q_dim = len(q_vec)
        if q_dim == 0:
            raise ValueError(
                "embedding reranker: query vector has length 0 — "
                "endpoint returned an empty embedding"
            )
        for (_, e), dvec in zip(bm25_all, doc_vecs):
            if len(dvec) == 0:
                raise ValueError(
                    f"embedding reranker: tool '{e.name}' vector has length 0"
                )
            if len(dvec) != q_dim:
                raise ValueError(
                    f"embedding reranker: dimension mismatch — query dim={q_dim}, "
                    f"tool '{e.name}' dim={len(dvec)}. Was the embedding model "
                    "changed mid-session? Falling back to BM25."
                )

        name_to_doc_vec: Dict[str, List[float]] = {
            e.name: vec for (_, e), vec in zip(bm25_all, doc_vecs)
        }

        if cfg.mode == "rerank":
            # Pure cosine rerank: sort all candidates by embedding similarity.
            embed_scored = [
                (_cosine(q_vec, name_to_doc_vec[e.name]), e)
                for _, e in bm25_all
            ]
            embed_scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in embed_scored[:top_k]]

        # mode == "rrf": fuse BM25 rank + embedding rank.
        embed_ranked: List[Tuple[float, CatalogEntry]] = sorted(
            [(_cosine(q_vec, name_to_doc_vec[e.name]), e) for _, e in bm25_all],
            key=lambda x: x[0],
            reverse=True,
        )
        return _rrf_fuse(bm25_all, embed_ranked, k=cfg.rrf_k, top_n=top_k)


# Module-level reranker singleton.
# Lazily constructed on first use; None means reranker is disabled or not
# yet built. The reranker embeds tool texts; the cache lives here across
# search calls within the same process, invalidated when the catalog changes.
_reranker: Optional[EmbeddingReranker] = None
_reranker_catalog_key: str = ""  # md5 of (endpoint + model + tool names)


def _get_reranker(
    cfg: RerankerConfig,
    catalog: List[CatalogEntry],
) -> Optional[EmbeddingReranker]:
    """Return a reranker singleton, creating or resetting it if the catalog changed.

    Why: A module-level singleton avoids re-embedding the full catalog on
    every search call. Cache invalidation ensures stale embeddings don't
    persist when tools are added or removed.
    What: Reuses the existing reranker if the catalog key (endpoint, model,
    and tool names) is unchanged; otherwise builds a fresh instance so the
    per-tool embedding cache starts clean.
    Test: Change the catalog; assert the reranker is rebuilt (new instance).
          Keep the catalog; assert the same instance is returned.
    """
    global _reranker, _reranker_catalog_key  # noqa: PLW0603

    if not cfg.enabled or not cfg.endpoint:
        return None

    catalog_key = hashlib.md5(
        (cfg.endpoint + cfg.model + ",".join(e.name for e in catalog)).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    # Fast path (lock-free read) — avoids lock contention on the hot path.
    if _reranker is not None and _reranker_catalog_key == catalog_key:
        return _reranker

    # Slow path: acquire lock and re-check (double-checked locking, fix HIGH-1).
    # Prevents concurrent requests from each building a separate singleton.
    with _GLOBAL_LOCK:
        if _reranker is None or _reranker_catalog_key != catalog_key:
            _reranker = EmbeddingReranker(cfg)
            _reranker_catalog_key = catalog_key

    return _reranker


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

    visible, deferrable = classify_tools(incoming)
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
        )

    bridge = bridge_tool_schemas(len(deferrable))
    result = visible + bridge
    threshold_tokens = int((context_length or 0) * (config.threshold_pct / 100.0))

    logger.info(
        "tool_search activated: %d core/visible tools kept, %d deferred (~%d tokens, threshold ~%d)",
        len(visible), len(deferrable), deferrable_tokens, threshold_tokens,
    )

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(deferrable),
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

    _, deferrable = classify_tools(current_tool_defs)
    catalog = build_catalog(deferrable)

    reranker: Optional[EmbeddingReranker] = None
    if config.reranker.enabled:
        reranker = _get_reranker(config.reranker, catalog)

    hits = search_catalog(catalog, query, limit=limit, config=config, reranker=reranker)
    return json.dumps({
        "query": query,
        "total_available": len(catalog),
        "matches": [_format_search_hit(h) for h in hits],
    }, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]]) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns a JSON string."""
    name = str(args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name is required"}, ensure_ascii=False)
    if not is_deferrable_tool_name(name):
        return json.dumps({
            "error": (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search."
            ),
        }, ensure_ascii=False)
    _, deferrable = classify_tools(current_tool_defs)
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
    "RerankerConfig",
    "CatalogEntry",
    "AssemblyResult",
    "EmbeddingReranker",
    "load_config",
    "is_deferrable_tool_name",
    "classify_tools",
    "estimate_tokens_from_schemas",
    "should_activate",
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
