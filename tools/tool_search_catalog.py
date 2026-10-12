"""Deferred-tool catalog for tool search: BM25 retrieval over deferrable tool
defs plus the budgeted, byte-stable catalog listing embedded in the bridge."""

from __future__ import annotations

import functools
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import snowballstemmer

# Reserved bridge names: a user/plugin/MCP tool may not take them (registry override
# protection rejects such registrations).
TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_CALL_NAME = "tool_call"
BRIDGE_TOOL_NAMES = frozenset({TOOL_SEARCH_NAME, TOOL_DESCRIBE_NAME, TOOL_CALL_NAME})
# Chars-per-token rule of thumb; 4.0 slightly underestimates (fewer false activations).
CHARS_PER_TOKEN = 4.0


@dataclass
class CatalogEntry:
    """One deferrable tool, in a form the bridge tools can search and serve."""

    name: str
    description: str
    schema: dict[str, Any]  # the full {"type":"function", "function": {...}} entry
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # toolset name, e.g. "mcp-github" or "kanban"
    _tokens: list[str] = field(default_factory=list)  # pre-tokenized for BM25


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_thread_local = threading.local()


@functools.lru_cache(maxsize=16384)
def _stem(token: str) -> str:
    """Stem one token, memoized across stateless catalog rebuilds. Snowball stemmers carry
    mutable parsing state and bridge dispatch runs on parallel tool-call threads, so the
    stemmer is one-per-thread, created lazily."""
    if getattr(_thread_local, "stemmer", None) is None:
        _thread_local.stemmer = snowballstemmer.stemmer("english")
    return _thread_local.stemmer.stemWord(token)


# Domain-specific synonym clusters of common agent action verbs.
_RAW_ACTION_CLUSTERS = [
    # fetch / scrape / extract / crawl / download / retrieve
    {"fetch", "scrape", "extract", "crawl", "download", "retrieve", "抓取", "获取", "提取", "爬取"},
    # inspect / examine / read / view / display
    {"inspect", "examine", "read", "view", "display", "查看", "显示"},
    # search / query / find / lookup / locate / scan
    {"search", "query", "find", "lookup", "locate", "scan", "搜索", "查询", "查找"},
    # list / enumerate / index
    {"list", "enumerate", "index", "列出"},
    # modify / update / edit / patch / change
    {"modify", "update", "edit", "patch", "change", "修改", "更新", "编辑"},
    # delete / remove / drop / purge / erase
    {"delete", "remove", "drop", "purge", "erase", "删除", "移除"},
    # send / post / publish / deliver / transmit
    {"send", "post", "publish", "deliver", "transmit", "发送", "推送"},
    # check / test / verify / validate / audit
    {"check", "test", "verify", "validate", "audit", "检查", "测试", "验证"},
]


def _build_action_synonyms() -> dict[str, frozenset[str]]:
    mapping: dict[str, set[str]] = {}
    for cluster in _RAW_ACTION_CLUSTERS:
        stemmed = {_stem(word) for word in cluster}
        for sword in stemmed:
            mapping.setdefault(sword, set()).update(stemmed - {sword})
    return {k: frozenset(v) for k, v in mapping.items()}


SEMANTIC_ACTION_SYNONYMS: dict[str, frozenset[str]] = _build_action_synonyms()


def _is_cjk(char: str) -> bool:
    """Return True if character is within standard CJK Unified Ideographs unicode ranges."""
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0xF900 <= cp <= 0xFAFF
    )


def _cjk_ngrams(text: str, n: int = 2) -> list[str]:
    """Extract unigrams from contiguous CJK character sequences plus recognized multi-character action words."""
    if not text:
        return []
    tokens: list[str] = [ch for ch in text if _is_cjk(ch)]
    # Include recognized multi-character semantic action terms if present in text
    for syn_word in SEMANTIC_ACTION_SYNONYMS:
        if len(syn_word) > 1 and syn_word in text:
            tokens.append(syn_word)
    return tokens


def _expand_semantic_tokens(tokens: list[str]) -> list[str]:
    """Expand stemmed tokens with their domain-equivalent semantic synonyms."""
    if not tokens:
        return []
    expanded = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        syns = SEMANTIC_ACTION_SYNONYMS.get(tok)
        if syns:
            for s in syns:
                if s not in seen:
                    seen.add(s)
                    expanded.append(s)
    return expanded


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, Snowball-stemmed (English); plus CJK n-grams."""
    if not text:
        return []
    tokens = [_stem(token.lower()) for token in _TOKEN_RE.findall(text)]
    cjk = _cjk_ngrams(text)
    if cjk:
        tokens.extend(cjk)
    return tokens


def _fn(td: dict[str, Any]) -> dict[str, Any]:
    """The ``function`` block of a tool-def (``{}`` when absent/None)."""
    return td.get("function") or {}


def _registry_entry(name: str) -> Any:
    """Registry entry for ``name``; None when unregistered OR the registry raises (lookup
    failures must never fail a bridge call). Lazy import: tests patch the registry."""
    try:
        from tools.registry import registry
        return registry.get_entry(name)
    except Exception:
        return None


def _registry_toolset(name: str) -> Optional[str]:
    """Toolset of a registered tool; None when unregistered or malformed (no str toolset)."""
    toolset = getattr(_registry_entry(name), "toolset", None)
    return toolset if isinstance(toolset, str) else None


def _entry_search_text(td: dict[str, Any], source_label: str = "") -> str:
    """Search-text blob: split name words + source label + description + top-level parameter
    names + action synonyms (the ``mcp__`` prefix is dropped — it is in every MCP document,
    so its IDF is ~0). The source label lets a service-name query ("linear") reach a tool
    whose own name omits the vendor."""
    fn = _fn(td)
    name = fn.get("name", "")
    name = name.removeprefix("mcp__")
    name_words = re.sub(r"[_.:-]", " ", name)
    extra = source_label if source_label and source_label not in name_words.split() else ""
    param_names = " ".join(((fn.get("parameters") or {}).get("properties") or {}).keys())
    base_text = f"{name_words} {extra} {fn.get('description', '') or ''} {param_names}"
    # Expand action synonyms found in tool name and description
    action_syns: list[str] = []
    seen: set[str] = set()
    for word in _TOKEN_RE.findall(f"{name_words} {fn.get('description', '') or ''}"):
        stemmed = _stem(word.lower())
        syns = SEMANTIC_ACTION_SYNONYMS.get(stemmed)
        if syns:
            for s in syns:
                if s not in seen:
                    seen.add(s)
                    action_syns.append(s)
    if action_syns:
        return f"{base_text} {' '.join(action_syns)}"
    return base_text


def _classify_source(name: str) -> tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    toolset = _registry_toolset(name)
    if toolset is None:
        return ("other", "")
    return ("mcp" if toolset.startswith("mcp-") else "plugin", toolset)


def build_catalog(tool_defs: list[dict[str, Any]]) -> list[CatalogEntry]:
    """Build the deferred-tool catalog from the deferrable subset of tool-defs."""
    catalog: list[CatalogEntry] = []
    for td in tool_defs:
        fn = _fn(td)
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        # Index the human-facing label ("linear", not "mcp-linear").
        source_label = _listing_group_label(source_name) if source_name else ""
        catalog.append(CatalogEntry(
            name=name, description=fn.get("description", "") or "", schema=td, source=source,
            source_name=source_name, _tokens=_tokenize(_entry_search_text(td, source_label))))
    return catalog


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], doc_lengths: list[int],
                avg_dl: float, doc_freq: dict[str, int], n_docs: int, k1: float = 1.5,
                b: float = 0.75) -> float:
    """Standard BM25 for one query against one document (inlined; the catalog is bounded —
    typically < 500 tools — so a dependency is not worth it)."""
    score = 0.0
    dl = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    for q in query_tokens:
        df, tf = doc_freq.get(q, 0), doc_tf.get(q, 0)
        if df and tf:
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
    return score


_CorpusStats = tuple[list[int], float, dict[str, int], int]  # doc_lengths, avg_dl, df, n_docs


def _corpus_stats(catalog: list[CatalogEntry]) -> _CorpusStats:
    """Compute the BM25 statistics shared by every query over a catalog."""
    doc_lengths = [len(entry._tokens) for entry in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq = Counter(tok for entry in catalog for tok in set(entry._tokens))
    return doc_lengths, avg_dl, dict(doc_freq), len(catalog)


def _gate_token(query_tokens: list[str], doc_freq: dict[str, int], n_docs: int) -> str:
    """The query token with the highest IDF: the word that names the intent. send,
    read, create sit in dozens of tool documents and separate nothing; gmail,
    github, incident sit in a few and separate everything. A document without this
    token answered a different question, however many common tokens it shares."""
    def _idf(token: str) -> float:
        df = doc_freq.get(token, 0)
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    return max(query_tokens, key=_idf)


# Relevance floor. The rarest-token gate stops queries whose intent word no tool carries; it
# does not stop a long hunt whose every word exists SOMEWHERE in the catalog while no single
# tool carries more than one of them (observed: "run shell command execute code python" -> a
# workflow-rerun tool sharing only "run"; the model read "results exist" as "it is in here"
# and re-searched 216 times). A document must also match MIN_QUERY_TERM_COVERAGE of the
# query's ANSWERABLE terms (present in at least one document) before it is offered. Coverage
# only engages from MIN_ANSWERABLE_TERMS_FOR_COVERAGE terms up: short queries ("list issues")
# legitimately differ from a tool by a word, and are where a coverage rule costs real recall.
MIN_QUERY_TERM_COVERAGE = 0.5
MIN_ANSWERABLE_TERMS_FOR_COVERAGE = 4


def _required_term_coverage(answerable_term_count: int) -> int:
    """How many of a query's ANSWERABLE unique terms a document must match to be offered:
    one below the engagement floor, else at least half, rounded up."""
    if answerable_term_count < MIN_ANSWERABLE_TERMS_FOR_COVERAGE:
        return 1
    return math.ceil(answerable_term_count * MIN_QUERY_TERM_COVERAGE)


def search_catalog(catalog: list[CatalogEntry], query: str, limit: int = 5, *,
                   corpus_stats: Optional[_CorpusStats] = None) -> list[CatalogEntry]:
    """Top-``limit`` catalog entries for ``query`` by BM25 (exact name match ranks first).

    Admission is by the query's rarest token (:func:`_gate_token`), not by ``score > 0``:
    BM25 is additive over the tokens a document shares with the query, so on a large catalog
    ``score > 0`` admits one-token matches and fills every slot with them (measured: "send
    gmail email" returned 5 incident tools that only shared ``email``). A token no document
    carries admits nothing; the caller's empty-group hint tells the model to retry without it.
    Long queries additionally need :func:`_required_term_coverage` of their answerable terms."""
    query_tokens = _tokenize(query) if catalog and limit > 0 else []
    if not query_tokens:
        return []
    corpus_stats = corpus_stats or _corpus_stats(catalog)
    doc_freq = corpus_stats[2]
    gate = _gate_token(query_tokens, doc_freq, corpus_stats[3])
    answerable = {
        t for t in query_tokens
        if doc_freq.get(t, 0) > 0
        or any(doc_freq.get(s, 0) > 0 for s in SEMANTIC_ACTION_SYNONYMS.get(t, ()))
    }
    required_terms = _required_term_coverage(len(answerable))
    exact_name = query.strip().lower()

    gate_synonyms = SEMANTIC_ACTION_SYNONYMS.get(gate, frozenset())

    def _admitted(entry: CatalogEntry) -> tuple[bool, float]:
        """Carries the intent word AND enough of the answerable terms (exact name is exempt).
        Returns (is_admitted, score_multiplier)."""
        if entry.name.lower() == exact_name:
            return True, 1.0
        tokens = set(entry._tokens)
        if gate in tokens and sum(1 for t in answerable if t in tokens) >= required_terms:
            return True, 1.0
        # Soft-gate fallback: gate token matches via semantic synonyms
        if gate_synonyms and any(s in tokens for s in gate_synonyms):
            matched = sum(
                1 for t in answerable
                if t in tokens or any(st in tokens for st in SEMANTIC_ACTION_SYNONYMS.get(t, ()))
            )
            if matched >= required_terms:
                return True, 0.85
        return False, 0.0

    scored = []
    for entry in catalog:
        admitted, multiplier = _admitted(entry)
        if admitted:
            if entry.name.lower() == exact_name:
                score = float("inf")
            else:
                score = _bm25_score(query_tokens, entry._tokens, *corpus_stats) * multiplier
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


# Sentence end: ., !, ? followed by whitespace/EOS, not inside e.g./i.e./etc.
_SENTENCE_END_RE = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?](?=\s|$)")


def _short_desc(description: str, max_chars: int = 60) -> str:
    """First sentence of a tool description, clipped to ``max_chars`` on a word boundary.
    ``e.g.``/``i.e.``/``etc.`` do not end a sentence; whitespace normalization and the regex
    search stay linear-time on hostile input."""
    text = " ".join((description or "").split())
    m = _SENTENCE_END_RE.search(text)
    text = text[:m.end()] if m else text
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    clipped = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    return clipped.rstrip(",;: ") + "…"


def _listing_group_label(source_name: str) -> str:
    """Human-facing group heading for a toolset, e.g. ``mcp-github`` -> ``github``."""
    label = source_name or "other"
    return label.removeprefix("mcp-")


def hidden_declared_sources() -> list[dict[str, Any]]:
    """Return deterministic summaries for declared MCP servers hidden by their check."""
    from hermes_platform import declaration
    from tools.mcp_liveness import unavailable_details
    from tools.registry import registry

    grouped: dict[str, list[Any]] = {}
    for entry in registry.get_all_entries():
        if entry.toolset.startswith("mcp-"):
            grouped.setdefault(entry.toolset[4:], []).append(entry)
    rows: list[dict[str, Any]] = []
    for server_name in sorted(grouped):
        if declaration.lookup(server_name) is None:
            continue
        entries = grouped[server_name]
        if any(entry.check_fn is None or bool(entry.check_fn()) for entry in entries):
            continue
        details = unavailable_details(server_name)
        if details is None:
            continue
        _decl, _current, sentence = details
        rows.append({"name": server_name, "tool_count": len(entries), "unavailable": sentence})
    return rows


def build_catalog_listing_with_form(
    deferrable: list[dict[str, Any]], *, max_tokens: int = 4000) -> tuple[Optional[str], str]:
    """Render the deferred-catalog manifest: ``- name: short desc`` lines grouped per source.
    Returns ``(text, form)``; form is ``"full"``, ``"names"``, ``"mixed"`` (oversized servers
    collapsed to a name + count line), ``"groups"`` (every server summarized) or ``"none"``
    (over budget even summarized -> text is None). Ordering is deterministic (sorted groups
    and tools) so the block is byte-stable — the request prefix stays cacheable. Degradation
    is PER SERVER, largest first: one huge server must not cost a small one its listing."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for td in deferrable:
        fn = _fn(td)
        name = fn.get("name", "")
        if name:
            # _classify_source gives ("other", "") when unregistered; the label of "" is "other".
            label = _listing_group_label(_classify_source(name)[1])
            groups.setdefault(label, []).append((name, _short_desc(fn.get("description", ""))))
    unavailable = hidden_declared_sources()
    if not groups and not unavailable:
        return None, "none"

    def render_group(label: str, mode: str) -> str:
        """Render one server's block. mode: 'full' | 'names' | 'summary'."""
        tools = sorted(groups[label])
        if mode == "summary":
            return (f"{label} ({len(tools)} tools — names not listed; "
                    f"discover via `{TOOL_SEARCH_NAME}`)")
        lines = [f"{label} tools ({len(tools)}):"]
        if mode == "full":
            lines.extend(f"- {name}: {desc}" if desc else f"- {name}" for name, desc in tools)
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    header = ("Deferred tool catalog (call schemas via "
              f"`{TOOL_DESCRIBE_NAME}`, invoke via `{TOOL_CALL_NAME}`):")

    def assemble_if_fits(modes: dict[str, str]) -> Optional[str]:
        available_blocks = {label: render_group(label, modes[label]) for label in groups}
        unavailable_blocks = {
            row["name"]: (
                f"{row['name']} ({row['tool_count']} tools unavailable: {row['unavailable']})"
                if row.get("tool_count") is not None
                else f"{row['name']} (tools unavailable: {row['unavailable']})"
            )
            for row in unavailable
        }
        blocks = [
            available_blocks[label] if label in available_blocks else unavailable_blocks[label]
            for label in sorted(available_blocks | unavailable_blocks)
        ]
        text = "\n".join([header] + blocks)
        return text if math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens else None

    for mode in ("full", "names"):  # 1. everything full; 2. everything names-only
        modes = {lbl: mode for lbl in groups}
        text = assemble_if_fits(modes)
        if text is not None:
            return text, mode
    # 3. Collapse the LARGEST rendered groups first (deterministic: size then label).
    for lbl in sorted(groups, key=lambda lbl: (-len(render_group(lbl, "names")), lbl)):
        modes[lbl] = "summary"
        text = assemble_if_fits(modes)
        if text is not None:
            return text, "groups" if all(m == "summary" for m in modes.values()) else "mixed"
    return None, "none"
