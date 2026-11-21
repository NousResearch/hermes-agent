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


# Unicode word characters minus the underscore (``[^\W_]``): letters and digits in any script.
# The index and the query share this pattern, so a catalog described in Ukrainian is
# searchable in Ukrainian — the ASCII-only class this replaced tokenized ``графік`` to
# nothing, and indexed a Ukrainian description without its words.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_thread_local = threading.local()


@functools.lru_cache(maxsize=16384)
def _stem(token: str) -> str:
    """Stem one token, memoized across stateless catalog rebuilds. Snowball stemmers carry
    mutable parsing state and bridge dispatch runs on parallel tool-call threads, so the
    stemmer is one-per-thread, created lazily.

    English only: the Snowball English stemmer mangles other scripts (``графік`` -> ``граф``,
    ``charts`` -/-> anything useful in Cyrillic), so a non-ASCII token is matched verbatim.
    Index and query still share the same vocabulary, which is what recall needs; morphology
    in non-English catalogs is out of scope here."""
    if not token.isascii():
        return token
    if getattr(_thread_local, "stemmer", None) is None:
        _thread_local.stemmer = snowballstemmer.stemmer("english")
    return _thread_local.stemmer.stemWord(token)


def _tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens, Snowball-stemmed when ASCII (English); shared by the
    index and query paths so "issues" matches ``create_issue`` and ``графік`` matches a
    Ukrainian description."""
    return [_stem(token.lower()) for token in _TOKEN_RE.findall(text)] if text else []


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
    names (schema bodies are noise with no recall gain). The ``mcp__`` prefix is dropped — it
    is in every MCP document, so its IDF is ~0. The source label lets a service-name query
    ("linear") reach a tool whose own name omits the vendor."""
    fn = _fn(td)
    name = fn.get("name", "")
    name = name.removeprefix("mcp__")
    name_words = re.sub(r"[_.:-]", " ", name)
    extra = source_label if source_label and source_label not in name_words.split() else ""
    param_names = " ".join(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return f"{name_words} {extra} {fn.get('description', '') or ''} {param_names}"


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


def _idf(token: str, doc_freq: dict[str, int], n_docs: int) -> float:
    """BM25 inverse document frequency of one token under this catalog."""
    df = doc_freq.get(token, 0)
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def _gate_token(query_tokens: list[str], doc_freq: dict[str, int], n_docs: int) -> str:
    """The query token with the highest IDF: the word that names the intent. ``send``,
    ``read``, ``create`` sit in dozens of tool documents and separate nothing; ``gmail``,
    ``github``, ``incident`` sit in a few and separate everything. A document without this
    token answered a different question, however many common tokens it shares."""
    return max(query_tokens, key=lambda token: _idf(token, doc_freq, n_docs))


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


# Second chance. The gate is strict on purpose — an unknown intent word means "no such
# capability here" — but it also refuses a LONG natural-language query whose intent word is a
# synonym the catalog does not carry (measured live, 2026-11-06: "create image chart plot file"
# -> ``matches: []`` while ``render_chart`` carried ``chart``/``imag``). When the strict pass
# admits nothing on a long query, documents whose NAME carries one of the query's words are
# offered instead, still subject to the coverage floor, and the caller flags them
# low-confidence so a near miss is never read as an exact capability. Queries shorter than the
# coverage engagement floor (four distinct words) stay strict: with one or two content words
# there is nothing to triangulate, and an unknown word there IS the intent ("create zzzz
# issue" keeps returning the empty group plus retry hint).
MIN_SECOND_CHANCE_QUERY_TOKENS = MIN_ANSWERABLE_TERMS_FOR_COVERAGE


def _name_tokens(entry: CatalogEntry) -> set[str]:
    """Tokens of the tool NAME alone (``render_chart`` -> {render, chart}). The second chance
    matches these, never the description: a name word is what the tool claims to be, a
    description word is a hint."""
    return set(_tokenize(re.sub(r"[_.:-]", " ", entry.name.removeprefix("mcp__"))))


def _rank(query_tokens: list[str], entries: list[CatalogEntry], corpus_stats: _CorpusStats,
          exact_name: str, limit: int,
          name_evidence: Optional[dict[str, float]] = None) -> list[CatalogEntry]:
    """Entries by BM25 (an exact tool-name query first, then score), truncated to ``limit``.

    ``name_evidence`` (second-chance ranking) maps a tool name to the IDF of the rarest query
    word its NAME carries: the second chance exists to find a name match, so the strongest name
    evidence leads and BM25 only breaks ties among equally-named candidates.
    """
    scored = []
    for entry in entries:
        exact = entry.name.lower() == exact_name
        score = float("inf") if exact else _bm25_score(
            query_tokens, entry._tokens, *corpus_stats)
        key = (exact, (name_evidence or {}).get(entry.name, 0.0), score)
        scored.append((key, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


def search_catalog(catalog: list[CatalogEntry], query: str, limit: int = 5, *,
                   corpus_stats: Optional[_CorpusStats] = None) -> list[CatalogEntry]:
    """Top-``limit`` catalog entries for ``query``. Thin wrapper over
    :func:`search_catalog_with_confidence` for callers that do not need the confidence flag."""
    return search_catalog_with_confidence(catalog, query, limit, corpus_stats=corpus_stats)[0]


def search_catalog_with_confidence(
        catalog: list[CatalogEntry], query: str, limit: int = 5, *,
        corpus_stats: Optional[_CorpusStats] = None) -> tuple[list[CatalogEntry], bool]:
    """Top-``limit`` entries for ``query`` plus whether they came from the second-chance pass.

    The strict pass admits by the query's rarest token (:func:`_gate_token`), not by
    ``score > 0``: BM25 is additive over the tokens a document shares with the query, so on a
    large catalog ``score > 0`` admits one-token matches and fills every slot with them
    (measured: "send gmail email" returned 5 incident tools that only shared ``email``). A
    token no document carries admits nothing; the caller's empty-group hint tells the model to
    retry without it. Long queries additionally need :func:`_required_term_coverage` of their
    answerable terms.

    When the strict pass admits nothing and the query is long, the second chance admits
    documents whose NAME carries one of the query's words and that still clear the coverage
    floor; those come back with the confidence flag True so the caller can mark them.
    """
    if not catalog or limit <= 0:
        return [], False
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [], False
    corpus_stats = corpus_stats or _corpus_stats(catalog)
    doc_freq = corpus_stats[2]
    gate = _gate_token(query_tokens, doc_freq, corpus_stats[3])
    answerable = {t for t in query_tokens if doc_freq.get(t, 0) > 0}
    required_terms = _required_term_coverage(len(answerable))
    exact_name = query.strip().lower()

    def _covers(tokens: set[str]) -> bool:
        return sum(1 for t in answerable if t in tokens) >= required_terms

    def _admitted(entry: CatalogEntry) -> bool:
        """Carries the intent word AND enough of the answerable terms (exact name is exempt)."""
        if entry.name.lower() == exact_name:
            return True
        tokens = set(entry._tokens)
        return gate in tokens and _covers(tokens)

    hits = _rank(query_tokens, [e for e in catalog if _admitted(e)], corpus_stats,
                 exact_name, limit)
    if hits:
        return hits, False

    unique = set(query_tokens)
    if len(unique) < MIN_SECOND_CHANCE_QUERY_TOKENS:
        return [], False
    n_docs = corpus_stats[3]
    near: list[CatalogEntry] = []
    evidence: dict[str, float] = {}
    for entry in catalog:
        matched = _name_tokens(entry) & unique
        if not matched or not _covers(set(entry._tokens)):
            continue
        near.append(entry)
        evidence[entry.name] = max(_idf(t, doc_freq, n_docs) for t in matched)
    if not near:
        return [], False
    return _rank(query_tokens, near, corpus_stats, exact_name, limit, evidence), True


# Sentence end: ., !, ? followed by whitespace/EOS, not inside e.g./i.e./etc.
_SENTENCE_END_RE = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?](?=\s|$)")


# A catalog line is its tool's first sentence: built-in tools keep that sentence within this cap
# and lead with the key capability (tests/tools/test_tool_search.py pins it), so no line clips.
CATALOG_SUMMARY_MAX_CHARS = 80


def _short_desc(description: str, max_chars: int = CATALOG_SUMMARY_MAX_CHARS) -> str:
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
        if mode == "full":
            lines = [f"- {name}: {desc}" if desc else f"- {name}" for name, desc in tools]
        else:
            lines = [", ".join(name for name, _ in tools)] if len(tools) > 1 else [f"- {tools[0][0]}"]
        # A one-tool group's "<label> tools (1):" heading says nothing its single line does not.
        return "\n".join(lines if len(tools) == 1 else [f"{label} tools ({len(tools)}):", *lines])

    header = ("Deferred tool catalog (call schemas via "
              f"`{TOOL_DESCRIBE_NAME}`, invoke via `{TOOL_CALL_NAME}`):")

    def assemble_if_fits(modes: dict[str, str]) -> Optional[str]:
        # Headless one-tool lines go first, straight under the catalog header: placed between
        # groups they would read as members of the group above them.
        singles = [render_group(label, modes[label]) for label in sorted(groups)
                   if len(groups[label]) == 1 and modes[label] != "summary"]
        available_blocks = {label: render_group(label, modes[label]) for label in groups
                            if len(groups[label]) > 1 or modes[label] == "summary"}
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
        text = "\n".join([header] + singles + blocks)
        return text if math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens else None

    for mode in ("full", "names"):  # 1. everything full; 2. everything names-only
        text = assemble_if_fits({lbl: mode for lbl in groups})
        if text is not None:
            return text, mode
    # 3. Collapse the LARGEST rendered groups first (deterministic: size then label); the servers
    # left uncollapsed get their full lines back when those fit, else names-only.
    summarized: set[str] = set()
    for lbl in sorted(groups, key=lambda lbl: (-len(render_group(lbl, "names")), lbl)):
        summarized.add(lbl)
        for rest in ("full", "names"):
            text = assemble_if_fits({g: "summary" if g in summarized else rest for g in groups})
            if text is not None:
                return text, "groups" if len(summarized) == len(groups) else "mixed"
    return None, "none"
