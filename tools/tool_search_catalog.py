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
    schema: Dict[str, Any]  # the full {"type":"function", "function": {...}} entry
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # toolset name, e.g. "mcp-github" or "kanban"
    _tokens: List[str] = field(default_factory=list)  # pre-tokenized for BM25


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


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, Snowball-stemmed (English); shared by the index and
    query paths so "issues" matches ``create_issue``."""
    return [_stem(token.lower()) for token in _TOKEN_RE.findall(text)] if text else []


def _fn(td: Dict[str, Any]) -> Dict[str, Any]:
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


def _entry_search_text(td: Dict[str, Any], source_label: str = "") -> str:
    """Search-text blob: split name words + source label + description + top-level parameter
    names (schema bodies are noise with no recall gain). The ``mcp__`` prefix is dropped — it
    is in every MCP document, so its IDF is ~0. The source label lets a service-name query
    ("linear") reach a tool whose own name omits the vendor."""
    fn = _fn(td)
    name = fn.get("name", "")
    if name.startswith("mcp__"):
        name = name[len("mcp__"):]
    name_words = re.sub(r"[_.:-]", " ", name)
    extra = source_label if source_label and source_label not in name_words.split() else ""
    param_names = " ".join(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return f"{name_words} {extra} {fn.get('description', '') or ''} {param_names}"


def _classify_source(name: str) -> Tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    toolset = _registry_toolset(name)
    if toolset is None:
        return ("other", "")
    return ("mcp" if toolset.startswith("mcp-") else "plugin", toolset)


def build_catalog(tool_defs: List[Dict[str, Any]]) -> List[CatalogEntry]:
    """Build the deferred-tool catalog from the deferrable subset of tool-defs."""
    catalog: List[CatalogEntry] = []
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


def _bm25_score(query_tokens: List[str], doc_tokens: List[str], doc_lengths: List[int],
                avg_dl: float, doc_freq: Dict[str, int], n_docs: int, k1: float = 1.5,
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


_CorpusStats = Tuple[List[int], float, Dict[str, int], int]  # doc_lengths, avg_dl, df, n_docs


def _corpus_stats(catalog: List[CatalogEntry]) -> _CorpusStats:
    """Compute the BM25 statistics shared by every query over a catalog."""
    doc_lengths = [len(entry._tokens) for entry in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq = Counter(tok for entry in catalog for tok in set(entry._tokens))
    return doc_lengths, avg_dl, dict(doc_freq), len(catalog)


def search_catalog(catalog: List[CatalogEntry], query: str, limit: int = 5, *,
                   corpus_stats: Optional[_CorpusStats] = None) -> List[CatalogEntry]:
    """Top-``limit`` catalog entries for ``query`` by BM25 (exact name match ranks first).
    Falls back to a name-substring match only when NO query token appears in any document
    (e.g. "hub" vs ``github_*``); the IDF variant is strictly positive, so a hit anywhere
    suppresses the fallback."""
    query_tokens = _tokenize(query) if catalog and limit > 0 else []
    if not query_tokens:
        return []
    corpus_stats = corpus_stats or _corpus_stats(catalog)
    scored: List[Tuple[float, CatalogEntry]] = []
    exact_name = query.strip().lower()
    for entry in catalog:
        s = (float("inf") if entry.name.lower() == exact_name
             else _bm25_score(query_tokens, entry._tokens, *corpus_stats))
        if s > 0:
            scored.append((s, entry))
    if not scored:
        scored = [(0.1, entry) for entry in catalog if query.lower() in entry.name.lower()]
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
    return label[4:] if label.startswith("mcp-") else label


def build_catalog_listing_with_form(
    deferrable: List[Dict[str, Any]], *, max_tokens: int = 4000) -> Tuple[Optional[str], str]:
    """Render the deferred-catalog manifest: ``- name: short desc`` lines grouped per source.
    Returns ``(text, form)``; form is ``"full"``, ``"names"``, ``"mixed"`` (oversized servers
    collapsed to a name + count line), ``"groups"`` (every server summarized) or ``"none"``
    (over budget even summarized -> text is None). Ordering is deterministic (sorted groups
    and tools) so the block is byte-stable — the request prefix stays cacheable. Degradation
    is PER SERVER, largest first: one huge server must not cost a small one its listing."""
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for td in deferrable:
        fn = _fn(td)
        name = fn.get("name", "")
        if name:
            # _classify_source gives ("other", "") when unregistered; the label of "" is "other".
            label = _listing_group_label(_classify_source(name)[1])
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
            lines.extend(f"- {name}: {desc}" if desc else f"- {name}" for name, desc in tools)
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    header = ("Deferred tool catalog (call schemas via "
              f"`{TOOL_DESCRIBE_NAME}`, invoke via `{TOOL_CALL_NAME}`):")

    def assemble_if_fits(modes: Dict[str, str]) -> Optional[str]:
        text = "\n".join([header] + [render_group(lbl, modes[lbl]) for lbl in sorted(groups)])
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
