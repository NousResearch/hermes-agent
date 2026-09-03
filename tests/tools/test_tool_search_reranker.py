"""Tests for the optional embedding reranker in tools/tool_search.py.

The reranker is OFF by default. The invariants pinned here:

* Disabled (or no endpoint) -> ``search_catalog`` / ``dispatch_tool_search``
  are byte-identical to the BM25-only path.
* Any endpoint failure (exception, malformed payload, dimension mismatch)
  -> fall back to the BM25 result, never raise into the bridge.
* Exact-name matches stay pinned first regardless of mode.
* Tool vectors are embedded once per scope and cached; subsequent queries
  cost exactly one query-embed call.
* The reranker cache is scope-keyed and bounded (A -> B -> A reuse,
  concurrent scopes retained, FIFO eviction, config fields in the key).
* Credentials come from ``HERMES_EMBED_API_KEY`` only, are sent as a Bearer
  header, and never appear in ``repr``.

No test talks to a network endpoint: ``EmbeddingReranker._embed`` is
monkeypatched with a deterministic fake that maps text -> vector.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _td(name: str, description: str = "", properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


# A flat REST-style catalog that reproduces the measured BM25 failure mode:
# "list accounts" ranks the long rules_lists names above get_accounts.
_CF_DEFS = [
    _td("get_accounts", "GET /accounts List Accounts. List all accounts you have access to."),
    _td("get_accounts_rules_lists", "GET /accounts/{account_id}/rules/lists Get lists. Fetches all lists in the account."),
    _td("get_accounts_rules_lists_by_list_id", "GET /accounts/{account_id}/rules/lists/{list_id} Get a list. Fetches the details of a list."),
    _td("get_accounts_rules_lists_items", "GET /accounts/{account_id}/rules/lists/{list_id}/items Get list items. Fetches all the items in the list."),
    _td("get_accounts_rules_lists_items_by_item_id", "GET /accounts/{account_id}/rules/lists/{list_id}/items/{item_id} Get a list item."),
    _td("get_zones", "GET /zones List Zones. Lists, searches, sorts, and filters your zones."),
    _td("post_accounts_pages_projects_deployments", "POST deployments Create deployment. Start a new deployment from production."),
    _td("calendar_create_event", "Add an event to the user's calendar at a given time."),
]


@pytest.fixture
def cf_defs():
    """``_CF_DEFS`` registered under an MCP toolset so dispatch defers them."""
    from tools.registry import registry
    for td in _CF_DEFS:
        registry.register(
            name=td["function"]["name"],
            handler=lambda args, **kwargs: "{}",
            schema=td,
            toolset="mcp-rerank-test",
        )
    return _CF_DEFS


# Deterministic fake embeddings: a fixed vocabulary of "concepts"; a text's
# vector is the indicator of which concept words it contains. Cosine
# similarity then rewards concept overlap, which is enough to make the
# reranker's ordering predictable in tests.
_CONCEPTS = ["account", "list", "zone", "deploy", "pages", "calendar", "remind", "rules", "item"]
_SYNONYMS = {
    "accounts": "account", "lists": "list", "zones": "zone", "deployment": "deploy",
    "deployments": "deploy", "deploy": "deploy", "tonight": "remind", "remind": "remind",
    "event": "calendar", "items": "item",
}


def _fake_vec(text: str, dim: int = len(_CONCEPTS)) -> List[float]:
    words = [w.strip(".,:/{}()'\"").lower() for w in text.replace("_", " ").split()]
    vec = [0.0] * dim
    for w in words:
        c = _SYNONYMS.get(w, w)
        if c in _CONCEPTS[:dim]:
            vec[_CONCEPTS.index(c)] += 1.0
    return vec


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace the HTTP call with a deterministic in-memory embedder.

    Returns a recorder: ``calls`` is the list of text batches sent.
    """
    from tools.tool_search import EmbeddingReranker

    calls: List[List[str]] = []

    def _embed(self, texts):
        calls.append(list(texts))
        return [_fake_vec(t) for t in texts]

    monkeypatch.setattr(EmbeddingReranker, "_embed", _embed)
    return calls


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    from tools import tool_search
    tool_search._reranker_cache.clear()
    yield
    tool_search._reranker_cache.clear()


def _cfg(**overrides) -> "RerankerConfig":  # noqa: F821
    from tools.tool_search import RerankerConfig
    raw = {"enabled": True, "endpoint": "http://localhost:11434/v1/embeddings"}
    raw.update(overrides)
    return RerankerConfig.from_raw(raw)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestRerankerConfig:
    def test_default_is_disabled(self):
        from tools.tool_search import RerankerConfig, ToolSearchConfig
        assert RerankerConfig.from_raw(None).enabled is False
        assert RerankerConfig.from_raw(None).active is False
        assert ToolSearchConfig.from_raw(None).reranker.enabled is False
        assert ToolSearchConfig.from_raw({}).reranker.active is False
        assert ToolSearchConfig.from_raw(True).reranker.active is False

    def test_shipped_defaults_keep_reranker_off(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw(DEFAULT_CONFIG["tools"]["tool_search"])
        assert cfg.reranker.enabled is False
        assert cfg.reranker.active is False

    def test_enabled_without_endpoint_is_inactive(self):
        from tools.tool_search import RerankerConfig
        cfg = RerankerConfig.from_raw({"enabled": True})
        assert cfg.enabled is True
        assert cfg.active is False

    def test_parses_all_fields(self):
        from tools.tool_search import ToolSearchConfig
        cfg = ToolSearchConfig.from_raw({"reranker": {
            "enabled": "yes",
            "endpoint": " http://embed.local/v1/embeddings ",
            "model": "text-embedding-3-small",
            "mode": "RRF",
            "rrf_k": 60,
            "query_prefix": "",
            "doc_prefix": None,
            "timeout": 2.5,
        }}).reranker
        assert cfg.active is True
        assert cfg.endpoint == "http://embed.local/v1/embeddings"
        assert cfg.model == "text-embedding-3-small"
        assert cfg.mode == "rrf"
        assert cfg.rrf_k == 60
        assert cfg.query_prefix == ""
        assert cfg.doc_prefix == ""
        assert cfg.timeout == 2.5

    def test_nomic_prefixes_are_the_default(self):
        cfg = _cfg()
        assert cfg.model == "nomic-embed-text"
        assert cfg.query_prefix == "search_query: "
        assert cfg.doc_prefix == "search_document: "

    def test_invalid_values_fall_back_safely(self):
        cfg = _cfg(mode="cross-encoder", rrf_k="lots", timeout="never")
        assert cfg.mode == "rerank"
        assert cfg.rrf_k == 10
        assert cfg.timeout == 5.0
        assert _cfg(rrf_k=0).rrf_k == 1
        assert _cfg(timeout=0).timeout == 0.1
        assert _cfg(timeout=10_000).timeout == 120.0

    def test_api_key_comes_from_env_only(self, monkeypatch):
        monkeypatch.delenv("HERMES_EMBED_API_KEY", raising=False)
        assert _cfg(api_key="from-config").api_key == ""
        monkeypatch.setenv("HERMES_EMBED_API_KEY", " sk-env ")
        cfg = _cfg(api_key="from-config")
        assert cfg.api_key == "sk-env"
        # A secret must never leak through a logged config.
        assert "sk-env" not in repr(cfg)
        assert "sk-env" not in str(cfg)


# ---------------------------------------------------------------------------
# Off / failure paths are byte-identical to BM25
# ---------------------------------------------------------------------------


class TestDisabledIsIdentical:
    def test_get_reranker_returns_none_when_inactive(self):
        from tools.tool_search import RerankerConfig, _get_reranker, build_catalog
        catalog = build_catalog(_CF_DEFS)
        assert _get_reranker(RerankerConfig(), catalog) is None
        assert _get_reranker(RerankerConfig(enabled=True, endpoint=""), catalog) is None

    def test_dispatch_without_reranker_never_embeds(self, fake_embed, cf_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search
        cfg = ToolSearchConfig.from_raw({"enabled": "on"})
        out = json.loads(dispatch_tool_search(
            {"queries": ["list accounts"]}, current_tool_defs=cf_defs, config=cfg,
        ))
        assert out["results"][0]["matches"]
        assert fake_embed == []

    @pytest.mark.parametrize("failure", ["raise", "short", "mismatch", "empty"])
    def test_endpoint_failure_falls_back_to_bm25(self, monkeypatch, failure):
        from tools.tool_search import EmbeddingReranker, _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        baseline = search_catalog(catalog, "list accounts", limit=5)
        assert baseline, "fixture must produce BM25 hits"

        def _embed(self, texts):
            if failure == "raise":
                raise OSError("connection refused")
            if failure == "short":
                return [[1.0, 0.0]] * (len(texts) - 1) if len(texts) > 1 else [[1.0, 0.0]]
            if failure == "mismatch":
                # documents get 3-dim vectors, the single query gets 2-dim
                return [[1.0, 0.0, 0.0] if len(texts) > 1 else [1.0, 0.0] for _ in texts]
            return [[] for _ in texts]

        monkeypatch.setattr(EmbeddingReranker, "_embed", _embed)
        reranker = _get_reranker(_cfg(), catalog)
        assert reranker is not None
        for query in ("list accounts", "zones", "zzzz"):
            assert search_catalog(catalog, query, limit=5, reranker=reranker) == \
                search_catalog(catalog, query, limit=5)

    def test_failure_keeps_substring_fallback(self, monkeypatch):
        from tools.tool_search import EmbeddingReranker, _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        monkeypatch.setattr(EmbeddingReranker, "_embed", lambda self, texts: (_ for _ in ()).throw(TimeoutError()))
        reranker = _get_reranker(_cfg(), catalog)
        # "cal" is not a token but is a name substring: BM25 path returns it
        # via the substring fallback, and the failing reranker must not eat it.
        assert [e.name for e in search_catalog(catalog, "cal", limit=5, reranker=reranker)] == \
            ["calendar_create_event"]

    def test_search_catalog_signature_default_is_none(self):
        import inspect
        from tools.tool_search import search_catalog
        assert inspect.signature(search_catalog).parameters["reranker"].default is None


# ---------------------------------------------------------------------------
# Rerank behavior
# ---------------------------------------------------------------------------


class TestRerankBehavior:
    def test_rerank_surfaces_short_name_bm25_buries(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        bm25 = [e.name for e in search_catalog(catalog, "list accounts", limit=3)]
        # The measured failure mode: BM25 prefers the long rules_lists names
        # (accounts + list co-occur) over the 2-token target. In the real
        # 1,938-tool catalog the target falls out of the top-5 entirely; in
        # this small fixture it is merely not first.
        assert bm25[0] != "get_accounts"

        reranker = _get_reranker(_cfg(), catalog)
        reranked = [e.name for e in search_catalog(catalog, "list accounts", limit=3, reranker=reranker)]
        assert reranked[0] == "get_accounts"

    def test_semantic_query_with_no_lexical_overlap(self, fake_embed):
        """'remind me tonight' has no token in common with calendar_create_event."""
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        assert search_catalog(catalog, "remind me tonight", limit=3) == []
        reranker = _get_reranker(_cfg(), catalog)
        # The fake embedder maps "tonight" and "event" onto neighbouring
        # concepts; a zero query vector would tie everything, so seed it.
        hits = search_catalog(catalog, "calendar remind me tonight", limit=3, reranker=reranker)
        assert hits[0].name == "calendar_create_event"

    def test_exact_name_pinned_first_in_both_modes(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        for mode in ("rerank", "rrf"):
            reranker = _get_reranker(_cfg(mode=mode), catalog)
            hits = search_catalog(catalog, "get_zones", limit=2, reranker=reranker)
            assert hits[0].name == "get_zones", mode

    def test_limit_is_honored(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        for mode in ("rerank", "rrf"):
            reranker = _get_reranker(_cfg(mode=mode), catalog)
            for limit in (1, 2, 5, 50):
                hits = search_catalog(catalog, "accounts", limit=limit, reranker=reranker)
                assert len(hits) == min(limit, len(catalog))
                assert len({h.name for h in hits}) == len(hits)

    def test_rerank_scores_whole_catalog_not_bm25_shortlist(self, fake_embed):
        """A target BM25 scores zero for must still be reachable."""
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        reranker = _get_reranker(_cfg(), catalog)
        # "deploy" stems differently from "deployments"? Either way the
        # fake embedder maps both to the same concept.
        hits = search_catalog(catalog, "deploy", limit=1, reranker=reranker)
        assert hits[0].name == "post_accounts_pages_projects_deployments"

    def test_dispatch_uses_reranker_when_enabled(self, fake_embed, cf_defs):
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        cfg = ToolSearchConfig.from_raw({
            "enabled": "on",
            "reranker": {"enabled": True, "endpoint": "http://localhost:11434/v1/embeddings"},
        })
        out = json.loads(dispatch_tool_search(
            {"queries": ["list accounts", "zones"], "limit": 2},
            current_tool_defs=cf_defs, config=cfg,
        ))
        assert out["results"][0]["matches"][0] == "get_accounts"
        assert out["results"][1]["matches"][0] == "get_zones"
        # Catalog embedded once; each query embedded once.
        assert len(fake_embed) == 3
        assert len(fake_embed[0]) == len(_CF_DEFS)
        assert fake_embed[1] == ["search_query: list accounts"]
        assert fake_embed[2] == ["search_query: zones"]

    def test_dispatch_response_shape_unchanged(self, fake_embed, cf_defs):
        """Enabling the reranker must not add or remove response fields."""
        from tools.tool_search import ToolSearchConfig, dispatch_tool_search

        off = ToolSearchConfig.from_raw({"enabled": "on"})
        on = ToolSearchConfig.from_raw({
            "enabled": "on",
            "reranker": {"enabled": True, "endpoint": "http://localhost:11434/v1/embeddings"},
        })
        a = json.loads(dispatch_tool_search({"queries": ["zones"]}, current_tool_defs=cf_defs, config=off))
        b = json.loads(dispatch_tool_search({"queries": ["zones"]}, current_tool_defs=cf_defs, config=on))
        assert a["results"][0]["matches"] and b["results"][0]["matches"]
        assert set(a) == set(b)
        assert set(a["results"][0]) == set(b["results"][0])
        assert a["total_available"] == b["total_available"]


# ---------------------------------------------------------------------------
# RRF math
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_exact_scores(self):
        from tools.tool_search import CatalogEntry, _rrf_fuse

        a = CatalogEntry(name="a", description="", schema={}, source="mcp", source_name="s")
        b = CatalogEntry(name="b", description="", schema={}, source="mcp", source_name="s")
        c = CatalogEntry(name="c", description="", schema={}, source="mcp", source_name="s")
        # BM25: a, b   embed: b, c   (k=10)
        # a = 1/11            = 0.0909
        # b = 1/12 + 1/11     = 0.1742
        # c = 1/12            = 0.0833
        fused = _rrf_fuse([(9.0, a), (8.0, b)], [(0.9, b), (0.8, c)], k=10, top_n=3)
        assert [e.name for e in fused] == ["b", "a", "c"]

    def test_rrf_top_n_and_dedup(self):
        from tools.tool_search import CatalogEntry, _rrf_fuse

        ents = [CatalogEntry(name=n, description="", schema={}, source="mcp", source_name="s")
                for n in "abcd"]
        fused = _rrf_fuse([(1.0, e) for e in ents], [(1.0, e) for e in reversed(ents)], k=10, top_n=2)
        assert len(fused) == 2
        assert len({e.name for e in fused}) == 2

    def test_rrf_mode_blends_lexical_and_semantic(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        reranker = _get_reranker(_cfg(mode="rrf", rrf_k=10), catalog)
        hits = [e.name for e in search_catalog(catalog, "list accounts", limit=3, reranker=reranker)]
        # get_accounts is top of the embedding list and present in BM25's;
        # it must be at or near the top after fusion.
        assert "get_accounts" in hits[:2]


# ---------------------------------------------------------------------------
# Payload, prefixes, credentials
# ---------------------------------------------------------------------------


class TestEmbedPayload:
    def test_prefixes_applied_to_docs_and_query(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        reranker = _get_reranker(_cfg(query_prefix="Q: ", doc_prefix="D: "), catalog)
        search_catalog(catalog, "zones", limit=1, reranker=reranker)
        docs, (query,) = fake_embed[0], fake_embed[1]
        assert all(d.startswith("D: ") for d in docs)
        assert query == "Q: zones"

    def test_empty_prefixes_send_bare_text(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        reranker = _get_reranker(_cfg(query_prefix="", doc_prefix=""), catalog)
        search_catalog(catalog, "zones", limit=1, reranker=reranker)
        assert fake_embed[1] == ["zones"]
        assert not any(d.startswith("search_document") for d in fake_embed[0])

    def test_embed_text_is_name_words_plus_description(self):
        from tools.tool_search import _entry_embed_text
        td = _td("mcp__cloudflare__get_accounts", "GET /accounts  List   Accounts.")
        assert _entry_embed_text(td, "cloudflare") == "cloudflare cloudflare  get accounts: GET /accounts List Accounts."
        assert _entry_embed_text(_td("get_zones", ""), "") == "get zones"

    def test_embed_text_clips_long_descriptions(self):
        from tools.tool_search import _entry_embed_text
        text = _entry_embed_text(_td("t", "x" * 5000))
        assert len(text) < 1100

    def test_http_request_shape_and_bearer_header(self, monkeypatch):
        """The only network path: OpenAI-compatible JSON body + optional Bearer."""
        import urllib.request
        from tools.tool_search import EmbeddingReranker

        seen: Dict[str, Any] = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]}).encode()

        def _urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            seen["headers"] = {k.lower(): v for k, v in req.header_items()}
            seen["body"] = json.loads(req.data)
            seen["timeout"] = timeout
            return _Resp()

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        monkeypatch.setenv("HERMES_EMBED_API_KEY", "sk-test")
        vecs = EmbeddingReranker(_cfg(model="m", timeout=3.0))._embed(["a", "b"])
        assert vecs == [[1.0, 0.0], [0.0, 1.0]]  # re-sorted by index
        assert seen["url"] == "http://localhost:11434/v1/embeddings"
        assert seen["method"] == "POST"
        assert seen["body"] == {"model": "m", "input": ["a", "b"]}
        assert seen["headers"]["authorization"] == "Bearer sk-test"
        assert seen["timeout"] == 3.0

    def test_no_bearer_header_without_key(self, monkeypatch):
        import urllib.request
        from tools.tool_search import EmbeddingReranker

        seen: Dict[str, Any] = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"data": [{"index": 0, "embedding": [1.0]}]}).encode()

        def _urlopen(req, timeout=None):
            seen["headers"] = {k.lower() for k, _ in req.header_items()}
            return _Resp()

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        monkeypatch.delenv("HERMES_EMBED_API_KEY", raising=False)
        EmbeddingReranker(_cfg())._embed(["a"])
        assert "authorization" not in seen["headers"]


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_catalog_embedded_once_then_one_call_per_query(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        reranker = _get_reranker(_cfg(), catalog)
        search_catalog(catalog, "zones", limit=1, reranker=reranker)
        search_catalog(catalog, "accounts", limit=1, reranker=reranker)
        search_catalog(catalog, "zones", limit=1, reranker=reranker)  # cached query
        assert [len(c) for c in fake_embed] == [len(_CF_DEFS), 1, 1]

    def test_vectors_are_unit_normalized_float32(self, fake_embed):
        import array
        import math
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        reranker = _get_reranker(_cfg(), catalog)
        search_catalog(catalog, "zones", limit=1, reranker=reranker)
        for vec in reranker._cache.values():
            assert isinstance(vec, array.array) and vec.typecode == "f"
            norm = math.sqrt(sum(x * x for x in vec))
            assert norm == 0.0 or abs(norm - 1.0) < 1e-5

    def test_cache_key_includes_model(self, fake_embed):
        from tools.tool_search import _get_reranker, build_catalog, search_catalog

        catalog = build_catalog(_CF_DEFS)
        r1 = _get_reranker(_cfg(model="m1"), catalog)
        r2 = _get_reranker(_cfg(model="m2"), catalog)
        assert r1 is not r2
        search_catalog(catalog, "zones", limit=1, reranker=r1)
        search_catalog(catalog, "zones", limit=1, reranker=r2)
        assert len(fake_embed) == 4  # both scopes embed the catalog + query

    def test_batches_are_bounded_and_progress_survives_a_failure(self, monkeypatch):
        from tools import tool_search
        from tools.tool_search import EmbeddingReranker, _get_reranker, build_catalog, search_catalog

        monkeypatch.setattr(tool_search, "_EMBED_BATCH_SIZE", 3)
        defs = [_td(f"tool_{i}", f"Tool number {i} does thing {i}.") for i in range(8)]
        catalog = build_catalog(defs)
        calls: List[int] = []
        state = {"fail_after": 2}

        def _embed(self, texts):
            calls.append(len(texts))
            if len(calls) > state["fail_after"]:
                raise OSError("timeout")
            return [[1.0, float(i)] for i, _ in enumerate(texts)]

        monkeypatch.setattr(EmbeddingReranker, "_embed", _embed)
        reranker = _get_reranker(_cfg(), catalog)
        # First search: 3 + 3 succeed, third batch fails -> BM25 fallback.
        bm25 = search_catalog(catalog, "tool 4", limit=2)
        assert search_catalog(catalog, "tool 4", limit=2, reranker=reranker) == bm25
        assert calls == [3, 3, 2]
        assert len(reranker._cache) == 6
        # Second search: only the remaining 2 docs + the query are embedded.
        state["fail_after"] = 99
        search_catalog(catalog, "tool 4", limit=2, reranker=reranker)
        assert calls == [3, 3, 2, 2, 1]
        assert len(reranker._cache) == 9


# ---------------------------------------------------------------------------
# Scope-keyed bounded reranker cache (review point 1 on #35457)
# ---------------------------------------------------------------------------


class TestScopeCache:
    def _catalog(self, *names):
        from tools.tool_search import build_catalog
        return build_catalog([_td(n, f"{n} description") for n in names])

    def test_same_scope_reuses_instance(self):
        from tools.tool_search import _get_reranker
        cat = self._catalog("a", "b")
        assert _get_reranker(_cfg(), cat) is _get_reranker(_cfg(), cat)

    def test_a_b_a_reuses_scope_a_without_rebuild(self, fake_embed):
        from tools import tool_search
        from tools.tool_search import _get_reranker, search_catalog

        cat_a = self._catalog("alpha_tool", "beta_tool")
        cat_b = self._catalog("gamma_tool")
        ra = _get_reranker(_cfg(), cat_a)
        search_catalog(cat_a, "alpha", limit=1, reranker=ra)
        rb = _get_reranker(_cfg(), cat_b)
        search_catalog(cat_b, "gamma", limit=1, reranker=rb)
        assert rb is not ra
        assert _get_reranker(_cfg(), cat_a) is ra
        assert len(tool_search._reranker_cache) == 2
        # Scope A's vectors survived scope B's creation: another A query
        # embeds only the query.
        n_before = len(fake_embed)
        search_catalog(cat_a, "beta", limit=1, reranker=_get_reranker(_cfg(), cat_a))
        assert [len(c) for c in fake_embed[n_before:]] == [1]

    def test_fifo_eviction_at_capacity(self, monkeypatch):
        from tools import tool_search
        from tools.tool_search import _get_reranker

        monkeypatch.setattr(tool_search, "_RERANKER_CACHE_MAX", 2)
        r1 = _get_reranker(_cfg(), self._catalog("t1"))
        r2 = _get_reranker(_cfg(), self._catalog("t2"))
        r3 = _get_reranker(_cfg(), self._catalog("t3"))
        assert len(tool_search._reranker_cache) == 2
        assert _get_reranker(_cfg(), self._catalog("t2")) is r2
        assert _get_reranker(_cfg(), self._catalog("t3")) is r3
        assert _get_reranker(_cfg(), self._catalog("t1")) is not r1  # evicted, rebuilt

    def test_scope_key_includes_behavior_fields(self):
        from tools.tool_search import _get_reranker
        cat = self._catalog("a")
        base = _get_reranker(_cfg(), cat)
        for override in (
            {"mode": "rrf"}, {"rrf_k": 60}, {"query_prefix": ""}, {"doc_prefix": ""},
            {"model": "other"}, {"endpoint": "http://other/v1/embeddings"},
        ):
            assert _get_reranker(_cfg(**override), cat) is not base, override

    def test_scope_key_is_not_comma_ambiguous(self):
        from tools.tool_search import _get_reranker
        assert _get_reranker(_cfg(), self._catalog("a,b")) is not \
            _get_reranker(_cfg(), self._catalog("a", "b"))

    def test_concurrent_scope_creation_is_thread_safe(self, monkeypatch):
        import threading
        from tools import tool_search
        from tools.tool_search import _get_reranker

        monkeypatch.setattr(tool_search, "_RERANKER_CACHE_MAX", 3)
        cats = [self._catalog(f"tool_{i}") for i in range(12)]
        errors: List[BaseException] = []

        def _worker(i):
            try:
                for _ in range(50):
                    _get_reranker(_cfg(), cats[i % len(cats)])
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(tool_search._reranker_cache) <= 3
