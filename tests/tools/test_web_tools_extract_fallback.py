"""Regression tests for the web_extract configurable fallback chain (``web.extract_backends``).

Covers the review findings on the chain dispatch (``tools.web_tools_extract._extract_with_fallback``,
driven from ``tools.web_tools.web_extract_tool``):

  1. Plugin discovery must run BEFORE chain entries are resolved, so a custom
     fallback provider that only becomes registered at discovery time (cold
     start — subprocess agent runs, delegate children, standalone scripts)
     is still attempted.
  2. An explicit chain entry that fails to resolve to a registered provider
     must be skipped (recorded as a typed error) — never silently replaced by
     the scalar "active" provider, which resolves independently from
     ``web.extract_backend`` / ``web.backend`` and may not even be a member
     of the configured chain.
  3. Duplicate entries in the configured chain (e.g. ``[a, b, a]``) must not
     short-circuit the "is this the last attempt" check — every distinct
     backend in the chain is still attempted.
  4. All-error / empty-response / exception / CONTENTLESS outcomes from a
     backend fall through to the next chain entry; the final entry's outcome
     is surfaced exactly as a single backend's would be.
  5. A ``blocked_by_policy`` result is a terminal decision, NOT a retryable
     all-error outcome — the next backend must not be asked for the same
     blocked URL, and the marker must survive into the tool output.
  6. Users who never configured ``web.extract_backends`` keep the pre-chain
     active-provider walk; explicit chains never take it.
  7. An empty provider response must not swallow the reconstructed
     invalid-URL / private-network diagnostics.
  8. Chain entries are normalized: blanks/None dropped, duplicates collapsed,
     configured order preserved; the chain wins over the scalar key.
  9. The one-shot keyless rescue is withheld from non-final entries (the
     configured chain, not the free ring, is the fallback) and kept for the
     final one.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import web_search_registry
from agent.web_search_provider import WebSearchProvider
from tools import web_tools
from tools import web_tools_extract as wte


class _FakeExtractProvider(WebSearchProvider):
    """Minimal configurable extract-only provider for dispatch tests."""

    def __init__(self, name, *, available=True, respond=None, raises=None, empty=False):
        self._name = name
        self._available = available
        self._respond = respond  # callable(urls) -> list[dict] | None
        self._raises = raises
        self._empty = empty
        self.calls = 0

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._name

    def is_available(self):
        return self._available

    def supports_extract(self):
        return True

    async def extract(self, urls, **kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        if self._empty:
            return []
        if self._respond is not None:
            return self._respond(urls)
        return [
            {
                "url": u, "title": "", "content": f"ok-from-{self._name}",
                "raw_content": f"ok-from-{self._name}",
            }
            for u in urls
        ]


class _SearchOnlyProvider(WebSearchProvider):
    @property
    def name(self):
        return "search-only"

    @property
    def display_name(self):
        return "Search Only"

    def is_available(self):
        return True

    def supports_search(self):
        return True

    def search(self, query, limit=5):
        return {"success": True, "data": {"web": []}}


def _error_results(name):
    """Build a ``respond`` callable whose results all carry an error."""
    def _respond(urls):
        return [
            {"url": u, "title": "", "content": "", "raw_content": "",
             "error": f"{name} failed"}
            for u in urls
        ]
    return _respond


def _contentless_results(name, content=""):
    """Build a ``respond`` callable shaped like an HTTP-200-but-empty page.

    Matches what a backend emits for an unhydrated SPA shell or a scrape payload
    with neither markdown nor HTML: a title, blank content, and NO ``error``.
    """
    def _respond(urls):
        return [
            {"url": u, "title": f"{name} shell", "content": content,
             "raw_content": content, "metadata": {}, "error": None}
            for u in urls
        ]
    return _respond


def _policy_blocked_results(name):
    """Build a ``respond`` callable shaped like a website-policy block.

    Matches what the firecrawl provider emits when the website policy denies a
    host: a per-URL ``error`` PLUS a ``blocked_by_policy`` marker.
    """
    def _respond(urls):
        return [
            {"url": u, "title": "", "content": "", "raw_content": "",
             "error": f"Blocked by website policy ({name})",
             "blocked_by_policy": {
                 "host": "blocked.test",
                 "rule": "blocked.test",
                 "source": "config",
             }}
            for u in urls
        ]
    return _respond


@pytest.fixture
def clean_registry():
    """Snapshot/restore the web provider registry around a test."""
    with web_search_registry._lock:
        previous = dict(web_search_registry._providers)
        web_search_registry._providers.clear()
    yield
    with web_search_registry._lock:
        web_search_registry._providers.clear()
        web_search_registry._providers.update(previous)


@pytest.fixture
def safe_urls(monkeypatch):
    """Bypass the SSRF probe so plain https:// test URLs dispatch normally."""
    async def _safe(_url):
        return True
    monkeypatch.setattr(web_tools, "async_is_safe_url", _safe)


@pytest.fixture
def no_discovery(monkeypatch):
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)


@pytest.fixture(autouse=True)
def _no_keyless_rescue(monkeypatch):
    """Hermetic default: the one-shot keyless rescue is its own feature
    (test_web_keyless_rescue.py). Left on, a deliberately failing FINAL entry
    here would turn into a live free-ring call. The rescue-gating tests below
    re-enable it explicitly with the ring patched out."""
    monkeypatch.setattr(wte, "_rescue_eligible", lambda _provider: False)


def _chain(monkeypatch, *names, **extra):
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: {"extract_backends": list(names), **extra},
    )


def _register(*providers):
    for p in providers:
        web_search_registry.register_provider(p)


# ─── Finding 1: discovery must precede chain resolution ─────────────────────


class TestColdStartPluginDiscoveryOrdering:
    @pytest.mark.asyncio
    async def test_custom_plugin_registered_at_discovery_is_attempted(
        self, clean_registry, safe_urls, monkeypatch
    ):
        # "already-loaded" simulates a provider registered before this call
        # (e.g. a built-in loaded earlier in process lifetime). It fails every
        # extraction, so the dispatcher must fall through to the next entry.
        already_loaded = _FakeExtractProvider(
            "already-loaded", respond=_error_results("already-loaded"),
        )
        _register(already_loaded)

        # "cold-start-plugin" is NOT registered until discovery runs — it
        # represents a custom fallback plugin whose registration only
        # happens via _ensure_web_plugins_loaded() in a fresh process.
        cold_start_plugin = _FakeExtractProvider("cold-start-plugin")

        def _discover():
            if web_search_registry.get_provider("cold-start-plugin") is None:
                web_search_registry.register_provider(cold_start_plugin)

        mock_hook = MagicMock(wraps=_discover)
        monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", mock_hook)
        _chain(monkeypatch, "already-loaded", "cold-start-plugin")

        # Sanity: the custom plugin genuinely isn't registered pre-discovery.
        assert web_search_registry.get_provider("cold-start-plugin") is None

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert mock_hook.called
        assert already_loaded.calls == 1
        assert cold_start_plugin.calls == 1, (
            "cold-start-plugin must be attempted after already-loaded fails. "
            "If entries were resolved BEFORE discovery, this entry would look "
            "unregistered and be skipped."
        )
        assert result["results"][0]["content"] == "ok-from-cold-start-plugin"


# ─── Finding 2: unregistered explicit entry must not fall back to the ──────
# ─── scalar active provider ─────────────────────────────────────────────────


class TestExplicitUnregisteredEntryNeverSubstitutesActiveProvider:
    @pytest.mark.asyncio
    async def test_unregistered_explicit_entry_is_skipped_not_replaced(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        # A fully valid, resolvable provider that ``get_active_extract_provider()``
        # would hand back (it reads web.extract_backend / web.backend, a
        # completely separate resolution path from web.extract_backends).
        # It must never be dispatched for a chain entry that itself fails
        # to resolve.
        wrong_active_provider = _FakeExtractProvider("wrong-active-provider")
        mock_active = MagicMock(return_value=wrong_active_provider)
        monkeypatch.setattr(web_search_registry, "get_active_extract_provider", mock_active)
        _chain(monkeypatch, "totally-unregistered-name")

        raw = await web_tools.web_extract_tool(["https://example.com"])
        result = json.loads(raw)

        mock_active.assert_not_called()
        assert wrong_active_provider.calls == 0
        assert "wrong-active-provider" not in raw
        assert result.get("success") is False
        assert "totally-unregistered-name" in result["error"]
        assert "web.extract_backends" in result["error"]

    @pytest.mark.asyncio
    async def test_unregistered_entry_is_skipped_and_next_entry_serves(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_b)
        _chain(monkeypatch, "not-registered", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_search_only_entry_is_a_typed_error_not_a_silent_switch(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        wrong_active_provider = _FakeExtractProvider("wrong-active-provider")
        monkeypatch.setattr(
            web_search_registry, "get_active_extract_provider",
            MagicMock(return_value=wrong_active_provider),
        )
        _register(_SearchOnlyProvider())
        _chain(monkeypatch, "search-only")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert wrong_active_provider.calls == 0
        assert result.get("success") is False
        assert "search-only backend" in result["error"]

    @pytest.mark.asyncio
    async def test_trailing_unresolvable_entry_does_not_erase_a_real_fetch_outcome(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """A typo in a LATER slot must not replace the per-URL errors the real
        backend produced with a config error."""
        chain_a = _FakeExtractProvider("chain-a", respond=_error_results("chain-a"))
        _register(chain_a)
        _chain(monkeypatch, "chain-a", "typo-name")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert result["results"][0]["error"] == "chain-a failed"


# ─── Finding 3: duplicate chain entries must not block a remaining ─────────
# ─── distinct fallback ───────────────────────────────────────────────────────


class TestDuplicateChainEntriesStillAttemptRemainingFallbacks:
    @pytest.mark.asyncio
    async def test_duplicate_first_and_last_entry_does_not_skip_middle_fallback(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        # Mirrors the reported [firecrawl, tavily, firecrawl] shape with
        # neutral names so the test doesn't depend on real backend env vars.
        chain_a = _FakeExtractProvider("chain-a", respond=_error_results("chain-a"))
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b", "chain-a")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1, (
            "chain-a (index 0) should be attempted once before falling through — "
            "the duplicate trailing 'chain-a' must not make index 0 look final"
        )
        assert chain_b.calls == 1, "chain-b is the distinct remaining fallback and must be attempted"
        assert result["results"][0]["content"] == "ok-from-chain-b"


# ─── Finding 4: all-error / empty / exception outcomes ──────────────────────


class TestAllErrorEmptyExceptionOutcomes:
    @pytest.mark.asyncio
    async def test_single_backend_all_error_falls_through_to_next(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider("chain-a", respond=_error_results("chain-a"))
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_all_backends_error_surfaces_last_attempted_results(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider("chain-a", respond=_error_results("chain-a"))
        chain_b = _FakeExtractProvider("chain-b", respond=_error_results("chain-b"))
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert result["results"][0]["error"] == "chain-b failed"

    @pytest.mark.asyncio
    async def test_partial_success_is_a_final_answer(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """One usable page out of two is not shopped to the next backend."""
        def _mixed(urls):
            return [
                {"url": urls[0], "title": "", "content": "page one", "raw_content": "page one"},
                {"url": urls[1], "title": "", "content": "", "raw_content": "", "error": "404"},
            ]

        chain_a = _FakeExtractProvider("chain-a", respond=_mixed)
        must_not_run = _FakeExtractProvider("chain-b")
        _register(chain_a, must_not_run)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(
            ["https://example.com/one", "https://example.com/two"]
        ))

        assert chain_a.calls == 1
        assert must_not_run.calls == 0
        assert result["results"][0]["content"] == "page one"
        assert result["results"][1]["error"] == "404"

    @pytest.mark.asyncio
    async def test_empty_response_falls_through_to_next_backend(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider("chain-a", empty=True)
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_all_backends_empty_surfaces_the_single_backend_error(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """The final entry's empty response is surfaced exactly as one backend's
        would be: the tool's "inaccessible" error, never a results list."""
        chain_a = _FakeExtractProvider("chain-a", empty=True)
        chain_b = _FakeExtractProvider("chain-b", empty=True)
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert "results" not in result
        assert "inaccessible" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_falls_through_to_next_backend(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider("chain-a", raises=RuntimeError("boom"))
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_all_backends_raise_surfaces_last_exception_error(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider("chain-a", raises=RuntimeError("first boom"))
        chain_b = _FakeExtractProvider("chain-b", raises=RuntimeError("second boom"))
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert "results" not in result
        assert "second boom" in result["error"]
        assert "first boom" not in result["error"]


# ─── Contentless rows (HTTP 200, empty body, no error) are retryable ────────


class TestContentlessRowsAreRetryable:
    """A provider may answer an unhydrated SPA (or a soft bot wall) with HTTP 200
    and an empty body — a row with blank ``content``/``raw_content`` and NO
    ``error``. Judging failure by ``error`` alone treats that as success and
    returns an empty page without consulting the next configured backend."""

    @pytest.mark.asyncio
    async def test_contentless_rows_fall_through_to_next_backend(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider("chain-a", respond=_contentless_results("chain-a"))
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://spa.example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1, "a contentless, errorless batch must not be treated as success"
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_whitespace_only_content_is_contentless(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        chain_a = _FakeExtractProvider(
            "chain-a", respond=_contentless_results("chain-a", content="  \n\t "),
        )
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://spa.example.com"]))

        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_raw_content_alone_counts_as_usable(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """Only ``raw_content`` populated (a provider that fills one field) is
        still a usable page — no fall-through."""
        def _raw_only(urls):
            return [{"url": u, "title": "", "content": "", "raw_content": "<p>raw</p>"} for u in urls]

        chain_a = _FakeExtractProvider("chain-a", respond=_raw_only)
        must_not_run = _FakeExtractProvider("chain-b")
        _register(chain_a, must_not_run)
        _chain(monkeypatch, "chain-a", "chain-b")

        await web_tools.web_extract_tool(["https://example.com"])

        assert chain_a.calls == 1
        assert must_not_run.calls == 0

    @pytest.mark.asyncio
    async def test_mixed_contentless_and_usable_rows_are_partial_success(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        def _mixed(urls):
            return [
                {"url": urls[0], "title": "shell", "content": "", "raw_content": ""},
                {"url": urls[1], "title": "", "content": "real page", "raw_content": "real page"},
            ]

        chain_a = _FakeExtractProvider("chain-a", respond=_mixed)
        must_not_run = _FakeExtractProvider("chain-b")
        _register(chain_a, must_not_run)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(
            ["https://spa.example.com", "https://example.com"]
        ))

        assert must_not_run.calls == 0
        assert result["results"][1]["content"] == "real page"

    @pytest.mark.asyncio
    async def test_contentless_rows_from_the_final_entry_are_returned_as_is(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """When every entry is contentless the final one's rows ride through
        unchanged — the same answer a single contentless backend gives."""
        chain_a = _FakeExtractProvider("chain-a", respond=_contentless_results("chain-a"))
        chain_b = _FakeExtractProvider("chain-b", respond=_contentless_results("chain-b"))
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://spa.example.com"]))

        assert chain_a.calls == 1
        assert chain_b.calls == 1
        assert result["results"][0]["title"] == "chain-b shell"
        assert result["results"][0]["content"] == ""
        assert not result["results"][0]["error"]

    def test_row_classification_helpers(self):
        assert wte._contentless({"content": "", "raw_content": ""})
        assert wte._contentless({"content": "   ", "raw_content": None})
        assert wte._contentless({"title": "only a title"})
        assert wte._contentless("not a dict")
        assert not wte._contentless({"content": "text"})
        assert not wte._contentless({"content": "", "raw_content": "<p>x</p>"})

        assert wte._batch_failed([])
        assert wte._batch_failed([{"error": "x"}, {"content": ""}])
        assert not wte._batch_failed([{"content": "ok"}, {"content": ""}])
        assert not wte._batch_failed([{"content": "ok"}, {"error": "x"}])


# ─── A website-policy block is terminal, not a retryable backend failure ────


class TestPolicyBlockIsTerminal:
    @pytest.mark.asyncio
    async def test_blocked_by_policy_does_not_fall_through_to_next_backend(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """A policy-blocked result carries an ``error``, so the all-error
        fallthrough would otherwise shop the forbidden URL around the chain
        until some provider isn't policy-aware. The block must stop the chain
        and its marker must survive into the tool output."""
        blocking = _FakeExtractProvider(
            "chain-a", respond=_policy_blocked_results("chain-a"),
        )
        must_not_run = _FakeExtractProvider("chain-b")
        _register(blocking, must_not_run)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(["https://blocked.test/x"]))

        assert blocking.calls == 1
        assert must_not_run.calls == 0, (
            "a policy block must be terminal — the next backend must never be "
            "asked for the same blocked URL"
        )
        entry = result["results"][0]
        assert entry["blocked_by_policy"]["rule"] == "blocked.test"
        assert entry["error"]

    @pytest.mark.asyncio
    async def test_partial_policy_block_still_stops_the_chain(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """One blocked URL among otherwise-failed ones is still a policy
        decision — the whole batch must not be retried elsewhere."""
        def _mixed(urls):
            return [
                {"url": urls[0], "title": "", "content": "", "raw_content": "",
                 "error": "Blocked by website policy",
                 "blocked_by_policy": {"host": "blocked.test",
                                       "rule": "blocked.test",
                                       "source": "config"}},
                {"url": urls[1], "title": "", "content": "", "raw_content": "",
                 "error": "chain-a failed"},
            ]

        blocking = _FakeExtractProvider("chain-a", respond=_mixed)
        must_not_run = _FakeExtractProvider("chain-b")
        _register(blocking, must_not_run)
        _chain(monkeypatch, "chain-a", "chain-b")

        result = json.loads(await web_tools.web_extract_tool(
            ["https://blocked.test/x", "https://example.com"]
        ))

        assert blocking.calls == 1
        assert must_not_run.calls == 0
        assert result["results"][0]["blocked_by_policy"]["rule"] == "blocked.test"


# ─── Legacy (non-chain) resolution keeps the active-provider walk ───────────


class TestLegacyScalarResolutionKeepsActiveProviderWalk:
    @pytest.mark.asyncio
    async def test_unregistered_scalar_backend_still_walks_to_active_provider(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        """Users who never set ``web.extract_backends`` (and have no stored web
        selection) keep the pre-chain behavior: an auto-detected name that
        isn't a registered provider falls through to
        ``get_active_extract_provider()`` instead of erroring out."""
        rescued = _FakeExtractProvider("rescued-active-provider")
        mock_active = MagicMock(return_value=rescued)
        monkeypatch.setattr(
            web_search_registry, "get_active_extract_provider", mock_active,
        )
        monkeypatch.setattr(wte, "selection_exists", lambda _section: False)
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backend": "not-registered-anywhere"},
        )

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert mock_active.called
        assert rescued.calls == 1
        assert result["results"][0]["content"] == "ok-from-rescued-active-provider"

    @pytest.mark.asyncio
    async def test_scalar_backend_is_dispatched_as_a_single_entry(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        only = _FakeExtractProvider("only-one", respond=_error_results("only-one"))
        _register(only)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_backend": "only-one"})

        result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        assert only.calls == 1
        assert result["results"][0]["error"] == "only-one failed"


# ─── Empty provider response must not eat the per-URL diagnostics ───────────


class TestEmptyResponsePreservesUrlDiagnostics:
    @pytest.mark.asyncio
    async def test_invalid_and_private_url_entries_survive_an_empty_response(
        self, clean_registry, no_discovery, monkeypatch
    ):
        """When URLs were rejected up front (malformed / private-network), the
        reconstructed per-URL diagnostics are the answer — a backend that then
        returns nothing must not replace them with a bare provider error."""
        empty_provider = _FakeExtractProvider("chain-a", empty=True)
        _register(empty_provider)

        async def _safe(url):
            return "169.254.169.254" not in url

        monkeypatch.setattr(web_tools, "async_is_safe_url", _safe)
        _chain(monkeypatch, "chain-a")

        result = json.loads(await web_tools.web_extract_tool(
            ["https://example.com", "http://169.254.169.254/latest/meta-data", 12345]
        ))

        assert empty_provider.calls == 1
        results = result["results"]
        assert len(results) == 3
        assert results[0]["error"] == "Extract backend returned no result for this URL"
        assert "private or internal" in results[1]["error"]
        assert "Invalid URL item at index 2" in results[2]["error"]


# ─── Keyless rescue: withheld from non-final entries, kept for the last ─────


class TestKeylessRescueIsReservedForTheFinalEntry:
    """With the rescue enabled, ``_rescue_eligible`` is true for any non-ring
    backend. A failing non-final entry must fall through to the configured
    chain, not to the free ring; the final entry keeps the one-shot rescue a
    single backend gets today."""

    @pytest.fixture(autouse=True)
    def _rescue_on(self, monkeypatch, _no_keyless_rescue):  # runs after the module default
        monkeypatch.setattr(wte, "_rescue_eligible", lambda _provider: True)

    @pytest.mark.asyncio
    async def test_non_final_failure_goes_to_the_next_entry_not_the_ring(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        from plugins.web import keyless_mcp

        chain_a = _FakeExtractProvider("chain-a", raises=RuntimeError("boom"))
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        ring.assert_not_called()
        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_non_final_all_error_batch_goes_to_the_next_entry_not_the_ring(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        from plugins.web import keyless_mcp

        chain_a = _FakeExtractProvider("chain-a", respond=_error_results("chain-a"))
        chain_b = _FakeExtractProvider("chain-b")
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")

        with patch.object(keyless_mcp, "extract_with_failover") as ring:
            result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        ring.assert_not_called()
        assert chain_b.calls == 1
        assert result["results"][0]["content"] == "ok-from-chain-b"

    @pytest.mark.asyncio
    async def test_final_entry_keeps_the_one_shot_rescue(
        self, clean_registry, safe_urls, no_discovery, monkeypatch
    ):
        from plugins.web import keyless_mcp

        chain_a = _FakeExtractProvider("chain-a", raises=RuntimeError("first boom"))
        chain_b = _FakeExtractProvider("chain-b", raises=RuntimeError("second boom"))
        _register(chain_a, chain_b)
        _chain(monkeypatch, "chain-a", "chain-b")
        rescued = [{"url": "https://example.com", "title": "R", "content": "from ring",
                    "raw_content": "from ring", "metadata": {}}]

        with patch.object(keyless_mcp, "extract_with_failover", return_value=rescued) as ring:
            result = json.loads(await web_tools.web_extract_tool(["https://example.com"]))

        ring.assert_called_once()
        assert result["results"][0]["content"] == "from ring"


# ─── Chain normalization: blanks dropped, duplicates collapsed, order kept ──


class TestChainNormalization:
    def test_blank_none_and_duplicate_entries_are_normalized(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backends":
                     ["Chain-A", "", None, "  chain-b  ", "chain-a"]},
        )

        assert web_tools._get_extract_backends() == ["chain-a", "chain-b"]

    def test_chain_wins_over_the_scalar_key(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backends": ["chain-a", "chain-b"], "extract_backend": "tavily"},
        )
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        assert web_tools._get_extract_backends() == ["chain-a", "chain-b"]
        # The scalar key itself is untouched — it is simply not what dispatch walks.
        assert web_tools._get_extract_backend() == "tavily"

    def test_non_list_value_counts_as_unset(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backends": "tavily", "extract_backend": "keenable"},
        )
        monkeypatch.setenv("KEENABLE_API_KEY", "test-key")

        assert web_tools._explicit_extract_chain() == []
        assert web_tools._get_extract_backends() == ["keenable"]

    def test_empty_chain_falls_through_to_scalar_resolution(self, monkeypatch):
        monkeypatch.setattr(
            web_tools, "_load_web_config",
            lambda: {"extract_backends": [], "extract_backend": "tavily"},
        )
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        assert web_tools._get_extract_backends() == ["tavily"]
