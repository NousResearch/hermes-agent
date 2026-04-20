"""Tests for Nimble web backend integration.

Coverage:
  _get_nimble_client / _get_async_nimble_client — API key handling, lazy init.
  _nimble_search() — SDK call kwargs, response normalization, interrupt.
  _normalize_nimble_search_results() — search response → standard format.
  _normalize_nimble_extract_result() — extract / task-results → standard doc shape.
  _nimble_extract() — single-URL path (no polling) + batch path (poll + fetch).
  web_search_tool / web_extract_tool — Nimble dispatch.
"""

import json
import os
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ─── Client factories ────────────────────────────────────────────────────────

class TestNimbleClientInit:
    """Sync + async client initialization."""

    def setup_method(self):
        # Reset module-level client singletons between tests.
        import tools.web_tools as wt
        wt._nimble_client = None
        wt._async_nimble_client = None

    def test_sync_client_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NIMBLE_API_KEY", None)
            with patch("tools.web_tools._nimble_client", None):
                from tools.web_tools import _get_nimble_client
                with pytest.raises(ValueError, match="NIMBLE_API_KEY"):
                    _get_nimble_client()

    def test_async_client_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NIMBLE_API_KEY", None)
            with patch("tools.web_tools._async_nimble_client", None):
                from tools.web_tools import _get_async_nimble_client
                with pytest.raises(ValueError, match="NIMBLE_API_KEY"):
                    _get_async_nimble_client()


# ─── _normalize_nimble_search_results ─────────────────────────────────────────

class TestNormalizeNimbleSearchResults:

    def test_basic_normalization(self):
        from tools.web_tools import _normalize_nimble_search_results
        raw = {
            "results": [
                {"title": "Python Docs", "url": "https://docs.python.org", "description": "Official docs"},
                {"title": "Tutorial", "url": "https://example.com", "description": "A tutorial"},
            ]
        }
        result = _normalize_nimble_search_results(raw)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0]["title"] == "Python Docs"
        assert web[0]["url"] == "https://docs.python.org"
        assert web[0]["description"] == "Official docs"
        assert web[0]["position"] == 1
        assert web[1]["position"] == 2

    def test_empty_results(self):
        from tools.web_tools import _normalize_nimble_search_results
        result = _normalize_nimble_search_results({"results": []})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_fields(self):
        from tools.web_tools import _normalize_nimble_search_results
        result = _normalize_nimble_search_results({"results": [{}]})
        web = result["data"]["web"]
        assert web[0]["title"] == ""
        assert web[0]["url"] == ""
        assert web[0]["description"] == ""
        assert web[0]["position"] == 1

    def test_snippet_falls_back_to_description(self):
        """Some Nimble responses use 'snippet' instead of 'description'."""
        from tools.web_tools import _normalize_nimble_search_results
        result = _normalize_nimble_search_results(
            {"results": [{"title": "T", "url": "https://x", "snippet": "s"}]}
        )
        assert result["data"]["web"][0]["description"] == "s"

    def test_results_under_data_envelope(self):
        """Nimble may return {data: {results: [...]}} for some endpoints."""
        from tools.web_tools import _normalize_nimble_search_results
        result = _normalize_nimble_search_results(
            {"data": {"results": [{"title": "T", "url": "https://x"}]}}
        )
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["url"] == "https://x"

    def test_garbage_input(self):
        from tools.web_tools import _normalize_nimble_search_results
        assert _normalize_nimble_search_results({})["data"]["web"] == []
        assert _normalize_nimble_search_results({"results": None})["data"]["web"] == []


# ─── _normalize_nimble_extract_result ─────────────────────────────────────────

class TestNormalizeNimbleExtractResult:

    def test_markdown_in_data_envelope(self):
        from tools.web_tools import _normalize_nimble_extract_result
        raw = {"data": {"markdown": "# Heading\n\nbody", "title": "Page", "url": "https://x"}}
        doc = _normalize_nimble_extract_result(raw)
        assert doc["url"] == "https://x"
        assert doc["title"] == "Page"
        assert doc["content"] == "# Heading\n\nbody"
        assert doc["raw_content"] == "# Heading\n\nbody"
        assert doc["metadata"]["sourceURL"] == "https://x"

    def test_falls_back_to_html(self):
        from tools.web_tools import _normalize_nimble_extract_result
        doc = _normalize_nimble_extract_result(
            {"data": {"html": "<p>x</p>"}}, fallback_url="https://fb"
        )
        assert doc["content"] == "<p>x</p>"
        assert doc["url"] == "https://fb"

    def test_falls_back_to_plain_text(self):
        from tools.web_tools import _normalize_nimble_extract_result
        doc = _normalize_nimble_extract_result({"data": {"plain_text": "txt"}})
        assert doc["content"] == "txt"

    def test_empty_payload_uses_fallback_url(self):
        from tools.web_tools import _normalize_nimble_extract_result
        doc = _normalize_nimble_extract_result({}, fallback_url="https://f")
        assert doc["url"] == "https://f"
        assert doc["content"] == ""


# ─── _nimble_search ──────────────────────────────────────────────────────────

class TestNimbleSearch:

    def test_calls_sdk_with_expected_kwargs(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}

        with patch("tools.web_tools._get_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_search
            _nimble_search("hello world", limit=7)
            mock_client.search.assert_called_once_with(
                query="hello world",
                max_results=7,
                search_depth="lite",
            )

    def test_clamps_limit_to_max_20(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        with patch("tools.web_tools._get_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_search
            _nimble_search("q", limit=999)
            assert mock_client.search.call_args.kwargs["max_results"] == 20

    def test_returns_normalized_shape(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": "T", "url": "https://x", "description": "d"}]
        }
        with patch("tools.web_tools._get_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_search
            out = _nimble_search("q", limit=1)
            assert out["success"] is True
            assert out["data"]["web"][0]["url"] == "https://x"

    def test_returns_early_on_interrupt(self):
        mock_client = MagicMock()
        with patch("tools.web_tools._get_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_tools import _nimble_search
            out = _nimble_search("q")
            assert out == {"error": "Interrupted", "success": False}
            mock_client.search.assert_not_called()


# ─── _nimble_extract — single URL path ───────────────────────────────────────

class TestNimbleExtractSingle:

    def test_uses_extract_for_single_url(self):
        """One URL → single AsyncNimble.extract() call, no polling."""
        mock_client = MagicMock()
        mock_client.extract = AsyncMock(return_value={
            "data": {"markdown": "body", "title": "T", "url": "https://x"}
        })
        with patch("tools.web_tools._get_async_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://x"])
            )
            mock_client.extract.assert_called_once_with(url="https://x", formats=["markdown"])
            mock_client.extract_batch.assert_not_called()
            assert len(results) == 1
            assert results[0]["content"] == "body"

    def test_single_url_extract_exception_returns_error_doc(self):
        mock_client = MagicMock()
        mock_client.extract = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("tools.web_tools._get_async_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://x"])
            )
            assert len(results) == 1
            assert results[0]["error"] == "boom"
            assert results[0]["url"] == "https://x"

    def test_empty_urls_returns_empty(self):
        mock_client = MagicMock()
        with patch("tools.web_tools._get_async_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(_nimble_extract([]))
            assert results == []

    def test_interrupt_returns_error_docs(self):
        mock_client = MagicMock()
        with patch("tools.web_tools._get_async_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://x", "https://y"])
            )
            assert all(r["error"] == "Interrupted" for r in results)
            assert len(results) == 2


# ─── _nimble_extract — batch path (polling) ───────────────────────────────────

class TestNimbleExtractBatch:

    def _make_mock_client(self, batch_resp, progress_seq, task_results):
        """Build a mock client wired for one extract_batch run."""
        client = MagicMock()
        client.extract = AsyncMock()  # should NOT be called in batch path
        client.extract_batch = AsyncMock(return_value=batch_resp)
        client.batches = MagicMock()
        client.batches.progress = AsyncMock(side_effect=progress_seq)
        client.tasks = MagicMock()
        # tasks.results returns by task_id lookup
        async def _results(task_id):
            return task_results[task_id]
        client.tasks.results = AsyncMock(side_effect=_results)
        return client

    def test_batch_submits_polls_and_fetches_results(self):
        batch_resp = {
            "batch_id": "b1",
            "tasks": [
                {"id": "t1", "state": "success", "input": {"url": "https://a"}},
                {"id": "t2", "state": "success", "input": {"url": "https://b"}},
            ],
        }
        progress_seq = [
            {"completed": False, "progress": 0.5},
            {"completed": True, "progress": 1.0},
        ]
        task_results = {
            "t1": {"data": {"markdown": "A body", "url": "https://a"}},
            "t2": {"data": {"markdown": "B body", "url": "https://b"}},
        }
        client = self._make_mock_client(batch_resp, progress_seq, task_results)

        with patch("tools.web_tools._get_async_nimble_client", return_value=client), \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://a", "https://b"])
            )

        # extract_batch called with right inputs + shared_inputs
        call = client.extract_batch.call_args.kwargs
        assert call["inputs"] == [{"url": "https://a"}, {"url": "https://b"}]
        assert call["shared_inputs"] == {"formats": ["markdown"]}
        # Polled twice (first not completed, second completed)
        assert client.batches.progress.await_count == 2
        # Results normalized
        assert len(results) == 2
        assert {r["content"] for r in results} == {"A body", "B body"}

    def test_batch_state_error_surfaces_error_doc(self):
        batch_resp = {
            "batch_id": "b1",
            "tasks": [
                {"id": "t1", "state": "error", "error": "extraction failed",
                 "input": {"url": "https://a"}},
                {"id": "t2", "state": "success", "input": {"url": "https://b"}},
            ],
        }
        progress_seq = [{"completed": True, "progress": 1.0}]
        task_results = {"t2": {"data": {"markdown": "B body"}}}
        client = self._make_mock_client(batch_resp, progress_seq, task_results)

        with patch("tools.web_tools._get_async_nimble_client", return_value=client), \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://a", "https://b"])
            )

        errs = [r for r in results if "error" in r and r["error"]]
        oks = [r for r in results if not r.get("error")]
        assert len(errs) == 1
        assert errs[0]["url"] == "https://a"
        assert errs[0]["error"] == "extraction failed"
        assert len(oks) == 1
        # tasks.results only called for the success task
        assert client.tasks.results.await_count == 1

    def test_batch_no_batch_id_returns_error_for_each_url(self):
        client = MagicMock()
        client.extract_batch = AsyncMock(return_value={"tasks": []})  # missing batch_id
        with patch("tools.web_tools._get_async_nimble_client", return_value=client), \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://a", "https://b"])
            )
        assert len(results) == 2
        assert all("Nimble batch submission failed" in r["error"] for r in results)

    def test_batch_submit_exception_returns_error_for_each_url(self):
        client = MagicMock()
        client.extract_batch = AsyncMock(side_effect=RuntimeError("api down"))
        with patch("tools.web_tools._get_async_nimble_client", return_value=client), \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://a", "https://b"])
            )
        assert all(r["error"] == "api down" for r in results)

    def test_batch_interrupt_mid_poll(self):
        """Interrupt during polling returns error docs for all URLs."""
        batch_resp = {
            "batch_id": "b1",
            "tasks": [
                {"id": "t1", "state": "success", "input": {"url": "https://a"}},
            ],
        }
        client = MagicMock()
        client.extract_batch = AsyncMock(return_value=batch_resp)
        client.batches = MagicMock()
        # First poll returns not-completed
        client.batches.progress = AsyncMock(return_value={"completed": False, "progress": 0.0})
        client.tasks = MagicMock()
        client.tasks.results = AsyncMock()

        # Interrupt fires false on entry (so we get past initial check and submit)
        # then true on the next pass (the poll-loop check).
        is_int_calls = iter([False, True])
        with patch("tools.web_tools._get_async_nimble_client", return_value=client), \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()), \
             patch("tools.interrupt.is_interrupted", side_effect=lambda: next(is_int_calls)):
            from tools.web_tools import _nimble_extract
            results = asyncio.get_event_loop().run_until_complete(
                _nimble_extract(["https://a", "https://b"])
            )
        assert all(r["error"] == "Interrupted" for r in results)
        # tasks.results never called once interrupted
        client.tasks.results.assert_not_called()


# ─── web_search_tool dispatch ────────────────────────────────────────────────

class TestWebSearchNimbleDispatch:

    def test_dispatches_to_nimble(self):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [{"title": "R", "url": "https://r", "description": "d"}]
        }
        with patch("tools.web_tools._get_backend", return_value="nimble"), \
             patch("tools.web_tools._get_nimble_client", return_value=mock_client), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool
            result = json.loads(web_search_tool("q", limit=3))
            assert result["success"] is True
            assert result["data"]["web"][0]["title"] == "R"


# ─── web_extract_tool dispatch ────────────────────────────────────────────────

class TestWebExtractNimbleDispatch:

    def test_dispatches_single_url_to_nimble(self):
        mock_client = MagicMock()
        mock_client.extract = AsyncMock(return_value={
            "data": {"markdown": "body", "url": "https://x", "title": "T"}
        })
        with patch("tools.web_tools._get_backend", return_value="nimble"), \
             patch("tools.web_tools._get_async_nimble_client", return_value=mock_client), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.web_tools.process_content_with_llm", return_value=None), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_extract_tool
            raw = asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://x"], use_llm_processing=False)
            )
            # Single-URL path → extract() called, not extract_batch()
            mock_client.extract.assert_called_once()
            mock_client.extract_batch.assert_not_called()
            assert "body" in raw

    def test_dispatches_multi_url_to_batch(self):
        mock_client = MagicMock()
        mock_client.extract = AsyncMock()  # should NOT be called
        mock_client.extract_batch = AsyncMock(return_value={
            "batch_id": "b1",
            "tasks": [
                {"id": "t1", "state": "success", "input": {"url": "https://a"}},
                {"id": "t2", "state": "success", "input": {"url": "https://b"}},
            ],
        })
        mock_client.batches = MagicMock()
        mock_client.batches.progress = AsyncMock(return_value={"completed": True, "progress": 1.0})
        mock_client.tasks = MagicMock()
        async def _results(task_id):
            return {"data": {"markdown": f"body-{task_id}"}}
        mock_client.tasks.results = AsyncMock(side_effect=_results)

        with patch("tools.web_tools._get_backend", return_value="nimble"), \
             patch("tools.web_tools._get_async_nimble_client", return_value=mock_client), \
             patch("tools.web_tools.is_safe_url", return_value=True), \
             patch("tools.web_tools.check_website_access", return_value=None), \
             patch("tools.web_tools.asyncio.sleep", new=AsyncMock()), \
             patch("tools.web_tools.process_content_with_llm", return_value=None), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_extract_tool
            raw = asyncio.get_event_loop().run_until_complete(
                web_extract_tool(["https://a", "https://b"], use_llm_processing=False)
            )
            mock_client.extract_batch.assert_called_once()
            mock_client.extract.assert_not_called()
            assert "body-t1" in raw or "body-t2" in raw

