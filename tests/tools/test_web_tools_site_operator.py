"""Tests for the ``site:<domain>`` search operator handling.

Background: ``site:nature.com CRISPR`` style queries used to be passed to
web-search providers verbatim, so providers with a native domain-filter
parameter never used it and the ``site:`` token reached the backend as a
literal search term.

These tests pin the shared ``parse_site_operator`` helper plus the
provider-specific wiring: both Exa and Tavily receive the residual query
plus their native domain filter (``include_domains``).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# ─── parse_site_operator (shared helper) ─────────────────────────────────────


class TestParseSiteOperator:
    def test_single_operator_leading(self):
        from tools.web_tools import parse_site_operator

        domains, residual = parse_site_operator("site:nature.com CRISPR")
        assert domains == ["nature.com"]
        assert residual == "CRISPR"

    def test_no_operator_passthrough(self):
        from tools.web_tools import parse_site_operator

        domains, residual = parse_site_operator("CRISPR gene editing")
        assert domains == []
        assert residual == "CRISPR gene editing"

    def test_operator_midquery(self):
        from tools.web_tools import parse_site_operator

        domains, residual = parse_site_operator("CRISPR site:nature.com review")
        assert domains == ["nature.com"]
        assert residual == "CRISPR review"

    def test_multiple_operators(self):
        from tools.web_tools import parse_site_operator

        domains, residual = parse_site_operator(
            "site:nature.com site:science.org foo bar"
        )
        assert domains == ["nature.com", "science.org"]
        assert residual == "foo bar"

    def test_strips_scheme_and_dedups(self):
        from tools.web_tools import parse_site_operator

        domains, residual = parse_site_operator(
            "site:https://nature.com site:nature.com foo"
        )
        assert domains == ["nature.com"]
        assert residual == "foo"

    def test_empty_query(self):
        from tools.web_tools import parse_site_operator

        domains, residual = parse_site_operator("")
        assert domains == []
        assert residual == ""


# ─── Exa: include_domains + residual query ───────────────────────────────────


class TestExaSiteOperator:
    def _mock_client(self):
        client = MagicMock()
        resp = MagicMock()
        resp.results = []
        client.search.return_value = resp
        return client

    def test_site_query_uses_include_domains_and_residual(self):
        from plugins.web.exa.provider import ExaWebSearchProvider

        client = self._mock_client()
        with patch.dict(os.environ, {"EXA_API_KEY": "exa-key"}):
            with patch(
                "plugins.web.exa.provider._get_exa_client", return_value=client
            ):
                ExaWebSearchProvider().search("site:nature.com CRISPR", limit=5)

        client.search.assert_called_once()
        call = client.search.call_args
        # residual query passed positionally, site: token stripped
        assert call.args[0] == "CRISPR"
        assert call.kwargs.get("include_domains") == ["nature.com"]

    def test_plain_query_omits_include_domains(self):
        from plugins.web.exa.provider import ExaWebSearchProvider

        client = self._mock_client()
        with patch.dict(os.environ, {"EXA_API_KEY": "exa-key"}):
            with patch(
                "plugins.web.exa.provider._get_exa_client", return_value=client
            ):
                ExaWebSearchProvider().search("CRISPR gene editing", limit=5)

        call = client.search.call_args
        assert call.args[0] == "CRISPR gene editing"
        assert call.kwargs.get("include_domains") is None


# ─── Tavily: include_domains + residual query ────────────────────────────────


class TestTavilySiteOperator:
    @staticmethod
    def _search(query, limit=5):
        from plugins.web.tavily.provider import TavilyWebSearchProvider

        captured = {}

        def fake_request(endpoint, payload):
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            return {"results": []}

        with patch(
            "plugins.web.tavily.provider._tavily_request", side_effect=fake_request
        ):
            TavilyWebSearchProvider().search(query, limit=limit)

        return captured["payload"]

    def test_site_query_uses_include_domains_and_residual(self):
        payload = self._search("site:nature.com CRISPR")

        assert payload["include_domains"] == ["nature.com"]
        # The operator is translated, not forwarded as a literal term.
        assert payload["query"] == "CRISPR"
        assert "site:" not in payload["query"]

    def test_site_query_keeps_default_depth(self):
        """Scoped queries must not silently upgrade to the 2-credit tier."""
        payload = self._search("site:nature.com CRISPR")

        assert "search_depth" not in payload

    def test_multiple_operators_forwarded(self):
        payload = self._search("site:nature.com site:science.org CRISPR")

        assert payload["include_domains"] == ["nature.com", "science.org"]
        assert payload["query"] == "CRISPR"

    def test_plain_query_omits_include_domains(self):
        payload = self._search("CRISPR gene editing")

        assert "include_domains" not in payload
        assert "search_depth" not in payload
        assert payload["query"] == "CRISPR gene editing"
