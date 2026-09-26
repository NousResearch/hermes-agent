"""Tests for the Exa provider's ``web.exa_max_age_hours`` freshness passthrough.

Covers:
- ``_get_exa_max_age_hours()`` config parsing — unset / 0 / positive int /
  float / numeric string / negative / non-numeric
- ``extract()`` keyed SDK path — ``max_age_hours`` is forwarded to
  ``exa-py get_contents`` only when configured; the default request stays
  bit-for-bit identical (``get_contents(urls, text=True)``)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _configured(exa_max_age_hours):
    """Patch ``tools.web_tools._load_web_config`` to a single-key web section."""
    return {"exa_max_age_hours": exa_max_age_hours}


def _sdk_result(url="https://example.com", title="Example", text="hello world"):
    m = MagicMock()
    m.url = url
    m.title = title
    m.text = text
    return m


def _mock_client(results):
    client = MagicMock()
    client.get_contents.return_value.results = results
    return client


class TestMaxAgeHoursParsing:
    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            (None, None),  # unset — Exa cache-first default
            (0, 0.0),  # always-live crawl
            (24, 24.0),
            (0.5, 0.5),  # fractional hours are meaningful
            ("24", 24.0),  # numeric string from hand-edited YAML
            (-1, None),  # negative — invalid, ignore
            ("not-a-number", None),
        ],
    )
    def test_parse_matrix(self, monkeypatch, configured, expected):
        monkeypatch.setattr(
            "tools.web_tools._load_web_config", lambda: _configured(configured)
        )
        from plugins.web.exa.provider import _get_exa_max_age_hours

        assert _get_exa_max_age_hours() == expected

    def test_absent_key_returns_none(self, monkeypatch):
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {})
        from plugins.web.exa.provider import _get_exa_max_age_hours

        assert _get_exa_max_age_hours() is None


class TestExtractForwardsMaxAgeHours:
    def _extract(self, monkeypatch, web_config):
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: web_config)
        client = _mock_client([_sdk_result()])
        from plugins.web.exa.provider import ExaWebSearchProvider

        with patch("plugins.web.exa.provider._get_exa_client", return_value=client):
            results = ExaWebSearchProvider().extract(["https://example.com"])
        return client, results

    def test_default_request_unchanged_when_unset(self, monkeypatch):
        client, results = self._extract(monkeypatch, {})
        client.get_contents.assert_called_once_with(["https://example.com"], text=True)
        assert results and results[0]["url"] == "https://example.com"

    def test_zero_forces_live_crawl(self, monkeypatch):
        client, _ = self._extract(monkeypatch, _configured(0))
        client.get_contents.assert_called_once_with(
            ["https://example.com"], text=True, max_age_hours=0.0
        )

    def test_positive_value_forwarded(self, monkeypatch):
        client, _ = self._extract(monkeypatch, _configured(24))
        kwargs = client.get_contents.call_args.kwargs
        assert kwargs == {"text": True, "max_age_hours": 24.0}

    def test_invalid_value_keeps_default_request(self, monkeypatch):
        client, _ = self._extract(monkeypatch, _configured("not-a-number"))
        client.get_contents.assert_called_once_with(["https://example.com"], text=True)
