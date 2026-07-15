"""Tests for the global webbrowser hermeticity guard."""

import webbrowser

import pytest


@pytest.mark.parametrize("method_name", ("open", "open_new", "open_new_tab"))
def test_webbrowser_methods_are_neutralized_by_default(
    _neutralize_webbrowser,
    method_name,
):
    url = "https://auth.x.ai/oauth2/authorize"

    assert getattr(webbrowser, method_name)(url) is True
    assert _neutralize_webbrowser == [url]


def test_tests_can_override_webbrowser_open(monkeypatch):
    opened = []

    def fake_open(url, *args, **kwargs):
        opened.append(url)
        return False

    monkeypatch.setattr(webbrowser, "open", fake_open)

    assert webbrowser.open("https://example.test/login") is False
    assert opened == ["https://example.test/login"]
