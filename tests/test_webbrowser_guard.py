"""Tests for the global webbrowser hermeticity guard."""

import webbrowser


def test_webbrowser_open_is_neutralized_by_default():
    assert webbrowser.open("https://auth.x.ai/oauth2/authorize") is True


def test_tests_can_override_webbrowser_open(monkeypatch):
    opened = []

    def fake_open(url, *args, **kwargs):
        opened.append(url)
        return False

    monkeypatch.setattr(webbrowser, "open", fake_open)

    assert webbrowser.open("https://example.test/login") is False
    assert opened == ["https://example.test/login"]
