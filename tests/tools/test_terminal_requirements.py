import logging

from tools.terminal_tool import check_terminal_requirements


def test_local_terminal_requirements_do_not_depend_on_minisweagent(monkeypatch, caplog):
    """Local backend uses Hermes' own LocalEnvironment wrapper and should not
    be marked unavailable just because `minisweagent` isn't importable."""
    monkeypatch.setenv("TERMINAL_ENV", "local")

    with caplog.at_level(logging.ERROR):
        ok = check_terminal_requirements()

    assert ok is True
    assert "Terminal requirements check failed" not in caplog.text
