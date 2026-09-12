"""Regression tests for the public browser JavaScript-evaluation boundary."""
import json

import pytest

from tools import browser_tool


@pytest.mark.parametrize("expression", [
    "fetch('http://' + '169.254.169.254/latest/meta-data/')",
    "globalThis['fe' + 'tch']('http://169.254.169.254/latest/meta-data/')",
    "location.href = 'http://' + '169.254.169.254/latest/meta-data/'",
])
def test_browser_console_expression_is_blocked_before_any_evaluation(monkeypatch, expression):
    monkeypatch.setattr(browser_tool, "_browser_eval", lambda *_: pytest.fail("evaluation must not be dispatched"))

    result = json.loads(browser_tool.browser_console(expression=expression))

    assert result["error"] == browser_tool.BROWSER_EVALUATION_DISABLED_ERROR


@pytest.mark.parametrize("camofox", [False, True])
def test_browser_eval_is_blocked_before_subprocess_or_camofox_transport(monkeypatch, camofox):
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: camofox)
    monkeypatch.setattr(
        browser_tool._session, "_run_browser_command", lambda *_args, **_kwargs: pytest.fail("eval must not run"),
    )
    monkeypatch.setattr(browser_tool, "_camofox_eval", lambda *_args, **_kwargs: pytest.fail("Camofox eval must not run"))

    result = json.loads(browser_tool._browser_eval(
        "fetch('http://' + '169.254.169.254/latest/meta-data/')", task_id="test"
    ))

    assert result["error"] == browser_tool.BROWSER_EVALUATION_DISABLED_ERROR


def test_camofox_eval_is_blocked_before_http_request(monkeypatch):
    import tools.browser_camofox as camofox

    monkeypatch.setattr(camofox, "_ensure_tab", lambda *_args: pytest.fail("Camofox eval must not create a tab"))

    result = json.loads(browser_tool._camofox_eval(
        "fetch('http://' + '169.254.169.254/latest/meta-data/')", task_id="test"
    ))

    assert result["error"] == browser_tool.BROWSER_EVALUATION_DISABLED_ERROR
