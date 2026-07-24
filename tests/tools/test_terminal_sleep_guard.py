"""Regression tests for the long foreground `sleep` interceptor.

A foreground `sleep N` (N > 30) holds the executor worker thread hostage for
the whole duration, risking upstream timeouts. While interruptible on main,
the guard rejects such commands early to enforce the asynchronous 
background + process(poll) pattern instead.
"""

import json

from tools.terminal_tool import terminal_tool

def test_long_foreground_sleep_is_rejected():
    result = json.loads(terminal_tool("sleep 120"))

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert "sleep 120" in result["error"]
    assert "background=true" in result["error"]

def test_long_foreground_sleep_with_whitespace_is_rejected():
    result = json.loads(terminal_tool("  sleep   90  "))

    assert result["exit_code"] == -1
    assert "background=true" in result["error"]

def test_short_sleep_is_not_intercepted(monkeypatch):
    import tools.terminal_tool as tt

    def _boom():
        raise RuntimeError("env-config-reached")

    monkeypatch.setattr(tt, "_get_env_config", _boom)
    result = json.loads(terminal_tool("sleep 5"))
    assert "env-config-reached" in result["error"]
    assert "background=true" not in result["error"]

def test_background_sleep_is_not_intercepted(monkeypatch):
    import tools.terminal_tool as tt

    def _boom():
        raise RuntimeError("env-config-reached")

    monkeypatch.setattr(tt, "_get_env_config", _boom)
    result = json.loads(terminal_tool("sleep 300", background=True))
    assert "env-config-reached" in result["error"]
    assert "background=true" not in result["error"]

def test_command_containing_sleep_substring_is_not_intercepted(monkeypatch):
    import tools.terminal_tool as tt

    def _boom():
        raise RuntimeError("env-config-reached")

    monkeypatch.setattr(tt, "_get_env_config", _boom)
    result = json.loads(terminal_tool("./run_sleep_test.sh 120"))
    assert "env-config-reached" in result["error"]
    assert "background=true" not in result["error"]
