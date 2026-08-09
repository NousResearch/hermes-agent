"""Terminal-tool exception paths must redact secrets before returning to the model.

The logger copy of the traceback already goes through RedactingFormatter, but
the JSON result returned to the model previously carried raw ``str(e)`` and
``traceback.format_exc()`` — exception text can embed the failing command line
(and any secrets inline in it). See issue #77484.
"""

import json

import tools.terminal_tool as terminal_tool

SECRET = "sk-proj-AbCdEf1234567890SecretValue999"


def _force_exception(monkeypatch, exc):
    def boom():
        raise exc
    monkeypatch.setattr(terminal_tool, "_get_env_config", boom)


def test_generic_exception_result_redacts_error_and_traceback(monkeypatch):
    _force_exception(monkeypatch, RuntimeError(f"connect failed OPENAI_API_KEY={SECRET}"))
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["status"] == "error"
    assert SECRET not in result["error"]
    assert SECRET not in result["traceback"]
    # The redaction must mask the value, not drop the message entirely.
    assert "OPENAI_API_KEY=" in result["error"]
    assert "Failed to execute command" in result["error"]


def test_degraded_fail_mode_result_redacts_error_and_traceback(monkeypatch):
    from tools.environments.base import EnvironmentConnectionError

    monkeypatch.setenv("TERMINAL_DEGRADED_MODE", "fail")
    exc = EnvironmentConnectionError(
        f"ssh auth failed TOKEN={SECRET}", retry_hint="retry later"
    )
    _force_exception(monkeypatch, exc)
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["status"] == "error"
    assert SECRET not in result["error"]
    assert SECRET not in result["traceback"]
