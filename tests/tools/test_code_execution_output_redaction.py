"""execute_code output redaction: env-dump-shaped stdout is masked, printed source is not (#95509)."""

import json
import time
from unittest.mock import patch

import pytest

from tools.code_execution_tool import _finish_remote_kernel_result, execute_code
from tools.code_kernel import shutdown_all_kernels

_OPAQUE = "abc123" + "randomstring"  # prefix-less: no vendor pattern can catch it
_ENV_DUMP = f"MY_SERVICE_TOKEN={_OPAQUE}\nexport DB_PASSWORD={_OPAQUE}9\n"
_SOURCE = "MAX_TOKENS = 100\n    API_KEY = 'fixture-value'\nprint(f'API_KEY={key}')\n"


@pytest.fixture(autouse=True)
def _redaction_on(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run_local(text: str) -> dict:
    with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 30}):
        return json.loads(execute_code(f"import sys\nsys.stdout.write({text!r})", task_id="redact-test"))


def _run_remote(text: str) -> dict:
    payload = {"status": "success", "stdout": text, "stderr": text, "tool_calls_made": 0}
    return json.loads(_finish_remote_kernel_result(payload, timeout=30, exec_start=time.monotonic()))


@pytest.mark.parametrize("run", [_run_local, _run_remote], ids=["local-kernel", "remote"])
def test_env_dump_lines_are_masked(run):
    result = run(_ENV_DUMP)
    assert _OPAQUE not in result["output"], result
    assert "MY_SERVICE_TOKEN=" in result["output"]
    assert "export DB_PASSWORD=" in result["output"]


@pytest.mark.parametrize("run", [_run_local, _run_remote], ids=["local-kernel", "remote"])
def test_printed_source_is_not_rewritten(run):
    result = run(_SOURCE)
    for line in _SOURCE.splitlines():
        assert line in result["output"], result
