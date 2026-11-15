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


def test_remote_error_field_is_masked():
    payload = {"status": "error", "stdout": "", "stderr": "", "traceback": f"DB_PASSWORD={_OPAQUE}\n",
               "error": f"DB_PASSWORD={_OPAQUE}", "tool_calls_made": 0}
    result = json.loads(_finish_remote_kernel_result(payload, timeout=30, exec_start=time.monotonic()))
    assert result["error"].startswith("DB_PASSWORD=") and _OPAQUE not in result["error"], result


@pytest.mark.parametrize("run", [_run_local, _run_remote], ids=["local-kernel", "remote"])
def test_printed_source_is_not_rewritten(run):
    result = run(_SOURCE)
    for line in _SOURCE.splitlines():
        assert line in result["output"], result


@pytest.mark.parametrize("line", ['DB_PASSWORD="first second third"', "export DB_PASSWORD='first second third'",
                                  "DB_PASSWORD=first second third"])
def test_multi_word_env_value_is_masked_whole(line):
    from agent.redact import redact_code_output
    out = redact_code_output(line + "\n")
    assert "second" not in out and "third" not in out, out
    assert out.startswith(line.split("=")[0] + "=")


def test_remote_cap_cannot_split_key_from_value():
    # Tail of the 50 KB cap starts right after ``DB_PASSWORD=``: redaction must run before the cut.
    from tools.code_execution_tool import MAX_STDOUT_BYTES
    from tools.tool_output_truncate import head_tail_split
    _, tail_bytes = head_tail_split(MAX_STDOUT_BYTES)
    tail = _OPAQUE + "\n" + "y" * (tail_bytes - len(_OPAQUE) - 2) + "\n"
    head = ("x" * 99 + "\n") * (MAX_STDOUT_BYTES // 100)
    result = _run_remote(head + "DB_PASSWORD=" + tail)
    assert result["stdout_truncated"], result
    assert _OPAQUE not in result["output"]


@pytest.mark.parametrize("line", ["GIT_COMMIT=0123456789abcdef0123456789abcdef01234567",
                                  "PATH=/usr/local/bin:/usr/bin", "MAX_TOKENS=100"])
def test_non_secret_env_lines_are_kept(line):
    from agent.redact import redact_code_output
    assert redact_code_output(line + "\n") == line + "\n"


def test_quotes_and_inline_assignments_keep_their_shape():
    from agent.redact import redact_code_output
    out = redact_code_output(f"export DB_PASSWORD='short pw'\nFOO=bar MY_SERVICE_TOKEN={_OPAQUE}\n").splitlines()
    assert out[0] == "export DB_PASSWORD='***'"
    assert out[1].startswith("FOO=bar MY_SERVICE_TOKEN=") and _OPAQUE not in out[1], out


def test_redaction_off_leaves_env_lines(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    from agent.redact import redact_code_output
    assert redact_code_output(_ENV_DUMP) == _ENV_DUMP
