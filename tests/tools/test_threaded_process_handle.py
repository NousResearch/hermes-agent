"""Tests for _ThreadedProcessHandle — the adapter for SDK backends."""

import json
import logging
import threading
import time

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.base_output import _ThreadedProcessHandle
from tools.terminal_tool_result import finalize_foreground_result


class TestBasicExecution:
    def test_successful_execution(self):
        def exec_fn():
            return ("hello world", 0)

        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=5)

        assert handle.returncode == 0
        output = handle.stdout.read()
        assert "hello world" in output




class TestPolling:
    def test_poll_returns_none_while_running(self):
        event = threading.Event()

        def exec_fn():
            event.wait(timeout=5)
            return ("done", 0)

        handle = _ThreadedProcessHandle(exec_fn)
        assert handle.poll() is None

        event.set()
        handle.wait(timeout=5)
        assert handle.poll() == 0



class TestCancelFn:
    def test_cancel_fn_called_on_kill(self):
        called = threading.Event()

        def cancel():
            called.set()

        def exec_fn():
            time.sleep(10)
            return ("", 0)

        handle = _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)
        handle.kill()
        assert called.is_set()


    def test_cancel_fn_exception_swallowed(self):
        def cancel():
            raise RuntimeError("cancel failed")

        def exec_fn():
            return ("ok", 0)

        handle = _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)
        handle.kill()  # should not raise despite cancel raising
        handle.wait(timeout=5)


class TestStdoutPipe:
    def test_stdout_is_readable(self):
        def exec_fn():
            return ("line1\nline2\nline3\n", 0)

        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=5)

        lines = handle.stdout.readlines()
        assert len(lines) == 3
        assert lines[0] == "line1\n"


    def test_unicode_output(self):
        def exec_fn():
            return ("hello 世界 🌍\n", 0)

        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=5)

        output = handle.stdout.read()
        assert "世界" in output
        assert "🌍" in output


def _raising_exec_fn():
    raise ConnectionError("sandbox sb-123 is not running")


class TestExecFnFailure:
    """An SDK call that raises (sandbox gone, network, auth) is the command's failure: its type and
    message become the output (exit code stays 1) and it is logged, instead of an empty exit 1."""

    def test_exception_is_logged_with_its_type_and_message(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.environments.base_output"):
            handle = _ThreadedProcessHandle(_raising_exec_fn)
            handle.wait(timeout=5)
        assert handle.returncode == 1
        assert any(r.levelno == logging.WARNING and "ConnectionError: sandbox sb-123 is not running" in r.getMessage()
                   for r in caplog.records)

    def test_host_credentials_in_the_exception_are_redacted(self):
        def exec_fn():
            raise ConnectionError(
                "401 for https://svc:hunter2pass@api.example.test/v1 with "
                "{'Authorization': 'Bearer abcdef0123456789abcdef0123456789'}")

        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=5)
        output = handle.stdout.read()
        assert "ConnectionError: 401" in output
        assert "hunter2pass" not in output and "abcdef0123456789abcdef0123456789" not in output

    @pytest.mark.parametrize("message,secret", [
        ("403 with headers {'X-Api-Key': 'k3y-abc123def456'}", "k3y-abc123def456"),
        ('401 sent {"x-auth-token": "tok9f8e7d6c5b4a3"}', "tok9f8e7d6c5b4a3"),
        ("request {'Client-Secret': 's3cr3t-value-xyz', 'User-Agent': 'sdk/1.0'}", "s3cr3t-value-xyz"),
        ("{'Authorization': 'Basic dXNlcjpwYXNzd29yZA=='}", "dXNlcjpwYXNzd29yZA=="),
        ('{"request": "{\\"X-Api-Key\\": \\"k3y-escaped-json-77\\"}"}', "k3y-escaped-json-77"),
        ("{b'x-api-key': b'k3y-bytes-repr-88'}", "k3y-bytes-repr-88"),
        ("{'Proxy-Authorization': 'Basic cHJveHk6c2VjcmV0'}", "cHJveHk6c2VjcmV0"),
        ("headers=[(b'x-api-key', b'k3y-tuple-list-99'), (b'accept', b'*/*')]", "k3y-tuple-list-99"),
        ('{"x-api-key"            : "k3y-aligned-json-66"}', "k3y-aligned-json-66"),
    ], ids=["x_api_key", "json_token", "client_secret", "authorization_basic", "escaped_json",
            "bytes_dict", "proxy_authorization", "header_tuple_list", "aligned_json"])
    def test_header_style_credentials_in_the_exception_are_masked(self, message, secret):
        def exec_fn():
            raise PermissionError(message)

        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=5)
        output = handle.stdout.read()
        assert output.startswith("[backend error: PermissionError: ")
        assert secret not in output

    def test_masking_stays_linear_on_an_unterminated_value(self):
        """A value cut off mid-escape (a run of backslashes, no closing quote) must not make the
        masking regex backtrack: ``re`` holds the GIL, so a slow match stalls the whole process."""
        message = "{'x-api-key': '" + "\\" * 34

        def exec_fn():
            raise ValueError(message)

        started = time.monotonic()
        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=30)
        assert handle.returncode == 1
        assert time.monotonic() - started < 2.0

    def test_the_redaction_pass_gets_a_bounded_input(self, monkeypatch):
        """The forced redaction pass is superlinear on long backslash runs and holds the GIL, so
        the message it sees is cut to twice the output cap first, however long the exception is."""
        import agent.redact
        from tools.environments.base_output import _BACKEND_ERROR_MAX_CHARS

        seen = []
        real = agent.redact.redact_sensitive_text

        def spy(text, **kwargs):
            if kwargs.get("redact_url_credentials"):  # the backend-error pass, not a log formatter
                seen.append(len(text))
            return real(text, **kwargs)

        monkeypatch.setattr(agent.redact, "redact_sensitive_text", spy)

        def exec_fn():
            raise ValueError('{\\"x-api-key\\": \\"' + "\\" * 50_000)

        handle = _ThreadedProcessHandle(exec_fn)
        handle.wait(timeout=30)
        handle.stdout.read()
        assert seen and max(seen) <= 2 * _BACKEND_ERROR_MAX_CHARS

    def test_debug_logging_carries_no_raw_message(self, caplog):
        """With debug logging on, the traceback record must not carry the raw exception message:
        only the masked, capped text is logged."""
        secret = "k3y-" + "debuglog-55"  # built at runtime: the logged frames show source lines

        def exec_fn():
            raise ConnectionError(f"403 {{'X-Api-Key': '{secret}'}}")

        with caplog.at_level(logging.DEBUG, logger="tools.environments.base_output"):
            handle = _ThreadedProcessHandle(exec_fn)
            handle.wait(timeout=5)
        formatter = logging.Formatter()
        logged = [formatter.format(r) for r in caplog.records]
        assert logged and not any(secret in text for text in logged)

    def test_a_failed_cancel_is_logged_without_its_raw_message(self, caplog):
        """cancel_fn calls the same SDK, so its error can carry the same credentials."""
        secret = "k3y-" + "cancellog-77"  # built at runtime: the logged frames show source lines

        def cancel():
            raise ConnectionError(f"403 {{'X-Api-Key': '{secret}'}}")

        with caplog.at_level(logging.DEBUG, logger="tools.environments.base_output"):
            handle = _ThreadedProcessHandle(lambda: ("ok", 0), cancel_fn=cancel)
            handle.kill()
            handle.wait(timeout=5)
        formatter = logging.Formatter()
        logged = [formatter.format(r) for r in caplog.records]
        assert any("SDK cancel failed" in text and "ConnectionError" in text for text in logged)
        assert not any(secret in text for text in logged)

    def test_exception_after_hermes_kills_the_command_adds_no_backend_error(self):
        stopped = threading.Event()

        def exec_fn():
            stopped.wait(timeout=5)
            raise RuntimeError("sandbox was stopped")

        handle = _ThreadedProcessHandle(exec_fn, cancel_fn=stopped.set)
        handle.kill()  # Hermes's own timeout/interrupt path: cancel_fn makes the SDK call raise
        handle.wait(timeout=5)
        assert handle.returncode == 1
        assert "[backend error" not in handle.stdout.read()

    def test_terminal_tool_result_names_the_backend_error(self):
        class _SdkEnv(BaseEnvironment):
            _stdin_mode = "heredoc"

            def __init__(self):
                super().__init__(cwd="/root", timeout=30)
                self._snapshot_ready = True

            def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
                return _ThreadedProcessHandle(_raising_exec_fn)

            def cleanup(self):
                pass

        env = _SdkEnv()
        tool_result = json.loads(finalize_foreground_result(
            command="ls", result=env.execute("ls"), env=env, env_type="sdk", effective_task_id="t",
            task_id="t", session_id="s", session_key="k", workdir=None, command_cwd=None,
            approval_note=None))
        assert tool_result["exit_code"] == 1
        assert "ConnectionError" in tool_result["output"]
        assert "sandbox sb-123 is not running" in tool_result["output"]
