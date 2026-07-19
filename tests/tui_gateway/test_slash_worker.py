"""Focused unit tests for the slash worker protocol and startup orchestration."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock, call

import pytest

import cli as cli_mod
from tui_gateway import slash_worker


_MISSING = object()


@pytest.fixture(autouse=True)
def _reset_in_flight_event():
    slash_worker._in_flight.clear()
    yield
    slash_worker._in_flight.clear()


def _make_cli() -> MagicMock:
    cli = MagicMock()
    cli.console = MagicMock()
    cli.process_command = MagicMock()
    return cli


def _invoke_main(
    monkeypatch,
    stdin_text: str,
    *,
    session_key: str = "test-session",
    model: str = "",
    run_side_effect=_MISSING,
):
    fake_stdout = io.StringIO()
    calls = MagicMock()
    mock_cli = MagicMock(name="mock_cli")
    calls.cli.return_value = mock_cli
    if run_side_effect is _MISSING:
        calls.run.return_value = "worker-output"
    else:
        calls.run.side_effect = run_side_effect

    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(
        sys,
        "argv",
        ["slash_worker", "--session-key", session_key, "--model", model],
    )
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(slash_worker.os, "getppid", lambda: 4242)
    monkeypatch.setattr(slash_worker, "_start_parent_death_watchdog", calls.watchdog)
    monkeypatch.setattr(slash_worker, "_prepare_slash_worker_runtime", calls.prepare)
    monkeypatch.setattr(slash_worker, "HermesCLI", calls.cli)
    monkeypatch.setattr(slash_worker, "_run", calls.run)

    slash_worker.main()

    responses = [json.loads(line) for line in fake_stdout.getvalue().splitlines() if line.strip()]
    return calls, mock_cli, responses


def _startup_calls(mock_cli: MagicMock, model: str | None):
    return [
        call.watchdog(4242),
        call.prepare(),
        call.cli(model=model, compact=True, resume="test-session", verbose=False),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 1.5),
        ("", 1.5),
        ("2.25", 2.25),
        ("not-a-number", 1.5),
    ],
)
def test_env_float_uses_valid_values_or_default(monkeypatch, raw, expected):
    name = "HERMES_TEST_SLASH_FLOAT"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)

    assert slash_worker._env_float(name, 1.5) == expected


@pytest.mark.parametrize("command", ["", "   ", None])
def test_run_ignores_empty_commands(command):
    cli = _make_cli()

    assert slash_worker._run(cli, command) == ""
    cli.process_command.assert_not_called()


@pytest.mark.parametrize(
    ("command", "expected"),
    [("help", "/help"), ("/status", "/status")],
)
def test_run_normalizes_command_prefix(command, expected):
    cli = _make_cli()

    slash_worker._run(cli, command)

    cli.process_command.assert_called_once_with(expected)


def test_run_restores_cprint_and_strips_trailing_whitespace(monkeypatch):
    cli = _make_cli()
    original_cprint = MagicMock(name="original_cprint")
    monkeypatch.setattr(cli_mod, "_cprint", original_cprint, raising=False)

    def process_command(_command):
        assert cli_mod._cprint is not original_cprint
        print("output\n")

    cli.process_command.side_effect = process_command

    assert slash_worker._run(cli, "/test") == "output"
    assert cli_mod._cprint is original_cprint


def test_run_restores_cprint_when_command_fails(monkeypatch):
    cli = _make_cli()
    original_cprint = MagicMock(name="original_cprint")
    monkeypatch.setattr(cli_mod, "_cprint", original_cprint, raising=False)

    def process_command(_command):
        assert cli_mod._cprint is not original_cprint
        raise RuntimeError("boom")

    cli.process_command.side_effect = process_command

    with pytest.raises(RuntimeError, match="boom"):
        slash_worker._run(cli, "/fail")

    assert cli_mod._cprint is original_cprint


@pytest.mark.parametrize(("model", "expected_model"), [("", None), ("gpt-4", "gpt-4")])
def test_main_runs_valid_command_after_ordered_startup(
    monkeypatch,
    model,
    expected_model,
):
    def run_command(_cli, _command):
        assert slash_worker._in_flight.is_set()
        return "worker-output"

    calls, mock_cli, responses = _invoke_main(
        monkeypatch,
        json.dumps({"id": "r1", "command": "/help"}) + "\n",
        model=model,
        run_side_effect=run_command,
    )

    assert responses == [{"id": "r1", "ok": True, "output": "worker-output"}]
    assert calls.mock_calls == [
        *_startup_calls(mock_cli, expected_model),
        call.run(mock_cli, "/help"),
    ]
    assert slash_worker.os.environ["HERMES_SESSION_KEY"] == "test-session"
    assert slash_worker.os.environ["HERMES_INTERACTIVE"] == "1"
    assert not slash_worker._in_flight.is_set()


def test_main_reports_invalid_json_without_dispatch(monkeypatch):
    calls, mock_cli, responses = _invoke_main(monkeypatch, "not json\n")

    assert len(responses) == 1
    assert responses[0]["id"] is None
    assert responses[0]["ok"] is False
    assert responses[0]["error"]
    assert calls.mock_calls == _startup_calls(mock_cli, None)
    calls.run.assert_not_called()
    assert not slash_worker._in_flight.is_set()


def test_main_skips_blank_lines(monkeypatch):
    calls, mock_cli, responses = _invoke_main(monkeypatch, "\n\n\n")

    assert responses == []
    assert calls.mock_calls == _startup_calls(mock_cli, None)
    calls.run.assert_not_called()
    assert not slash_worker._in_flight.is_set()


def test_main_preserves_request_id_and_clears_in_flight_on_error(monkeypatch):
    calls, mock_cli, responses = _invoke_main(
        monkeypatch,
        json.dumps({"id": "r1", "command": "/fail"}) + "\n",
        run_side_effect=RuntimeError("boom"),
    )

    assert responses == [{"id": "r1", "ok": False, "error": "boom"}]
    assert calls.mock_calls == [
        *_startup_calls(mock_cli, None),
        call.run(mock_cli, "/fail"),
    ]
    assert not slash_worker._in_flight.is_set()
