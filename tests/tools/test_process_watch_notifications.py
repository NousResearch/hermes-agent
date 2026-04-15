from types import SimpleNamespace

import pytest

from cli import _format_process_notification
from gateway.run import _format_gateway_process_notification
from tools.process_registry import process_registry


@pytest.fixture(autouse=True)
def reset_process_registry_get(monkeypatch):
    monkeypatch.setattr(process_registry, "get", lambda _sid: None)


def test_cli_watch_match_reports_failed_process_context(monkeypatch):
    monkeypatch.setattr(
        process_registry,
        "get",
        lambda _sid: SimpleNamespace(
            exited=True,
            exit_code=1,
            output_buffer=(
                "INFO:     Application startup complete.\n"
                "ERROR:    [Errno 98] error while attempting to bind on address "
                "('127.0.0.1', 18085): address already in use\n"
            ),
        ),
    )
    evt = {
        "type": "watch_match",
        "session_id": "proc_test",
        "command": "python3 -m uvicorn app:app --port 18085",
        "pattern": "Application startup complete",
        "output": "INFO:     Application startup complete.",
        "suppressed": 0,
    }

    text = _format_process_notification(evt)

    assert "exited with code 1" in text
    assert "address already in use" in text
    assert "Matched output" in text
    assert "Final output" in text


def test_gateway_watch_match_reports_failed_process_context(monkeypatch):
    monkeypatch.setattr(
        process_registry,
        "get",
        lambda _sid: SimpleNamespace(
            exited=True,
            exit_code=1,
            output_buffer="INFO:     Application startup complete.\nERROR: bind failed\n",
        ),
    )
    evt = {
        "type": "watch_match",
        "session_id": "proc_test",
        "command": "python3 -m uvicorn app:app --port 18085",
        "pattern": "Application startup complete",
        "output": "INFO:     Application startup complete.",
        "suppressed": 0,
    }

    text = _format_gateway_process_notification(evt)

    assert "exited with code 1" in text
    assert "bind failed" in text
    assert "Final output" in text
