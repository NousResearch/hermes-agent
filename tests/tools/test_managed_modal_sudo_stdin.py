"""Managed Modal carries sudo password lines on the request's stdin field, not in the command.

The lines lead the caller's own stdin, the same merge ``BaseEnvironment.execute`` does, and the
command string is the prepared command with nothing prefixed to it.
"""

import types

import pytest

from tools import terminal_tool as tt
from tools import terminal_tool_sudo as tts

CONFIGURED = "canary-configured-pw-7f3a"
PROMPTED = "canary-typed-at-prompt-91c2"
SUDO_STDIN_FLAGS = " -S -p ''"


@pytest.fixture(autouse=True)
def _clean_sudo_state(monkeypatch):
    for name in ("SUDO_PASSWORD", "HERMES_INTERACTIVE", "HERMES_SINGLE_QUERY_SESSION",
                 "HERMES_GATEWAY_SESSION"):
        monkeypatch.delenv(name, raising=False)
    tts._reset_cached_sudo_passwords()
    tt.set_sudo_password_callback(None)
    yield
    tt.set_sudo_password_callback(None)
    tts._reset_cached_sudo_passwords()


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def _exec_payloads(monkeypatch, command, stdin_data):
    """Run *command* through the real ManagedModalEnvironment.execute over a fake HTTP transport."""
    from tools import credential_files
    from tools.environments import managed_modal

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", list)
    monkeypatch.setattr(managed_modal, "resolve_managed_tool_gateway", lambda vendor: types.SimpleNamespace(
        vendor=vendor, gateway_origin="https://modal-gateway.example.com",
        nous_user_token="user-token", managed_mode=True))
    payloads = []

    def fake_request(method, url, headers=None, json=None, timeout=None):
        if method == "POST" and url.endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if method == "POST" and url.endswith("/execs"):
            payloads.append(json)
            return _FakeResponse(200, {"execId": json["execId"], "status": "completed",
                                       "output": "ok", "returncode": 0})
        if method == "POST" and url.endswith("/terminate"):
            return _FakeResponse(200, {"status": "terminated"})
        raise AssertionError(f"Unexpected request: {method} {url}")

    monkeypatch.setattr(managed_modal.requests, "request", fake_request)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")
    monkeypatch.setattr(env, "_sudo_nopasswd_works", lambda: False)
    try:
        result = env.execute(command, stdin_data=stdin_data)
    finally:
        env.cleanup()
    assert result == {"output": "ok", "returncode": 0}
    return payloads


def test_prompted_password_goes_on_the_stdin_field_ahead_of_caller_stdin(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(tts, "_prompt_for_sudo_password",
                        lambda timeout_seconds=45, *, command="": PROMPTED)

    (payload,) = _exec_payloads(monkeypatch, "sudo tee /tmp/out", "caller-data\n")

    assert payload["command"] == "sudo" + SUDO_STDIN_FLAGS + " tee /tmp/out"
    assert PROMPTED not in payload["command"]
    assert payload["stdinData"] == PROMPTED + "\n" + "caller-data\n"


def test_configured_password_goes_on_the_stdin_field_one_line_per_sudo(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", CONFIGURED)

    (payload,) = _exec_payloads(monkeypatch, "sudo true && sudo id", None)

    assert payload["command"] == "sudo" + SUDO_STDIN_FLAGS + " true && sudo" + SUDO_STDIN_FLAGS + " id"
    assert CONFIGURED not in payload["command"]
    assert payload["stdinData"] == (CONFIGURED + "\n") * 2


def test_command_without_sudo_keeps_its_command_and_stdin(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", CONFIGURED)

    (payload,) = _exec_payloads(monkeypatch, "cat > /tmp/out", "caller-data\n")

    assert payload["command"] == "cat > /tmp/out"
    assert payload.get("stdinData") == "caller-data\n"
