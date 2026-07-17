"""Kanban owner credentials never cross a model-authored child boundary."""

from __future__ import annotations


_OWNER_ENV = {
    "_HERMES_KANBAN_BOOTSTRAP_STDIN": "1",
    "HERMES_KANBAN_SESSION": "1",
    "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE": "bootstrap-secret",
    "HERMES_KANBAN_CLAIM_LOCK": "claim-secret",
    "HERMES_KANBAN_APPROVAL_ID": "ka_request",
    "HERMES_KANBAN_APPROVAL_NONCE": "approval-secret",
}


def _assert_delegate_only(env):
    for key in _OWNER_ENV:
        assert key not in env
    assert env["HERMES_KANBAN_DELEGATE_SESSION"] == "1"


def test_execute_code_scrubs_owner_credentials_even_when_passthrough_allows_them():
    from tools.code_execution_tool import _scrub_child_env

    child = _scrub_child_env(
        {**_OWNER_ENV, "HERMES_KANBAN_DELEGATE_SESSION": "1"},
        is_passthrough=lambda _key: True,
        is_windows=False,
    )

    _assert_delegate_only(child)


def test_terminal_foreground_and_background_envs_scrub_owner_credentials(
    monkeypatch,
):
    from tools.environments.local import _make_run_env, _sanitize_subprocess_env

    for key, value in _OWNER_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HERMES_KANBAN_DELEGATE_SESSION", "1")

    _assert_delegate_only(_make_run_env({}))
    _assert_delegate_only(
        _sanitize_subprocess_env(
            {**_OWNER_ENV, "HERMES_KANBAN_DELEGATE_SESSION": "1"},
        )
    )
