"""Tests for the allowlist_match plugin hook.

A command auto-approved via command_allowlist (an exact or glob match in
_command_matches_permanent_allowlist) never reaches pre_approval_request /
post_approval_response at all -- it short-circuits before either fires. That
left plugins observing approval activity (an audit log, a notifier, metrics)
completely blind to allowlist-approved commands. allowlist_match is the one
event that path can fire.
"""
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval_context import set_current_session_key
from tools.approval_floors import _command_matches_permanent_allowlist


@pytest.fixture
def isolated_session():
    """Fresh session_key and clean permanent-allowlist state per test."""
    session_key = "test:session:allowlist_match_hook"
    token = set_current_session_key(session_key)
    saved_permanent = approval_module._permanent_approved.copy()
    approval_module._permanent_approved.clear()
    try:
        yield session_key
    finally:
        approval_module._permanent_approved.clear()
        approval_module._permanent_approved.update(saved_permanent)
        try:
            from tools import approval_context
            approval_context._approval_session_key.reset(token)
        except Exception:
            pass


def test_allowlist_match_fires_with_expected_kwargs(isolated_session):
    approval_module._permanent_approved.add("git status")

    captured = []

    def fake_invoke_hook(hook_name, **kwargs):
        captured.append((hook_name, kwargs))
        return []

    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
        matched = _command_matches_permanent_allowlist("git status")

    assert matched is True
    hook_calls = [(name, kw) for name, kw in captured if name == "allowlist_match"]
    assert len(hook_calls) == 1
    _, kwargs = hook_calls[0]
    assert kwargs["command"] == "git status"
    assert kwargs["pattern"] == "git status"
    assert kwargs["session_key"] == isolated_session
    assert kwargs["surface"] == "allowlist"


def test_allowlist_match_fires_for_a_glob_pattern_with_the_pattern_not_the_command(
    isolated_session,
):
    approval_module._permanent_approved.add("docker *")

    captured = []

    def fake_invoke_hook(hook_name, **kwargs):
        captured.append((hook_name, kwargs))
        return []

    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
        matched = _command_matches_permanent_allowlist("docker ps -a")

    assert matched is True
    _, kwargs = next((n, kw) for n, kw in captured if n == "allowlist_match")
    assert kwargs["command"] == "docker ps -a"
    assert kwargs["pattern"] == "docker *"  # the pattern that matched, not the raw command


def test_no_hook_fires_when_nothing_matches(isolated_session):
    approval_module._permanent_approved.add("git status")

    captured = []

    def fake_invoke_hook(hook_name, **kwargs):
        captured.append((hook_name, kwargs))
        return []

    with patch("hermes_cli.plugins.invoke_hook", side_effect=fake_invoke_hook):
        matched = _command_matches_permanent_allowlist("rm -rf /tmp/x")

    assert matched is False
    assert not any(name == "allowlist_match" for name, _ in captured)


def test_hook_dispatch_failure_does_not_break_the_match(isolated_session):
    """_fire_approval_hook never raises: a broken plugin must not turn an
    allowlisted command into a denied one."""
    approval_module._permanent_approved.add("git status")

    with patch("hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("plugin boom")):
        matched = _command_matches_permanent_allowlist("git status")

    assert matched is True
