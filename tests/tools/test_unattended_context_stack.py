"""Stacked unattended contexts: any deny blocks.

Regression for #156677. Single-query is listed ahead of cron or an unattended
platform. An earlier approve must not waive a later deny for execute_code,
dangerous commands, or plugin approvals.
"""

import os

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools import approval, approval_context
from tools.file_tools_write_guards import _check_approval_required_write

_CLEARED = (
    "HERMES_GATEWAY_SESSION",
    "HERMES_EXEC_ASK",
    "HERMES_YOLO_MODE",
    "HERMES_CRON_SESSION",
    "HERMES_SINGLE_QUERY_SESSION",
    "HERMES_SESSION_PLATFORM",
)


def _mode(name: str, value: str):
    """Bind one approvals.<name>_mode getter for the duration of a patch."""
    return approval_context, f"_get_{name}_approval_mode", lambda value=value: value


def setup_function():
    """Drop allowlist grants and session vars so each case starts clean."""
    approval._permanent_approved.clear()
    reset_session_vars()


def teardown_function():
    approval._permanent_approved.clear()
    reset_session_vars()


def _prepare(monkeypatch, **modes):
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)
    # hermes chat -q exports this so sudo prompts work; nobody answers approvals.
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    for name, value in modes.items():
        target, attr, getter = _mode(name, value)
        monkeypatch.setattr(target, attr, getter)


def test_single_query_approve_does_not_waive_cron_deny(monkeypatch):
    """approve on -q plus deny on cron blocks code, shell, and plugin calls."""
    _prepare(monkeypatch, single_query="approve", cron="deny")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    tokens = set_session_vars(cron_session="1")
    try:
        code = approval.check_execute_code_guard("import os", "local")
        dangerous = approval.check_dangerous_command("rm -rf /tmp/stuff", "local")
        plugin = approval.request_tool_approval("memory", "write a note")
        shell = approval.check_all_command_guards("rm -rf /tmp/stuff", "local")
        write = _check_approval_required_write([os.path.expanduser("~/.ssh/config")])
    finally:
        clear_session_vars(tokens)

    assert code["approved"] is False and code["outcome"] == "blocked"
    assert "cron_mode" in code["message"]
    for result in (dangerous, plugin, shell):
        assert result["approved"] is False
        assert "cron_mode" in result["message"]
    assert "cron session denies it" in (write or "")


def test_single_query_approve_does_not_waive_unattended_deny(monkeypatch):
    """approve on -q plus deny on a webhook session blocks execute_code."""
    _prepare(monkeypatch, single_query="approve", unattended="deny")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    tokens = set_session_vars(platform="webhook")
    try:
        code = approval.check_execute_code_guard("import os", "local")
        dangerous = approval.check_dangerous_command("rm -rf /tmp/stuff", "local")
    finally:
        clear_session_vars(tokens)

    assert code["approved"] is False and "unattended_mode" in code["message"]
    assert dangerous["approved"] is False and "unattended_mode" in dangerous["message"]


def test_first_deny_supplies_the_block_message(monkeypatch):
    """When several contexts deny, the earliest one is the one named."""
    _prepare(monkeypatch, single_query="deny", cron="deny")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    tokens = set_session_vars(cron_session="1")
    try:
        code = approval.check_execute_code_guard("import os", "local")
    finally:
        clear_session_vars(tokens)

    assert code["approved"] is False
    assert "single_query_mode" in code["message"]


def test_single_query_approve_alone_still_approves(monkeypatch):
    """A lone single-query approve still skips the fail-closed branch."""
    _prepare(monkeypatch, single_query="approve")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    code = approval.check_execute_code_guard("import os", "local")
    plugin = approval.request_tool_approval("memory", "write a note")

    assert code["approved"] is True
    assert plugin["approved"] is True


def test_cron_deny_alone_still_blocks_execute_code(monkeypatch):
    """A cron session with no single-query marker still honors cron_mode."""
    _prepare(monkeypatch, cron="deny")
    tokens = set_session_vars(cron_session="1")
    try:
        code = approval.check_execute_code_guard("import os", "local")
    finally:
        clear_session_vars(tokens)

    assert code["approved"] is False and "cron_mode" in code["message"]


def test_stacked_approve_modes_still_approve_execute_code(monkeypatch):
    """Approve on every active context still allows the script."""
    _prepare(monkeypatch, single_query="approve", cron="approve")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    tokens = set_session_vars(cron_session="1")
    try:
        code = approval.check_execute_code_guard("import os", "local")
    finally:
        clear_session_vars(tokens)

    assert code["approved"] is True
