"""Self-disruption commands stay gated even under the container-backend bypass.

Isolated container backends (docker/singularity/modal/daytona) skip
dangerous-command approval on the rationale that the container is the
host-safety boundary — a destructive command inside the sandbox can't reach the
host. That rationale is sound for ``rm -rf`` / ``mkfs`` / ``dd`` but does NOT
cover *self-termination*: killing the agent's own gateway process is a
self-inflicted service DoS, not host damage, so it must still require approval
even in a container backend (issue #71957).

These tests assert that carve-out: self-disruption commands route through the
approval gate under every container backend, while genuinely host-scoped
destructive commands are still waived by the bypass.
"""
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    check_execute_code_guard,
    set_current_session_key,
    clear_session,
)

CONTAINER_BACKENDS = ["docker", "singularity", "modal", "daytona"]


@pytest.fixture
def interactive_session(monkeypatch):
    """Fresh session with an interactive CLI surface and clean approval state,
    so a dangerous command that reaches the gate consults our callback."""
    session_key = "test:session:self-disruption-bypass"
    token = set_current_session_key(session_key)
    monkeypatch.setenv("HERMES_SESSION_KEY", session_key)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    # manual mode so check_all_command_guards actually reaches the prompt site
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")

    saved_permanent = approval_module._permanent_approved.copy()
    saved_session = {k: v.copy() for k, v in approval_module._session_approved.items()}
    approval_module._permanent_approved.clear()
    approval_module._session_approved.clear()
    try:
        yield session_key
    finally:
        approval_module._permanent_approved.update(saved_permanent)
        approval_module._session_approved.update(saved_session)
        try:
            approval_module._approval_session_key.reset(token)
        except Exception:
            pass
        clear_session(session_key)


def _deny_cb_recording(calls):
    def cb(command, description, *, allow_permanent=True):
        calls.append((command, description))
        return "deny"
    return cb


# ---------------------------------------------------------------------------
# check_dangerous_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", CONTAINER_BACKENDS)
def test_self_termination_gated_in_container_backend(interactive_session, backend):
    calls = []
    result = check_dangerous_command(
        "pkill hermes", backend, approval_callback=_deny_cb_recording(calls),
    )
    # The command reached the approval gate (callback invoked) and the deny
    # decision stuck — the container bypass did NOT silently approve it.
    assert calls, f"self-termination command was not gated under {backend}"
    assert result["approved"] is False


@pytest.mark.parametrize("backend", CONTAINER_BACKENDS)
def test_host_destructive_still_bypassed_in_container(interactive_session, backend):
    calls = []
    result = check_dangerous_command(
        "rm -rf /workspace", backend, approval_callback=_deny_cb_recording(calls),
    )
    # Host-scoped destruction is still waived by the container boundary: the
    # callback is never consulted and the command is approved.
    assert not calls, f"host-destructive command was gated under {backend}"
    assert result["approved"] is True


def test_pkill_dash9_variant_still_gated_in_docker(interactive_session):
    """`pkill -9 hermes` matches the generic "force kill processes" rule before
    the self-termination rule, but it is still self-termination and must stay
    gated — the carve-out checks the self-disruption patterns directly."""
    calls = []
    result = check_dangerous_command(
        "pkill -9 hermes", "docker", approval_callback=_deny_cb_recording(calls),
    )
    assert calls
    assert result["approved"] is False


def test_kill_pgrep_expansion_gated_in_docker(interactive_session):
    calls = []
    result = check_dangerous_command(
        "kill -9 $(pgrep -f hermes)", "docker",
        approval_callback=_deny_cb_recording(calls),
    )
    assert calls
    assert result["approved"] is False


def test_gateway_lifecycle_gated_in_docker(interactive_session):
    calls = []
    result = check_dangerous_command(
        "hermes gateway restart", "docker",
        approval_callback=_deny_cb_recording(calls),
    )
    assert calls
    assert result["approved"] is False


def test_self_termination_approved_when_user_allows(interactive_session):
    """The carve-out only re-introduces the approval prompt — an interactive
    user who approves still gets the command run."""
    def cb(command, description, *, allow_permanent=True):
        return "once"

    result = check_dangerous_command(
        "pkill hermes", "docker", approval_callback=cb,
    )
    assert result["approved"] is True


def test_local_backend_self_termination_unchanged(interactive_session):
    """Sanity: the local backend already gated self-termination and still does
    (the carve-out only affects the container-bypass path)."""
    calls = []
    result = check_dangerous_command(
        "pkill hermes", "local", approval_callback=_deny_cb_recording(calls),
    )
    assert calls
    assert result["approved"] is False


# ---------------------------------------------------------------------------
# check_all_command_guards (combined tirith + dangerous-command gate)
# ---------------------------------------------------------------------------

def test_check_all_guards_gates_self_termination_in_docker(interactive_session):
    calls = []
    result = check_all_command_guards(
        "pkill hermes", "docker", approval_callback=_deny_cb_recording(calls),
    )
    assert calls
    assert result["approved"] is False


def test_check_all_guards_host_destructive_bypassed_in_docker(interactive_session):
    calls = []
    result = check_all_command_guards(
        "rm -rf /workspace", "docker", approval_callback=_deny_cb_recording(calls),
    )
    assert not calls
    assert result["approved"] is True


# ---------------------------------------------------------------------------
# check_execute_code_guard (the equivalent bypass surface)
# ---------------------------------------------------------------------------
# execute_code runs arbitrary local Python and shares the container skip, so it
# is a parallel bypass: a script can subprocess/os.system its way to a
# self-termination command. The carve-out matches self-disruption *shell*
# commands in the script text so they still route through approval, while
# ordinary scripts stay waived by the container boundary. The whole-script
# approval only fires in a gateway/ask surface, so these run under ask-mode.

@pytest.fixture
def ask_session(interactive_session, monkeypatch):
    """interactive_session plus ask-mode, so execute_code reaches the
    whole-script approval site instead of the local-non-gateway auto-approve."""
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    return interactive_session


@pytest.mark.parametrize("backend", CONTAINER_BACKENDS)
def test_execute_code_self_disruption_gated_in_container(ask_session, backend):
    """A script that shells out to self-termination is NOT waived by the
    container bypass — it routes through approval (pending, not auto-approved)."""
    code = 'import os\nos.system("pkill hermes")\n'
    result = check_execute_code_guard(code, backend)
    assert result["approved"] is False, (
        f"execute_code self-termination auto-approved under {backend}"
    )


@pytest.mark.parametrize("backend", CONTAINER_BACKENDS)
def test_execute_code_ordinary_still_bypassed_in_container(ask_session, backend):
    """A script with no self-disruption command is still waived by the
    container boundary — unchanged behavior."""
    code = 'import subprocess\nsubprocess.run(["rm", "-rf", "/workspace"])\n'
    result = check_execute_code_guard(code, backend)
    assert result["approved"] is True, (
        f"ordinary execute_code was gated under {backend}"
    )


def test_execute_code_gateway_stop_gated_in_docker(ask_session):
    code = 'import subprocess\nsubprocess.run("hermes gateway stop", shell=True)\n'
    result = check_execute_code_guard(code, "docker")
    assert result["approved"] is False


def test_execute_code_vercel_sandbox_still_waived(ask_session):
    """vercel_sandbox has no local gateway to self-terminate and is handled by a
    separate always-skip outside _should_skip_container_guards; the carve-out
    intentionally leaves it unchanged."""
    code = 'import os\nos.system("pkill hermes")\n'
    result = check_execute_code_guard(code, "vercel_sandbox")
    assert result["approved"] is True


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------

def test_self_disruption_descriptions_exist_in_dangerous_patterns():
    """Every description in the carve-out set must still be a real
    DANGEROUS_PATTERNS description, or the carve-out silently stops matching."""
    all_descriptions = {desc for _pattern, desc in approval_module.DANGEROUS_PATTERNS}
    missing = approval_module._SELF_DISRUPTION_DESCRIPTIONS - all_descriptions
    assert not missing, f"self-disruption descriptions not in DANGEROUS_PATTERNS: {missing}"


def test_self_disruption_compiled_subset_nonempty():
    assert approval_module._SELF_DISRUPTION_PATTERNS_COMPILED
    for _regex, desc in approval_module._SELF_DISRUPTION_PATTERNS_COMPILED:
        assert desc in approval_module._SELF_DISRUPTION_DESCRIPTIONS
