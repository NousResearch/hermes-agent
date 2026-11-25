"""The dispatcher-owned worker's terminal session expands ``$HERMES_KANBAN_TASK`` as a plain
(non-exported) shell variable.

The worker protocol documents the var to shells (done-when receipts, evidence naming;
#160138), while the child-env scrub deliberately keeps worker identity away from every child
process a worker starts (#90782, #112817). A plain shell variable threads that needle: the
command and its subshells expand it, ``env``/child processes never see it.
"""

import os
import shlex
import subprocess

import pytest

from agent.delegation_context import worker_plain_shell_vars
from tools.environments.base import BaseEnvironment
from tools.environments.base_session_env import _wrap_command_script

def _wrapper(command: str, plain_vars=None) -> str:
    return _wrap_command_script(
        command,
        quoted_cwd="'/tmp'",
        quoted_snap="'/tmp/hermes-snap.sh'",
        snap_tmp_template="'/tmp/hermes-snap.sh.tmp.XXXXXXXXXX'",
        passthrough_names=(),
        snapshot_ready=False,
        cwd_marker="HERMES_TEST_CWD:",
        plain_vars=plain_vars)


def _with_worker(monkeypatch, task_id="task-abc-123", *, owned=True):
    if task_id is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setattr(
        "agent.delegation_context.is_dispatcher_owned_worker_context", lambda: owned)


# --- worker_plain_shell_vars (the policy) ---

def test_worker_gets_its_task_as_plain_var(monkeypatch):
    _with_worker(monkeypatch)
    assert worker_plain_shell_vars() == {"HERMES_KANBAN_TASK": "task-abc-123"}


def test_worker_value_is_stripped(monkeypatch):
    _with_worker(monkeypatch, "  task-9  ")
    assert worker_plain_shell_vars() == {"HERMES_KANBAN_TASK": "task-9"}


def test_non_worker_process_gets_nothing(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert worker_plain_shell_vars() == {}


def test_delegate_child_of_worker_gets_nothing(monkeypatch):
    # An in-process delegate_task child / cron run shares the worker's os.environ but does
    # not own the task — its shells must stay as inert as its env already is.
    _with_worker(monkeypatch, owned=False)
    assert worker_plain_shell_vars() == {}


# --- wrapper rendering ---

def test_wrapper_sets_plain_assignment_before_the_command(monkeypatch):
    wrapped = _wrapper("echo hi", plain_vars={"HERMES_KANBAN_TASK": "task-abc-123"})
    # shlex.quote leaves a clean token unquoted — the assignment is still shell-safe.
    assert "HERMES_KANBAN_TASK=task-abc-123" in wrapped
    # Plain assignment, never an export: the re-dump and child envs must not see it.
    assert "export HERMES_KANBAN_TASK" not in wrapped
    assignment_at = wrapped.index("HERMES_KANBAN_TASK=")
    run_at = wrapped.index("eval 'echo hi'")
    assert assignment_at < run_at


def test_wrapper_skips_malformed_names_and_non_string_values():
    wrapped = _wrapper("echo hi", plain_vars={"bad name": "x", 7: "y"})
    assert "bad name" not in wrapped
    assert "7=" not in wrapped


# --- end-to-end semantics (real bash) ---

def _run_wrapper(script: str) -> subprocess.CompletedProcess:
    # Spawn env WITHOUT the var: in production the terminal spawn env is scrubbed
    # (delegated_child_subprocess_env), so the wrapper's plain assignment is the only
    # place the value exists — nothing may leak in via the inherited process env.
    child_env = {k: v for k, v in os.environ.items() if k != "HERMES_KANBAN_TASK"}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=30, env=child_env, check=False)


def test_command_and_subshell_expand_the_var_children_do_not(monkeypatch):
    _with_worker(monkeypatch)
    command = (
        'printf "shell=%s" "$HERMES_KANBAN_TASK"; '
        'printf " sub=%s" "$(printf %s "$HERMES_KANBAN_TASK")"; '
        'printf " child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"'
    )
    result = _run_wrapper(_wrapper(command, plain_vars=worker_plain_shell_vars()))
    assert result.returncode == 0, result.stderr
    assert "shell=task-abc-123 sub=task-abc-123 child=[]" in result.stdout


def test_hostile_value_expands_verbatim_without_injection(monkeypatch):
    hostile = "x'; rm -rf /; echo '"
    _with_worker(monkeypatch, hostile)
    result = _run_wrapper(_wrapper("printf %s \"$HERMES_KANBAN_TASK\"",
                                   plain_vars=worker_plain_shell_vars()))
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(hostile)
    assert "rm -rf" not in result.stdout.replace(hostile, "")


def test_env_dump_never_carries_the_plain_var(monkeypatch):
    # persist_env re-dumps `export -p` into the snapshot; a plain variable must never land
    # there, or the NEXT command (and its children) would inherit worker identity.
    _with_worker(monkeypatch)
    command = 'printf %s "$HERMES_KANBAN_TASK"'
    wrapped = _wrap_command_script(
        command, plain_vars=worker_plain_shell_vars(),
        quoted_cwd="'/tmp'", quoted_snap="'/tmp/hermes-snap.sh'",
        snap_tmp_template="'/tmp/hermes-snap.sh.tmp.XXXXXXXXXX'",
        passthrough_names=(), snapshot_ready=True, persist_env=True,
        cwd_marker="HERMES_TEST_CWD:")
    dump_at = wrapped.index("export -p")
    assert "HERMES_KANBAN_TASK" not in wrapped[dump_at:]


# --- LocalEnvironment wiring ---

class _QuietLocalEnvironment:
    """Lazy proxy: construct the real LocalEnvironment without its login-shell snapshot."""

    def __init__(self):
        from tools.environments.local import LocalEnvironment

        class _NoInit(LocalEnvironment):
            def init_session(self):
                self._snapshot_ready = False

        self._env = _NoInit(cwd="/tmp", timeout=10)

    def __getattr__(self, name):
        return getattr(self._env, name)


def test_local_environment_override_uses_the_policy(monkeypatch):
    _with_worker(monkeypatch)
    env = _QuietLocalEnvironment()
    assert env._plain_shell_vars() == {"HERMES_KANBAN_TASK": "task-abc-123"}


def test_local_environment_threading_reaches_the_wrapper(monkeypatch):
    _with_worker(monkeypatch)
    env = _QuietLocalEnvironment()
    wrapped = env._wrap_command("echo $HERMES_KANBAN_TASK", "/tmp")
    assert "HERMES_KANBAN_TASK=task-abc-123" in wrapped


def test_base_environment_default_is_empty():
    env = _TestableBaseEnv()
    assert env._plain_shell_vars() == {}


class _TestableBaseEnv(BaseEnvironment):
    _passwordless_sudo_probe_supported = True

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError

    def cleanup(self):
        pass


@pytest.mark.parametrize("value", ["with space", "with'quote", "$HOME", '`id`'])
def test_values_are_shlex_quoted_safely(value):
    # The rendering is exactly shlex.quote's (never a bare interpolation), so the value
    # expands verbatim inside the command and is never re-parsed as syntax.
    wrapped = _wrapper("echo hi", plain_vars={"V": value})
    assert f"V={shlex.quote(value)}" in wrapped
