"""The dispatcher-owned worker's terminal session expands ``$HERMES_KANBAN_TASK`` as a plain
(non-exported) shell variable.

The worker protocol documents the var to shells (done-when receipts, evidence naming;
#160138), while the child-env scrub deliberately keeps worker identity away from every child
process a worker starts (#90782, #112817). A plain shell variable threads that needle: the
command and its subshells expand it, ``env``/child processes never see it. ``builtin export -n``
after the assignment keeps the export attribute off the name even when it was already exported
(a sourced snapshot that carried an in-command ``export``) or under ``set -a``.
"""

import os
import subprocess

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
    non_dispatcher_owned_context,
    worker_plain_shell_command_prefix,
    worker_plain_shell_vars,
)
from tools.environments.base import BaseEnvironment
from tools.environments.base_session_env import _wrap_command_script
from tools.environments.local import LocalEnvironment, _find_bash

TASK_ID = "task-abc-123"
try:
    _BASH = _find_bash()  # raises (not returns None) on hosts without bash
except Exception:  # pragma: no cover - POSIX CI always has bash
    _BASH = None

needs_bash = pytest.mark.skipif(not _BASH, reason="bash required for real-shell tests")


def _wrapper(command: str, plain_vars=None, *, snapshot_ready=False, persist_env=True) -> str:
    return _wrap_command_script(
        command,
        quoted_cwd="'/tmp'",
        quoted_snap="'/tmp/hermes-snap.sh'",
        snap_tmp_template="'/tmp/hermes-snap.sh.tmp.XXXXXXXXXX'",
        passthrough_names=(),
        snapshot_ready=snapshot_ready,
        persist_env=persist_env,
        cwd_marker="HERMES_TEST_CWD:",
        plain_vars=plain_vars)


def _set_worker_env(monkeypatch, task_id=TASK_ID):
    if task_id is None:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)


def _scrubbed_spawn_env(monkeypatch) -> dict:
    """Env WITHOUT the var: in production the terminal spawn env is scrubbed
    (delegated_child_subprocess_env), so the wrapper's assignment is the only place the
    value exists — nothing may leak in via the inherited process env."""
    _set_worker_env(monkeypatch)
    return {k: v for k, v in os.environ.items() if k != "HERMES_KANBAN_TASK"}


def _run_bash(script: str, cwd, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([_BASH, "-c", script], capture_output=True, text=True,
                          timeout=30, check=False, cwd=str(cwd), env=env)


# --- the policy: real contexts, no mocked predicates ---

def test_worker_gets_its_task_as_plain_var(monkeypatch):
    _set_worker_env(monkeypatch)
    assert worker_plain_shell_vars() == {"HERMES_KANBAN_TASK": TASK_ID}


def test_spawned_descendant_marked_in_env_gets_nothing(monkeypatch):
    _set_worker_env(monkeypatch)
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "/home/x/.hermes/kanban")
    assert worker_plain_shell_vars() == {}


def test_in_process_delegate_child_of_worker_gets_nothing(monkeypatch):
    _set_worker_env(monkeypatch)
    with delegated_child_context():
        assert worker_plain_shell_vars() == {}


def test_cron_run_inside_a_worker_gets_nothing(monkeypatch):
    _set_worker_env(monkeypatch)
    with non_dispatcher_owned_context():
        assert worker_plain_shell_vars() == {}


def test_whitespace_only_value_is_not_a_task(monkeypatch):
    _set_worker_env(monkeypatch, "   ")
    assert worker_plain_shell_vars() == {}


def test_command_prefix_is_empty_for_non_workers(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert worker_plain_shell_command_prefix() == ""


# --- rendering: the one structural contract + the loud failure mode ---

def test_wrapper_sets_plain_assignment_before_the_command():
    wrapped = _wrapper("echo hi", {"HERMES_KANBAN_TASK": TASK_ID})
    # shlex.quote leaves a clean token unquoted — the assignment is still shell-safe.
    # Two lines: an assignment that fails outright must not take the export -n with it.
    assert f"HERMES_KANBAN_TASK={TASK_ID}\nbuiltin export -n HERMES_KANBAN_TASK" in wrapped
    assignment_at = wrapped.index("HERMES_KANBAN_TASK=")
    run_at = wrapped.index("eval 'echo hi'")
    assert assignment_at < run_at


def test_wrapper_rejects_names_that_are_not_shell_identifiers():
    with pytest.raises(ValueError):
        _wrapper("echo hi", {"bad name": "x"})


@needs_bash
def test_command_prefix_is_unset_first(monkeypatch, tmp_path):
    """The background/PTY prefix must behave, not merely spell: with the name present AND
    exported in the spawn env (an rc file could have done it), the unset-first prefix still
    leaves the command's child with nothing."""
    _set_worker_env(monkeypatch)
    prefix = worker_plain_shell_command_prefix()
    result = _run_bash(
        f'{prefix}printf "child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"',
        tmp_path, env=dict(os.environ))  # env deliberately NOT scrubbed: var inherited
    assert result.returncode == 0, result.stderr
    assert "child=[]" in result.stdout


@needs_bash
def test_background_prefix_survives_allexport(monkeypatch, tmp_path):
    """F2: an rc that left ``set -a`` on would turn the prefix assignment into an export;
    the POSIX ``case $-`` guard closes it for every background shell."""
    env = _scrubbed_spawn_env(monkeypatch)
    result = _run_bash(
        f"set -a\n{worker_plain_shell_command_prefix()}set +m; "
        'printf "child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"',
        tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert "child=[]" in result.stdout


# --- end-to-end semantics (real bash; spawn env mirrors production: no var) ---

@needs_bash
def test_command_and_subshell_expand_the_var_children_do_not(monkeypatch, tmp_path):
    env = _scrubbed_spawn_env(monkeypatch)
    command = (
        'printf "shell=%s" "$HERMES_KANBAN_TASK"; '
        'printf " sub=%s" "$(printf %s "$HERMES_KANBAN_TASK")"; '
        'printf " child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"'
    )
    result = _run_bash(_wrapper(command, worker_plain_shell_vars()), tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert "shell=task-abc-123 sub=task-abc-123 child=[]" in result.stdout


@needs_bash
@pytest.mark.parametrize("hostile", [
    "x'; rm -rf /; echo '",
    "$(id)",
    "`id`",
    "with space",
])
def test_hostile_values_expand_verbatim_without_injection(monkeypatch, tmp_path, hostile):
    env = _scrubbed_spawn_env(monkeypatch)
    result = _run_bash(
        _wrapper('printf %s "$HERMES_KANBAN_TASK"', {"HERMES_KANBAN_TASK": hostile}), tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(hostile)
    assert "rm -rf" not in result.stdout.replace(hostile, "")


@needs_bash
def test_mid_command_export_never_persists_across_commands(monkeypatch, tmp_path):
    """P0-1 regression: command 1 exports the name; command 2 must still see it as a plain
    variable only — the snapshot re-dump drops the name and ``export -n`` clears the
    attribute, so command 2's child inherits nothing."""
    env = _scrubbed_spawn_env(monkeypatch)
    snap = tmp_path / "snap.sh"
    first = _wrap_command_script(
        "export HERMES_KANBAN_TASK", plain_vars={"HERMES_KANBAN_TASK": TASK_ID},
        quoted_cwd=f"'{tmp_path}'", quoted_snap=f"'{snap}'",
        snap_tmp_template=f"'{snap}.tmp.XXXXXXXXXX'",
        passthrough_names=(), snapshot_ready=True, persist_env=True,
        cwd_marker="HERMES_TEST_CWD:")
    second = _wrap_command_script(
        'printf "child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"',
        plain_vars={"HERMES_KANBAN_TASK": TASK_ID},
        quoted_cwd=f"'{tmp_path}'", quoted_snap=f"'{snap}'",
        snap_tmp_template=f"'{snap}.tmp.XXXXXXXXXX'",
        passthrough_names=(), snapshot_ready=True, persist_env=True,
        cwd_marker="HERMES_TEST_CWD:")
    # Two separate invocations, like production's spawn-per-call: each wrapper ends in
    # `exit`, so concatenating them into one bash would never reach the second.
    first_run = _run_bash(first, tmp_path, env)
    assert first_run.returncode == 0, first_run.stderr
    second_run = _run_bash(second, tmp_path, env)
    assert second_run.returncode == 0, second_run.stderr
    assert "child=[]" in second_run.stdout
    # And the snapshot file itself never carries the export line.
    assert "HERMES_KANBAN_TASK" not in snap.read_text()


@needs_bash
def test_allexport_cannot_turn_the_assignment_into_an_export(monkeypatch, tmp_path):
    """P0-1: under ``set -a`` (an rc file that left allexport on) a bare assignment would
    export; ``builtin export -n`` in the wrapper keeps the attribute off."""
    env = _scrubbed_spawn_env(monkeypatch)
    inner = _wrapper('printf "child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"',
                     {"HERMES_KANBAN_TASK": TASK_ID})
    result = _run_bash(f"set -a\n{inner}", tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert "child=[]" in result.stdout


@needs_bash
def test_background_prefix_expands_in_shell_but_not_in_children(monkeypatch, tmp_path):
    env = _scrubbed_spawn_env(monkeypatch)
    result = _run_bash(
        f"{worker_plain_shell_command_prefix()}set +m; "
        'printf "shell=%s" "$HERMES_KANBAN_TASK"; '
        'printf " child=[%s]" "$(sh -c \'printf %s "$HERMES_KANBAN_TASK"\')"',
        tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert "shell=task-abc-123 child=[]" in result.stdout


# --- the production chain: LocalEnvironment.execute, red on base, green here ---

@needs_bash
def test_local_environment_execute_the_full_chain(monkeypatch, tmp_path):
    """The #160138 repro through the real machinery: scrubbed spawn env, worker os.environ,
    login snapshot, per-command wrapper. The shell expands the id; the command's child gets
    no identity but does carry the non-owner fence marker. Subsumes the policy-override and
    wrapper-threading shape checks."""
    _set_worker_env(monkeypatch)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        result = env.execute(
            'printf %s "$HERMES_KANBAN_TASK"; printf "|%s" '
            '"$(sh -c \'printf %s "$HERMES_KANBAN_TASK"; printf "|%s" "${HERMES_DELEGATED_CHILD_CONTEXT-}"\')"',
            cwd=str(tmp_path))
    finally:
        env.cleanup()
    assert result["returncode"] == 0, result["output"]
    shell_out, _, rest = result["output"].partition("|")
    child_task, _, child_marker = rest.partition("|")
    assert shell_out == TASK_ID
    assert child_task == ""  # the child process inherited no identity
    assert child_marker.strip()  # …but it carries the non-owner fence


@needs_bash
def test_scope_argv_wiring_carries_the_prefix(monkeypatch, tmp_path):
    """F1: the background/PTY fix lives in ProcessRegistry._scope_argv — deleting the prefix
    there must turn this test red. Unbound call: the method reads no registry state, only
    the session's systemd_unit."""
    from types import SimpleNamespace
    from typing import cast

    from tools.process_registry import ProcessRegistry, ProcessSession

    _set_worker_env(monkeypatch)
    session = cast(ProcessSession, SimpleNamespace(systemd_unit=None))
    argv = ProcessRegistry._scope_argv(cast(ProcessRegistry, None), session, "echo hi", "u1", "test")
    assert len(argv) == 3
    assert worker_plain_shell_command_prefix() in argv[2]
    assert "echo hi" in argv[2]


def test_base_environment_default_is_empty(tmp_path):
    env = _TestableBaseEnv(cwd=str(tmp_path))
    assert env._plain_shell_vars() == {}


class _TestableBaseEnv(BaseEnvironment):
    _passwordless_sudo_probe_supported = True

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError

    def cleanup(self):
        pass
