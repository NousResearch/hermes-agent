"""Unit tests for BaseEnvironment's pure helpers and process-lifecycle seams.

``test_base_environment.py`` covers the execute/snapshot integration flow. This
file covers the seams that can be exercised without spawning a backend process:
the activity heartbeat, the snapshot JSON store, the shell-quoting helpers, the
stdin-staging helpers, CWD-marker parsing, the non-login bash probe, the
foreground-command registry used by the process-exit funnel, and the PEP 562
plugin-compat shims.

For #36543 (bring ``tools/environments/base.py`` to >= 70% statement coverage).
"""

import subprocess
import threading
import time
from unittest.mock import MagicMock

import pytest

from tools.environments import base as base_mod
from tools.environments.base import (
    BaseEnvironment,
    EnvironmentConnectionError,
    _file_mtime_key,
    _load_json_store,
    _quiet_kill,
    _save_json_store,
    get_activity_callback,
    get_sandbox_dir,
    kill_live_foreground_processes,
    set_activity_callback,
    touch_activity_if_due,
)


class _TestableEnv(BaseEnvironment):
    """Minimal concrete backend: every subprocess seam is mocked out."""

    _sudo_nopasswd_probe_supported = True

    def __init__(self, cwd="/tmp", timeout=10):
        super().__init__(cwd=cwd, timeout=timeout)

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        raise NotImplementedError("Use mock")

    def cleanup(self):
        pass


@pytest.fixture
def clean_foreground_registry():
    """Snapshot/restore the process-exit funnel's module globals.

    ``_exit_fenced`` is one-way and ``_live_foreground`` is process-wide, so a
    test that raises the fence would silently disable foreground spawning for
    every later test in the session.
    """
    live = dict(base_mod._live_foreground)
    fenced = base_mod._exit_fenced
    inflight = base_mod._spawns_in_flight
    try:
        yield
    finally:
        base_mod._live_foreground.clear()
        base_mod._live_foreground.update(live)
        base_mod._exit_fenced = fenced
        base_mod._spawns_in_flight = inflight


@pytest.fixture
def clean_activity_callback():
    """The activity callback is thread-local; don't leak one into other tests."""
    yield
    if hasattr(base_mod._activity_callback_local, "callback"):
        del base_mod._activity_callback_local.callback


# --------------------------------------------------------------------------
# EnvironmentConnectionError
# --------------------------------------------------------------------------


def test_connection_error_defaults_to_infrastructure_retry_hint():
    err = EnvironmentConnectionError("ssh host unreachable")

    assert err.reason == "ssh host unreachable"
    assert str(err) == "ssh host unreachable"
    # The default hint must tell the operator this is NOT a command failure and
    # must name the remediation step, not just the symptom.
    assert "not a command failure" in err.retry_hint
    assert "retry" in err.retry_hint.lower()
    # Subclassing RuntimeError keeps every `except RuntimeError` catcher working.
    assert isinstance(err, RuntimeError)


def test_connection_error_keeps_explicit_retry_hint():
    err = EnvironmentConnectionError("docker daemon down", retry_hint="start Docker Desktop")

    assert err.retry_hint == "start Docker Desktop"


# --------------------------------------------------------------------------
# Activity heartbeat
# --------------------------------------------------------------------------


def test_activity_callback_round_trips_and_clears(clean_activity_callback):
    assert get_activity_callback() is None

    cb = lambda label: None  # noqa: E731
    set_activity_callback(cb)
    assert get_activity_callback() is cb

    set_activity_callback(None)
    assert get_activity_callback() is None


def test_activity_callback_is_thread_local(clean_activity_callback):
    set_activity_callback(lambda label: None)
    seen = {}

    def worker():
        seen["callback"] = get_activity_callback()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    # A freshly spawned thread cannot read the parent's callback back, which is
    # exactly why get_activity_callback() exists as a public accessor.
    assert seen["callback"] is None


def test_touch_activity_fires_after_interval_with_elapsed_seconds(clean_activity_callback):
    seen = []
    set_activity_callback(seen.append)
    state = {"last_touch": 0.0, "start": 100.0, "interval": 10.0}

    # Now (monotonic) is far past last_touch=0, so the heartbeat is due.
    touch_activity_if_due(state, "compiling")

    assert len(seen) == 1
    assert seen[0].startswith("compiling (")
    assert seen[0].endswith("s elapsed)")
    assert state["last_touch"] > 0.0


def test_touch_activity_is_throttled_by_interval(clean_activity_callback):
    seen = []
    set_activity_callback(seen.append)
    now = time.monotonic()
    state = {"last_touch": now, "start": now - 5.0, "interval": 3600.0}

    touch_activity_if_due(state, "compiling")

    # A heartbeat that fired 5s ago must stay quiet for the rest of the interval,
    # and a throttled touch must not reschedule the timer.
    assert seen == []
    assert state["last_touch"] == now


def test_touch_activity_respects_custom_interval(clean_activity_callback):
    seen = []
    set_activity_callback(seen.append)
    state = {"last_touch": 0.0, "start": 0.0, "interval": 0.0}

    touch_activity_if_due(state, "tail")
    assert len(seen) == 1

    # A zero interval is always due, so a second call fires again.
    touch_activity_if_due(state, "tail")
    assert len(seen) == 2


def test_touch_activity_swallows_callback_exceptions(clean_activity_callback):
    def boom(label):
        raise RuntimeError("gateway websocket gone")

    set_activity_callback(boom)
    state = {"last_touch": 0.0, "start": 0.0, "interval": 0.0}

    # A dead activity sink must never take down a long-running command.
    touch_activity_if_due(state, "compiling")
    assert state["last_touch"] > 0.0


def test_touch_activity_without_registered_callback_still_records_time(clean_activity_callback):
    state = {"last_touch": 0.0, "start": 0.0, "interval": 0.0}

    touch_activity_if_due(state, "compiling")

    assert state["last_touch"] > 0.0


# --------------------------------------------------------------------------
# Sandbox dir + snapshot JSON store
# --------------------------------------------------------------------------


def test_get_sandbox_dir_honours_override_and_creates_it(tmp_path, monkeypatch):
    custom = tmp_path / "nested" / "custom-sandboxes"
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(custom))

    resolved = get_sandbox_dir()

    assert resolved == custom
    assert custom.is_dir()


def test_get_sandbox_dir_defaults_under_hermes_home(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
    monkeypatch.setattr(base_mod, "get_hermes_home", lambda: tmp_path)

    resolved = get_sandbox_dir()

    assert resolved == tmp_path / "sandboxes"
    assert resolved.is_dir()


def test_save_json_store_round_trips_and_creates_parents(tmp_path):
    path = tmp_path / "deep" / "nested" / "snapshots.json"
    data = {"task": "snapshot", "vars": {"PATH": "/usr/bin"}}

    _save_json_store(path, data)

    assert path.is_file()
    assert _load_json_store(path) == data
    # Written pretty-printed so a hand-inspected snapshot stays readable.
    assert "\n  " in path.read_text(encoding="utf-8")


def test_file_mtime_key_reports_size_and_mtime(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"x" * 17)

    key = _file_mtime_key(str(target))

    assert key is not None
    mtime, size = key
    assert size == 17
    assert mtime == target.stat().st_mtime


def test_file_mtime_key_is_none_for_unreadable_path(tmp_path):
    assert _file_mtime_key(str(tmp_path / "does-not-exist")) is None
    assert _file_mtime_key(str(tmp_path / "nope" / "deeper")) is None


# --------------------------------------------------------------------------
# Shell quoting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("~", "~"),
        ("~/", "$HOME"),
        ("~/work", "$HOME/work"),
        # A space in the suffix must stay one word, which is why ~/ goes through
        # $HOME instead of a bare `cd` on the raw string.
        ("~/my work", "$HOME/'my work'"),
        ("/srv/app", "/srv/app"),
        ("/srv/my app", "'/srv/my app'"),
    ],
)
def test_quote_cwd_for_cd_preserves_tilde_expansion(cwd, expected):
    assert BaseEnvironment._quote_cwd_for_cd(cwd) == expected


def test_quote_shell_path_defaults_to_shlex_quote():
    env = _TestableEnv()

    assert env._quote_shell_path("/srv/my app") == "'/srv/my app'"


# --------------------------------------------------------------------------
# stdin staging
# --------------------------------------------------------------------------


def test_staged_stdin_path_is_unique_and_under_temp_dir():
    env = _TestableEnv()
    env.get_temp_dir = lambda: "/tmp/hermes-env"  # type: ignore[assignment]

    first = env._staged_stdin_path()
    second = env._staged_stdin_path()

    assert first.startswith("/tmp/hermes-env/.hermes-stdin-")
    assert first != second, "each staged payload needs its own file"


def test_staged_stdin_path_tolerates_trailing_slash_temp_dir():
    env = _TestableEnv()
    env.get_temp_dir = lambda: "/tmp/hermes-env/"  # type: ignore[assignment]

    assert env._staged_stdin_path().startswith("/tmp/hermes-env/.hermes-stdin-")


def test_redirect_stdin_from_file_unlinks_before_running_command():
    command = BaseEnvironment._redirect_stdin_from_file("wc -l", "/tmp/env/.hermes-stdin-abc")

    lines = command.splitlines()
    # Shell owns the payload: read it, unlink it, and only then run the command.
    # A failed redirect or unlink aborts instead of running the command with no
    # stdin, so the model never sees a silent empty-input result.
    assert lines[0] == "exec 0< /tmp/env/.hermes-stdin-abc || exit $?"
    assert lines[1] == "rm -f -- /tmp/env/.hermes-stdin-abc || exit $?"
    assert lines[2] == "wc -l"


def test_redirect_stdin_from_file_quotes_paths_with_spaces():
    command = BaseEnvironment._redirect_stdin_from_file("cat", "/tmp/my env/payload file")

    assert "'/tmp/my env/payload file'" in command
    assert command.endswith("cat")


# --------------------------------------------------------------------------
# CWD marker parsing
# --------------------------------------------------------------------------


def test_extract_cwd_leaves_cwd_alone_when_no_marker_emitted():
    env = _TestableEnv()
    env.cwd = "/original"
    result = {"output": "just some output\n", "returncode": 0}

    env._update_cwd(result)

    # A killed/timed-out command emits no marker, so cwd_observed must stay
    # absent rather than claiming an observation that never happened.
    assert env.cwd == "/original"
    assert "cwd_observed" not in result
    assert "cwd" not in result
    assert result["output"] == "just some output\n"


def test_extract_cwd_records_observation_and_strips_marker():
    env = _TestableEnv()
    marker = env._cwd_marker
    result = {"output": f"before\n{marker}/srv/app{marker}\nafter\n"}

    env._update_cwd(result)

    assert env.cwd == "/srv/app"
    assert result["cwd_observed"] is True
    assert result["cwd"] == "/srv/app"
    assert marker not in result["output"]
    assert "before" in result["output"] and "after" in result["output"]


def test_extract_cwd_empty_marker_cleans_output_without_claiming_cwd():
    env = _TestableEnv()
    env.cwd = "/original"
    marker = env._cwd_marker
    result = {"output": f"before\n{marker}{marker}\n"}

    env._update_cwd(result)

    assert env.cwd == "/original"
    assert "cwd_observed" not in result
    assert marker not in result["output"]


# --------------------------------------------------------------------------
# Non-login bash probe
# --------------------------------------------------------------------------


def test_probe_nonlogin_prefers_nonlogin_when_probe_succeeds():
    env = _TestableEnv()
    env._run_bash = MagicMock(return_value=object())  # type: ignore[assignment]
    env._wait_for_process = MagicMock(return_value={"returncode": 0})  # type: ignore[assignment]

    prefer, detail = env._probe_nonlogin_fallback("login bash dead")

    assert prefer is True
    assert detail == "login bash dead"
    # The probe must not run under a login shell, and must be time-capped.
    assert env._run_bash.call_args.kwargs["login"] is False
    assert env._run_bash.call_args.kwargs["timeout"] <= 15


def test_probe_nonlogin_reports_probe_stdout_when_it_fails():
    env = _TestableEnv()
    env._run_bash = MagicMock(return_value=object())  # type: ignore[assignment]
    env._wait_for_process = MagicMock(  # type: ignore[assignment]
        return_value={"returncode": 1, "stdout": "bash: /etc/profile: Permission denied\n"}
    )

    prefer, detail = env._probe_nonlogin_fallback("login bash dead")

    assert prefer is False
    assert "Permission denied" in detail


def test_probe_nonlogin_falls_back_to_detail_when_probe_is_silent():
    env = _TestableEnv()
    env._run_bash = MagicMock(return_value=object())  # type: ignore[assignment]
    env._wait_for_process = MagicMock(return_value={"returncode": 1, "stdout": "  "})  # type: ignore[assignment]

    _prefer, detail = env._probe_nonlogin_fallback("original reason")

    assert detail == "original reason"


def test_probe_nonlogin_reports_probe_exception():
    env = _TestableEnv()

    def boom(*_args, **_kwargs):
        raise OSError("ssh: connect to host refused")

    env._run_bash = boom  # type: ignore[assignment]

    prefer, detail = env._probe_nonlogin_fallback("snapshot failed")

    # Fails closed: an unprobeable backend must not be assumed healthy.
    assert prefer is False
    assert "snapshot failed" in detail
    assert "non-login probe" in detail
    assert "refused" in detail


# --------------------------------------------------------------------------
# Snapshot exclusions + script kwargs
# --------------------------------------------------------------------------


def test_no_profile_scoped_passthrough_means_no_snapshot_exclusions():
    env = _TestableEnv()
    env._profile_scoped_passthrough = False

    assert env._snapshot_excluded_passthrough_names() == ()


def test_snapshot_exclusions_refresh_from_live_passthrough_allowlist(monkeypatch):
    import agent.secret_scope as secret_scope
    import tools.env_passthrough as env_passthrough

    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    monkeypatch.setattr(
        env_passthrough, "get_all_passthrough", lambda: {"HERMES_PROFILE_ID", "BAD NAME", "9BAD"}
    )

    env = _TestableEnv()
    env._profile_scoped_passthrough = True
    env._snapshot_passthrough_names = set()

    names = env._snapshot_excluded_passthrough_names()

    # Only well-formed shell env names are kept, and the result is sorted so the
    # generated script is byte-stable across runs.
    assert names == ("HERMES_PROFILE_ID",)
    # The captured set is retained even if the allowlist is later cleared, so an
    # old value cannot leak into a later profile.
    assert env._snapshot_passthrough_names == {"HERMES_PROFILE_ID"}
    assert env._snapshot_excluded_passthrough_names() == ("HERMES_PROFILE_ID",)


def test_snapshot_exclusions_survive_a_raising_allowlist(monkeypatch):
    import agent.secret_scope as secret_scope

    def boom():
        raise RuntimeError("passthrough registry unavailable")

    monkeypatch.setattr(secret_scope, "is_multiplex_active", boom)

    env = _TestableEnv()
    env._profile_scoped_passthrough = True
    env._snapshot_passthrough_names = {"HERMES_PROFILE_ID"}

    # A refresh failure degrades to the retained set rather than dropping
    # exclusions and leaking a previous profile's value.
    assert env._snapshot_excluded_passthrough_names() == ("HERMES_PROFILE_ID",)


def test_snapshot_script_kwargs_delegates_quoting_to_the_backend():
    env = _TestableEnv()
    env._snapshot_path = "/tmp/env/snapshot.sh"

    kwargs = env._snapshot_script_kwargs("/srv/my app")

    assert kwargs["quoted_cwd"] == "'/srv/my app'"
    assert kwargs["quoted_snap"] == "/tmp/env/snapshot.sh"
    assert kwargs["snap_tmp_template"] == env._quote_shell_path(env._snapshot_path + ".tmp.XXXXXXXXXX")
    assert kwargs["cwd_marker"] == env._cwd_marker


def test_additional_profile_scoped_passthrough_names_defaults_to_empty():
    assert _TestableEnv()._additional_profile_scoped_passthrough_names() == ()


# --------------------------------------------------------------------------
# Process kill seams
# --------------------------------------------------------------------------


def test_kill_process_calls_kill():
    proc = MagicMock()
    env = _TestableEnv()

    env._kill_process(proc)

    proc.kill.assert_called_once_with()


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError, OSError])
def test_kill_process_swallows_already_dead_process(error):
    proc = MagicMock()
    proc.kill.side_effect = error("gone")
    env = _TestableEnv()

    # Racing a process that already exited must not raise out of the kill path.
    env._kill_process(proc)


def test_force_kill_process_delegates_to_kill_process():
    proc = MagicMock()
    env = _TestableEnv()
    env._kill_process = MagicMock()  # type: ignore[assignment]

    env._force_kill_process(proc)

    env._kill_process.assert_called_once_with(proc)


def test_quiet_kill_swallows_kill_failures():
    def boom(_proc):
        raise RuntimeError("kill failed")

    # The exit-time funnel must never raise, whatever the backend's kill does.
    _quiet_kill(boom, object())


def test_before_execute_is_a_noop_and_mark_recreated_sets_the_notice_flag():
    env = _TestableEnv()

    # Local backends have nothing to sync before a command.
    env._before_execute()
    assert getattr(env, "_recreated_notice_pending", False) is False

    env._mark_recreated()
    assert env._recreated_notice_pending is True


def test_del_swallows_cleanup_exceptions():
    class Exploding(_TestableEnv):
        def cleanup(self):
            raise RuntimeError("cleanup blew up")

    # __del__ runs during GC; a raising cleanup must not escalate.
    Exploding().__del__()


# --------------------------------------------------------------------------
# Foreground command registry (process-exit funnel)
# --------------------------------------------------------------------------


def test_quiet_foreground_kill_uses_gentle_kill(clean_foreground_registry):
    proc = object()
    env = _TestableEnv()
    env._kill_process = MagicMock()  # type: ignore[assignment]
    base_mod._live_foreground[id(proc)] = (env, proc)

    signalled = kill_live_foreground_processes()

    assert signalled == 1
    env._kill_process.assert_called_once_with(proc)


def test_no_foreground_commands_reports_zero(clean_foreground_registry):
    assert kill_live_foreground_processes() == 0


def test_leave_foreground_spawn_with_failed_spawn_registers_nothing(clean_foreground_registry):
    env = _TestableEnv()

    fenced = base_mod._leave_foreground_spawn(env, None)

    assert fenced is False
    assert base_mod._live_foreground == {}


def test_leave_foreground_spawn_publishes_the_handle(clean_foreground_registry):
    proc = object()
    env = _TestableEnv()

    base_mod._enter_foreground_spawn()
    fenced = base_mod._leave_foreground_spawn(env, proc)

    assert fenced is False
    assert base_mod._live_foreground[id(proc)] == (env, proc)
    assert base_mod._spawns_in_flight == 0


def test_hard_kill_raises_the_exit_fence(clean_foreground_registry):
    env = _TestableEnv()
    env._force_kill_process = MagicMock()  # type: ignore[assignment]
    popen = MagicMock(spec=subprocess.Popen)
    base_mod._live_foreground[id(popen)] = (env, popen)

    signalled = kill_live_foreground_processes(now=True)

    assert signalled == 1
    # A hard exit force-kills: a SIGTERM-ignoring command must not outlive it.
    env._force_kill_process.assert_called_once_with(popen)
    assert base_mod._exit_fenced is True
    # Once fenced, no new foreground command may spawn.
    assert base_mod._enter_foreground_spawn() is False
    assert base_mod._spawns_in_flight == 0


def test_hard_kill_runs_remote_handle_kills_off_thread(clean_foreground_registry):
    class RemoteHandle:
        """Not a local Popen, so the kill belongs on a daemon thread."""

    handle = RemoteHandle()
    env = _TestableEnv()
    done = threading.Event()
    env._force_kill_process = MagicMock(side_effect=lambda _p: done.set())  # type: ignore[assignment]
    base_mod._live_foreground[id(handle)] = (env, handle)

    signalled = kill_live_foreground_processes(now=True)

    assert signalled == 1
    assert done.wait(2.0), "remote handle kill never ran"


# --------------------------------------------------------------------------
# PEP 562 plugin-compat shims
# --------------------------------------------------------------------------


def test_plugin_compat_resolves_lazy_attribute():
    from tools.environments.path_utils import sanitize_task_id_for_path

    assert base_mod.sanitize_task_id_for_path is sanitize_task_id_for_path


def test_plugin_compat_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'no_such_thing'"):
        base_mod.no_such_thing


# --------------------------------------------------------------------------
# fetch_realpath
# --------------------------------------------------------------------------


def test_fetch_realpath_returns_none_on_nonzero_exit():
    env = _TestableEnv()
    env.execute = MagicMock(return_value={"returncode": 1, "output": "no such link"})  # type: ignore[assignment]

    assert env.fetch_realpath("/tmp/missing") is None
    assert "2>/dev/null" in env.execute.call_args.args[0]


def test_fetch_realpath_takes_last_absolute_line():
    env = _TestableEnv()
    env.execute = MagicMock(  # type: ignore[assignment]
        return_value={"returncode": 0, "output": "resolve: not found\n/real/target\n"}
    )

    assert env.fetch_realpath("/tmp/link") == "/real/target"


def test_fetch_realpath_returns_none_when_output_has_no_absolute_path():
    env = _TestableEnv()
    env.execute = MagicMock(return_value={"returncode": 0, "output": "relative/path\n"})  # type: ignore[assignment]

    assert env.fetch_realpath("/tmp/link") is None
