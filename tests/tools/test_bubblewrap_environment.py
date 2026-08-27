"""Tests for BubblewrapEnvironment: the LocalEnvironment subclass that runs
every spawn inside a bwrap sandbox.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import builtins
import inspect
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments.base import get_sandbox_dir
from tools.environments import bubblewrap
from tools.environments.bubblewrap import (
    BindMount,
    BubblewrapConfig,
    BubblewrapEnvironment,
    chdir_failed,
    masked_inside,
    HOST_SOCKET_VARS,
    build_bwrap_args,
)
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _bwrap_probe_passed(monkeypatch):
    """Unit constructions never spawn: count the process-wide bwrap probe as passed."""
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")


def _bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    try:
        probe = subprocess.run(
            ["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
            capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


BWRAP_USABLE = _bwrap_usable()
needs_bwrap = pytest.mark.skipif(not BWRAP_USABLE, reason="bwrap missing or its namespace probe failed")


@pytest.fixture
def sandbox_root(tmp_path, monkeypatch):
    root = tmp_path / "sandboxes"
    monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
    return root


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    return d


def _no_session():
    return patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None)


class TestStructure:
    def test_is_a_local_environment_subclass(self):
        assert issubclass(BubblewrapEnvironment, LocalEnvironment)

    def test_unknown_profile_rejected_at_construction(self, sandbox_root, work_dir):
        with _no_session(), pytest.raises(ValueError) as excinfo:
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(profile="bogus"))
        for name in ("restricted", "workspace", "network"):
            assert name in str(excinfo.value)
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())


class TestStateDir:
    def test_temp_dir_is_a_fresh_subdir_of_the_sandbox_root(self, sandbox_root, work_dir):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        temp = Path(env.get_temp_dir())
        assert temp.parent == get_sandbox_dir() == sandbox_root
        assert temp.is_dir()
        assert env._snapshot_path.startswith(str(temp) + os.sep)
        assert env._cwd_file.startswith(str(temp) + os.sep)

    def test_temp_dir_is_unique_per_instance(self, sandbox_root, work_dir):
        with _no_session():
            a = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
            b = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        assert a.get_temp_dir() != b.get_temp_dir()

    def test_temp_dir_exists_before_init_session_runs(self, sandbox_root, work_dir):
        seen = {}

        def fake_init(self):
            seen["exists"] = os.path.isdir(self.get_temp_dir())

        with patch.object(LocalEnvironment, "init_session", autospec=True, side_effect=fake_init):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        assert seen == {"exists": True}

    def test_cleanup_removes_the_state_dir(self, sandbox_root, work_dir):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        temp = env.get_temp_dir()
        Path(env._cwd_file).write_text("x")
        env.cleanup()
        assert not os.path.exists(temp)

    def test_state_dir_is_bound_read_write_in_the_argv(self, sandbox_root, work_dir):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        argv = env._wrap_popen_args(["/bin/bash", "-c", "true"])
        temp = env.get_temp_dir()
        i = argv.index("--bind", argv.index(temp) - 1)
        assert argv[i:i + 3] == ["--bind", temp, temp]
        assert argv[-3:] == ["/bin/bash", "-c", "true"]

    def test_socket_variables_are_unset_in_the_argv(self, sandbox_root, work_dir):
        """The host agent and bus socket variables are dropped by the
        bwrap prefix, so LocalEnvironment's run env is left untouched."""
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        argv = env._wrap_popen_args(["/bin/bash", "-c", "true"])
        prefix = argv[:argv.index("--")]
        assert set(HOST_SOCKET_VARS) == {"SSH_AUTH_SOCK", "GPG_AGENT_INFO", "DBUS_SESSION_BUS_ADDRESS"}
        for name in HOST_SOCKET_VARS:
            i = prefix.index(name)
            assert prefix[i - 1] == "--unsetenv", name

    def test_wrap_command_unsets_the_socket_variables_in_front_of_the_command(self, sandbox_root, work_dir):
        """A shell init file sourced into the snapshot can export them
        again, so the wrapped script unsets them before the command runs;
        LocalEnvironment's wrapper is untouched."""
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
            local = LocalEnvironment(cwd=str(work_dir), timeout=10)
        prefix = "unset " + " ".join(HOST_SOCKET_VARS) + "; "
        assert f"eval '{prefix}echo hi'" in env._wrap_command("echo hi", str(work_dir))
        assert "eval 'echo hi'" in local._wrap_command("echo hi", str(work_dir))
        assert "unset SSH_AUTH_SOCK" not in local._wrap_command("echo hi", str(work_dir))


@needs_bwrap
class TestSandboxIntegration:
    @pytest.fixture
    def env(self, sandbox_root, work_dir):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            yield env
        finally:
            env.cleanup()

    def test_snapshot_lands_in_the_host_state_dir(self, env):
        assert env._snapshot_ready
        assert Path(env._snapshot_path).is_file()
        assert Path(env._snapshot_path).parent == Path(env.get_temp_dir())

    def test_pid_namespace_hides_host_processes(self, env):
        host = len([p for p in os.listdir("/proc") if p.isdigit()])
        result = env.execute("ls /proc | grep -c '^[0-9]'")
        inside = int(result["output"].strip())
        assert inside < 10 < host

    def test_root_is_read_only(self, env):
        result = env.execute("touch /usr/hermes-probe")
        assert result["returncode"] != 0
        assert "ead-only" in result["output"]
        assert not os.path.exists("/usr/hermes-probe")

    def test_host_root_visible_at_host_paths(self, env):
        result = env.execute("cat /etc/os-release")
        assert result["returncode"] == 0
        assert result["output"].strip() == Path("/etc/os-release").read_text().strip()

    def test_tmp_is_fresh_per_spawn(self, env):
        assert env.execute("touch /tmp/probe-a")["returncode"] == 0
        assert env.execute("test -e /tmp/probe-a")["returncode"] != 0

    def test_cwd_persists_between_commands(self, env):
        assert env.execute("cd /usr/share")["returncode"] == 0
        assert env.execute("pwd")["output"].strip() == "/usr/share"
        assert env.cwd == "/usr/share"

    def test_shell_init_file_variable_survives_into_later_commands(self, sandbox_root, work_dir):
        # The file must be visible inside the sandbox: pytest's tmp_path sits
        # under /tmp, which is a fresh tmpfs per spawn, so use the bound cwd.
        init_file = work_dir / "init.sh"
        init_file.write_text("export HERMES_BWRAP_INIT_MARK=seen-in-sandbox\n")
        with patch("tools.environments.local._read_terminal_shell_init_config", return_value=([str(init_file)], False)):
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert env._snapshot_ready
            assert env.execute("echo $HERMES_BWRAP_INIT_MARK")["output"].strip() == "seen-in-sandbox"
        finally:
            env.cleanup()

    def test_socket_variables_exported_by_a_shell_init_file_stay_unset(self, sandbox_root, work_dir):
        # --unsetenv strips them from what bwrap receives, but the login
        # bootstrap sources the init files into the snapshot afterwards
        # (a 1Password or gpg-agent setup exports SSH_AUTH_SOCK from
        # ~/.bashrc), so the wrapper unsets them again per command.
        init_file = work_dir / "init.sh"
        init_file.write_text(
            "export HERMES_BWRAP_INIT_PLAIN=plain-from-init\n"
            + "".join(f"export {name}=/run/user/1000/{name.lower()}\n" for name in HOST_SOCKET_VARS)
        )
        with patch("tools.environments.local._read_terminal_shell_init_config", return_value=([str(init_file)], False)):
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert env._snapshot_ready
            # Twice: the second command sources the snapshot the first re-dumped.
            for _ in range(2):
                out = env.execute("env")["output"]
                assert "HERMES_BWRAP_INIT_PLAIN=plain-from-init" in out
                for name in HOST_SOCKET_VARS:
                    assert f"{name}=" not in out, name
                assert "/run/user/1000/" not in out
        finally:
            env.cleanup()


def _host_interfaces() -> set[str]:
    lines = Path("/proc/net/dev").read_text().splitlines()[2:]
    return {line.split(":")[0].strip() for line in lines if ":" in line}


# /proc/net/dev reflects the network namespace of the process reading it
# because /proc is mounted fresh per spawn. /sys/class/net does not: the
# bind-mounted host sysfs is tagged with the host's namespace and lists the
# host interfaces in every profile.
INTERFACES_CMD = "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' '"


@needs_bwrap
class TestProfileNetworkAndWritableSet:
    @pytest.fixture
    def make_env(self, sandbox_root, work_dir):
        envs = []

        def factory(profile="network", binds=()):
            env = BubblewrapEnvironment(
                cwd=str(work_dir), timeout=30,
                config=BubblewrapConfig(profile=profile, binds=tuple(binds)),
            )
            envs.append(env)
            return env

        try:
            yield factory
        finally:
            for env in envs:
                env.cleanup()

    def _interfaces(self, env) -> set[str]:
        result = env.execute(INTERFACES_CMD)
        assert result["returncode"] == 0, result["output"]
        return set(result["output"].split())

    def test_network_profile_shares_host_interfaces(self, make_env):
        host = _host_interfaces()
        assert len(host) > 1, "host needs more than lo for this check to mean anything"
        assert self._interfaces(make_env("network")) == host

    @pytest.mark.parametrize("profile", ["workspace", "restricted"])
    def test_isolated_profiles_have_only_loopback_network(self, make_env, profile):
        assert self._interfaces(make_env(profile)) == {"lo"}

    @pytest.mark.parametrize("profile", ["workspace", "network"])
    def test_cwd_writable_under_workspace_and_network(self, make_env, work_dir, profile):
        result = make_env(profile).execute(f"touch {work_dir}/probe-{profile}")
        assert result["returncode"] == 0, result["output"]
        assert (work_dir / f"probe-{profile}").exists()

    def test_cwd_not_writable_under_restricted(self, make_env, work_dir):
        env = make_env("restricted")
        assert env.execute("pwd")["output"].strip() == str(work_dir)
        result = env.execute(f"touch {work_dir}/probe-restricted")
        assert result["returncode"] != 0
        assert "ead-only" in result["output"]
        assert not (work_dir / "probe-restricted").exists()

    def test_rw_bind_entry_is_writable_inside_the_sandbox(self, make_env, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        env = make_env("workspace", binds=[BindMount(src=str(shared), dest=str(shared), readonly=False)])
        result = env.execute(f"touch {shared}/from-sandbox")
        assert result["returncode"] == 0, result["output"]
        assert (shared / "from-sandbox").exists()

    def test_ro_bind_entry_is_not_writable_inside_the_sandbox(self, make_env, tmp_path):
        shared = tmp_path / "shared-ro"
        shared.mkdir()
        (shared / "seed").write_text("host")
        # dest defaults to src: bwrap cannot create a new mount point on the
        # read-only root, so a dest must already exist there (or sit under a
        # writable mount such as /tmp).
        env = make_env("workspace", binds=[BindMount(src=str(shared), dest=str(shared), readonly=True)])
        assert env.execute(f"cat {shared}/seed")["output"].strip() == "host"
        result = env.execute(f"touch {shared}/from-sandbox")
        assert result["returncode"] != 0
        assert "ead-only" in result["output"]


MOUNT_FLAGS = {"--bind": 2, "--ro-bind": 2, "--bind-try": 2, "--ro-bind-try": 2, "--tmpfs": 1, "--dev": 1, "--proc": 1}


def _mounts(argv):
    """The mount directives of a bwrap argv as (flag, *operands) tuples, in order."""
    out, i = [], 0
    while i < len(argv):
        n = MOUNT_FLAGS.get(argv[i])
        if n is None:
            i += 1
            continue
        out.append(tuple(argv[i:i + 1 + n]))
        i += 1 + n
    return out


def _chdir(argv):
    return argv[argv.index("--chdir") + 1]


class TestConstructionTimeMounts:
    """Only --chdir follows the tracked cwd; the mount set is fixed."""

    def test_builder_signature_takes_only_construction_inputs_and_tracked_cwd(self):
        params = list(inspect.signature(build_bwrap_args).parameters)
        # hidden_paths is the sensitive set the environment resolved at
        # construction; it never comes from inside a sandbox.
        assert params == ["config", "initial_cwd", "state_dir", "home", "hermes_home", "tracked_cwd", "bwrap_path", "hidden_paths"]

    def test_chdir_follows_tracked_cwd_with_fixed_mounts(self, sandbox_root, work_dir):
        home = os.path.expanduser("~")
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        first = env._wrap_popen_args(["bash"])
        assert _chdir(first) == str(work_dir)

        # The same path a real `cd $HOME` reports through the cwd marker.
        env._update_cwd({"output": f"{env._cwd_marker}{home}{env._cwd_marker}\n", "returncode": 0})
        assert env.cwd == home

        second = env._wrap_popen_args(["bash"])
        assert _chdir(second) == home
        assert _mounts(second) == _mounts(first)
        assert (str(work_dir), str(work_dir)) in [m[1:] for m in _mounts(second) if m[0] == "--bind-try"]
        assert home not in {p for m in _mounts(second) for p in m[1:]}

    def test_builder_reads_nothing_from_state_dir(self, sandbox_root, work_dir, monkeypatch):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        state_dir = env.get_temp_dir()

        def guard(name, real):
            def wrapped(path, *args, **kwargs):
                if str(path).startswith(state_dir):
                    raise AssertionError(f"{name}() touched the state dir: {path}")
                return real(path, *args, **kwargs)
            return wrapped

        monkeypatch.setattr(builtins, "open", guard("open", builtins.open))
        monkeypatch.setattr(os, "listdir", guard("os.listdir", os.listdir))
        monkeypatch.setattr(os, "scandir", guard("os.scandir", os.scandir))

        for tracked in (str(work_dir), os.path.expanduser("~"), "/usr/share"):
            env.cwd = tracked
            argv = env._wrap_popen_args(["bash"])
            assert _chdir(argv) == tracked
            build_bwrap_args(env._config, str(work_dir), state_dir, env._home, env._hermes_home, tracked)


@needs_bwrap
class TestConstructionTimeMountsIntegration:
    @pytest.fixture
    def env(self, sandbox_root, work_dir):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            yield env
        finally:
            env.cleanup()

    def test_chdir_to_home_keeps_root_read_only(self, env, work_dir):
        home = os.path.expanduser("~")
        before = _mounts(env._wrap_popen_args(["bash"]))
        assert env.execute(f"cd {home}")["returncode"] == 0
        assert env.cwd == home
        assert env.execute("pwd")["output"].strip() == home
        assert _mounts(env._wrap_popen_args(["bash"])) == before

        result = env.execute("touch ./probe-hermes-bwrap")
        assert result["returncode"] != 0
        assert "ead-only" in result["output"]

    def test_state_dir_holds_only_the_state_files(self, env, work_dir):
        for command in ("true", f"cd {os.path.expanduser('~')}", "export HERMES_PROBE=1"):
            assert env.execute(command)["returncode"] == 0
        state_dir = Path(env.get_temp_dir())
        allowed = {Path(env._snapshot_path).name, Path(env._cwd_file).name}
        listing = {p.name for p in state_dir.iterdir()}
        # cwd tracking uses the output marker, so the cwd file name is
        # reserved but never written; the snapshot must be there.
        assert listing <= allowed, listing
        assert Path(env._snapshot_path).name in listing


def _host_pids_with_argv0(marker: str) -> list[int]:
    """PIDs on the host whose argv[0] is *marker* (set with bash's exec -a)."""
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/cmdline", "rb") as fh:
                argv0 = fh.read().split(b"\0", 1)[0]
        except OSError:
            continue
        if argv0 == marker.encode():
            found.append(int(name))
    return found


def _bwrap_children_of_this_process() -> list[int]:
    me = os.getpid()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            with open(f"/proc/{name}/cmdline", "rb") as fh:
                argv0 = fh.read().split(b"\0", 1)[0]
        except (OSError, IndexError):
            continue
        if int(fields[1]) == me and argv0.endswith(b"bwrap"):
            found.append(int(name))
    return found


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class TestKillArgv:
    def test_die_with_parent_present_in_argv(self, sandbox_root, work_dir):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        argv = env._wrap_popen_args(["bash"])
        assert "--die-with-parent" in argv
        assert argv.index("--die-with-parent") < argv.index("--")


@needs_bwrap
class TestKillAndCleanupIntegration:
    """Timeout and cleanup leave no sandboxed process behind."""

    def test_timeout_kills_background_children_too(self, sandbox_root, work_dir):
        marker = f"hermes-timeout-{uuid.uuid4().hex[:8]}"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            # sleep 300 & sleep 300, each carrying the marker as argv[0].
            result = env.execute(f"(exec -a {marker} sleep 300) & (exec -a {marker} sleep 300)", timeout=2)
            assert "timed out" in result["output"], result
            assert _wait_until(lambda: not _host_pids_with_argv0(marker), 3.0), _host_pids_with_argv0(marker)
        finally:
            env.cleanup()

    def test_cleanup_kills_an_in_flight_sandbox(self, sandbox_root, work_dir):
        marker = f"hermes-cleanup-{uuid.uuid4().hex[:8]}"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=60)
        results = []
        worker = threading.Thread(
            target=lambda: results.append(env.execute(f"(exec -a {marker} sleep 300)", timeout=60)),
            daemon=True,
        )
        worker.start()
        try:
            assert _wait_until(lambda: bool(_host_pids_with_argv0(marker)), 10.0), "sandbox never started"
            assert env._live_sandbox_pids()
            env.cleanup()
            worker.join(timeout=10)
            assert not worker.is_alive(), "execute did not return after cleanup"
            assert _wait_until(lambda: not _host_pids_with_argv0(marker), 3.0), _host_pids_with_argv0(marker)
            assert _bwrap_children_of_this_process() == []
            assert env._live_sandbox_pids() == []
            assert not os.path.exists(env.get_temp_dir())
        finally:
            if worker.is_alive():
                env._kill_live_sandboxes()


@needs_bwrap
class TestEnvPassthroughParity:
    """The sandbox gets the env LocalEnvironment would build, minus the
    host agent and bus socket variables, nothing more."""

    @pytest.fixture
    def clean_passthrough(self):
        from agent import secret_scope as ss
        from tools import env_passthrough as ep

        def reset():
            ep.clear_env_passthrough()
            ep._config_passthrough = None
            ss.set_multiplex_active(False)

        reset()
        yield ep
        reset()

    @pytest.fixture
    def pair(self, sandbox_root, work_dir):
        local = LocalEnvironment(cwd=str(work_dir), timeout=30)
        bwrap = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            yield local, bwrap
        finally:
            local.cleanup()
            bwrap.cleanup()

    def test_passthrough_omits_a_provider_key_absent_from_passthrough(self, clean_passthrough, monkeypatch, sandbox_root, work_dir):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-hermes-marker")
        monkeypatch.setenv("HERMES_PLAIN", "plain-marker")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            out = env.execute("env")["output"]
        finally:
            env.cleanup()
        assert "OPENROUTER_API_KEY" not in out
        assert "sk-or-hermes-marker" not in out
        # The plain variable proves the process env does reach the sandbox.
        assert "HERMES_PLAIN=plain-marker" in out

    def test_passthrough_registered_token_is_visible_as_for_local(self, clean_passthrough, monkeypatch, pair):
        clean_passthrough.register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token")
        local, bwrap = pair
        cmd = "printf '%s' \"${SERVICE_TOKEN-unset}\""
        assert bwrap.execute(cmd)["output"] == local.execute(cmd)["output"] == "token"

    def test_socket_variables_are_removed_from_the_sandbox_env(self, clean_passthrough, monkeypatch, sandbox_root, work_dir):
        for name in HOST_SOCKET_VARS:
            monkeypatch.setenv(name, f"/run/user/1000/{name.lower()}")
        monkeypatch.setenv("HERMES_PLAIN", "plain-marker")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            out = env.execute("env")["output"]
        finally:
            env.cleanup()
        for name in HOST_SOCKET_VARS:
            assert f"{name}=" not in out, name
        assert "/run/user/1000/" not in out
        # The plain variable proves the process env does reach the sandbox.
        assert "HERMES_PLAIN=plain-marker" in out

    def test_passthrough_variable_names_match_local(self, clean_passthrough, monkeypatch, pair):
        for name in HOST_SOCKET_VARS:
            monkeypatch.setenv(name, f"/run/user/1000/{name.lower()}")
        local, bwrap = pair
        cmd = "python3 -c 'import os; print(chr(10).join(sorted(os.environ)))'"
        local_names = set(local.execute(cmd)["output"].split())
        bwrap_names = set(bwrap.execute(cmd)["output"].split())
        assert bwrap_names - local_names == set(), "variables injected into the sandbox"
        # Exactly the socket variables are missing, nothing else.
        assert local_names - bwrap_names == set(HOST_SOCKET_VARS), "variables missing from the sandbox"

    @pytest.mark.parametrize("mode", ["auto", "profile"])
    def test_passthrough_home_follows_home_mode_as_for_local(self, clean_passthrough, monkeypatch, sandbox_root, work_dir, mode):
        from hermes_constants import get_hermes_home

        profile_home = get_hermes_home() / "home"
        profile_home.mkdir(exist_ok=True)
        monkeypatch.setenv("TERMINAL_HOME_MODE", mode)
        local = LocalEnvironment(cwd=str(work_dir), timeout=30)
        bwrap = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            cmd = "printf '%s' \"$HOME\""
            home_local = local.execute(cmd)["output"]
            assert bwrap.execute(cmd)["output"] == home_local
            expected = str(profile_home) if mode == "profile" else os.path.expanduser("~")
            assert home_local == expected
            if mode == "profile":
                assert bwrap.execute("touch \"$HOME/probe\"")["returncode"] == 0
                assert (profile_home / "probe").is_file()
        finally:
            local.cleanup()
            bwrap.cleanup()


class TestInitialCwdGuard:
    """The cwd is the writable set: / is refused and HOME gets a warning."""

    @pytest.fixture
    def fake_home(self, tmp_path, monkeypatch):
        # One level down so the parent-of-home cwd does not cover the
        # sandbox dir, which the sandbox dir guard would refuse.
        home = tmp_path / "homes" / "home"
        home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        return home

    def test_root_cwd_is_refused_before_touching_disk(self, sandbox_root, fake_home):
        with _no_session(), pytest.raises(ValueError, match="terminal.cwd must not be /"):
            BubblewrapEnvironment(cwd="/", timeout=10)
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    @pytest.mark.parametrize("which", ["home", "parent-of-home"])
    def test_home_or_its_ancestor_as_cwd_warns(self, sandbox_root, fake_home, caplog, which):
        cwd = fake_home if which == "home" else fake_home.parent
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"), _no_session():
            env = BubblewrapEnvironment(cwd=str(cwd), timeout=10)
        env.cleanup()
        assert any("covers the home directory" in r.getMessage() for r in caplog.records)

    def test_project_dir_as_cwd_is_silent(self, sandbox_root, fake_home, work_dir, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"), _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        env.cleanup()
        assert not any("covers the home directory" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("rel", ["hermes-agent", "logs/deep"])
    def test_cwd_under_hermes_home_is_refused_before_touching_disk(self, sandbox_root, fake_home, monkeypatch, rel):
        # ~/.hermes/hermes-agent is the documented clone location; the
        # HERMES_HOME overlay would mask it in every spawn.
        hermes_home = fake_home / ".hermes"
        cwd = hermes_home / rel
        cwd.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with _no_session(), pytest.raises(ValueError, match="terminal.cwd") as exc:
            BubblewrapEnvironment(cwd=str(cwd), timeout=10)
        assert str(hermes_home) in str(exc.value)
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    @pytest.mark.parametrize("rel", [".ssh", ".config/gcloud/sub"])
    def test_cwd_at_or_under_a_sensitive_entry_is_refused(self, sandbox_root, fake_home, rel):
        cwd = fake_home / rel
        cwd.mkdir(parents=True)
        with _no_session(), pytest.raises(ValueError, match="terminal.cwd"):
            BubblewrapEnvironment(cwd=str(cwd), timeout=10)

    @pytest.mark.parametrize("mode, constructs", [("profile", True), ("auto", False)])
    def test_cwd_under_the_profile_home_follows_home_mode(self, sandbox_root, fake_home, monkeypatch, mode, constructs):
        hermes_home = fake_home / ".hermes"
        cwd = hermes_home / "home" / "proj"
        cwd.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        config = BubblewrapConfig(home_mode=mode)
        if constructs:
            with _no_session():
                env = BubblewrapEnvironment(cwd=str(cwd), timeout=10, config=config)
            env.cleanup()
        else:
            with _no_session(), pytest.raises(ValueError, match="terminal.cwd"):
                BubblewrapEnvironment(cwd=str(cwd), timeout=10, config=config)

    @pytest.mark.parametrize("spelling", ["link", "target"])
    def test_cwd_under_a_profile_home_linked_inside_hermes_home_is_refused(self, sandbox_root, fake_home, monkeypatch, spelling):
        # The profile-home bind lands at the link path, so a cwd under a
        # link target inside HERMES_HOME sits under the tmpfs in every
        # spawn and nothing binds it back.
        hermes_home = fake_home / ".hermes"
        target = hermes_home / "profiles" / "x"
        (target / "proj").mkdir(parents=True)
        (hermes_home / "home").symlink_to(target)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        cwd = (hermes_home / "home" if spelling == "link" else target) / "proj"
        with _no_session(), pytest.raises(ValueError, match="terminal.cwd"):
            BubblewrapEnvironment(cwd=str(cwd), timeout=10, config=BubblewrapConfig(home_mode="profile"))
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    @pytest.mark.parametrize("spelling", ["link", "target"])
    def test_cwd_under_a_profile_home_linked_outside_constructs_under_home_mode_profile(self, sandbox_root, fake_home, tmp_path, monkeypatch, spelling):
        # Its real path is outside every hidden path: no exemption needed.
        hermes_home = fake_home / ".hermes"
        hermes_home.mkdir()
        target = tmp_path / "elsewhere" / "home"
        (target / "proj").mkdir(parents=True)
        (hermes_home / "home").symlink_to(target)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        cwd = (hermes_home / "home" if spelling == "link" else target) / "proj"
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(cwd), timeout=10, config=BubblewrapConfig(home_mode="profile"))
        try:
            assert env._initial_cwd == os.path.realpath(target / "proj")
        finally:
            env.cleanup()

    def test_cwd_under_tmp_constructs(self, sandbox_root, fake_home):
        # /tmp is a fresh tmpfs inside, but the cwd is bound at its own path on top of it.
        cwd = Path(tempfile.mkdtemp(prefix="hermes-", dir="/tmp"))
        try:
            with _no_session():
                env = BubblewrapEnvironment(cwd=str(cwd), timeout=10)
            env.cleanup()
        finally:
            cwd.rmdir()

    def test_failed_bootstrap_leaves_no_state_behind(self, sandbox_root, work_dir):
        with patch.object(LocalEnvironment, "init_session", autospec=True, side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())


@pytest.fixture
def host_dir(tmp_path):
    """A scratch dir outside /tmp (which is a fresh tmpfs inside the sandbox)."""
    if not str(tmp_path.resolve()).startswith("/tmp/"):
        yield tmp_path
        return
    try:
        base = Path(tempfile.mkdtemp(prefix="hermes-bwrap-r1-", dir=Path.home()))
    except OSError:
        pytest.skip("no writable directory outside /tmp")
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


class TestSandboxDirGuard:
    """A sandbox dir a sandbox could write to is refused at construction: the
    empty file bound over hidden files lives beside the state dir, and a
    sandbox that can replace it with a symlink would choose the next
    spawn's bind source."""

    @pytest.mark.parametrize("profile", ["workspace", "network"])
    def test_sandbox_dir_under_a_writable_cwd_is_refused(self, tmp_path, work_dir, monkeypatch, profile):
        root = work_dir / "sandboxes"
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
        with _no_session(), pytest.raises(ValueError, match="terminal.sandbox_dir"):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(profile=profile))
        assert not root.exists() or not any(root.iterdir())

    def test_sandbox_dir_under_a_read_only_cwd_constructs(self, tmp_path, work_dir, monkeypatch):
        root = work_dir / "sandboxes"
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(profile="restricted"))
        try:
            assert Path(env.get_temp_dir()).parent == root
        finally:
            env.cleanup()

    def test_sandbox_dir_under_a_rw_operator_bind_is_refused_and_under_a_ro_bind_constructs(self, tmp_path, work_dir, monkeypatch):
        shared = tmp_path / "shared"
        shared.mkdir()
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(shared / "sandboxes"))
        rw = BubblewrapConfig(binds=(BindMount(src=str(shared), dest="/mnt/shared", readonly=False),))
        with _no_session(), pytest.raises(ValueError, match="terminal.sandbox_dir"):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=rw)
        ro = BubblewrapConfig(binds=(BindMount(src=str(shared), dest="/mnt/shared", readonly=True),))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=ro)
        env.cleanup()

    def test_default_hermes_home_sandboxes_under_cwd_home_constructs(self, tmp_path, monkeypatch):
        # The HERMES_HOME overlay covers the default sandbox dir, so a
        # sandbox never sees the empty file even with HOME writable.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(home), timeout=10)
        try:
            assert Path(env.get_temp_dir()).parent == home / ".hermes" / "sandboxes"
        finally:
            env.cleanup()

    @pytest.mark.parametrize("mode", ["profile", "isolated"])
    def test_sandbox_dir_under_the_profile_home_is_refused_under_home_mode_profile(self, tmp_path, work_dir, monkeypatch, mode):
        # home_mode=profile binds HERMES_HOME/home read-write on top of the
        # HERMES_HOME overlay, which would show the empty file.
        hermes_home = tmp_path / "hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root = profile_home / "sandboxes"
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
        with _no_session(), pytest.raises(ValueError, match="terminal.sandbox_dir"):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode=mode))
        # get_sandbox_dir() creates the empty root; no state dir or empty file follows.
        assert not root.exists() or not any(root.iterdir())

    @pytest.mark.parametrize("mode", ["profile", "auto", "real"])
    def test_default_sandbox_dir_constructs_under_every_home_mode(self, tmp_path, work_dir, monkeypatch, mode):
        hermes_home = tmp_path / "hermes"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode=mode))
        try:
            assert Path(env.get_temp_dir()).parent == hermes_home / "sandboxes"
        finally:
            env.cleanup()

    @pytest.mark.parametrize("spelling", ["link", "target"])
    @pytest.mark.parametrize("link", ["absolute", "relative"])
    def test_sandbox_dir_under_a_symlinked_profile_home_is_refused_under_home_mode_profile(self, tmp_path, work_dir, monkeypatch, link, spelling):
        # HERMES_HOME/home symlinked outside HERMES_HOME: the sandbox root
        # resolves outside every hidden path, but the profile-home bind still
        # makes it writable inside.
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        target = tmp_path / "elsewhere" / "home"
        target.mkdir(parents=True)
        (hermes_home / "home").symlink_to(target if link == "absolute" else os.path.relpath(target, hermes_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root = (hermes_home / "home" if spelling == "link" else target) / "sandboxes"
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
        with _no_session(), pytest.raises(ValueError, match="terminal.sandbox_dir"):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode="profile"))
        assert not root.exists() or not any(root.iterdir())
        # Under auto nothing binds the profile home back, so the same root constructs.
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode="auto"))
        env.cleanup()

    def test_rw_bind_of_home_elsewhere_with_the_default_sandbox_dir_constructs_with_the_bind_dropped(self, tmp_path, monkeypatch):
        # A rw mirror of HOME at /mnt gave a second writable view of
        # HERMES_HOME/sandboxes and the empty file. filter_binds drops the
        # mirror, so the exemption for a sandbox dir under a hidden path holds.
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
        config = BubblewrapConfig(binds=(BindMount(src=str(home), dest="/mnt", readonly=False),))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(home), timeout=10, config=config)
        try:
            assert Path(env.get_temp_dir()).parent == home / ".hermes" / "sandboxes"
            assert "/mnt" not in env._wrap_popen_args(["bash"])
        finally:
            env.cleanup()


class TestProfileHomeGuard:
    """Under a profile home_mode the profile home is bound read-write on top
    of every overlay, so a HERMES_HOME/home that resolves to a tree holding
    a hidden path, or lying under one, would show the secrets again at the
    bind."""

    @pytest.fixture
    def fake_home(self, tmp_path, monkeypatch):
        home = tmp_path / "homes" / "home"
        for rel in (".ssh", ".config"):
            (home / rel).mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        return home

    @pytest.fixture
    def hermes_home(self, tmp_path, monkeypatch):
        hh = tmp_path / "hermes"
        hh.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hh))
        return hh

    @staticmethod
    def _link(hermes_home, target, link="absolute"):
        (hermes_home / "home").symlink_to(target if link == "absolute" else os.path.relpath(target, hermes_home))

    @staticmethod
    def _constructs(work_dir, mode):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode=mode))
        env.cleanup()

    @pytest.mark.parametrize("link", ["absolute", "relative"])
    @pytest.mark.parametrize("rel", ["", ".config", ".ssh"])
    def test_profile_home_linked_into_the_home_tree_is_refused_under_home_mode_profile(self, sandbox_root, work_dir, fake_home, hermes_home, link, rel):
        # HOME and HOME/.config contain hidden paths; HOME/.ssh is one.
        target = fake_home / rel if rel else fake_home
        self._link(hermes_home, target, link)
        with _no_session(), pytest.raises(ValueError, match="terminal.home_mode") as exc:
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode="profile"))
        assert os.path.realpath(target) in str(exc.value)
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    def test_profile_home_linked_to_hermes_home_itself_is_refused(self, sandbox_root, work_dir, fake_home, hermes_home):
        # The bind would show config.yaml and .env at HERMES_HOME/home.
        self._link(hermes_home, hermes_home)
        with _no_session(), pytest.raises(ValueError, match="terminal.home_mode"):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=BubblewrapConfig(home_mode="profile"))
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    def test_plain_profile_home_constructs_under_home_mode_profile(self, sandbox_root, work_dir, fake_home, hermes_home):
        (hermes_home / "home").mkdir()
        self._constructs(work_dir, "profile")

    @pytest.mark.parametrize("where", ["outside", "under-hermes-home"])
    def test_profile_home_linked_to_a_clean_directory_constructs_under_home_mode_profile(self, sandbox_root, work_dir, fake_home, hermes_home, tmp_path, where):
        target = tmp_path / "elsewhere" / "home" if where == "outside" else hermes_home / "profiles" / "x"
        target.mkdir(parents=True)
        self._link(hermes_home, target)
        self._constructs(work_dir, "profile")

    @pytest.mark.parametrize("rel", ["", ".ssh"])
    def test_the_same_links_construct_under_home_mode_auto(self, sandbox_root, work_dir, fake_home, hermes_home, rel):
        # Nothing binds the profile home back under auto.
        self._link(hermes_home, fake_home / rel if rel else fake_home)
        self._constructs(work_dir, "auto")


class TestChdirFailureDetection:
    """The retry fires only for a spawn bwrap aborted before the shell ran."""

    CWD = "/srv/hermes/logs"
    MSG = "bwrap: Can't chdir to /srv/hermes/logs: No such file or directory\n"

    def test_bwrap_chdir_error_alone_is_detected(self):
        assert chdir_failed({"output": self.MSG, "returncode": 1}, self.CWD)

    def test_a_failing_command_that_ran_is_not(self):
        # cwd_observed means the wrapper printed the cwd marker: the shell ran.
        assert not chdir_failed({"output": "boom\n", "returncode": 1, "cwd_observed": True}, self.CWD)
        assert not chdir_failed({"output": self.MSG, "returncode": 1, "cwd_observed": True}, self.CWD)

    def test_a_timeout_is_not(self):
        out = self.MSG + "[Command timed out after 1s]"
        assert not chdir_failed({"output": out, "returncode": -1}, self.CWD)

    def test_a_clean_exit_or_another_cwd_is_not(self):
        assert not chdir_failed({"output": self.MSG, "returncode": 0}, self.CWD)
        assert not chdir_failed({"output": self.MSG, "returncode": 1}, "/srv/hermes")


class TestMaskedInside:
    """masked_inside reads the fixed mount layout, so a masked tracked cwd
    is reset before the spawn and every command runs once."""

    @pytest.fixture
    def layout(self, tmp_path, monkeypatch):
        home = tmp_path / "homes" / "home"
        (home / ".ssh" / "deep").mkdir(parents=True)
        hermes_home = home / ".hermes"
        (hermes_home / "logs").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
        work = tmp_path / "work"
        (work / "sub").mkdir(parents=True)
        src = tmp_path / "scratch"
        (src / "present").mkdir(parents=True)
        dest = tmp_path / "data"
        (dest / "present").mkdir(parents=True)
        (dest / "absent").mkdir()
        (tmp_path / "other").mkdir()
        config = BubblewrapConfig(binds=(BindMount(src=str(src), dest=str(dest), readonly=False),))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work), timeout=10, config=config)
        (Path(env.get_temp_dir()) / "inner").mkdir()
        try:
            yield env, {"home": home, "hermes_home": hermes_home, "work": work, "dest": dest, "tmp": tmp_path}
        finally:
            env.cleanup()

    def test_visible_and_masked_paths(self, layout):
        env, p = layout
        argv = env._wrap_popen_args(["bash"])
        state = Path(env.get_temp_dir())
        visible = [p["work"] / "sub", p["home"] / ".ssh", p["hermes_home"], state / "inner", p["dest"] / "present", Path("/usr/share"), Path("/tmp")]
        masked = [p["home"] / ".ssh" / "deep", p["hermes_home"] / "logs", p["dest"] / "absent", p["tmp"] / "other"]
        assert [str(x) for x in visible if masked_inside(argv, str(x))] == []
        assert [str(x) for x in masked if not masked_inside(argv, str(x))] == []

    def test_runtime_dir_is_masked(self, layout):
        env, _ = layout
        runtime_dir = f"/run/user/{os.getuid()}"
        if not os.path.isdir(runtime_dir):
            pytest.skip("no runtime dir on this host")
        argv = env._wrap_popen_args(["bash"])
        assert not masked_inside(argv, runtime_dir)
        assert masked_inside(argv, runtime_dir + "/keyring")

    def test_execute_resets_a_masked_tracked_cwd_before_the_spawn(self, layout, caplog):
        env, p = layout
        env.cwd = str(p["work"] / "sub")
        assert env._reset_masked_cwd() is None
        assert env.cwd == str(p["work"] / "sub")
        masked = str(p["hermes_home"] / "logs")
        env.cwd = masked
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"):
            note = env._reset_masked_cwd()
        assert env.cwd == str(p["work"])
        assert masked in note and str(p["work"]) in note
        assert _chdir(env._wrap_popen_args(["bash"])) == str(p["work"])
        assert any(masked in r.getMessage() for r in caplog.records)
        # The initial cwd is never reset, even when it is gone from the host.
        env.cwd = str(p["work"])
        assert env._reset_masked_cwd() is None

    def test_a_deleted_tracked_cwd_is_left_to_local_recovery(self, layout):
        # masked_inside reports a host directory that is gone as masked
        # (the root bind shows nothing there); the deleted cwd takes
        # LocalEnvironment's parent recovery instead.
        env, p = layout
        gone = p["work"] / "sub"
        env.cwd = str(gone)
        shutil.rmtree(gone)
        assert env._reset_masked_cwd() is None
        assert env.cwd == str(gone)


@needs_bwrap
class TestMaskedCwdRecovery:
    """A tracked cwd that exists on the host but not inside the sandbox
    (masked by an overlay tmpfs or by the fresh /tmp) is reset to the
    initial cwd before the spawn instead of wedging every later spawn."""

    @staticmethod
    def _check_recovery(env, masked, work_dir):
        result = env.execute(f"mkdir -p {masked} && cd {masked} && pwd")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == str(masked)
        assert env.cwd == str(masked)
        result = env.execute("pwd")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].splitlines()[-1] == str(work_dir)
        assert str(masked) in result["output"] and str(work_dir) in result["output"]
        assert env.cwd == str(work_dir)
        result = env.execute("touch probe && pwd")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == str(work_dir)
        assert (work_dir / "probe").is_file()

    def test_host_dir_under_hermes_home_recovers(self, sandbox_root, host_dir, work_dir, monkeypatch):
        hermes_home = host_dir / "hermes"
        logs = hermes_home / "logs"
        logs.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            self._check_recovery(env, logs, work_dir)
        finally:
            env.cleanup()

    def test_host_dir_under_tmp_recovers(self, sandbox_root, work_dir):
        masked = Path(tempfile.mkdtemp(prefix="hermes-", dir="/tmp"))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            self._check_recovery(env, masked, work_dir)
        finally:
            env.cleanup()
            masked.rmdir()

    def test_a_forged_chdir_line_never_runs_the_command_twice(self, sandbox_root, work_dir):
        # The backstop only resets the tracked cwd; it never re-runs the command.
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            env.execute(f"mkdir -p {work_dir}/sub && cd {work_dir}/sub")
            assert env.cwd == str(work_dir / "sub")
            forged = r"""echo x >> counter; printf 'bwrap: Can'"'"'t chdir to %s: boom\n' "$PWD"; exec false"""
            result = env.execute(forged)
            assert result["returncode"] != 0
            assert (work_dir / "sub" / "counter").read_text() == "x\n"
            assert "run the command again" in result["output"]
            assert env.cwd == str(work_dir)
            assert env.execute("pwd")["output"].strip() == str(work_dir)
        finally:
            env.cleanup()


class TestBindsFrozenAtConstruction:
    """Operator binds are filtered and their dests resolved once, at
    construction."""

    def test_a_dropped_bind_warns_once_per_environment(self, sandbox_root, work_dir, tmp_path, monkeypatch, caplog):
        home = tmp_path / "homes" / "home"
        (home / ".ssh").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        config = BubblewrapConfig(binds=(
            BindMount(src=str(home / ".ssh"), dest="/keys"),
            BindMount(src=str(home), dest="/mnt"),
            BindMount(src=str(work_dir), dest="/proj"),
        ))
        with caplog.at_level(logging.WARNING, logger="tools.environments.bubblewrap"), _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        try:
            dropped = [r.getMessage() for r in caplog.records if "Ignoring terminal.bubblewrap_binds" in r.getMessage()]
            assert len(dropped) == 2
            assert any("/keys" in m or str(home / ".ssh") in m for m in dropped)
            assert any("/mnt" in m for m in dropped)
            caplog.clear()
            for _ in range(5):
                argv = env._wrap_popen_args(["bash"])
            assert [r for r in caplog.records if "bubblewrap_binds" in r.getMessage()] == []
            assert (str(work_dir), "/proj") in [m[1:] for m in _mounts(argv) if m[0] == "--ro-bind"]
            assert [b.dest for b in env._config.binds] == ["/proj"]
        finally:
            env.cleanup()

    def test_a_symlink_planted_under_a_dest_after_construction_does_not_move_the_mount(self, sandbox_root, work_dir, tmp_path):
        src = tmp_path / "scratch"
        src.mkdir()
        (work_dir / "proj").mkdir()
        dest = work_dir / "proj" / "data"
        config = BubblewrapConfig(binds=(BindMount(src=str(src), dest=str(dest), readonly=False),))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        try:
            before = [m for m in _mounts(env._wrap_popen_args(["bash"])) if m[0] == "--bind" and m[1] == str(src)]
            assert before == [("--bind", str(src), str(dest))]
            # What a sandbox with the cwd writable could do between spawns.
            (work_dir / "proj").rename(work_dir / "proj-moved")
            (work_dir / "elsewhere").mkdir()
            (work_dir / "proj").symlink_to(work_dir / "elsewhere")
            after = [m for m in _mounts(env._wrap_popen_args(["bash"])) if m[0] == "--bind" and m[1] == str(src)]
            assert after == before
        finally:
            env.cleanup()


class TestBindSourceGuard:
    """bwrap resolves a bind source on the host at every spawn, so a
    read-write source a sandbox can rename (a plain directory inside the
    writable set) with a dest elsewhere lets a command choose the next
    spawn's mount; at its own path the source is a mount point inside and
    cannot be moved."""

    @staticmethod
    def _config(profile="network", binds=(), **kw):
        return BubblewrapConfig(profile=profile, binds=tuple(binds), **kw)

    @staticmethod
    def _refused(work_dir, config, sandbox_root):
        with _no_session(), pytest.raises(ValueError, match="terminal.bubblewrap_binds") as exc:
            BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        assert "dest equal to src" in str(exc.value)
        assert not sandbox_root.exists() or not any(sandbox_root.iterdir())

    @staticmethod
    def _constructs(work_dir, config):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10, config=config)
        env.cleanup()

    @pytest.mark.parametrize("profile", ["network", "workspace"])
    def test_rw_bind_from_inside_a_writable_cwd_to_another_dest_is_refused(self, sandbox_root, work_dir, profile):
        cache = work_dir / "cache"
        cache.mkdir()
        config = self._config(profile, [BindMount(src=str(cache), dest="/mnt/cache", readonly=False)])
        self._refused(work_dir, config, sandbox_root)

    def test_the_same_bind_constructs_under_restricted(self, sandbox_root, work_dir):
        cache = work_dir / "cache"
        cache.mkdir()
        self._constructs(work_dir, self._config("restricted", [BindMount(src=str(cache), dest="/mnt/cache", readonly=False)]))

    def test_rw_bind_from_inside_another_rw_bind_source_to_another_dest_is_refused(self, sandbox_root, work_dir, tmp_path):
        shared = tmp_path / "shared"
        (shared / "models").mkdir(parents=True)
        config = self._config(binds=[
            BindMount(src=str(shared), dest=str(shared), readonly=False),
            BindMount(src=str(shared / "models"), dest="/mnt", readonly=False),
        ])
        self._refused(work_dir, config, sandbox_root)

    @pytest.mark.parametrize("mode, constructs", [("profile", False), ("auto", True)])
    def test_rw_bind_from_inside_the_profile_home_follows_home_mode(self, sandbox_root, work_dir, tmp_path, monkeypatch, mode, constructs):
        # A source under the plain HERMES_HOME/home is under the hidden
        # HERMES_HOME and filter_binds drops it; the shape that reaches
        # the sandbox is a profile home linked to a clean directory outside.
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        target = tmp_path / "elsewhere" / "home"
        data = target / "data"
        data.mkdir(parents=True)
        (hermes_home / "home").symlink_to(target)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        config = self._config(binds=[BindMount(src=str(data), dest="/mnt/data", readonly=False)], home_mode=mode)
        if constructs:
            self._constructs(work_dir, config)
        else:
            self._refused(work_dir, config, sandbox_root)

    def test_rw_bind_at_its_own_path_and_the_ro_form_construct(self, sandbox_root, work_dir):
        cache = work_dir / "cache"
        cache.mkdir()
        self._constructs(work_dir, self._config(binds=[BindMount(src=str(cache), dest=str(cache), readonly=False)]))
        self._constructs(work_dir, self._config(binds=[BindMount(src=str(cache), dest="/mnt/cache", readonly=True)]))

    def test_rw_bind_of_the_cwd_itself_elsewhere_constructs(self, sandbox_root, work_dir):
        # The cwd is bound at its own path first, so it is a mount point inside.
        self._constructs(work_dir, self._config(binds=[BindMount(src=str(work_dir), dest="/proj", readonly=False)]))

    def test_self_bound_rw_source_two_levels_below_the_cwd_is_refused(self, sandbox_root, work_dir):
        # Only the mount point itself gets EBUSY: a command renames cwd/a,
        # recreates it and plants a relative symlink at cwd/a/cache, and
        # the next spawn mounts the link target read-write.
        cache = work_dir / "a" / "cache"
        cache.mkdir(parents=True)
        self._refused(work_dir, self._config(binds=[BindMount(src=str(cache), dest=str(cache), readonly=False)]), sandbox_root)

    def test_self_bound_rw_source_directly_under_a_self_bound_rw_source_constructs(self, sandbox_root, work_dir):
        # cwd/a is a mount point inside, so cwd/a/cache has an unrenamable parent.
        cache = work_dir / "a" / "cache"
        cache.mkdir(parents=True)
        self._constructs(work_dir, self._config(binds=[
            BindMount(src=str(work_dir / "a"), dest=str(work_dir / "a"), readonly=False),
            BindMount(src=str(cache), dest=str(cache), readonly=False),
        ]))

    @pytest.mark.parametrize("rel", ["link", "link/sub"])
    def test_rw_source_through_a_symlink_inside_the_cwd_is_refused(self, sandbox_root, work_dir, tmp_path, rel):
        # The link sits in the writable cwd, so a command can repoint it
        # whatever its target; self-bound or not.
        target = tmp_path / "data"
        (target / "sub").mkdir(parents=True)
        (work_dir / "link").symlink_to(target)
        src = work_dir / rel
        self._refused(work_dir, self._config(binds=[BindMount(src=str(src), dest=str(src), readonly=False)]), sandbox_root)
        self._refused(work_dir, self._config(binds=[BindMount(src=str(src), dest="/mnt/data", readonly=False)]), sandbox_root)
        self._constructs(work_dir, self._config(binds=[BindMount(src=str(src), dest="/mnt/data", readonly=True)]))

    @needs_bwrap
    def test_a_source_bound_at_its_own_path_cannot_be_renamed_inside(self, sandbox_root, work_dir):
        cache = work_dir / "cache"
        cache.mkdir()
        (cache / "seed").write_text("host")
        env = BubblewrapEnvironment(
            cwd=str(work_dir), timeout=30,
            config=self._config("workspace", [BindMount(src=str(cache), dest=str(cache), readonly=False)]),
        )
        try:
            result = env.execute(f"mv {cache} {cache}.bak")
            assert result["returncode"] != 0
            assert "busy" in result["output"].lower(), result["output"]
            assert cache.is_dir() and not (work_dir / "cache.bak").exists()
            assert env.execute(f"cat {cache}/seed")["output"].strip() == "host"
            assert (str(cache), str(cache)) in [m[1:] for m in _mounts(env._wrap_popen_args(["bash"])) if m[0] == "--bind"]
        finally:
            env.cleanup()


@needs_bwrap
class TestDeletedCwdRecovery:
    def test_deleted_cwd_recovers_to_the_parent_and_rebinds_when_recreated(self, sandbox_root, host_dir):
        work = host_dir / "work"
        work.mkdir()
        env = BubblewrapEnvironment(cwd=str(work), timeout=30)
        try:
            assert env.execute("pwd")["output"].strip() == str(work)
            shutil.rmtree(work)
            # LocalEnvironment recovers the tracked cwd on the first spawn after
            # the deletion, whose wrapper still targets the old dir (same as the
            # local backend); every spawn after that must run, not wedge on the
            # missing bind source.
            env.execute("true")
            result = env.execute("pwd")
            assert result["returncode"] == 0, result["output"]
            recovered = result["output"].strip()
            assert recovered == env.cwd != str(work)
            assert os.path.isdir(recovered)
            # The recovered dir is on the read-only root: visible, not writable.
            assert env.execute("touch ./r1-probe")["returncode"] != 0
            work.mkdir()
            result = env.execute(f"cd {work} && touch r1-probe")
            assert result["returncode"] == 0, result["output"]
            assert (work / "r1-probe").is_file()
        finally:
            env.cleanup()

    @pytest.mark.parametrize("how", ["host", "inside"])
    def test_deleted_non_initial_cwd_takes_local_recovery_not_the_masked_reset(self, sandbox_root, host_dir, caplog, how):
        # The same landing as the local backend, with local's warning and
        # no bubblewrap note.
        work = host_dir / "work"
        sub = work / "sub"
        sub.mkdir(parents=True)
        env = BubblewrapEnvironment(cwd=str(work), timeout=30)
        try:
            assert env.execute(f"cd {sub} && pwd")["output"].strip() == str(sub)
            assert env.cwd == str(sub)
            if how == "host":
                shutil.rmtree(sub)
            else:
                assert env.execute('rm -rf "$PWD"')["returncode"] == 0
            assert env.cwd == str(sub) and not sub.exists()
            with caplog.at_level(logging.WARNING):
                first = env.execute("pwd")
            # The wrapper still cd's to the gone directory on this spawn, as
            # for the local backend; the tracked cwd is recovered from here on.
            assert first["returncode"] == 126, first["output"]
            assert "not visible inside the sandbox" not in first["output"]
            assert "bwrap:" not in first["output"]
            assert env.cwd == str(work)
            messages = [r.getMessage() for r in caplog.records]
            assert any("missing on disk" in m and str(sub) in m for m in messages)
            assert not any("bubblewrap" in m for m in messages)
            result = env.execute("pwd")
            assert result["returncode"] == 0, result["output"]
            assert result["output"].strip() == str(work)
        finally:
            env.cleanup()


@needs_bwrap
class TestSymlinkedCwd:
    def test_cwd_that_is_an_absolute_symlink_runs_commands_at_its_target(self, sandbox_root, tmp_path):
        # bwrap resolves a mount destination inside the sandbox root, where
        # an absolute symlink points nowhere, so the bind goes on the target.
        real = tmp_path / "real-work"
        real.mkdir()
        link = tmp_path / "work-link"
        link.symlink_to(real)
        env = BubblewrapEnvironment(cwd=str(link), timeout=30)
        try:
            result = env.execute("touch probe && pwd")
            assert result["returncode"] == 0, result["output"]
            assert result["output"].strip() == str(real)
            assert (real / "probe").is_file()
        finally:
            env.cleanup()


@needs_bwrap
class TestRuntimeDirAndDockerSocketMasked:
    @pytest.fixture
    def env(self, sandbox_root, work_dir):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            yield env
        finally:
            env.cleanup()

    def test_runtime_dir_is_empty_inside(self, env):
        runtime_dir = f"/run/user/{os.getuid()}"
        if not os.path.isdir(runtime_dir) or not os.listdir(runtime_dir):
            pytest.skip("host has no populated runtime dir")
        result = env.execute(f"ls -A {runtime_dir} | wc -l")
        assert result["returncode"] == 0, result["output"]
        assert result["output"].strip() == "0"

    def test_host_socket_in_runtime_dir_is_not_connectable(self, env):
        import socket

        runtime_dir = f"/run/user/{os.getuid()}"
        if not os.path.isdir(runtime_dir):
            pytest.skip("host has no runtime dir")
        path = f"{runtime_dir}/hermes-r1-{uuid.uuid4().hex[:8]}.sock"
        server = socket.socket(socket.AF_UNIX)
        server.bind(path)
        server.listen(1)
        try:
            probe = f"python3 -c \"import socket; socket.socket(socket.AF_UNIX).connect('{path}')\""
            result = env.execute(probe)
            assert result["returncode"] != 0
            assert "No such file" in result["output"]
        finally:
            server.close()
            os.unlink(path)

    def test_docker_socket_is_a_plain_empty_file_inside(self, env):
        from tools.environments.bubblewrap import DOCKER_SOCKETS

        present = [s for s in DOCKER_SOCKETS if os.path.exists(s)]
        if not present:
            pytest.skip("host has no docker socket")
        for sock in present:
            result = env.execute(f"test -S {sock}")
            assert result["returncode"] != 0, sock
            result = env.execute(f"test -f {sock} && wc -c < {sock}")
            assert result["returncode"] == 0 and result["output"].strip() == "0", (sock, result)
