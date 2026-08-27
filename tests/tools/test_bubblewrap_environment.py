"""Tests for BubblewrapEnvironment: the LocalEnvironment subclass that runs
every spawn inside a bwrap sandbox.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import builtins
import inspect
import os
import shutil
import subprocess
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


MOUNT_FLAGS = {"--bind": 2, "--ro-bind": 2, "--tmpfs": 1, "--dev": 1, "--proc": 1}


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
        assert params == ["config", "initial_cwd", "state_dir", "home", "hermes_home", "tracked_cwd", "bwrap_path"]

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
        assert (str(work_dir), str(work_dir)) in [m[1:] for m in _mounts(second) if m[0] == "--bind"]
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
