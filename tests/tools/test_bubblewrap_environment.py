"""Tests for BubblewrapEnvironment: the LocalEnvironment subclass that runs
every spawn inside a bwrap sandbox.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments.base import get_sandbox_dir
from tools.environments.bubblewrap import BindMount, BubblewrapConfig, BubblewrapEnvironment
from tools.environments.local import LocalEnvironment


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
