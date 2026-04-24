"""Tests for NsjailEnvironment CLI-arg construction and preflight.

These tests stub out the real nsjail binary and sandbox process spawn — they
verify the Python-side contract (which flags Hermes asks nsjail for, what the
backend does when nsjail is missing/broken, how config_file mode behaves)
without actually entering a Linux namespace, so they run on any OS.
"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from tools.environments import nsjail as nsjail_mod


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    nsjail_mod._nsjail_executable = None
    monkeypatch.delenv("HERMES_NSJAIL_BINARY", raising=False)
    yield
    nsjail_mod._nsjail_executable = None


@pytest.fixture
def _fake_nsjail_preflight():
    """Pretend nsjail exists on PATH and responds to --help successfully."""
    fake_help = subprocess.CompletedProcess(
        args=["/usr/bin/nsjail", "--help"],
        returncode=0,
        stdout="Usage: nsjail [options] -- cmd\n",
        stderr="",
    )
    with patch("tools.environments.nsjail.shutil.which", return_value="/usr/bin/nsjail"), \
         patch("tools.environments.nsjail._IS_LINUX", True), \
         patch("tools.environments.nsjail.subprocess.run", return_value=fake_help):
        yield


def _make_env(tmp_path, **kwargs):
    """Construct an NsjailEnvironment with init_session() stubbed out.

    init_session() would call _run_bash() which spawns the real nsjail;
    we skip it so the test stays hermetic.
    """
    with patch.object(nsjail_mod.NsjailEnvironment, "init_session", lambda self: None):
        return nsjail_mod.NsjailEnvironment(
            cwd=str(tmp_path), timeout=30, **kwargs
        )


class TestPreflight:
    def test_missing_binary_raises(self):
        with patch("tools.environments.nsjail.shutil.which", return_value=None), \
             patch("tools.environments.nsjail._IS_LINUX", True):
            os.environ.pop("HERMES_NSJAIL_BINARY", None)
            with pytest.raises(RuntimeError, match="nsjail binary not found"):
                nsjail_mod._ensure_nsjail_available()

    def test_non_linux_raises(self):
        with patch("tools.environments.nsjail._IS_LINUX", False):
            with pytest.raises(RuntimeError, match="only supported on Linux"):
                nsjail_mod._ensure_nsjail_available()

    def test_help_timeout_raises(self):
        with patch("tools.environments.nsjail.shutil.which", return_value="/usr/bin/nsjail"), \
             patch("tools.environments.nsjail._IS_LINUX", True), \
             patch(
                 "tools.environments.nsjail.subprocess.run",
                 side_effect=subprocess.TimeoutExpired(cmd="nsjail", timeout=5),
             ):
            with pytest.raises(RuntimeError, match="timed out"):
                nsjail_mod._ensure_nsjail_available()


class TestNsjailArgs:
    def test_defaults_include_rootfs_ro_and_cwd_rw(self, tmp_path, _fake_nsjail_preflight):
        env = _make_env(tmp_path)
        args = env._build_nsjail_args(timeout=30)

        assert args[0] == "/usr/bin/nsjail"
        assert "-Mo" in args
        assert "--quiet" in args
        assert "--bindmount_ro" in args and "/" in args
        # cwd bindmount uses the host path verbatim (rw inside the jail).
        assert f"--bindmount" in args
        assert str(tmp_path) in args
        # time_limit is timeout + cushion (5s).
        tl_idx = args.index("--time_limit")
        assert int(args[tl_idx + 1]) >= 30

    def test_network_off_by_default(self, tmp_path, _fake_nsjail_preflight):
        env = _make_env(tmp_path)
        args = env._build_nsjail_args(timeout=30)
        assert "--disable_clone_newnet" not in args

    def test_allow_net_true_disables_new_netns(self, tmp_path, _fake_nsjail_preflight):
        env = _make_env(tmp_path, allow_net=True)
        args = env._build_nsjail_args(timeout=30)
        assert "--disable_clone_newnet" in args

    def test_memory_limit_passed_as_rlimit_as(self, tmp_path, _fake_nsjail_preflight):
        env = _make_env(tmp_path, memory=777)
        args = env._build_nsjail_args(timeout=30)
        i = args.index("--rlimit_as")
        assert args[i + 1] == "777"

    def test_disk_limit_passed_as_rlimit_fsize(self, tmp_path, _fake_nsjail_preflight):
        env = _make_env(tmp_path, disk=555)
        args = env._build_nsjail_args(timeout=30)
        i = args.index("--rlimit_fsize")
        assert args[i + 1] == "555"

    def test_provider_keys_stripped_from_env(self, tmp_path, _fake_nsjail_preflight):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-should-be-stripped",
            "PATH": "/usr/bin",
        }, clear=False):
            env = _make_env(tmp_path)
            args = env._build_nsjail_args(timeout=30)

        env_flags = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
        assert not any("OPENAI_API_KEY" in v for v in env_flags)
        assert any(v.startswith("PATH=") for v in env_flags)

    def test_forward_env_includes_named_var(self, tmp_path, _fake_nsjail_preflight):
        with patch.dict(os.environ, {"CUSTOM_RESEARCH_TOKEN": "xyz"}, clear=False):
            env = _make_env(tmp_path, forward_env=["CUSTOM_RESEARCH_TOKEN"])
            args = env._build_nsjail_args(timeout=30)
        env_flags = [args[i + 1] for i, a in enumerate(args) if a == "--env"]
        assert any(v == "CUSTOM_RESEARCH_TOKEN=xyz" for v in env_flags)


class TestConfigFileMode:
    def test_config_file_overrides_default_args(self, tmp_path, _fake_nsjail_preflight):
        cfg = tmp_path / "jail.cfg"
        cfg.write_text("# nsjail text-format config\n")

        env = _make_env(tmp_path, config_file=str(cfg))
        args = env._build_nsjail_args(timeout=30)

        # Config mode: only executable, --config <path>, --time_limit.
        assert "--config" in args
        assert str(cfg) in args
        assert "--bindmount_ro" not in args
        assert "--tmpfsmount" not in args

    def test_missing_config_file_raises(self, tmp_path, _fake_nsjail_preflight):
        with pytest.raises(RuntimeError, match="non-existent file"):
            _make_env(tmp_path, config_file=str(tmp_path / "nope.cfg"))


class TestInitialCwdPin:
    def test_cwd_change_does_not_change_rw_bindmount(self, tmp_path, _fake_nsjail_preflight):
        """If the user cd's elsewhere mid-session, the new location must stay
        read-only. Only ``--cwd`` updates; ``--bindmount`` keeps the original."""
        env = _make_env(tmp_path)
        env.cwd = "/etc"
        args = env._build_nsjail_args(timeout=30)

        def _values_after(flag: str) -> list[str]:
            return [args[i + 1] for i, a in enumerate(args) if a == flag]

        rw_bindmounts = [b for b in _values_after("--bindmount") if ":" not in b]
        start_cwds = _values_after("--cwd")

        assert str(tmp_path) in rw_bindmounts  # initial cwd still rw-mounted
        assert "/etc" not in rw_bindmounts      # new cwd NOT bindmounted rw
        assert start_cwds == ["/etc"]           # but process starts in /etc
