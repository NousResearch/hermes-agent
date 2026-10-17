"""Tests for how BubblewrapEnvironment keeps a working cwd: a tracked cwd
masked inside the sandbox, a cwd deleted on the host or from inside, and a
cwd that is a symlink.

Unit tests never spawn bwrap. Integration tests are skipped when bwrap is
missing or its runtime probe fails, so CI without bwrap stays green.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tools.environments import bubblewrap
from tools.environments.bubblewrap import BubblewrapEnvironment


@pytest.fixture(autouse=True)
def _bwrap_probe_passed(monkeypatch):
    """Unit constructions never spawn: count the process-wide bwrap probe as passed."""
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")
    monkeypatch.setattr(bubblewrap, "_process_limit_scoped", True)


def _bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    try:
        probe = subprocess.run(
            ["bwrap", "--unshare-user", "--ro-bind", "/", "/", "true"],
            check=False, capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(not _bwrap_usable(), reason="bwrap missing or its namespace probe failed")


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
