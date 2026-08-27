"""Tests for hiding the sensitive HOME set and HERMES_HOME inside the bwrap
sandbox.

On bwrap 0.9.0, ``--tmpfs`` on a file path fails ("Can't mkdir
...: Not a directory"). A ro-bind of /dev/null mounts, but reading it
fails with EACCES because bwrap remounts binds nodev inside the user
namespace. A ro-bind of a zero-length host file works, so directories get
``--tmpfs`` and files get the empty-file bind.

Unit tests never spawn bwrap. Integration tests are skipped as a module
when bwrap is missing or its runtime probe fails, so CI without bwrap
stays green.
"""

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments.bubblewrap import (
    SENSITIVE_HOME_PATHS,
    BindMount,
    BubblewrapConfig,
    BubblewrapEnvironment,
    build_bwrap_args,
    empty_file_path,
    sensitive_overlay_args,
)
from tools.environments.local import LocalEnvironment

MARKER = "HERMES-SECRET-MARKER"
VISIBLE = "HERMES-VISIBLE-MARKER"
# The entries of the sensitive set that are files; every other entry is a directory.
FILE_ENTRIES = frozenset({".npmrc", ".pypirc", ".netrc", ".env"})
DIR_ENTRIES = tuple(rel for rel in SENSITIVE_HOME_PATHS if rel not in FILE_ENTRIES)


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


def _no_session():
    return patch.object(LocalEnvironment, "init_session", autospec=True, return_value=None)


def populate_home(home: Path) -> None:
    """Create every sensitive entry with marker content, plus two visible controls."""
    for rel in SENSITIVE_HOME_PATHS:
        path = home / rel
        if rel in FILE_ENTRIES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{MARKER} {rel}\n")
        else:
            path.mkdir(parents=True)
            (path / "secret").write_text(f"{MARKER} {rel}\n")
    (home / "visible.txt").write_text(VISIBLE + "\n")
    (home / ".config" / "visible.txt").write_text(VISIBLE + "\n")


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
    """A scratch dir the sandbox sees at its host path.

    /tmp is a fresh tmpfs inside every spawn, so a fake HOME under pytest's
    tmp_path would be hidden by that alone and prove nothing about the
    overlays. Use tmp_path when it lives elsewhere, otherwise a dir under
    the real home.
    """
    if not str(tmp_path.resolve()).startswith("/tmp/"):
        yield tmp_path
        return
    try:
        base = Path(tempfile.mkdtemp(prefix="hermes-bwrap-", dir=Path.home()))
    except OSError:
        pytest.skip("no writable directory outside /tmp for the fake HOME")
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def fake_home(host_dir, monkeypatch):
    home = host_dir / "home"
    home.mkdir()
    populate_home(home)
    monkeypatch.setenv("HOME", str(home))
    return home


class TestOverlayArgs:
    """The builder emits one overlay per sensitive path that exists on the host."""

    @pytest.fixture
    def paths(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        work = tmp_path / "work"
        work.mkdir()
        hermes_home = home / ".hermes"
        return {
            "initial_cwd": str(work),
            "state_dir": str(hermes_home / "sandboxes" / "bwrap-abc123"),
            "home": str(home),
            "hermes_home": str(hermes_home),
            "tracked_cwd": str(work),
        }

    def _overlays(self, paths):
        return _mounts(sensitive_overlay_args(paths["home"], paths["hermes_home"], paths["state_dir"]))

    def test_empty_file_is_a_sibling_of_the_state_dir(self, paths):
        empty = empty_file_path(paths["state_dir"])
        assert empty == paths["state_dir"] + ".empty"
        assert Path(empty).parent == Path(paths["state_dir"]).parent

    def test_nothing_emitted_when_no_sensitive_path_exists(self, paths):
        assert self._overlays(paths) == []

    def test_dirs_get_tmpfs_and_files_get_the_empty_file_bind(self, paths):
        home = Path(paths["home"])
        populate_home(home)
        mounts = self._overlays(paths)
        empty = empty_file_path(paths["state_dir"])
        for rel in DIR_ENTRIES:
            assert ("--tmpfs", str(home / rel)) in mounts, rel
        for rel in FILE_ENTRIES:
            assert ("--ro-bind", empty, str(home / rel)) in mounts, rel
        # HERMES_HOME does not exist here, so the HOME set is the whole set.
        assert len(mounts) == len(SENSITIVE_HOME_PATHS)

    def test_hermes_home_gets_tmpfs_after_the_home_set(self, paths):
        home = Path(paths["home"])
        (home / ".ssh").mkdir()
        Path(paths["hermes_home"]).mkdir(parents=True)
        assert self._overlays(paths) == [
            ("--tmpfs", str(home / ".ssh")),
            ("--tmpfs", paths["hermes_home"]),
        ]

    def test_symlinked_entries_follow_the_target_type(self, paths, tmp_path):
        home = Path(paths["home"])
        real_dir = tmp_path / "elsewhere-ssh"
        real_dir.mkdir()
        real_file = tmp_path / "elsewhere-npmrc"
        real_file.write_text(MARKER)
        (home / ".ssh").symlink_to(real_dir)
        (home / ".npmrc").symlink_to(real_file)
        (home / ".netrc").symlink_to(tmp_path / "dangling")
        assert self._overlays(paths) == [
            ("--tmpfs", str(home / ".ssh")),
            ("--ro-bind", empty_file_path(paths["state_dir"]), str(home / ".npmrc")),
        ]

    def test_overlays_sit_after_operator_binds_and_before_the_state_dir(self, paths, tmp_path):
        populate_home(Path(paths["home"]))
        Path(paths["hermes_home"]).mkdir(parents=True)
        shared = tmp_path / "shared"
        shared.mkdir()
        config = BubblewrapConfig(binds=(BindMount(src=str(shared), dest=str(shared)),))
        mounts = _mounts(build_bwrap_args(config, **paths))
        overlay_idx = [mounts.index(m) for m in self._overlays(paths)]
        assert len(overlay_idx) == len(SENSITIVE_HOME_PATHS) + 1
        i_cwd = mounts.index(("--bind", paths["initial_cwd"], paths["initial_cwd"]))
        i_shared = mounts.index(("--ro-bind", str(shared), str(shared)))
        i_state = mounts.index(("--bind", paths["state_dir"], paths["state_dir"]))
        assert i_cwd < i_shared < min(overlay_idx)
        assert max(overlay_idx) < i_state


class TestEmptyFileLifecycle:
    def test_created_read_only_beside_the_state_dir_and_removed_on_cleanup(self, sandbox_root, work_dir):
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        state_dir = Path(env.get_temp_dir())
        empty = Path(env._empty_file)
        assert empty == Path(empty_file_path(str(state_dir)))
        assert empty.parent == state_dir.parent
        assert not str(empty).startswith(str(state_dir) + os.sep)
        assert empty.is_file()
        assert empty.stat().st_size == 0
        assert stat.S_IMODE(empty.stat().st_mode) == 0o400
        env.cleanup()
        assert not empty.exists()
        assert not state_dir.exists()

    def test_argv_binds_that_file_over_sensitive_files(self, sandbox_root, work_dir, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        (home / ".npmrc").write_text(MARKER)
        (home / ".ssh").mkdir()
        monkeypatch.setenv("HOME", str(home))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        mounts = _mounts(env._wrap_popen_args(["bash"]))
        assert ("--ro-bind", env._empty_file, str(home / ".npmrc")) in mounts
        assert ("--tmpfs", str(home / ".ssh")) in mounts


@needs_bwrap
class TestSensitiveHomePathsIntegration:
    @pytest.fixture
    def env(self, sandbox_root, work_dir, fake_home):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            yield env
        finally:
            env.cleanup()

    def test_every_sensitive_path_shows_no_marker(self, env, fake_home):
        leaks = {}
        for rel in SENSITIVE_HOME_PATHS:
            path = fake_home / rel
            out = env.execute(f"cat {path} 2>/dev/null; ls -A {path} 2>/dev/null")["output"]
            # A leak shows as marker content from cat or the inner file name
            # from ls; ls on a hidden file prints only the file's own path.
            if MARKER in out or "secret" in out.split():
                leaks[rel] = out
        assert leaks == {}

    def test_hidden_dirs_are_empty_and_hidden_files_are_zero_length(self, env, fake_home):
        for rel in DIR_ENTRIES:
            result = env.execute(f"ls -A {fake_home / rel} | wc -l")
            assert result["returncode"] == 0, (rel, result["output"])
            assert result["output"].strip() == "0", rel
        for rel in FILE_ENTRIES:
            result = env.execute(f"test -f {fake_home / rel} && wc -c < {fake_home / rel}")
            assert result["returncode"] == 0, (rel, result["output"])
            assert result["output"].strip() == "0", rel

    def test_non_sensitive_home_content_stays_visible(self, env, fake_home):
        # The hiding must come from the overlays, not from an unrelated mask.
        assert env.execute(f"cat {fake_home}/visible.txt")["output"].strip() == VISIBLE
        assert env.execute(f"cat {fake_home}/.config/visible.txt")["output"].strip() == VISIBLE
        listing = set(env.execute(f"ls -A {fake_home}/.config")["output"].split())
        assert listing == {"gcloud", "visible.txt"}

    def test_writes_into_a_hidden_dir_never_reach_the_host(self, env, fake_home):
        env.execute(f"touch {fake_home}/.ssh/from-sandbox; echo {VISIBLE} > {fake_home}/.npmrc")
        assert [p.name for p in (fake_home / ".ssh").iterdir()] == ["secret"]
        assert (fake_home / ".ssh" / "secret").read_text().startswith(MARKER)
        assert (fake_home / ".npmrc").read_text().startswith(MARKER)

    def test_no_sensitive_paths_present_runs_true(self, sandbox_root, work_dir, host_dir, monkeypatch):
        home = host_dir / "bare-home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert env.execute("true")["returncode"] == 0
            dests = {m[-1] for m in _mounts(env._wrap_popen_args(["bash"]))}
            assert not any(d.startswith(str(home)) for d in dests)
        finally:
            env.cleanup()


@needs_bwrap
class TestHermesHomeIntegration:
    @pytest.fixture
    def hermes_home(self, host_dir, monkeypatch):
        hh = host_dir / "hermes"
        hh.mkdir()
        (hh / "config.yaml").write_text(f"# {MARKER}\n")
        (hh / ".env").write_text(f"HERMES_MARKER={MARKER}\n")
        monkeypatch.setenv("HERMES_HOME", str(hh))
        # No TERMINAL_SANDBOX_DIR: the state dir lands in HERMES_HOME/sandboxes,
        # the one entry allowed to show through the overlay.
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
        return hh

    def test_hidden_and_lists_only_the_state_dir(self, work_dir, hermes_home):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            state_dir = Path(env.get_temp_dir())
            assert state_dir.parent == hermes_home / "sandboxes"
            out = env.execute(f"cat {hermes_home}/config.yaml {hermes_home}/.env; ls -A {hermes_home}")["output"]
            assert MARKER not in out
            assert env.execute(f"ls -A {hermes_home}")["output"].split() == ["sandboxes"]
            assert env.execute(f"ls -A {hermes_home}/sandboxes")["output"].split() == [state_dir.name]
        finally:
            env.cleanup()
        assert (hermes_home / "config.yaml").read_text() == f"# {MARKER}\n"
        assert (hermes_home / ".env").read_text() == f"HERMES_MARKER={MARKER}\n"
