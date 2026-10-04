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
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments import bubblewrap
from tools.environments.bubblewrap import (
    SENSITIVE_HOME_PATHS,
    BindMount,
    BubblewrapConfig,
    BubblewrapEnvironment,
    build_bwrap_args,
    empty_file_path,
    load_bubblewrap_config,
    sensitive_overlay_args,
    sensitive_paths,
)
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _bwrap_probe_passed(monkeypatch):
    """Unit constructions never spawn: count the process-wide bwrap probe as passed."""
    monkeypatch.setattr(bubblewrap, "_probed_bwrap_path", shutil.which("bwrap") or "/usr/bin/bwrap")
    monkeypatch.setattr(bubblewrap, "_process_limit_scoped", True)

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

MOUNT_FLAGS = {
    "--bind": 2, "--ro-bind": 2, "--bind-try": 2, "--ro-bind-try": 2, "--tmpfs": 1, "--dev": 1, "--proc": 1,
    "--symlink": 2, "--remount-ro": 1,
}
# What a sandbox lists at the top of HERMES_HOME: the state dir parent, the
# scratch dir parent and the top-level staged data roots.
HERMES_VISIBLE = ["attachments", "cache", "composer-pastes", "images", "sandboxes"]
# Sensitive entries that sit below an entry the default allowlist shows.
# The top-level entries are hidden by the HOME layout and get no mount.
BELOW_VISIBLE = (
    ".config/git/credentials", ".cargo/credentials", ".cargo/credentials.toml",
    ".m2/settings.xml", ".gradle/gradle.properties", ".cache/huggingface/token",
)


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


@pytest.fixture
def paths(tmp_path):
    """Builder inputs for the unit tests; home exists, HERMES_HOME does not."""
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


@pytest.fixture
def hermes_home(host_dir, monkeypatch):
    """A fake HERMES_HOME the sandbox can see, holding marker files."""
    hh = host_dir / "hermes"
    hh.mkdir()
    (hh / "config.yaml").write_text(f"# {MARKER}\n")
    (hh / ".env").write_text(f"HERMES_MARKER={MARKER}\n")
    monkeypatch.setenv("HERMES_HOME", str(hh))
    # As in a Hermes process: the temp variables name the scratch dir of the home.
    for name in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(name, str(hh / "cache" / "scratch"))
    # No TERMINAL_SANDBOX_DIR: the state dir lands in HERMES_HOME/sandboxes,
    # the one entry allowed to show through the overlay.
    monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
    return hh


class TestOverlayArgs:
    """The builder emits one overlay per sensitive path that exists on the host."""

    def _overlays(self, paths):
        hidden = sensitive_paths(paths["home"], paths["hermes_home"])
        return _mounts(sensitive_overlay_args(hidden, paths["state_dir"]))

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
        # bwrap resolves a mount destination inside the sandbox root, where an
        # absolute symlink points nowhere, so the overlays go on the targets.
        assert self._overlays(paths) == [
            ("--tmpfs", str(real_dir)),
            ("--ro-bind", empty_file_path(paths["state_dir"]), str(real_file)),
        ]

    def test_overlays_sit_after_operator_binds_and_before_the_state_dir(self, paths, tmp_path):
        home = Path(paths["home"])
        populate_home(home)
        Path(paths["hermes_home"]).mkdir(parents=True)
        shared = tmp_path / "shared"
        shared.mkdir()
        config = BubblewrapConfig(binds=(BindMount(src=str(shared), dest=str(shared)),))
        mounts = _mounts(build_bwrap_args(config, **paths))
        overlay_idx = [mounts.index(("--tmpfs", str(home / rel))) for rel in BELOW_VISIBLE]
        i_cwd = mounts.index(("--bind-try", paths["initial_cwd"], paths["initial_cwd"]))
        i_shared = mounts.index(("--ro-bind", str(shared), str(shared)))
        i_home = mounts.index(("--tmpfs", str(home)))
        i_state = mounts.index(("--bind", paths["state_dir"], paths["state_dir"]))
        i_seal = mounts.index(("--remount-ro", str(home)))
        assert i_cwd < i_shared < i_home < min(overlay_idx)
        assert max(overlay_idx) < i_state < i_seal
        # A top-level entry is not visible, so nothing is mounted on it.
        top_level = {str(home / rel) for rel in SENSITIVE_HOME_PATHS if "/" not in rel}
        assert top_level.isdisjoint(m[-1] for m in mounts)



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
        (home / ".cargo").mkdir(parents=True)
        (home / ".cargo" / "credentials.toml").write_text(MARKER)
        (home / ".npmrc").write_text(MARKER)
        (home / ".ssh").mkdir()
        monkeypatch.setenv("HOME", str(home))
        with _no_session():
            env = BubblewrapEnvironment(cwd=str(work_dir), timeout=10)
        mounts = _mounts(env._wrap_popen_args(["bash"]))
        assert ("--ro-bind", env._empty_file, str(home / ".cargo" / "credentials.toml")) in mounts
        # The top-level entries are off the allowlist: nothing is mounted there.
        dests = {m[-1] for m in mounts}
        assert str(home / ".npmrc") not in dests
        assert str(home / ".ssh") not in dests



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

    def test_hidden_entries_are_absent_or_empty(self, env, fake_home):
        # A top-level entry does not exist in the sandbox at all; an entry
        # below a visible one shows as an empty directory or a zero-length file.
        for rel in SENSITIVE_HOME_PATHS:
            path = fake_home / rel
            result = env.execute(
                f"if [ -d {path} ]; then ls -A {path} | wc -l; elif [ -e {path} ]; then wc -c < {path}; else echo absent; fi"
            )
            assert result["returncode"] == 0, (rel, result["output"])
            assert result["output"].strip() == ("absent" if rel not in BELOW_VISIBLE else "0"), rel


    def test_non_sensitive_home_content_stays_visible(self, env, fake_home):
        # The hiding must come from the layout, not from an unrelated mask.
        assert env.execute(f"cat {fake_home}/visible.txt")["output"].strip() == VISIBLE
        # .config is default-deny: only an allowed child shows, and a file
        # no list names does not.
        assert VISIBLE not in env.execute(f"cat {fake_home}/.config/visible.txt 2>&1")["output"]
        listing = set(env.execute(f"ls -A {fake_home}/.config")["output"].split())
        assert listing == {"git"}


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
            mounts = _mounts(env._wrap_popen_args(["bash"]))
            # An empty HOME gets the layout and nothing inside it.
            assert [m for m in mounts if m[-1].startswith(str(home))] == [
                ("--tmpfs", str(home)), ("--remount-ro", str(home)),
            ]
        finally:
            env.cleanup()



@needs_bwrap
class TestAncestorPinIntegration:
    """With cwd=HOME the sandbox may not rename the parent of a hidden path
    out from under its overlay."""

    @pytest.fixture
    def env(self, sandbox_root, fake_home):
        env = BubblewrapEnvironment(cwd=str(fake_home), timeout=30)
        try:
            yield env
        finally:
            env.cleanup()

    def test_renaming_the_parent_of_a_hidden_dir_fails(self, env, fake_home):
        result = env.execute(f"mv {fake_home}/.config {fake_home}/.config2")
        assert result["returncode"] != 0, result["output"]
        assert not (fake_home / ".config2").exists()
        assert (fake_home / ".config" / "gcloud" / "secret").read_text().startswith(MARKER)

    def test_secret_stays_hidden_on_the_next_spawn(self, env, fake_home):
        env.execute(f"mv {fake_home}/.config {fake_home}/.config2; rmdir {fake_home}/.config")
        out = env.execute(
            f"cat {fake_home}/.config2/gcloud/secret {fake_home}/.config/gcloud/secret 2>/dev/null; "
            f"ls -A {fake_home}/.config/gcloud {fake_home}/.config2/gcloud 2>/dev/null"
        )["output"]
        assert MARKER not in out
        assert "secret" not in out.split()

    def test_default_deny_dir_is_read_only(self, env, fake_home):
        result = env.execute(f"touch {fake_home}/.config/probe")
        assert result["returncode"] != 0
        assert "Read-only file system" in result["output"]
        assert not (fake_home / ".config" / "probe").exists()



@needs_bwrap
class TestAncestorBindIntegration:
    """An operator bind of HOME at another destination would show the secrets
    at that destination; it is dropped."""

    def test_ro_bind_of_home_elsewhere_is_dropped_and_shows_nothing(self, sandbox_root, work_dir, fake_home):
        config = BubblewrapConfig(binds=(BindMount(src=str(fake_home), dest="/mnt", readonly=True),))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=config)
        try:
            assert not any(m[-1] == "/mnt" for m in _mounts(env._wrap_popen_args(["bash"])))
            out = env.execute(
                "test -e /mnt/.ssh && echo MNT-SSH-EXISTS; cat /mnt/.ssh/secret 2>/dev/null; "
                f"ls -A /mnt/.ssh {fake_home}/.ssh 2>/dev/null"
            )["output"]
            assert "MNT-SSH-EXISTS" not in out
            assert MARKER not in out
            assert "secret" not in out.split()
        finally:
            env.cleanup()


@needs_bwrap
class TestSymlinkedEntryIntegration:
    """A sensitive entry that is a symlink (a dotfiles repository) is hidden
    at its target, and with cwd=HOME the parent of the target cannot be
    renamed out from under the overlay."""

    @pytest.fixture
    def linked_home(self, sandbox_root, host_dir, monkeypatch):
        home = host_dir / "home"
        (home / "dotfiles" / "ssh").mkdir(parents=True)
        (home / "dotfiles" / "ssh" / "key").write_text(MARKER + "\n")
        monkeypatch.setenv("HOME", str(home))
        return home

    @staticmethod
    def _readable(env, home):
        paths = " ".join(f"{home}/{rel}/key" for rel in (".ssh", "dotfiles/ssh", "dotfiles2/ssh"))
        listing = " ".join(f"{home}/{rel}" for rel in (".ssh", "dotfiles/ssh"))
        out = env.execute(f"cat {paths} 2>/dev/null; ls -A {listing} 2>/dev/null")["output"]
        return MARKER in out or "key" in out.split()

    def test_relative_symlink_target_hidden_and_its_parent_pinned(self, linked_home):
        (linked_home / ".ssh").symlink_to("dotfiles/ssh")
        env = BubblewrapEnvironment(cwd=str(linked_home), timeout=30)
        try:
            assert not self._readable(env, linked_home)
            result = env.execute(f"mv {linked_home}/dotfiles {linked_home}/dotfiles2")
            assert result["returncode"] != 0, result["output"]
            assert not self._readable(env, linked_home)
        finally:
            env.cleanup()
        assert (linked_home / "dotfiles" / "ssh" / "key").read_text().startswith(MARKER)

    def test_absolute_symlink_runs_commands_and_hides_the_target(self, linked_home):
        (linked_home / ".ssh").symlink_to(linked_home / "dotfiles" / "ssh")
        env = BubblewrapEnvironment(cwd=str(linked_home), timeout=30)
        try:
            result = env.execute("echo ok")
            assert result["returncode"] == 0, result["output"]
            assert result["output"].strip() == "ok"
            assert not self._readable(env, linked_home)
        finally:
            env.cleanup()


@needs_bwrap
class TestSymlinkedBindDestIntegration:
    """A read-write operator bind whose dest is a symlink into the home tree
    gets the same pins as the real path, so the parent of a hidden path
    cannot be renamed through it."""

    @pytest.mark.parametrize("cwd", ["home", "work"])
    @pytest.mark.parametrize("target", ["relative", "absolute"])
    def test_parent_of_a_hidden_dir_cannot_be_renamed_through_the_link(self, sandbox_root, host_dir, work_dir, monkeypatch, target, cwd):
        real = host_dir / "tree" / "home"
        real.mkdir(parents=True)
        populate_home(real)
        link = host_dir / "home-link"
        link.symlink_to(os.path.relpath(real, host_dir) if target == "relative" else real)
        monkeypatch.setenv("HOME", str(real))
        config = BubblewrapConfig(binds=(BindMount(src=str(link), dest=str(link), readonly=False),))
        env = BubblewrapEnvironment(cwd=str(real if cwd == "home" else work_dir), timeout=30, config=config)
        try:
            assert env.execute("echo ok")["output"].strip() == "ok"
            result = env.execute(f"mv {link}/.config {link}/.config2")
            assert result["returncode"] != 0, result["output"]
            out = env.execute(
                f"cat {real}/.config2/gcloud/secret {link}/.config2/gcloud/secret 2>/dev/null; "
                f"ls -A {real}/.config/gcloud {link}/.config/gcloud 2>/dev/null"
            )["output"]
            assert MARKER not in out
            assert "secret" not in out.split()
        finally:
            env.cleanup()
        assert (real / ".config" / "gcloud" / "secret").read_text().startswith(MARKER)


@needs_bwrap
class TestHermesHomeIntegration:
    def test_hidden_and_lists_only_the_state_dir(self, work_dir, hermes_home):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            state_dir = Path(env.get_temp_dir())
            assert state_dir.parent == hermes_home / "sandboxes"
            out = env.execute(f"cat {hermes_home}/config.yaml {hermes_home}/.env; ls -A {hermes_home}")["output"]
            assert MARKER not in out
            # Besides the state dir only the scratch dir and the staged data
            # roots show through the overlay.
            assert env.execute(f"ls -A {hermes_home}")["output"].split() == HERMES_VISIBLE
            assert "scratch" in env.execute(f"ls -A {hermes_home}/cache")["output"].split()
            assert "bws_cache.json" not in env.execute(f"ls -A {hermes_home}/cache")["output"].split()
            assert env.execute(f"ls -A {hermes_home}/sandboxes")["output"].split() == [state_dir.name]
        finally:
            env.cleanup()
        assert (hermes_home / "config.yaml").read_text() == f"# {MARKER}\n"
        assert (hermes_home / ".env").read_text() == f"HERMES_MARKER={MARKER}\n"

    def test_default_home_is_hidden_when_hermes_home_is_relocated(self, work_dir, hermes_home, fake_home):
        # With HERMES_HOME elsewhere, HOME/.hermes (the default home, with
        # its .env and auth.json) is hidden as well; a live test read both
        # files from inside before this.
        default_home = fake_home / ".hermes"
        default_home.mkdir(exist_ok=True)
        (default_home / ".env").write_text(f"DEFAULT_MARKER={MARKER}\n")
        (default_home / "auth.json").write_text(f'{{"token": "{MARKER}"}}\n')
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            out = env.execute(f"cat {default_home}/.env {default_home}/auth.json 2>&1; ls -A {default_home}")["output"]
            assert MARKER not in out
            # Off the allowlist, so it does not exist in the sandbox at all.
            assert env.execute(f"ls -A {default_home} 2>/dev/null")["output"].split() == []
            assert env.execute(f"ls -A {hermes_home}")["output"].split() == HERMES_VISIBLE
        finally:
            env.cleanup()
        assert (default_home / ".env").read_text() == f"DEFAULT_MARKER={MARKER}\n"
        assert (default_home / "auth.json").read_text() == f'{{"token": "{MARKER}"}}\n'


class TestHomeModeCarveOut:
    """HERMES_HOME/home is bound back over the overlay only under home_mode=profile."""

    def test_home_mode_defaults_to_auto_and_reads_terminal_home_mode(self):
        assert BubblewrapConfig().home_mode == "auto"
        assert load_bubblewrap_config({}).home_mode == "auto"
        assert load_bubblewrap_config({"TERMINAL_HOME_MODE": " Profile "}).home_mode == "profile"

    @pytest.mark.parametrize("mode", ["profile", "isolated", "profile_home", "profile-home"])
    def test_home_mode_profile_binds_profile_home_between_overlay_and_state_dir(self, paths, mode):
        hermes_home = Path(paths["hermes_home"])
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        (Path(paths["home"]) / ".ssh").mkdir()
        mounts = _mounts(build_bwrap_args(BubblewrapConfig(home_mode=mode), **paths))
        # HERMES_HOME is HOME/.hermes here: the HOME layout hides it, and
        # the profile home and the state dir are bound inside the layout,
        # before HOME is sealed.
        i_layout = mounts.index(("--tmpfs", paths["home"]))
        i_profile = mounts.index(("--bind", str(profile_home), str(profile_home)))
        i_state = mounts.index(("--bind", paths["state_dir"], paths["state_dir"]))
        i_seal = mounts.index(("--remount-ro", paths["home"]))
        assert i_layout < i_profile < i_state < i_seal
        # The real HOME set stays hidden regardless of the subprocess HOME.
        assert str(Path(paths["home"]) / ".ssh") not in {m[-1] for m in mounts}

    @pytest.mark.parametrize("mode", ["auto", "real"])
    def test_home_mode_auto_and_real_add_no_profile_home_bind(self, paths, mode):
        hermes_home = Path(paths["hermes_home"])
        (hermes_home / "home").mkdir(parents=True)
        argv = build_bwrap_args(BubblewrapConfig(home_mode=mode), **paths)
        assert str(hermes_home / "home") not in argv
        assert ("--tmpfs", paths["home"]) in _mounts(argv)

    def test_home_mode_profile_without_the_dir_adds_no_bind(self, paths):
        hermes_home = Path(paths["hermes_home"])
        hermes_home.mkdir(parents=True)
        argv = build_bwrap_args(BubblewrapConfig(home_mode="profile"), **paths)
        assert str(hermes_home / "home") not in argv
        assert ("--tmpfs", paths["home"]) in _mounts(argv)


@needs_bwrap
class TestHomeModeIntegration:
    @pytest.fixture
    def profile_home(self, hermes_home):
        ph = hermes_home / "home"
        ph.mkdir()
        return ph

    def test_home_mode_profile_home_writable_and_rest_of_hermes_home_hidden(self, work_dir, hermes_home, profile_home, monkeypatch):
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            # LocalEnvironment's run env already points HOME at the profile home.
            assert env.execute("echo $HOME")["output"].strip() == str(profile_home)
            result = env.execute("touch $HOME/probe")
            assert result["returncode"] == 0, result["output"]
            assert (profile_home / "probe").is_file()
            assert set(env.execute(f"ls -A {hermes_home}")["output"].split()) == {"home", *HERMES_VISIBLE}
            out = env.execute(f"cat {hermes_home}/config.yaml {hermes_home}/.env 2>/dev/null")["output"]
            assert MARKER not in out
        finally:
            env.cleanup()

    @pytest.mark.parametrize("mode", ["auto", "real"])
    def test_home_mode_auto_and_real_keep_profile_home_hidden(self, work_dir, hermes_home, profile_home, monkeypatch, mode):
        monkeypatch.setenv("TERMINAL_HOME_MODE", mode)
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert str(profile_home) not in env._wrap_popen_args(["bash"])
            assert env.execute(f"ls -A {hermes_home}")["output"].split() == HERMES_VISIBLE
            # The dir is absent inside the HERMES_HOME tmpfs; a write there stays in the tmpfs.
            assert env.execute(f"test -e {profile_home}")["returncode"] != 0
            assert env.execute(f"mkdir -p {profile_home} && touch {profile_home}/probe")["returncode"] == 0
            assert not (profile_home / "probe").exists()
        finally:
            env.cleanup()

    def test_home_mode_profile_with_the_profile_home_linked_outside_keeps_the_hidden_set_hidden(self, work_dir, hermes_home, fake_home, host_dir, monkeypatch):
        # The same layout with the link pointing at a clean directory: the
        # bind follows the link, and the hidden set stays hidden.
        target = host_dir / "elsewhere" / "home"
        target.mkdir(parents=True)
        (hermes_home / "home").symlink_to(target)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            result = env.execute("touch $HOME/probe")
            assert result["returncode"] == 0, result["output"]
            assert (target / "probe").is_file()
            leaks = {}
            for rel in SENSITIVE_HOME_PATHS:
                path = fake_home / rel
                out = env.execute(f"cat {path} 2>/dev/null; ls -A {path} 2>/dev/null")["output"]
                if MARKER in out or "secret" in out.split():
                    leaks[rel] = out
            assert leaks == {}
            out = env.execute(f"cat {hermes_home}/config.yaml {hermes_home}/.env 2>/dev/null")["output"]
            assert MARKER not in out
        finally:
            env.cleanup()

    @pytest.fixture
    def profile_layout(self, fake_home, monkeypatch):
        """The standard profile layout: HERMES_HOME at HOME/.hermes/profiles/<name>.

        No TERMINAL_SANDBOX_DIR: the state dir lands in HERMES_HOME/sandboxes,
        two overlays deep (HOME/.hermes, then HERMES_HOME inside it).
        """
        default_home = fake_home / ".hermes"
        default_home.mkdir(exist_ok=True)
        (default_home / ".env").write_text(f"DEFAULT_MARKER={MARKER}\n")
        (default_home / "auth.json").write_text(f'{{"token": "{MARKER}"}}\n')
        hh = default_home / "profiles" / "coder"
        (hh / "home").mkdir(parents=True)
        (hh / "config.yaml").write_text(f"# {MARKER}\n")
        (hh / ".env").write_text(f"PROFILE_MARKER={MARKER}\n")
        monkeypatch.setenv("HERMES_HOME", str(hh))
        monkeypatch.delenv("TERMINAL_SANDBOX_DIR", raising=False)
        return hh

    def test_profile_layout_under_the_default_home_keeps_the_profile_reachable(self, work_dir, fake_home, profile_layout, monkeypatch):
        # The profile home and the state dir are bound back through
        # both overlays; the default home and the rest of the profile stay hidden.
        hermes_home = profile_layout
        profile_home = hermes_home / "home"
        default_home = fake_home / ".hermes"
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            state_dir = Path(env.get_temp_dir())
            assert state_dir.parent == hermes_home / "sandboxes"
            assert env.execute("echo $HOME")["output"].strip() == str(profile_home)
            result = env.execute("touch $HOME/probe")
            assert result["returncode"] == 0, result["output"]
            assert (profile_home / "probe").is_file()
            (work_dir / "sub").mkdir()
            assert env.execute("cd sub")["returncode"] == 0
            assert env.execute("pwd")["output"].strip() == str(work_dir / "sub")
            assert env.execute(f"ls -A {default_home}")["output"].split() == ["profiles"]
            assert set(env.execute(f"ls -A {hermes_home}")["output"].split()) == {"home", *HERMES_VISIBLE}
            out = env.execute(
                f"cat {default_home}/.env {default_home}/auth.json {hermes_home}/config.yaml {hermes_home}/.env 2>/dev/null"
            )["output"]
            assert MARKER not in out
        finally:
            env.cleanup()
        assert (default_home / ".env").read_text() == f"DEFAULT_MARKER={MARKER}\n"



def _write(path: Path, text: str = MARKER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return path


UNLISTED_TOP = (".git-credentials", ".pgpass", ".boto", ".ssh/id_ed25519", ".zz-unlisted/secret", ".zz-unlisted-file")
UNLISTED_NESTED = (".config/zz-unlisted/token", ".config/zz-file", ".local/share/zz-unlisted/db", ".local/state/history")
BELOW_ALLOWED = (".config/gh/hosts.yml", ".cargo/credentials.toml")
ALLOWED_SAMPLES = (".gitconfig", ".profile", ".cargo/bin/x", ".cache/x", ".config/git/config", ".local/bin/x")


@pytest.fixture
def deny_home(host_dir, monkeypatch):
    """A HOME with unlisted secrets, allowed entries, a linked rc file and two tools on PATH."""
    home = host_dir / "home"
    home.mkdir()
    for rel in UNLISTED_TOP + UNLISTED_NESTED + BELOW_ALLOWED:
        _write(home / rel)
    for rel in ALLOWED_SAMPLES:
        _write(home / rel, VISIBLE)
    # The target is a dot entry of HOME: read-only inside the sandbox even
    # with the cwd at HOME, so the link does not stop construction.
    _write(home / ".cargo" / "bin" / "bashrc", f"# {VISIBLE}")
    (home / ".bashrc").symlink_to(".cargo/bin/bashrc")
    (home / ".config" / "pip").symlink_to("git")
    _write(home / "proj" / "inside.txt", VISIBLE)
    _write(home / "sibling" / "data.txt", VISIBLE)
    _write(home / ".local" / "share" / "other" / "marker")
    for rel in (".zz-tool/bin/zz-top", ".local/share/zz/bin/zz-nested"):
        tool = _write(home / rel, f"#!/bin/sh\necho {VISIBLE}")
        tool.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(home / ".zz-tool" / "bin"), str(home / ".local/share/zz/bin"), os.environ["PATH"]]),
    )
    return home


def _host_names(home: Path) -> set[str]:
    return {p.name for p in home.iterdir()}


@needs_bwrap
class TestHomeDefaultDenyIntegration:
    """HOME is default-deny for dot entries: what no list names is hidden too."""

    @staticmethod
    def _env(cwd, **kwargs):
        return BubblewrapEnvironment(cwd=str(cwd), timeout=30, **kwargs)

    @pytest.fixture
    def env(self, sandbox_root, deny_home):
        env = self._env(deny_home / "proj")
        try:
            yield env
        finally:
            env.cleanup()

    @pytest.fixture
    def env_at_home(self, sandbox_root, deny_home):
        env = self._env(deny_home)
        try:
            yield env
        finally:
            env.cleanup()

    @pytest.mark.parametrize("profile", sorted(bubblewrap.PROFILES))
    @pytest.mark.parametrize("at_home", [False, True])
    def test_unlisted_entries_cannot_be_read_or_listed(self, sandbox_root, deny_home, profile, at_home):
        env = self._env(deny_home if at_home else deny_home / "proj", config=BubblewrapConfig(profile=profile))
        try:
            for rel in UNLISTED_TOP + UNLISTED_NESTED + BELOW_ALLOWED:
                out = env.execute(f"cat {deny_home / rel} 2>&1")["output"]
                assert MARKER not in out, (profile, rel)
            listed = set()
            for rel in ("", ".config", ".local", ".local/share"):
                listed |= set(env.execute(f"ls -A {deny_home / rel}")["output"].split())
            assert listed.isdisjoint(
                {".git-credentials", ".pgpass", ".boto", ".ssh", ".zz-unlisted", ".zz-unlisted-file",
                 "zz-unlisted", "zz-file", "state", "gh", "other"}
            ), (profile, listed)
        finally:
            env.cleanup()

    def test_absent_names_cannot_be_created_with_cwd_at_home(self, env_at_home, deny_home):
        home = deny_home
        (home / ".ssh").joinpath("id_ed25519").unlink()
        (home / ".ssh").rmdir()
        before = _host_names(home)
        for command in (
            f"mkdir -p {home}/.ssh && printf injected > {home}/.ssh/authorized_keys",
            f"printf injected > {home}/.netrc",
            f"ln -s /etc {home}/.aws",
            f"printf injected > {home}/newfile",
        ):
            result = env_at_home.execute(command)
            assert result["returncode"] != 0, (command, result["output"])
        assert _host_names(home) == before
        assert not (home / ".ssh").exists()

    def test_sibling_is_read_only_from_a_project_cwd(self, env, deny_home):
        assert env.execute(f"cat {deny_home}/sibling/data.txt")["output"].strip() == VISIBLE
        result = env.execute(f"printf changed > {deny_home}/sibling/data.txt")
        assert result["returncode"] != 0
        assert (deny_home / "sibling" / "data.txt").read_text().strip() == VISIBLE
        assert env.execute("printf ok > made-here.txt")["returncode"] == 0
        assert (deny_home / "proj" / "made-here.txt").read_text() == "ok"

    def test_write_inside_an_existing_directory_reaches_the_host_with_cwd_at_home(self, env_at_home, deny_home):
        result = env_at_home.execute(f"printf ok > {deny_home}/proj/from-sandbox.txt")
        assert result["returncode"] == 0, result["output"]
        assert (deny_home / "proj" / "from-sandbox.txt").read_text() == "ok"

    def test_directory_made_on_the_host_later_is_visible_to_the_next_command(self, env, deny_home):
        assert env.execute("true")["returncode"] == 0
        _write(deny_home / "later" / "file.txt", VISIBLE)
        assert env.execute(f"cat {deny_home}/later/file.txt")["output"].strip() == VISIBLE

    def test_path_change_after_construction_does_not_change_the_mounts(self, env, deny_home, monkeypatch):
        before = _mounts(env._wrap_popen_args(["bash"]))
        _write(deny_home / ".zz-late" / "bin" / "x")
        monkeypatch.setenv("PATH", str(deny_home / ".zz-late" / "bin") + os.pathsep + os.environ["PATH"])
        after = _mounts(env._wrap_popen_args(["bash"]))
        assert after == before
        assert str(deny_home / ".zz-late") not in {m[-1] for m in after}

    def test_linked_rc_file_is_read_through_its_link(self, env, deny_home):
        assert VISIBLE in env.execute(f"cat {deny_home}/.bashrc")["output"]
        assert env.execute(f"test -L {deny_home}/.bashrc")["returncode"] == 0

    def test_allowed_entries_are_visible(self, env, deny_home):
        for rel in ALLOWED_SAMPLES + (".bashrc",):
            assert VISIBLE in env.execute(f"cat {deny_home / rel}")["output"], rel

    def test_tools_on_path_run_by_name_and_their_neighbours_stay_hidden(self, env, deny_home):
        for tool in ("zz-top", "zz-nested"):
            result = env.execute(f"PATH={deny_home}/.zz-tool/bin:{deny_home}/.local/share/zz/bin:$PATH {tool}")
            assert result["output"].strip() == VISIBLE, (tool, result["output"])
        assert MARKER not in env.execute(f"cat {deny_home}/.local/share/other/marker 2>&1")["output"]

    def test_nothing_can_be_created_in_a_default_deny_directory(self, env_at_home, deny_home):
        for command in (f"mkdir {deny_home}/.config/gh2", f"mkdir -p {deny_home}/.local/share/keyrings && "
                        f"printf x > {deny_home}/.local/share/keyrings/x", f"mkdir {deny_home}/.local/zz-new"):
            result = env_at_home.execute(command)
            assert result["returncode"] != 0, (command, result["output"])
        assert not (deny_home / ".config" / "gh2").exists()
        assert not (deny_home / ".local" / "share" / "keyrings").exists()
        assert not (deny_home / ".local" / "zz-new").exists()

    def test_allowed_child_that_is_a_symlink_stays_a_symlink(self, env, deny_home):
        assert env.execute(f"readlink {deny_home}/.config/pip")["output"].strip() == "git"

    def test_allowed_entries_are_read_only_with_cwd_at_home(self, env_at_home, deny_home):
        for command in (
            f"echo injected >> {deny_home}/.profile",
            f"echo injected >> {deny_home}/.gitconfig",
            f"printf x > {deny_home}/.cache/y",
        ):
            result = env_at_home.execute(command)
            assert result["returncode"] != 0, command
            assert "Read-only file system" in result["output"], (command, result["output"])
        assert (deny_home / ".profile").read_text() == VISIBLE + "\n"
        assert (deny_home / ".gitconfig").read_text() == VISIBLE + "\n"
        assert not (deny_home / ".cache" / "y").exists()

    def test_linked_dot_entry_and_its_target_cannot_be_changed(self, env_at_home, deny_home):
        for command in (
            f"rm {deny_home}/.bashrc",
            f"ln -sf /dev/null {deny_home}/.bashrc",
            f"echo changed >> {deny_home}/.bashrc",
            f"echo changed >> {deny_home}/.cargo/bin/bashrc",
        ):
            assert env_at_home.execute(command)["returncode"] != 0, command
        assert (deny_home / ".bashrc").is_symlink()
        assert VISIBLE in env_at_home.execute(f"cat {deny_home}/.bashrc")["output"]
        assert "changed" not in (deny_home / ".cargo" / "bin" / "bashrc").read_text()

    def test_every_home_path_of_the_file_safety_policy_is_hidden(self, sandbox_root, deny_home):
        from agent.file_safety import build_write_denied_paths, build_write_denied_prefixes

        home = str(deny_home)
        files = [Path(p) for p in build_write_denied_paths(home) if p.startswith(home + os.sep)]
        dirs = [Path(p.rstrip(os.sep)) for p in build_write_denied_prefixes(home) if p.startswith(home + os.sep)]
        assert files and dirs
        for path in dirs:
            if not path.exists():
                _write(path / "secret")
        for path in files:
            if not path.exists():
                _write(path)
        env = self._env(deny_home / "proj")
        try:
            for path in files + [d / "secret" for d in dirs]:
                assert MARKER not in env.execute(f"cat {path} 2>&1")["output"], path
        finally:
            env.cleanup()

    def test_a_path_added_to_the_file_safety_policy_is_hidden(self, sandbox_root, deny_home):
        from agent import file_safety

        invented = _write(deny_home / "sibling" / "zz-policy-secret")
        real = file_safety.build_write_denied_paths
        with patch.object(file_safety, "build_write_denied_paths", lambda home: real(home) | {str(invented)}):
            env = self._env(deny_home / "proj")
        try:
            assert MARKER not in env.execute(f"cat {invented} 2>&1")["output"]
            assert env.execute(f"cat {deny_home}/sibling/data.txt")["output"].strip() == VISIBLE
        finally:
            env.cleanup()

    def test_read_write_bind_below_an_allowed_entry_is_writable(self, sandbox_root, deny_home):
        zz = deny_home / ".cache" / "zz"
        zz.mkdir()
        _write(deny_home / ".cache" / "other" / "keep", VISIBLE)
        config = BubblewrapConfig(binds=(BindMount(src=str(zz), dest=str(zz), readonly=False),))
        env = self._env(deny_home / "proj", config=config)
        try:
            result = env.execute(f"printf ok > {zz}/from-sandbox")
            assert result["returncode"] == 0, result["output"]
            result = env.execute(f"printf no > {deny_home}/.cache/other/from-sandbox")
            assert result["returncode"] != 0
            assert "Read-only file system" in result["output"]
        finally:
            env.cleanup()
        assert (zz / "from-sandbox").read_text() == "ok"
        assert not (deny_home / ".cache" / "other" / "from-sandbox").exists()

    def test_read_write_bind_of_a_default_deny_directory_stays_writable(self, sandbox_root, deny_home):
        config_dir = deny_home / ".config"
        for rel in SENSITIVE_HOME_PATHS:
            if rel.startswith(".config/") and not (deny_home / rel).exists():
                (deny_home / rel).mkdir(parents=True)
        config = BubblewrapConfig(binds=(BindMount(src=str(config_dir), dest=str(config_dir), readonly=False),))
        env = self._env(deny_home / "proj", config=config)
        try:
            result = env.execute(f"printf ok > {config_dir}/zz-new")
            assert result["returncode"] == 0, result["output"]
            assert MARKER not in env.execute(f"cat {config_dir}/gh/hosts.yml 2>&1")["output"]
        finally:
            env.cleanup()
        assert (config_dir / "zz-new").read_text() == "ok"

    def test_hide_key_hides_a_non_dot_path_and_allow_key_shows_a_dot_entry(self, sandbox_root, deny_home):
        _write(deny_home / "Documents" / "keys" / "id")
        _write(deny_home / "Documents" / "notes.txt", VISIBLE)
        _write(deny_home / ".zz-extra" / "x", VISIBLE)
        _write(deny_home / ".config" / "zz-app" / "x", VISIBLE)
        config = BubblewrapConfig(hide=("~/Documents/keys",), home_allow=(".zz-extra", ".config/zz-app", ".config/gh"))
        env = self._env(deny_home / "proj", config=config)
        try:
            assert MARKER not in env.execute(f"cat {deny_home}/Documents/keys/id 2>&1")["output"]
            assert env.execute(f"cat {deny_home}/Documents/notes.txt")["output"].strip() == VISIBLE
            assert env.execute(f"cat {deny_home}/.zz-extra/x")["output"].strip() == VISIBLE
            assert env.execute(f"cat {deny_home}/.config/zz-app/x")["output"].strip() == VISIBLE
            assert MARKER not in env.execute(f"cat {deny_home}/.config/gh/hosts.yml 2>&1")["output"]
        finally:
            env.cleanup()


def _tmpdir(env) -> Path:
    return Path(env.execute('printf %s "$TMPDIR"')["output"].strip())


@needs_bwrap
class TestScratchDirIntegration:
    """HERMES_HOME/cache/scratch is TMPDIR of every Hermes process, and the
    overlay would hide it: the sandbox gets a scratch dir of its own inside
    that path, at the same path as on the host, kept from one command to
    the next."""

    def test_scratch_file_written_in_one_command_is_read_in_the_next(self, work_dir, hermes_home):
        scratch = hermes_home / "cache" / "scratch"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            private = _tmpdir(env)
            assert private.parent == scratch and private.is_dir()
            result = env.execute("printf kept > $TMPDIR/scratch-probe")
            assert result["returncode"] == 0, result["output"]
            assert env.execute("cat $TMPDIR/scratch-probe")["output"].strip() == "kept"
            made = env.execute("mktemp")["output"].strip()
            assert made.startswith(str(private) + os.sep)
            assert env.execute(f"test -f {made}")["returncode"] == 0
            assert env.execute('echo "$TMP $TEMP"')["output"].split() == [str(private)] * 2
            # The host-side tools read the same file at the same path.
            assert (private / "scratch-probe").read_text() == "kept"
            # Nothing else lands in the directory the other Hermes processes use.
            assert [entry.name for entry in scratch.iterdir()] == [private.name]
        finally:
            env.cleanup()
        assert list(scratch.iterdir()) == []

    def test_write_to_the_shared_scratch_path_fails_where_the_command_sees_it(self, work_dir, hermes_home):
        scratch = hermes_home / "cache" / "scratch"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            result = env.execute(f"printf no > {scratch}/scratch-probe")
            assert result["returncode"] != 0
            assert "Read-only file system" in result["output"]
            assert env.execute(f"ls -A {scratch}")["output"].split() == [_tmpdir(env).name]
        finally:
            env.cleanup()
        assert not (scratch / "scratch-probe").exists()

    def test_cd_into_a_scratch_subdirectory_is_tracked(self, work_dir, hermes_home):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            build = _tmpdir(env) / "build"
            result = env.execute("mkdir -p $TMPDIR/build && cd $TMPDIR/build")
            assert result["returncode"] == 0, result["output"]
            assert env.cwd == str(build)
            assert env.execute("pwd")["output"].strip() == str(build)
        finally:
            env.cleanup()

    def test_temp_variable_the_operator_set_elsewhere_is_left_alone(self, work_dir, hermes_home, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(work_dir))
        monkeypatch.delenv("TMP")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert _tmpdir(env) == work_dir
            assert env.execute('printf %s "${TMP-unset}"')["output"].strip() == "unset"
            assert Path(env.execute('printf %s "$TEMP"')["output"].strip()).parent == hermes_home / "cache" / "scratch"
        finally:
            env.cleanup()

    def test_temp_variables_are_retargeted_with_a_symlinked_hermes_home(self, work_dir, hermes_home, host_dir, monkeypatch):
        # HERMES_HOME, and with it the temp variables, name the home through a link.
        link = host_dir / "hermes-link"
        link.symlink_to(hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(link))
        for name in ("TMPDIR", "TMP", "TEMP"):
            monkeypatch.setenv(name, str(link / "cache" / "scratch"))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            private = _tmpdir(env)
            assert private.parent == hermes_home / "cache" / "scratch"
            made = env.execute("mktemp")
            assert made["returncode"] == 0, made["output"]
            assert made["output"].strip().startswith(str(private) + os.sep)
        finally:
            env.cleanup()

    def test_scratch_dir_pruned_while_the_environment_is_idle_is_made_again(self, work_dir, hermes_home):
        from hermes_constants import prune_scratch_dir

        scratch = hermes_home / "cache" / "scratch"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            private = _tmpdir(env)
            long_ago = time.time() - 25 * 3600
            os.utime(private, (long_ago, long_ago))
            assert prune_scratch_dir(scratch) == 1 and not private.exists()
            result = env.execute("printf again > $TMPDIR/f && cat $TMPDIR/f")
            assert result["returncode"] == 0, result["output"]
            assert result["output"].strip() == "again"
        finally:
            env.cleanup()

    def test_a_command_refreshes_the_scratch_dir_so_a_used_environment_is_not_pruned(self, work_dir, hermes_home):
        from hermes_constants import prune_scratch_dir

        scratch = hermes_home / "cache" / "scratch"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            private = _tmpdir(env)
            assert env.execute("printf kept > $TMPDIR/f")["returncode"] == 0
            long_ago = time.time() - 25 * 3600
            for path in (private / "f", private):
                os.utime(path, (long_ago, long_ago))
            assert env.execute("true")["returncode"] == 0
            assert prune_scratch_dir(scratch) == 0
            assert env.execute("cat $TMPDIR/f")["output"].strip() == "kept"
        finally:
            env.cleanup()

    def test_scratch_is_read_only_under_the_restricted_profile(self, work_dir, hermes_home):
        scratch = hermes_home / "cache" / "scratch"
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=BubblewrapConfig(profile="restricted"))
        try:
            private = _tmpdir(env)
            for target in (f"{scratch}/scratch-probe", "$TMPDIR/scratch-probe"):
                result = env.execute(f"printf no > {target}")
                assert result["returncode"] != 0, target
                assert "Read-only file system" in result["output"], target
            assert list(private.iterdir()) == []
        finally:
            env.cleanup()
        assert not (scratch / "scratch-probe").exists()

    def test_scratch_under_the_default_home_survives_the_home_layout(self, sandbox_root, work_dir, fake_home, monkeypatch):
        hermes_home = fake_home / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TMPDIR", str(hermes_home / "cache" / "scratch"))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            private = _tmpdir(env)
            assert private.parent == hermes_home / "cache" / "scratch"
            assert env.execute("printf kept > $TMPDIR/scratch-probe")["returncode"] == 0
            assert env.execute("cat $TMPDIR/scratch-probe")["output"].strip() == "kept"
            assert (private / "scratch-probe").read_text() == "kept"
        finally:
            env.cleanup()

    def test_two_environments_do_not_share_scratch(self, work_dir, hermes_home):
        scratch = hermes_home / "cache" / "scratch"
        first = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        second = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            mine = _tmpdir(first)
            assert mine != _tmpdir(second)
            assert first.execute(f"printf {MARKER} > $TMPDIR/from-first")["returncode"] == 0
            out = second.execute(f"ls -AR {scratch}; cat {mine}/from-first {scratch}/*/* 2>&1")["output"]
            assert MARKER not in out
            assert mine.name not in out.replace(f"{mine}/from-first: No such file", "")
            assert first.execute("cat $TMPDIR/from-first")["output"].strip() == MARKER
        finally:
            first.cleanup()
            second.cleanup()

    def test_scratch_directory_is_removed_with_the_environment(self, work_dir, hermes_home):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        sandboxes = Path(env.get_temp_dir()).parent
        try:
            assert env.execute("printf x > $TMPDIR/f")["returncode"] == 0
        finally:
            env.cleanup()
        assert list(sandboxes.iterdir()) == []
        assert list((hermes_home / "cache" / "scratch").iterdir()) == []


@needs_bwrap
class TestOtherEnvironmentStateIntegration:
    """A sandbox dir outside HERMES_HOME shows through the read-only root. The
    state dirs in it hold the shell snapshot of each environment and the
    scripts of its execute_code calls: no other environment may read them."""

    def test_state_of_another_environment_is_not_visible(self, host_dir, work_dir, monkeypatch):
        root = host_dir / "sandboxes"
        mirror_src = host_dir
        monkeypatch.setenv("TERMINAL_SANDBOX_DIR", str(root))
        # The same directory once more, through an operator bind at another path.
        config = BubblewrapConfig(binds=(BindMount(str(mirror_src), "/mnt/mirror", True),))
        first = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        second = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=config)
        try:
            theirs = Path(first.get_temp_dir())
            assert theirs.parent == root
            _write(theirs / "note.txt")
            assert first.execute(f"cat {theirs}/note.txt")["output"].strip() == MARKER
            out = second.execute(
                f"ls -A {root} /mnt/mirror/sandboxes 2>&1; "
                f"cat {theirs}/note.txt {root}/*/* /mnt/mirror/sandboxes/*/* 2>&1"
            )["output"]
            assert MARKER not in out
            mine = Path(second.get_temp_dir())
            assert theirs.name not in out.replace(f"{theirs}/note.txt: No such file", "")
            assert second.execute(f"ls -A {root}")["output"].split() == [mine.name]
            # Its own state dir still works: the shell state survives.
            assert second.execute("export KEPT=1")["returncode"] == 0
            assert second.execute("echo $KEPT")["output"].strip() == "1"
        finally:
            first.cleanup()
            second.cleanup()


def _listen(path: Path):
    """A listening unix socket at *path*, bound by a relative name (test paths are too long for one)."""
    import socket

    sock = socket.socket(socket.AF_UNIX)
    here = os.getcwd()
    os.chdir(path.parent)
    try:
        sock.bind(path.name)
    finally:
        os.chdir(here)
    sock.listen(1)
    return sock


@needs_bwrap
class TestSharedScratchIsNotInViewIntegration:
    """The host scratch dir is TMPDIR of every Hermes process: it holds
    control sockets and the message payloads of other sessions, and a new
    one can appear at any time. None of it shows in a sandbox."""

    @pytest.mark.parametrize("profile", ["network", "restricted"])
    def test_entries_made_before_and_during_a_command_are_not_visible(self, work_dir, hermes_home, profile):
        import threading

        scratch = hermes_home / "cache" / "scratch"
        _write(scratch / "hermes-dm-1000" / "msg.txt")
        _write(scratch / "hermes-relay-dm-early.txt")
        _write(scratch / "agent-browser-s1" / "info.txt")
        sock = _listen(scratch / "top.sock")
        # Both sides signal through a directory bound read-write under every profile.
        signal = work_dir.parent / "signal"
        signal.mkdir()
        started, go = signal / "started", signal / "go"
        config = BubblewrapConfig(profile=profile, binds=(BindMount(str(signal), str(signal), False),))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=60, config=config)
        result: dict = {}
        command = (
            f"touch {started}; while [ ! -e {go} ]; do sleep 0.05; done; "
            f"ls -AR {scratch}; cat {scratch}/* {scratch}/*/* 2>&1"
        )
        worker = threading.Thread(target=lambda: result.update(env.execute(command)))
        try:
            worker.start()
            deadline = time.monotonic() + 30
            while not started.exists():
                assert time.monotonic() < deadline, "the command did not start"
                time.sleep(0.05)
            # Made while the command runs, after every mount of its sandbox is in place.
            _write(scratch / "hermes-relay-dm-late.txt")
            go.touch()
            worker.join(60)
            assert not worker.is_alive()
        finally:
            go.touch()
            worker.join(60)
            env.cleanup()
            sock.close()
        out = result["output"]
        assert MARKER not in out
        for name in ("hermes-dm-1000", "hermes-relay-dm-early.txt", "hermes-relay-dm-late.txt", "agent-browser-s1", "top.sock"):
            assert name not in out, (name, out)
        # The listing did run: it shows the scratch dir of this environment.
        assert "hermes-bwrap-" in out
        assert (scratch / "hermes-relay-dm-late.txt").read_text().startswith(MARKER)


@needs_bwrap
class TestStagedDataIntegration:
    """Hermes hands the model host paths under HERMES_HOME for attachments and
    cached documents; a command must be able to open them."""

    def test_staged_files_are_readable_at_their_host_path_and_read_only(self, work_dir, hermes_home):
        (hermes_home / "cache").mkdir(exist_ok=True)
        (hermes_home / "cache" / "bws_cache.json").write_text(MARKER)
        (hermes_home / "auth.json").write_text(MARKER)
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        attachment = _write(hermes_home / "attachments" / "report.pdf", VISIBLE)
        document = _write(hermes_home / "cache" / "documents" / "notes.txt", VISIBLE)
        try:
            for path in (attachment, document):
                assert env.execute(f"cat {path}")["output"].strip() == VISIBLE, path
                result = env.execute(f"printf changed > {path}")
                assert result["returncode"] != 0
                assert "Read-only file system" in result["output"]
                assert env.execute(f"printf new > {path.parent}/from-sandbox")["returncode"] != 0
            out = env.execute(
                f"cat {hermes_home}/.env {hermes_home}/auth.json {hermes_home}/config.yaml "
                f"{hermes_home}/cache/bws_cache.json 2>&1"
            )["output"]
            assert MARKER not in out
        finally:
            env.cleanup()
        assert attachment.read_text().strip() == VISIBLE
        assert not (hermes_home / "attachments" / "from-sandbox").exists()

    def test_spillover_archive_of_the_profile_is_not_readable(self, work_dir, hermes_home):
        # Oversized tool results of every session land in one directory.
        other = _write(hermes_home / "cache" / "spillover" / "call_of_another_session.txt")
        attachment = _write(hermes_home / "attachments" / "report.pdf", VISIBLE)
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            out = env.execute(f"ls -A {other.parent} 2>&1; cat {other} {other.parent}/* 2>&1")["output"]
            assert MARKER not in out
            assert other.name not in out.replace(f"{other}: No such file", "")
            assert "spillover" not in env.execute(f"ls -A {hermes_home}/cache")["output"].split()
            assert env.execute(f"cat {attachment}")["output"].strip() == VISIBLE
            assert env.execute(f"printf changed > {attachment}")["returncode"] != 0
        finally:
            env.cleanup()

    def test_oversized_tool_result_is_readable_by_the_commands_of_its_own_environment(self, work_dir, hermes_home):
        from tools.tool_result_storage import extract_persisted_path, maybe_persist_tool_result

        spillover = hermes_home / "cache" / "spillover"
        other = _write(spillover / "call_of_another_session.txt")
        content = "".join(f"line {i} of the result\n" for i in range(4000))
        first = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        second = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            message = maybe_persist_tool_result(content, "terminal", "call_1", env=first, threshold=1000)
            path = extract_persisted_path(message)
            # The canonical archive path: it outlives the environment.
            assert path and Path(path).parent == spillover
            assert first.execute(f"wc -c < {path}")["output"].strip() == str(len(content))
            assert first.execute(f"tail -n 1 {path}")["output"].strip() == "line 3999 of the result"
            assert first.execute(f"printf changed >> {path}")["returncode"] != 0
            # Of the whole archive this environment sees that one file.
            assert first.execute(f"ls -A {spillover}")["output"].split() == [Path(path).name]
            assert MARKER not in first.execute(f"cat {other} 2>&1")["output"]
            out = second.execute(f"ls -A {spillover} 2>&1; cat {path} 2>&1 | head -c 200")["output"]
            assert "of the result" not in out and Path(path).name not in out.replace(f"{path}: No such file", "")
        finally:
            first.cleanup()
            second.cleanup()
        # The reference in the conversation stays good for the host-side file tools.
        assert Path(path).read_text() == content

    def test_two_sessions_with_the_same_tool_call_id_each_read_their_own_result(self, work_dir, hermes_home):
        # The archive name comes from the tool call id alone, and ids repeat
        # between sessions: the later write replaces the file on the host.
        from tools.tool_result_storage import extract_persisted_path, maybe_persist_tool_result

        def result_of(session: str) -> str:
            return "".join(f"{session} line {i}\n" for i in range(4000))

        first = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        second = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            paths = []
            for env, session in ((first, "FIRST-SESSION"), (second, "SECOND-SESSION")):
                message = maybe_persist_tool_result(result_of(session), "terminal", "functions.terminal:0", env=env, threshold=1000)
                paths.append(extract_persisted_path(message))
            assert paths[0] == paths[1]
            out_first = first.execute(f"head -n 1 {paths[0]}; grep -c SECOND-SESSION {paths[0]}")["output"].split("\n")
            assert out_first[0] == "FIRST-SESSION line 0" and out_first[1] == "0"
            assert second.execute(f"head -n 1 {paths[0]}")["output"].strip() == "SECOND-SESSION line 0"
            # The copies are bind sources only: no command finds them at their own path.
            sandboxes = Path(first.get_temp_dir()).parent
            assert first.execute(f"ls -A {sandboxes}")["output"].split() == [Path(first.get_temp_dir()).name]
            assert "SESSION line" not in first.execute(f"cat {sandboxes}/*/* 2>&1")["output"]
        finally:
            first.cleanup()
            second.cleanup()
        assert list(sandboxes.iterdir()) == []

    def test_archive_file_pruned_on_the_host_does_not_fail_the_spawn(self, work_dir, hermes_home):
        archive = _write(hermes_home / "cache" / "spillover" / "call_1.txt", VISIBLE)
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert env.expose_spillover_file(str(archive))
            assert env.execute(f"cat {archive}")["output"].strip() == VISIBLE
            archive.unlink()
            assert env.execute("echo ok")["output"].strip() == "ok"
        finally:
            env.cleanup()

    def test_staged_root_removed_after_construction_does_not_fail_the_spawn(self, work_dir, hermes_home):
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30)
        try:
            assert (hermes_home / "attachments").is_dir()
            (hermes_home / "attachments").rmdir()
            assert env.execute("echo ok")["output"].strip() == "ok"
        finally:
            env.cleanup()


@needs_bwrap
class TestCoveredDefaultDenyDirIntegration:
    """A bind on ~/.local covers the tmpfs of ~/.local/share below it: the
    layout must not remount a path that is no longer a mount point."""

    @pytest.fixture
    def local_home(self, host_dir, monkeypatch):
        home = host_dir / "home"
        _write(home / ".local" / "share" / "app" / "x", VISIBLE)
        _write(home / ".local" / "bin" / "t", VISIBLE)
        for rel in SENSITIVE_HOME_PATHS:
            if rel.startswith(".local/"):
                (home / rel).mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        return home

    def test_covered_by_a_read_only_operator_bind_runs(self, sandbox_root, work_dir, local_home):
        local = local_home / ".local"
        config = BubblewrapConfig(binds=(BindMount(src=str(local), dest=str(local), readonly=True),))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=config)
        try:
            result = env.execute(f"cat {local}/share/app/x")
            assert result["returncode"] == 0, result["output"]
            assert result["output"].strip() == VISIBLE
            assert env.execute(f"ls -A {local}/share/keyrings")["output"].strip() == ""
        finally:
            env.cleanup()

    def test_covered_by_a_restricted_cwd_runs(self, sandbox_root, local_home):
        env = BubblewrapEnvironment(cwd=str(local_home / ".local"), timeout=30, config=BubblewrapConfig(profile="restricted"))
        try:
            result = env.execute("echo hello")
            assert result["returncode"] == 0, result["output"]
            assert result["output"].strip() == "hello"
        finally:
            env.cleanup()

    def test_covered_by_a_read_write_operator_bind_is_writable(self, sandbox_root, work_dir, local_home):
        local = local_home / ".local"
        config = BubblewrapConfig(binds=(BindMount(src=str(local), dest=str(local), readonly=False),))
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=config)
        try:
            result = env.execute(f"printf ok > {local}/share/app/new")
            assert result["returncode"] == 0, result["output"]
        finally:
            env.cleanup()
        assert (local / "share" / "app" / "new").read_text() == "ok"


@needs_bwrap
class TestHideBelowRestoredRootsIntegration:
    """A hide entry under a staged data root stays hidden although its root
    is bound back on top of the HERMES_HOME overlay."""

    @staticmethod
    def _check(hermes_home, cwd):
        secret = _write(hermes_home / "cache" / "documents" / "secret.txt")
        _write(hermes_home / "cache" / "documents" / "open.txt", VISIBLE)
        config = BubblewrapConfig(hide=(str(secret),))
        env = BubblewrapEnvironment(cwd=str(cwd), timeout=30, config=config)
        try:
            assert MARKER not in env.execute(f"cat {secret} 2>&1")["output"]
            assert env.execute(f"cat {hermes_home}/cache/documents/open.txt")["output"].strip() == VISIBLE
            assert env.execute("printf ok > $TMPDIR/probe")["returncode"] == 0
        finally:
            env.cleanup()

    def test_hide_entry_below_a_restored_root_stays_hidden(self, work_dir, hermes_home):
        self._check(hermes_home, work_dir)

    def test_hide_entry_below_a_restored_root_stays_hidden_with_hermes_home_under_home(self, sandbox_root, work_dir, fake_home, monkeypatch):
        hermes_home = fake_home / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TMPDIR", str(hermes_home / "cache" / "scratch"))
        self._check(hermes_home, work_dir)

    def test_hide_entry_equal_to_a_staged_root_keeps_that_root_hidden(self, work_dir, hermes_home):
        documents = hermes_home / "cache" / "documents"
        _write(documents / "secret.txt")
        _write(hermes_home / "attachments" / "open.txt", VISIBLE)
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=BubblewrapConfig(hide=(str(documents),)))
        try:
            assert MARKER not in env.execute(f"cat {documents}/secret.txt 2>&1")["output"]
            assert env.execute(f"cat {hermes_home}/attachments/open.txt")["output"].strip() == VISIBLE
        finally:
            env.cleanup()

    def test_hide_entry_equal_to_the_scratch_dir_leaves_the_sandbox_its_own_scratch(self, work_dir, hermes_home):
        scratch = hermes_home / "cache" / "scratch"
        _write(scratch / "secret.txt")
        env = BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=BubblewrapConfig(hide=(str(scratch),)))
        try:
            assert MARKER not in env.execute(f"cat {scratch}/secret.txt 2>&1")["output"]
            assert env.execute("printf kept > $TMPDIR/probe")["returncode"] == 0
            assert env.execute("cat $TMPDIR/probe")["output"].strip() == "kept"
        finally:
            env.cleanup()

    def test_hide_entry_inside_the_profile_home_is_refused(self, work_dir, hermes_home):
        # The profile home is bound read-write on top of the overlay, so a
        # hidden path inside it would show again: the backend does not start.
        keys = hermes_home / "home" / "a" / "keys"
        _write(keys / "id")
        config = BubblewrapConfig(home_mode="profile", hide=(str(keys),))
        with pytest.raises(ValueError, match="terminal.home_mode"):
            BubblewrapEnvironment(cwd=str(work_dir), timeout=30, config=config)
