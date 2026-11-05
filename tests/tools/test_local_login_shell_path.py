"""The terminal snapshot uses a zsh user's login PATH (#129103).

The snapshot runs ``bash -l``, which never reads ``~/.zprofile``, so on macOS
Homebrew's ``python3`` sat behind ``/usr/bin/python3``. These tests use a fake
``zsh`` executable so they do not depend on the machine's shells or dotfiles.

See: https://github.com/NousResearch/hermes-agent/issues/129103
"""

import logging
import os
import pathlib
import re
import subprocess
import time

import pytest

from tests.waiting import wait_until
from tools.environments import local_login_path
from tools.environments.local import LocalEnvironment
from tools.environments.local_login_path import (
    _PATH_END, _PATH_START, _PROBE_BASE_PATH, _extract_path, login_shell_path,
    prepend_login_shell_path)

_PATH_LINE = re.compile(r'^(?:declare -x|export) PATH="(.*)"$', re.MULTILINE)


@pytest.fixture(autouse=True)
def _fresh_probe_cache(monkeypatch):
    monkeypatch.setattr(local_login_path, "_probe_results", {})


def _fake_zsh(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    """An executable named ``zsh``; ``$2`` is the probe command after ``-lc``."""
    bin_dir = tmp_path / "shells"
    bin_dir.mkdir(exist_ok=True)
    shell = bin_dir / "zsh"
    shell.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    shell.chmod(0o755)
    return shell


def _login_zsh(tmp_path, monkeypatch, login_dir: pathlib.Path) -> pathlib.Path:
    """A fake zsh whose login init puts *login_dir* first, like ``brew shellenv``."""
    shell = _fake_zsh(tmp_path, f'echo "profile banner"\nPATH="{login_dir}:$PATH"\neval "$2"\n')
    monkeypatch.setenv("SHELL", str(shell))
    return shell


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestExtractPath:
    def test_value_between_markers_ignores_profile_output(self):
        assert _extract_path(f"motd\n{_PATH_START}/a:/b{_PATH_END}\n") == "/a:/b"

    def test_last_start_marker_wins(self):
        text = f"{_PATH_START}/old{_PATH_END} {_PATH_START}/new{_PATH_END}"
        assert _extract_path(text) == "/new"

    @pytest.mark.parametrize("text", ["", "no markers", f"{_PATH_START}/a", f"{_PATH_START}{_PATH_END}"])
    def test_missing_or_empty_value_is_none(self, text):
        assert _extract_path(text) is None


@pytest.mark.platforms("posix")
class TestLoginShellPath:
    def test_zsh_login_path_is_captured_from_a_system_base_path(self, tmp_path, monkeypatch):
        login_dir = tmp_path / "brew-bin"
        _login_zsh(tmp_path, monkeypatch, login_dir)
        assert login_shell_path() == f"{login_dir}:{_PROBE_BASE_PATH}"

    def test_process_path_never_reaches_the_capture(self, tmp_path, monkeypatch):
        """One gateway serves many sessions: a client's venv on the process (or a
        session's) PATH must not be captured and replayed into other sessions."""
        _login_zsh(tmp_path, monkeypatch, tmp_path / "brew-bin")
        monkeypatch.setenv("PATH", f"/projA/.venv/bin:{os.environ['PATH']}")
        monkeypatch.setenv("VIRTUAL_ENV", "/projA/.venv")
        captured = login_shell_path()
        assert "/projA/.venv/bin" not in captured.split(":")

    def test_probe_env_keeps_only_fixed_keys(self, tmp_path, monkeypatch):
        shell = _fake_zsh(tmp_path, 'env > "$HOME/probe-env"\neval "$2"\n')
        monkeypatch.setenv("SHELL", str(shell))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("VIRTUAL_ENV", "/projA/.venv")
        login_shell_path()
        seen = dict(line.split("=", 1) for line in
                    (tmp_path / "probe-env").read_text(encoding="utf-8").splitlines() if "=" in line)
        assert "VIRTUAL_ENV" not in seen
        assert seen["HOME"] == str(tmp_path)

    @pytest.mark.parametrize("name", ["bash", "fish", "sh"])
    def test_other_shells_are_not_probed(self, tmp_path, monkeypatch, name):
        shell = tmp_path / name
        shell.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        shell.chmod(0o755)
        monkeypatch.setenv("SHELL", str(shell))
        monkeypatch.setattr(local_login_path, "_run_probe", pytest.fail)
        assert login_shell_path() is None

    def test_unset_or_missing_shell_is_not_probed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(local_login_path, "_run_probe", pytest.fail)
        monkeypatch.delenv("SHELL", raising=False)
        assert login_shell_path() is None
        monkeypatch.setenv("SHELL", str(tmp_path / "zsh"))
        assert login_shell_path() is None

    def test_windows_is_not_probed(self, tmp_path, monkeypatch):
        _login_zsh(tmp_path, monkeypatch, tmp_path / "brew-bin")
        monkeypatch.setattr(local_login_path, "_IS_WINDOWS", True)
        monkeypatch.setattr(local_login_path, "_run_probe", pytest.fail)
        assert login_shell_path() is None

    def test_probe_runs_once_per_shell_and_home(self, tmp_path, monkeypatch):
        _login_zsh(tmp_path, monkeypatch, tmp_path / "brew-bin")
        calls = []
        monkeypatch.setattr(local_login_path, "_run_probe",
                            lambda shell, env: calls.append(env["HOME"]) or "/x")
        monkeypatch.setenv("HOME", str(tmp_path / "a"))
        login_shell_path()
        login_shell_path()
        monkeypatch.setenv("HOME", str(tmp_path / "b"))
        login_shell_path()
        assert calls == [str(tmp_path / "a"), str(tmp_path / "b")]

    def test_failing_profile_gives_none_and_warns_once(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("SHELL", str(_fake_zsh(tmp_path, "echo broken >&2\nexit 1\n")))
        with caplog.at_level(logging.WARNING, logger=local_login_path.__name__):
            assert login_shell_path() is None
            assert login_shell_path() is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "terminal.shell_init_files" in warnings[0].getMessage()

    def test_hanging_profile_times_out_and_kills_its_daemon(self, tmp_path, monkeypatch):
        """A profile daemon that keeps stdout open (Powerlevel10k's gitstatusd, #107109)
        must not hold the probe past its timeout or outlive it."""
        pid_file = tmp_path / "daemon.pid"
        monkeypatch.setenv("SHELL", str(_fake_zsh(
            tmp_path, f"sleep 60 &\necho $! > '{pid_file}'\nwait\n")))
        monkeypatch.setattr(local_login_path, "_PROBE_TIMEOUT_SECONDS", 0.5)
        started = time.monotonic()
        assert login_shell_path() is None
        assert time.monotonic() - started < 3
        daemon = int(pid_file.read_text(encoding="utf-8"))
        wait_until(lambda: not _alive(daemon), timeout=2,
                   what="the profile's background daemon to die with the probe")


@pytest.mark.platforms("posix")
class TestPrependLoginShellPath:
    def _merged(self, login_path, shell_path):
        script = prepend_login_shell_path('printf %s "$PATH"', login_path)
        done = subprocess.run(["/bin/sh", "-c", script], env={"PATH": shell_path},
                              capture_output=True, text=True, timeout=10, check=True)
        return done.stdout

    def test_login_dirs_first_then_missing_shell_dirs_once(self):
        assert self._merged("/brew/bin:/usr/bin:/brew/bin", "/usr/bin:/bin::/usr/bin:/managed") \
            == "/brew/bin:/usr/bin:/bin:/managed"

    def test_entries_are_not_glob_expanded(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        star = f"{tmp_path}/*"
        assert self._merged("/brew/bin", f"/usr/bin:{star}") == f"/brew/bin:/usr/bin:{star}"

    @pytest.mark.parametrize("login_path", [None, "", ":::"])
    def test_nothing_captured_leaves_command_unchanged(self, login_path):
        assert prepend_login_shell_path("echo hi", login_path) == "echo hi"


def _snapshot_path_value(env: LocalEnvironment) -> list[str]:
    content = pathlib.Path(env._snapshot_path).read_text(encoding="utf-8-sig")
    match = _PATH_LINE.search(content)
    assert match, f"no PATH line in snapshot:\n{content}"
    return match.group(1).split(":")


@pytest.mark.platforms("posix")
class TestSnapshotUsesZshLoginPath:
    def _setup(self, tmp_path, monkeypatch, profile_body: str):
        """An init file plus a Hermes-managed bin dir."""
        profile = tmp_path / "profile.sh"
        profile.write_text(profile_body, encoding="utf-8")
        managed = tmp_path / "managed-bin"
        managed.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("tools.environments.local._resolve_shell_init_files",
                            lambda: [str(profile)])
        monkeypatch.setattr("tools.environments.local._resolve_hermes_bin_dir",
                            lambda: str(managed))
        monkeypatch.setattr("tools.environments.local._managed_runtime_path_entries", list)
        return managed

    def _snapshot(self, tmp_path, path: str) -> list[str]:
        env = LocalEnvironment(cwd=str(tmp_path), env={"PATH": path})
        try:
            return _snapshot_path_value(env)
        finally:
            env.cleanup()

    def test_zsh_user_gets_login_dirs_first_and_init_file_dirs_ahead_of_them(
            self, tmp_path, monkeypatch):
        """brew (zsh login) before /usr/bin, and nvm (a listed init file such as
        ~/.zshrc) before brew and before Hermes's own dirs, as in the user's shell."""
        nvm = tmp_path / "nvm-bin"
        nvm.mkdir()
        managed = self._setup(tmp_path, monkeypatch, f'PATH="{nvm}:$PATH"\n')
        brew = tmp_path / "brew-bin"
        brew.mkdir()
        _login_zsh(tmp_path, monkeypatch, brew)
        entries = self._snapshot(tmp_path, f"{managed}:/usr/bin:/bin")
        assert entries[:2] == [str(nvm), str(brew)]
        assert entries.index(str(brew)) < entries.index("/usr/bin")
        assert entries.index(str(nvm)) < entries.index(str(managed))
        assert len(entries) == len(set(entries))

    def test_one_sessions_path_does_not_reach_the_next_session(self, tmp_path, monkeypatch):
        managed = self._setup(tmp_path, monkeypatch, ": # init file that keeps PATH\n")
        _login_zsh(tmp_path, monkeypatch, tmp_path / "brew-bin")
        project_a, project_b = tmp_path / "projA-venv-bin", tmp_path / "projB-venv-bin"
        project_a.mkdir()
        project_b.mkdir()
        self._snapshot(tmp_path, f"{project_a}:{managed}:/usr/bin:/bin")
        entries = self._snapshot(tmp_path, f"{project_b}:{managed}:/usr/bin:/bin")
        assert str(project_a) not in entries
        assert str(project_b) in entries

    def test_bash_user_snapshot_is_unchanged(self, tmp_path, monkeypatch):
        managed = self._setup(tmp_path, monkeypatch, 'PATH="/usr/bin:/bin:$PATH"\n')
        monkeypatch.setenv("SHELL", "/bin/bash")
        entries = self._snapshot(tmp_path, f"{managed}:/usr/bin:/bin")
        assert entries[:2] == ["/usr/bin", "/bin"]
        assert str(managed) in entries
