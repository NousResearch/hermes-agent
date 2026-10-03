"""macOS terminal PATH must follow the login shell, not path_helper (#118220).

On macOS the login snapshot comes from ``bash -l``, which sources
/etc/profile -> path_helper and puts /usr/bin ahead of Homebrew — even when
the user's login shell (zsh) has Homebrew first. No macOS host here, so every
macOS fact is a test double: sys.platform is faked to darwin, the zsh login
PATH query is faked via subprocess.run, and the bash snapshot file is written
by a fake _run_bash exactly as macOS ``bash -l`` would emit it.
"""

import re
import shlex
import subprocess
import sys
from tempfile import TemporaryFile
from unittest.mock import MagicMock, patch

from tools.environments import local as local_mod
from tools.environments.local import LocalEnvironment

# As macOS ``bash -l`` (path_helper) emits it: /usr/bin ahead of Homebrew.
BASH_ORDER = ("/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:"
              "/usr/sbin:/sbin:/opt/homebrew/bin:/opt/homebrew/sbin")
# As the user's ``zsh -l`` reports it: Homebrew first.
ZSH_ORDER = ("/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:"
             "/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin")


def _make_env(tmp_path, monkeypatch):
    """LocalEnvironment without running the real snapshot (hermetic: temp dir
    only, no host shell spawned)."""
    monkeypatch.setattr(LocalEnvironment, "get_temp_dir", lambda self: str(tmp_path))
    with patch.object(LocalEnvironment, "init_session", lambda self: None):
        env = LocalEnvironment(cwd=str(tmp_path))
    env._snapshot_ready = False
    return env


def _mock_login_snapshot(env, path_value):
    """Fake _run_bash that writes the snapshot file the way macOS ``bash -l``
    would (bash-ordered PATH via ``export -p``)."""

    def mock_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
        assert login, "init_session must snapshot via the login shell"
        with open(env._snapshot_path, "w", encoding="utf-8") as fh:
            fh.write(f'declare -x PATH="{path_value}"\n')
        mock = MagicMock()
        mock.poll.return_value = 0
        mock.returncode = 0
        mock.stdout = TemporaryFile(mode="w+b")
        return mock

    env._run_bash = mock_run_bash


def _fake_macos(monkeypatch):
    """Fake the macOS facts: platform + zsh login-shell PATH query."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(local_mod, "_find_shell", lambda: "/bin/zsh")

    def fake_run(argv, **kwargs):
        assert argv[0] == "/bin/zsh" and "-l" in argv, argv
        return subprocess.CompletedProcess(argv, 0, stdout=ZSH_ORDER, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _snapshot_effective_path(text):
    """PATH after sourcing the snapshot: the last PATH assignment wins."""
    last = None
    for line in text.splitlines():
        m = re.match(r"^(?:declare -x PATH=|export PATH=)(.*)$", line)
        if m:
            last = m.group(1)
    assert last is not None, "snapshot must assign PATH"
    return shlex.split(last)[0]


class TestMacosLoginShellPath:
    def test_snapshot_puts_homebrew_first(self, tmp_path, monkeypatch):
        """Behavioral RED: with a bash-ordered snapshot and a Homebrew-first
        zsh login PATH, the sourced snapshot must resolve Homebrew first."""
        _fake_macos(monkeypatch)
        env = _make_env(tmp_path, monkeypatch)
        _mock_login_snapshot(env, BASH_ORDER)
        env.init_session()

        assert env._snapshot_ready
        with open(env._snapshot_path, encoding="utf-8") as fh:
            effective = _snapshot_effective_path(fh.read())
        entries = effective.split(":")
        assert entries.index("/opt/homebrew/bin") < entries.index("/usr/bin"), (
            f"Homebrew must precede /usr/bin, got: {effective}")

    def test_off_macos_snapshot_untouched(self, tmp_path, monkeypatch):
        """Off darwin the snapshot file must not gain a PATH fixup line."""
        env = _make_env(tmp_path, monkeypatch)
        _mock_login_snapshot(env, BASH_ORDER)
        env.init_session()

        assert env._snapshot_ready
        with open(env._snapshot_path, encoding="utf-8") as fh:
            text = fh.read()
        assert "export PATH=" not in text

    def test_non_zsh_shell_skips_fixup(self, tmp_path, monkeypatch):
        """A bash login shell on macOS already defines the order: no fixup."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(local_mod, "_find_shell", lambda: "/bin/bash")
        env = _make_env(tmp_path, monkeypatch)
        _mock_login_snapshot(env, BASH_ORDER)
        env.init_session()

        assert env._snapshot_ready
        with open(env._snapshot_path, encoding="utf-8") as fh:
            text = fh.read()
        assert "export PATH=" not in text


    def test_login_shell_query_uses_scrubbed_env(self, tmp_path, monkeypatch):
        """The zsh login-shell spawn must reuse the environment's own env through
        the shared scrubber, like every other spawn on this path: the raw parent
        env (provider credentials) must never reach its startup files."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(local_mod, "_find_shell", lambda: "/bin/zsh")
        built = {}

        def fake_make_run_env(env):
            built["env"] = env
            return {"PATH": "/scrubbed/bin", "SCRUBBED": "1"}

        monkeypatch.setattr(local_mod, "_make_run_env", fake_make_run_env)
        seen = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, stdout=ZSH_ORDER, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        env = _make_env(tmp_path, monkeypatch)
        env.env = {"TERMINAL_MARKER": "1"}
        _mock_login_snapshot(env, BASH_ORDER)
        env.init_session()

        assert built["env"] == {"TERMINAL_MARKER": "1"}, (
            "login-shell query must use the environment's own env")
        assert seen["env"] == {"PATH": "/scrubbed/bin", "SCRUBBED": "1"}, (
            "login-shell spawn must pass the scrubbed env, not inherit the raw parent env")


class TestReorderPathToReference:
    def test_shared_entries_follow_login_shell_order(self):
        from tools.environments.local import _reorder_path_to_reference
        fixed = _reorder_path_to_reference(BASH_ORDER, ZSH_ORDER).split(":")
        assert fixed.index("/opt/homebrew/bin") < fixed.index("/usr/bin")
        assert fixed.index("/opt/homebrew/sbin") < fixed.index("/usr/local/bin")

    def test_snapshot_only_entries_keep_precedence(self):
        from tools.environments.local import _reorder_path_to_reference
        snap = "/root/.nvm/versions/node/bin:" + BASH_ORDER
        fixed = _reorder_path_to_reference(snap, ZSH_ORDER).split(":")
        assert fixed[0] == "/root/.nvm/versions/node/bin"
        assert fixed.index("/opt/homebrew/bin") < fixed.index("/usr/bin")

    def test_already_ordered_is_stable(self):
        from tools.environments.local import _reorder_path_to_reference
        assert _reorder_path_to_reference(ZSH_ORDER, ZSH_ORDER) == ZSH_ORDER

    def test_empties_and_dupes_collapsed(self):
        from tools.environments.local import _reorder_path_to_reference
        fixed = _reorder_path_to_reference("/usr/bin::/usr/bin:/opt/homebrew/bin", ZSH_ORDER)
        assert fixed.split(":").count("/usr/bin") == 1
        assert "" not in fixed.split(":")


class TestReadSnapshotPath:
    def test_reads_dquote_value(self, tmp_path):
        from tools.environments.local import _read_snapshot_path
        snap = tmp_path / "s.sh"
        snap.write_text('declare -x PATH="/opt/homebrew/bin:/usr/bin"\n', encoding="utf-8")
        assert _read_snapshot_path(str(snap)) == "/opt/homebrew/bin:/usr/bin"

    def test_escaped_chars_decoded(self, tmp_path):
        from tools.environments.local import _read_snapshot_path
        snap = tmp_path / "s.sh"
        snap.write_text('declare -x PATH="/a\\$b:/c\\"d"\n', encoding="utf-8")
        assert _read_snapshot_path(str(snap)) == "/a$b:/c\"d"

    def test_missing_file_is_none(self):
        from tools.environments.local import _read_snapshot_path
        assert _read_snapshot_path("/nonexistent-hermes-snap-118220.sh") is None

    def test_no_path_line_is_none(self, tmp_path):
        from tools.environments.local import _read_snapshot_path
        snap = tmp_path / "s.sh"
        snap.write_text('declare -x HOME="/Users/x"\n', encoding="utf-8")
        assert _read_snapshot_path(str(snap)) is None
