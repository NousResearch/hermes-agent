"""MSYS POSIX-drive spelling of the home folds like every other spelling of it.

``/c/Users/alice/x``, ``C:/Users/alice/x`` and ``~/x`` name the same file. The home-prefix
fold knew only the Windows dialect, so an MSYS-spelled write — the dialect this host's bash
terminal writes natively — reached no home rule at all: ``sed -i`` on the Hermes config (the
approvals policy itself), on a sibling profile's config, or on a shell rc ran ungated under
``approvals.single_query_mode: deny`` while the ``C:/`` spelling of the identical command was
flagged.

The home is pinned through ``HOME`` so the test means the same thing on every host: a POSIX CI
box resolves no drive-letter home at all, and a Windows box's real home is a different name.
Detection itself is not host-gated — see ``tests/tools/test_approval_windows.py``, which pins
the same contract for the Windows path patterns (a Linux-hosted Hermes can drive a Windows box
over SSH).
"""
from __future__ import annotations

import pytest

from tools import approval_detection

detect_dangerous_command = approval_detection.detect_dangerous_command

# A Windows-shaped home that is nobody's real home, so the fold resolves it identically on
# POSIX (expanduser reads HOME) and on Windows (the explicit $HOME entry in the fold list).
HOME = r"C:\Users\msysdialect"
MSYS = "/c/Users/msysdialect"          # what git-bash writes
MSYS_UPPER = "/C/Users/msysdialect"    # the letter case MSYS also accepts
WINDOWS = "C:/Users/msysdialect"       # native forward-slash form
TILDE = "~"                            # the folded form the rules are written against
ROOT = "AppData/Local/hermes"
PROFILE = "other-profile"              # a SIBLING profile, never the active one


@pytest.fixture(autouse=True)
def _pinned_home(monkeypatch):
    monkeypatch.setenv("HOME", HOME)


def _verdict(command: str) -> tuple[bool, str | None]:
    flagged, key, _ = detect_dangerous_command(command)
    return bool(flagged), key


def _flagged(command: str) -> bool:
    return _verdict(command)[0]


class TestMsysHomeSpellingIsFlagged:
    """The three writes the fold was missing, in the dialect the shell actually uses."""

    @pytest.mark.parametrize("drive", [MSYS, MSYS_UPPER, WINDOWS])
    def test_root_config_write(self, drive):
        assert _flagged(f"sed -i 's/a/b/' {drive}/{ROOT}/config.yaml")

    @pytest.mark.parametrize("drive", [MSYS, MSYS_UPPER, WINDOWS])
    def test_sibling_profile_config_write(self, drive):
        assert _flagged(f"sed -i 's/a/b/' {drive}/{ROOT}/profiles/{PROFILE}/config.yaml")

    @pytest.mark.parametrize("name", [".bashrc", ".zshrc", ".profile"])
    def test_shell_rc_write(self, name):
        assert _flagged(f"sed -i 's/a/b/' {MSYS}/{name}")

    @pytest.mark.parametrize("name", [".netrc", ".npmrc", ".pgpass"])
    def test_credential_file_write(self, name):
        for drive in (MSYS, MSYS_UPPER, WINDOWS):
            assert _flagged(f"cp evil.txt {drive}/{name}"), f"{drive}/{name}"


class TestAlreadyFlaggedSpellingsStayFlagged:
    """Coverage that did NOT need the fold must not lose it."""

    @pytest.mark.parametrize("drive", [MSYS, MSYS_UPPER, WINDOWS])
    @pytest.mark.parametrize("rel", [f"{ROOT}/.env", f"{ROOT}/profiles/{PROFILE}/.env"])
    def test_env_read(self, drive, rel):
        assert _flagged(f"cat {drive}/{rel}")

    @pytest.mark.parametrize("drive", [MSYS, MSYS_UPPER, WINDOWS])
    def test_ssh_key_write(self, drive):
        """A write into ``~/.ssh`` is gated by the POSIX rule once the prefix folds."""
        assert _flagged(f"sed -i 's/a/b/' {drive}/.ssh/authorized_keys")


class TestDialectParity:
    """The same command must classify the same in every dialect of the same path.

    This is the contract the missing fold broke: ``C:/`` and ``~`` were flagged while ``/c/``
    was not, and after the fix the ``/c/`` read lands on the verdict ``C:/`` and ``~`` already
    had (probe_guard's C10 pins ``type ~/.ssh/id_rsa`` as must-not-flag, so the read half of
    the Windows ``Users/<x>/.ssh`` rule is unreachable for the owner's own key on every
    spelling — key *writes* stay gated, above).
    """

    @pytest.mark.parametrize("rel", [
        f"{ROOT}/config.yaml",
        f"{ROOT}/profiles/{PROFILE}/config.yaml",
        ".bashrc",
        ".ssh/authorized_keys",
    ])
    def test_write_verdicts_match(self, rel):
        spellings = [f"{MSYS}/{rel}", f"{MSYS_UPPER}/{rel}", f"{WINDOWS}/{rel}", f"{TILDE}/{rel}"]
        verdicts = [_flagged(f"sed -i 's/a/b/' {s}") for s in spellings]
        assert len(set(verdicts)) == 1, f"dialects disagree: {dict(zip(spellings, verdicts))}"
        assert verdicts[0], f"should be flagged in every dialect: {rel}"

    def test_ssh_key_read_parity(self):
        """Reads of the owner's own key: all three dialects agree."""
        spellings = [f"{MSYS}/.ssh/id_rsa", f"{WINDOWS}/.ssh/id_rsa", f"{TILDE}/.ssh/id_rsa"]
        verdicts = [_flagged(f"cat {s}") for s in spellings]
        assert len(set(verdicts)) == 1, f"dialects disagree: {dict(zip(spellings, verdicts))}"

    def test_bare_home_never_folds_in_any_dialect(self):
        """A bare home has no tail for the fold to anchor on — deliberately unchanged, and now
        consistent across dialects rather than Windows-only."""
        spellings = [MSYS, WINDOWS, TILDE]
        verdicts = [_flagged(f"sed -i 's/a/b/' {s}") for s in spellings]
        assert len(set(verdicts)) == 1, f"dialects disagree: {dict(zip(spellings, verdicts))}"


class TestControlsDoNotFlip:
    """Paths that must stay unflagged: no drive, no home, a benign relative operand."""

    @pytest.mark.parametrize("command", [
        "sed -i 's/a/b/' /srv/app/config.yaml",
        "cp config.yaml backup.yaml",
        "cat ./readme.txt",
        "cat /c/Users/msysdialect/notes.txt",
        "ls /c/Users",
        "echo reboot the gateway please",
    ])
    def test_not_flagged(self, command):
        assert not _flagged(command), f"should NOT be flagged: {command}"


class TestDriveDialectSpellings:
    """The dialect map itself: two ways to name one directory, nothing else rewritten."""

    @pytest.mark.parametrize("path,expected", [
        (r"C:\Users\alice", ["/c/Users/alice", "/C/Users/alice"]),
        ("C:/Users/alice", ["/c/Users/alice", "/C/Users/alice"]),
        (r"d:\work\repo", ["/d/work/repo", "/D/work/repo"]),
        ("/c/Users/alice", ["C:/Users/alice", "c:/Users/alice"]),
        ("/D/work/repo", ["D:/work/repo", "d:/work/repo"]),
        # no drive letter -> nothing to translate (POSIX hosts, "", degenerate input)
        ("/home/alice", []),
        ("/tmp", []),
        ("", []),
        (r"\\server\share\dir", []),
        ("relative/path", []),
    ])
    def test_spellings(self, path, expected):
        assert approval_detection._drive_dialect_spellings(path) == expected
