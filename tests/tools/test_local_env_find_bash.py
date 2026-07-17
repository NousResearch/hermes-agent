"""Tests for ``_find_bash()`` Windows candidate selection and fallback ordering.

The runtime, Desktop preflight, and installer script each maintain an
independent bash candidate list.  These tests verify the runtime list stays
complete and correctly ordered across all package-manager install locations.
"""

import os
from unittest.mock import patch

import pytest

from tools.environments import local as local_mod
from tools.environments.local import _find_bash


# ---------------------------------------------------------------------------
# POSIX — skip the candidate hunt entirely
# ---------------------------------------------------------------------------

class TestFindBashPosix:
    def test_returns_shutil_which_bash_on_linux(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        with patch("shutil.which", return_value="/bin/bash"):
            assert _find_bash() == "/bin/bash"

    def test_falls_back_to_standard_paths_on_linux(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=lambda p: p == "/usr/bin/bash"):
            assert _find_bash() == "/usr/bin/bash"


# ---------------------------------------------------------------------------
# Windows — candidate selection and ordering
# ---------------------------------------------------------------------------

class TestFindBashWindows:
    def test_hermes_git_bash_path_env_wins_over_everything(self, monkeypatch, tmp_path):
        """HERMES_GIT_BASH_PATH is the escape hatch — when set and valid it
        must return before probing any hard-coded location."""
        custom_bash = tmp_path / "custom-bash.exe"
        custom_bash.write_text("")
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", str(custom_bash))

        # shutil.which must NOT be called when HERMES_GIT_BASH_PATH is valid.
        with patch("shutil.which", side_effect=AssertionError("should not be called")):
            assert _find_bash() == str(custom_bash)

    def test_hermes_git_bash_path_ignored_when_file_missing(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", r"C:\does\not\exist\bash.exe")

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", return_value=False):
            with pytest.raises(RuntimeError, match="Git Bash not found"):
                _find_bash()

    def test_portable_git_checked_before_system_paths(self, monkeypatch):
        """Portable Git (installed by install.ps1) must win over a system
        Git in Program Files."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

        portable = r"C:\Users\tester\AppData\Local\hermes\git\bin\bash.exe"
        system = r"C:\Program Files\Git\bin\bash.exe"

        # Portable is "present" and gets checked first — return it before
        # probing system locations.
        probe_order = []
        def isfile(p):
            probe_order.append(p)
            return p == portable  # only portable "exists"

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == portable
        # Portable checked first — system paths never even reached
        assert portable in probe_order
        assert system not in probe_order

    def test_shutil_which_consulted_before_hardcoded_paths(self, monkeypatch):
        """``shutil.which('bash')`` is the PATH fallback and must be tried
        before any hard-coded candidates (Program Files, Scoop, etc.)."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        with patch("shutil.which", return_value=r"C:\tools\bash.exe") as mock_which, \
             patch("os.path.isfile", return_value=False):
            assert _find_bash() == r"C:\tools\bash.exe"
            mock_which.assert_called_once_with("bash")

    # -- Package-manager candidate tests -------------------------------------

    def test_scoop_path_found(self, monkeypatch):
        """Scoop installs Git at
        ``%USERPROFILE%\\scoop\\apps\\git\\current\\usr\\bin\\bash.exe``."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

        scoop_path = r"C:\Users\tester\scoop\apps\git\current\usr\bin\bash.exe"

        def isfile(p):
            return p == scoop_path

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == scoop_path

    def test_chocolatey_path_found(self, monkeypatch):
        """Chocolatey installs Git at
        ``C:\\ProgramData\\chocolatey\\lib\\git\\tools\\git\\bin\\bash.exe``."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")

        choco_path = r"C:\ProgramData\chocolatey\lib\git\tools\git\bin\bash.exe"

        def isfile(p):
            return p == choco_path

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == choco_path

    def test_msys2_path_found(self, monkeypatch):
        """MSYS2 installs bash at ``C:\\msys64\\usr\\bin\\bash.exe``."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")

        msys2_path = r"C:\msys64\usr\bin\bash.exe"

        def isfile(p):
            return p == msys2_path

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == msys2_path

    # -- Ordering tests ------------------------------------------------------

    def test_system_git_beats_scoop_when_both_present(self, monkeypatch):
        """Program Files Git (explicitly installed) takes priority over a
        Scoop-managed Git."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

        system_path = r"C:\Program Files\Git\bin\bash.exe"
        scoop_path = r"C:\Users\tester\scoop\apps\git\current\usr\bin\bash.exe"

        def isfile(p):
            return p in (system_path, scoop_path)

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == system_path

    def test_scoop_beats_chocolatey_when_both_present(self, monkeypatch):
        """Scoop appears before Chocolatey in the candidate list."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

        scoop_path = r"C:\Users\tester\scoop\apps\git\current\usr\bin\bash.exe"
        choco_path = r"C:\ProgramData\chocolatey\lib\git\tools\git\bin\bash.exe"

        def isfile(p):
            return p in (scoop_path, choco_path)

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == scoop_path

    def test_chocolatey_beats_msys2_when_both_present(self, monkeypatch):
        """Chocolatey appears before MSYS2 in the candidate list."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

        choco_path = r"C:\ProgramData\chocolatey\lib\git\tools\git\bin\bash.exe"
        msys2_path = r"C:\msys64\usr\bin\bash.exe"

        def isfile(p):
            return p in (choco_path, msys2_path)

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", side_effect=isfile):
            assert _find_bash() == choco_path

    def test_fallback_ordering_complete(self, monkeypatch):
        """End-to-end ordering verification: HERMES_GIT_BASH_PATH >
        PortableGit (bin) > PortableGit (usr/bin) > shutil.which('bash') >
        Program Files > Program Files (x86) > LocalAppData Programs >
        Scoop > Chocolatey > MSYS2."""
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

        seen = []

        def isfile_with_tracking(p):
            result = p in _fbs_paths
            if result:
                seen.append(p)
            return result

        _fbs_paths = {
            # PortableGit
            r"C:\Users\tester\AppData\Local\hermes\git\bin\bash.exe",
            r"C:\Users\tester\AppData\Local\hermes\git\usr\bin\bash.exe",
            # shutil.which is checked before hard-coded paths
            r"C:\on\path\bash.exe",
            # Standard Git for Windows
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            r"C:\Users\tester\AppData\Local\Programs\Git\bin\bash.exe",
            # Package managers
            r"C:\Users\tester\scoop\apps\git\current\usr\bin\bash.exe",
            r"C:\ProgramData\chocolatey\lib\git\tools\git\bin\bash.exe",
            r"C:\msys64\usr\bin\bash.exe",
        }

        with patch("shutil.which", return_value=r"C:\on\path\bash.exe"), \
             patch("os.path.isfile", side_effect=isfile_with_tracking):
            assert _find_bash() == r"C:\Users\tester\AppData\Local\hermes\git\bin\bash.exe"
            # PortableGit won — nothing else was probed
            assert seen == [r"C:\Users\tester\AppData\Local\hermes\git\bin\bash.exe"]

    def test_raises_when_nothing_found(self, monkeypatch):
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        monkeypatch.delenv("HERMES_GIT_BASH_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")
        monkeypatch.setenv("USERPROFILE", r"C:\Users\tester")

        with patch("shutil.which", return_value=None), \
             patch("os.path.isfile", return_value=False):
            with pytest.raises(RuntimeError, match="Git Bash not found"):
                _find_bash()
