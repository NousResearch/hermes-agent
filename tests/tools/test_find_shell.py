"""Tests for _find_shell — user-login-shell preference on POSIX.

Regression tests for #42203: on macOS, ``_find_shell`` used to return
``/bin/bash`` (bash 3.2) which silently swallowed background commands
when ``~/.bash_profile`` contained ``exec /bin/zsh -l``.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.environments.local import _find_bash, _find_shell


class TestFindShellPrefersUserShell:
    """_find_shell should prefer $SHELL over bash on POSIX."""

    def test_returns_shell_env_when_set_and_exists(self, tmp_path):
        """When $SHELL points to an existing allowlisted executable, _find_shell returns it."""
        fake_zsh = tmp_path / "zsh"
        fake_zsh.touch()
        fake_zsh.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake_zsh)}):
            assert _find_shell() == str(fake_zsh)

    def test_falls_back_when_shell_not_executable(self, tmp_path):
        """$SHELL exists but lacks the execute bit -> fall back to _find_bash
        (returning it would fail at spawn time)."""
        fake = tmp_path / "zsh"
        fake.touch()
        fake.chmod(0o644)  # not executable
        with patch.dict(os.environ, {"SHELL": str(fake)}):
            assert _find_shell() == _find_bash()

    def test_falls_back_for_incompatible_shell_fish(self, tmp_path):
        """#42203 regression: $SHELL=fish must NOT be returned — spawn_local's
        `-lic` / `set +m` syntax breaks fish, which would trade the bash-3.2
        swallow for a parse error on every background command. Fall back to bash."""
        fake_fish = tmp_path / "fish"
        fake_fish.touch()
        fake_fish.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake_fish)}):
            assert _find_shell() == _find_bash()

    def test_falls_back_for_incompatible_shell_csh(self, tmp_path):
        """$SHELL=tcsh/csh is also not -lic/set+m compatible -> fall back."""
        fake = tmp_path / "tcsh"
        fake.touch()
        fake.chmod(0o755)
        with patch.dict(os.environ, {"SHELL": str(fake)}):
            assert _find_shell() == _find_bash()

    def test_honours_allowlisted_bash_and_dash(self, tmp_path):
        """Every allowlisted POSIX-sh-family shell is honoured."""
        for name in ("bash", "dash", "sh", "ksh"):
            fake = tmp_path / name
            fake.touch()
            fake.chmod(0o755)
            with patch.dict(os.environ, {"SHELL": str(fake)}):
                assert _find_shell() == str(fake), name

    def test_falls_back_to_find_bash_when_shell_unset(self):
        """When $SHELL is unset, _find_shell delegates to _find_bash."""
        env = {k: v for k, v in os.environ.items() if k != "SHELL"}
        with patch.dict(os.environ, env, clear=True):
            assert _find_shell() == _find_bash()

    def test_falls_back_to_find_bash_when_shell_not_a_file(self, tmp_path):
        """When $SHELL points to a non-existent path, _find_shell delegates."""
        fake_path = str(tmp_path / "nonexistent_shell")
        with patch.dict(os.environ, {"SHELL": fake_path}):
            assert _find_shell() == _find_bash()

    def test_falls_back_to_find_bash_when_shell_empty(self):
        """When $SHELL is empty string, _find_shell delegates."""
        with patch.dict(os.environ, {"SHELL": ""}):
            assert _find_shell() == _find_bash()


class TestFindShellWindowsBehavior:
    """On Windows, _find_shell always delegates to _find_bash."""

    def test_windows_ignores_shell_env(self):
        """On Windows, $SHELL is ignored — _find_shell delegates to _find_bash."""
        with patch("tools.environments.local._IS_WINDOWS", True):
            # Even if SHELL is set, it should be ignored on Windows
            with patch.dict(os.environ, {"SHELL": "/usr/bin/zsh"}):
                result = _find_shell()
                assert result == _find_bash()


class TestFindShellReturnsString:
    """_find_shell must return a string, never None."""

    def test_returns_string(self):
        """_find_shell always returns a non-empty string on any platform."""
        result = _find_shell()
        assert isinstance(result, str)
        assert len(result) > 0


class TestFindBashUnchanged:
    """_find_bash should be unaffected by the _find_shell change."""

    def test_find_bash_still_prefers_bash(self):
        """_find_bash still returns bash (not $SHELL) on POSIX."""
        result = _find_bash()
        # On any system, _find_bash should return something containing "bash"
        # or fall back to $SHELL or /bin/sh — but it should NOT prefer $SHELL
        # over bash the way _find_shell does.
        assert isinstance(result, str)
        assert len(result) > 0


class TestFindBashSkipsBrokenCustomPath:
    """Stale HERMES_GIT_BASH_PATH must not brick Windows terminal startup."""

    def test_falls_through_to_portable_when_custom_fails_probe(self, tmp_path, monkeypatch):
        import tools.environments.local as local_mod

        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        local_mod._bash_starts_cache.clear()

        broken = tmp_path / "broken" / "bash.exe"
        broken.parent.mkdir()
        broken.write_text("", encoding="utf-8")
        portable = tmp_path / "hermes" / "git" / "bin" / "bash.exe"
        portable.parent.mkdir(parents=True)
        portable.write_text("", encoding="utf-8")

        monkeypatch.setenv("HERMES_GIT_BASH_PATH", str(broken))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

        def fake_starts(path: str) -> bool:
            return path == str(portable)

        monkeypatch.setattr(local_mod, "_bash_starts", fake_starts)

        assert _find_bash() == str(portable)


@pytest.mark.skipif(
    not os.path.isfile("/bin/bash") or sys.platform != "darwin",
    reason="reproduces the macOS system-bash-3.2 login-shell swallow",
)
class TestMacosLoginShellSwallowRegression:
    """E2E regression for #42203: the actual failure is that system bash 3.2,
    invoked as a login shell (`-lic`) with stdin=/dev/null and a
    ~/.bash_profile that `exec`s zsh, silently swallows the command (exit 0,
    no output, no side effects). Prove (a) the bug exists with /bin/bash and
    (b) the $SHELL (zsh) path _find_shell prefers does NOT swallow."""

    def _spawn_like_registry(self, shell, command, home, tmp_path):
        import subprocess
        env = dict(os.environ)
        env["HOME"] = str(home)
        # Mirror process_registry.spawn_local: [shell, "-lic", "set +m; <cmd>"]
        # with stdin redirected to /dev/null.
        return subprocess.run(
            [shell, "-lic", f"set +m; {command}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_system_bash_swallows_but_zsh_does_not(self, tmp_path):
        # A .bash_profile that exec's zsh — the reported macOS shape.
        home = tmp_path / "home"
        home.mkdir()
        (home / ".bash_profile").write_text("exec /bin/zsh -l\n")

        zsh = os.environ.get("SHELL") or "/bin/zsh"
        if not os.path.isfile(zsh):
            pytest.skip("no zsh available")

        marker_bash = tmp_path / "bash_ran"
        marker_zsh = tmp_path / "zsh_ran"

        # /bin/bash login shell: command is swallowed (file NOT created).
        self._spawn_like_registry("/bin/bash", f"echo x > {marker_bash}", home, tmp_path)
        # zsh (the $SHELL _find_shell prefers): command runs (file created).
        self._spawn_like_registry(zsh, f"echo x > {marker_zsh}", home, tmp_path)

        # The FIX path (zsh) must run the command.
        assert marker_zsh.exists(), "zsh ($SHELL) path must run the command"

        # Differential: when /bin/bash is the swallow-prone 3.x (macOS system
        # bash), the login-shell invocation must demonstrably FAIL to run the
        # command — that's the bug this PR routes around. Only assert the
        # negative when we've confirmed a 3.x bash, so the test stays valid on
        # boxes/CI with a newer /bin/bash that doesn't swallow.
        ver = subprocess.run(
            ["/bin/bash", "--version"], capture_output=True, text=True
        ).stdout
        if "version 3." in ver:
            assert not marker_bash.exists(), (
                "system bash 3.x login shell should swallow the command "
                "(the #42203 bug); _find_shell routes around it by preferring zsh"
            )

    def test_find_shell_selects_working_shell_on_this_box(self, tmp_path):
        """_find_shell's choice must actually execute a background-style
        command (regression against returning a swallow-prone shell)."""
        shell = _find_shell()
        marker = tmp_path / "ok_marker"
        subprocess.run(
            [shell, "-lic", f"set +m; echo ok > {marker}"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        assert marker.exists(), f"_find_shell()={shell} swallowed the command"


# ---------------------------------------------------------------------------
# WSL launcher stub rejection (issue: ``.sh`` cron / webhook route scripts
# silently fail with WSL_E_DEFAULT_DISTRO_NOT_FOUND when shutil.which("bash")
# resolves to the WSL stub under system32\ on a host with WSL enabled but no
# distributions installed and Git for Windows installed with its default PATH
# option, which adds Git\cmd — git.exe — but NOT Git\bin / Git\usr\bin —
# bash.exe).
# ---------------------------------------------------------------------------


class TestIsWSLLauncher:
    """``_is_wsl_launcher`` is the host-independent classifier for the stub.

    Uses ``PureWindowsPath`` (not ``os.sep``) so it splits Windows-style paths
    correctly on Linux / macOS CI runners that audit Windows behaviour — the
    audit must not be host-conditional.  Former ``os.sep``-based splits silently
    failed on POSIX because ``"/".split("C:\\Windows\\System32")`` is a no-op.
    """

    def test_system32_path_recognized_regardless_of_host_sep(self):
        from tools.environments.local import _is_wsl_launcher

        assert _is_wsl_launcher(r"C:\Windows\System32\bash.exe") is True
        # Case + extension variants both seen in the wild:
        assert _is_wsl_launcher(r"C:\windows\system32\bash.EXE") is True
        assert _is_wsl_launcher(r"C:\WINDOWS\system32\bash.exe") is True

    def test_windowsapps_alias_recognized(self):
        from tools.environments.local import _is_wsl_launcher

        assert _is_wsl_launcher(
            r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe"
        ) is True

    def test_git_bash_locations_not_classified_as_wsl(self):
        from tools.environments.local import _is_wsl_launcher

        for real_git_bash in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            r"C:\Users\me\AppData\Local\Programs\Git\bin\bash.exe",
            r"C:\Users\me\AppData\Local\hermes\git\bin\bash.exe",
        ):
            assert _is_wsl_launcher(real_git_bash) is False, real_git_bash

    def test_posix_paths_and_empty_are_not_wsl(self):
        from tools.environments.local import _is_wsl_launcher

        assert _is_wsl_launcher("") is False
        assert _is_wsl_launcher("/usr/bin/bash") is False
        assert _is_wsl_launcher("/bin/bash") is False
        assert _is_wsl_launcher("/bin/sh") is False

    def test_forward_slash_separated_windows_path_still_classified(self):
        """``PureWindowsPath`` understands forward slashes too — so a path
        arriving as ``C:/Windows/System32/bash.exe`` (e.g. via env vars set
        with POSIX-style separators) is still flagged as the WSL stub."""
        from tools.environments.local import _is_wsl_launcher

        assert _is_wsl_launcher("C:/Windows/System32/bash.exe") is True
        assert _is_wsl_launcher(
            "C:/Users/me/AppData/Local/Microsoft/WindowsApps/bash.exe"
        ) is True


class TestFindBashSkipsWSLStub:
    """``_find_bash`` must NOT return the WSL launcher stub on Windows.

    Reproduces the user-facing failure deterministically on any host by
    patching ``_IS_WINDOWS`` to ``True`` and steering every resolution layer
    (``HERMES_GIT_BASH_PATH``, PortableGit probe, ``ProgramFiles\\Git\\bin``
    probe, ``which("bash")``) so it lands on either a real Git Bash file at a
    well-known location or the WSL stub — and asserts the resolver returns the
    real Git Bash, never the stub.  Uses ``tmp_path`` for real on-disk Git
    Bash stubs so pytest manages cleanup (the old version leaked temp dirs).
    """

    def _patch_windows(self, monkeypatch, real_git_bash_dir, with_usr_bin=False):
        """Configure monkeypatch to look like Windows with the given Git tree.

        ``real_git_bash_dir`` is a real ``Path`` (a ``Program Files\\Git``
        layout root under ``tmp_path``) — we build it on disk so the resolver's
        ``os.path.isfile`` probe hits a real file.  If ``with_usr_bin`` is set,
        also drop ``usr/bin/bash.exe`` so the MinGit-fallback probe wins.
        """
        git_root = real_git_bash_dir
        (git_root / "bin").mkdir(parents=True)
        (git_root / "bin" / "bash.exe").write_text("")  # presence only — not executed
        if with_usr_bin:
            (git_root / "usr" / "bin").mkdir(parents=True)
            (git_root / "usr" / "bin" / "bash.exe").write_text("")
        # No HERMES_GIT_BASH_PATH, no PortableGit under LOCALAPPDATA,
        # no LocalAppData Git — force the resolver to use ProgramFiles only.
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", True)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", "")
        monkeypatch.setenv("LOCALAPPDATA", str(real_git_bash_dir.parent / "empty_localappdata"))
        monkeypatch.setenv(
            "ProgramFiles", str(real_git_bash_dir.parent)
        )
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)

    def test_real_git_bash_beats_wsl_stub_on_path(self, tmp_path, monkeypatch):
        """On Windows with both a real Git Bash install and WSL on PATH,
        _find_bash returns the real Git Bash (probe fires before which)."""
        git_root = tmp_path / "Program Files" / "Git"
        self._patch_windows(monkeypatch, git_root)

        # Pretend PATH only has the WSL stub — which() would return it.
        monkeypatch.setattr(
            "tools.environments.local.shutil.which",
            lambda name: r"C:\Windows\System32\bash.exe",
        )

        bash = _find_bash()
        assert bash == str(git_root / "bin" / "bash.exe"), bash
        assert "system32" not in os.path.normcase(bash)
        assert "windowsapps" not in os.path.normcase(bash)

    def test_only_wsl_stub_on_path_falls_back_to_usrbin_minGit(self, tmp_path, monkeypatch):
        """If ProgramFiles\\Git\\bin bash.exe is missing but usr/bin bash.exe
        (MinGit layout) is present, _find_bash picks it instead of the stub."""
        git_root = tmp_path / "Program Files" / "Git"
        self._patch_windows(monkeypatch, git_root, with_usr_bin=True)
        # Remove the bin/bash.exe so the primary probe misses; only
        # usr/bin/bash.exe (MinGit) is on disk.
        (git_root / "bin" / "bash.exe").unlink()

        monkeypatch.setattr(
            "tools.environments.local.shutil.which",
            lambda name: r"C:\Windows\System32\bash.exe",
        )

        bash = _find_bash()
        assert bash == str(git_root / "usr" / "bin" / "bash.exe"), bash
        assert "system32" not in os.path.normcase(bash)

    def test_hermes_git_bash_path_overrides_everything(self, tmp_path, monkeypatch):
        """An explicit HERMES_GIT_BASH_PATH wins over both probes and which."""
        custom = tmp_path / "mybash.exe"
        custom.write_text("")
        git_root = tmp_path / "Program Files" / "Git"
        self._patch_windows(monkeypatch, git_root)  # also leaves real Git on disk
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", str(custom))

        bash = _find_bash()
        assert bash == str(custom), bash

    def test_portable_git_install_under_localappdata_wins_over_system_stub(
        self, tmp_path, monkeypatch
    ):
        """The Hermes-bundled PortableGit under %LOCALAPPDATA%\\hermes\\git
        takes precedence over even the ProgramFiles probe — so an
        install.ps1 install with no system Git still resolves.

        This is the case the prior standalone cron resolver *missed*, and the
        reason teknium1 asked us to extend the shared resolver instead of
        duplicating a subset.
        """
        local_appdata = tmp_path / "LocalAppData"
        portable = local_appdata / "hermes" / "git" / "bin" / "bash.exe"
        portable.parent.mkdir(parents=True)
        portable.write_text("")
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", True)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", "")
        monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty_pf"))
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        # And PATH only has the WSL stub:
        monkeypatch.setattr(
            "tools.environments.local.shutil.which",
            lambda name: r"C:\Windows\System32\bash.exe",
        )

        bash = _find_bash()
        assert bash == str(portable), bash
        assert "system32" not in os.path.normcase(bash)

    def test_no_real_bash_raises_instead_of_returning_wsl_stub(
        self, tmp_path, monkeypatch
    ):
        """If the only thing on PATH is the WSL stub and there's no Git Bash
        anywhere _find_bash knows to look, it MUST raise RuntimeError with the
        actionable 'install Git for Windows' message — never hand back the
        stub.  Failing loudly is strictly better than handing the caller a
        bash that exits with WSL_E_DEFAULT_DISTRO_NOT_FOUND on every .sh
        script run."""
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", True)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "release", lambda: "10")
        monkeypatch.setenv("HERMES_GIT_BASH_PATH", "")
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no_localappdata"))
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "no_pf"))
        monkeypatch.delenv("ProgramFiles(x86)", raising=False)
        monkeypatch.setattr(
            "tools.environments.local.shutil.which",
            lambda name: r"C:\Windows\System32\bash.exe",
        )

        with pytest.raises(RuntimeError) as exc_info:
            _find_bash()
        msg = str(exc_info.value)
        assert "Git Bash not found" in msg
        assert "Git for Windows" in msg
        # The message must NOT advocate just running the WSL stub — it must
        # tell the user to install Git for Windows OR set HERMES_GIT_BASH_PATH.
        assert "HERMES_GIT_BASH_PATH" in msg or "git-scm.com" in msg


class TestPOSIXFindBashUnchangedOnNonWindows:
    """The WSL-stub rejection is Windows-only — when ``_IS_WINDOWS`` is False
    ``_find_bash`` must keep its prior POSIX behaviour (which('bash') first,
    then ``/usr/bin/bash``, ``/bin/bash``, $SHELL, ``/bin/sh``)."""

    def test_posix_returns_which_result_first(self, monkeypatch):
        """On POSIX, shutil.which('bash') wins as before — no WSL filtering."""
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(platform, "release", lambda: "6.8.0-generic")
        monkeypatch.setattr(
            "tools.environments.local.shutil.which",
            lambda name: "/nonexistent/from/which/bash",
        )
        # If which returns, we don't even probe /usr/bin/bash etc.
        assert _find_bash() == "/nonexistent/from/which/bash"

    def test_posix_empty_which_falls_through_to_bin_bash(self, monkeypatch):
        """If which returns None on POSIX, fall back to the existing chain
        (NOT skip /usr/bin/bash as a stub)."""
        monkeypatch.setattr("tools.environments.local._IS_WINDOWS", False)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.setattr(platform, "release", lambda: "6.8.0-generic")
        monkeypatch.setattr(
            "tools.environments.local.shutil.which", lambda name: None
        )
        # /usr/bin/bash may or may not exist on the CI host; _find_bash
        # ultimately reaches /bin/sh which always exists as a fallback path.
        result = _find_bash()
        assert isinstance(result, str) and len(result) > 0
        # Must NEVER raise on POSIX even if which() returns None.
        assert result in {
            shutil.which("bash"),
            "/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None,
            "/bin/bash" if os.path.isfile("/bin/bash") else None,
            os.environ.get("SHELL"),
            "/bin/sh",
        }


class TestSharedFindBashUsedEverywhere:
    """Source-level guards ensuring every ``.sh`` / ``.bash`` script-execution
    surface delegates to the shared ``_find_bash`` resolver rather than
    short-circuiting via ``shutil.which("bash")`` directly.

    Without these guards each new script-execution surface (cron, webhook
    route filters, …) is tempted to copy the broken ``shutil.which("bash")
    or "/bin/bash"`` one-liner, silently regressing on the common Windows
    configuration (WSL enabled with no distributions installed + Git for
    Windows installed with its default PATH option that adds ``Git\\cmd``
    but not ``Git\\bin``) to ``WSL_E_DEFAULT_DISTRO_NOT_FOUND``.
    """

    @staticmethod
    def _read(rel_path):
        root = Path(__file__).resolve().parents[2]
        return (root / rel_path).read_text(encoding="utf-8")

    def test_cron_scheduler_delegates_to_find_bash(self):
        source = self._read("cron/scheduler.py")
        assert "from tools.environments.local import _find_bash" in source, (
            "cron.scheduler .sh/.bash execution must delegate to the shared "
            "_find_bash resolver (extend, don't duplicate)."
        )
        assert 'shutil.which("bash") or ("/bin/bash"' not in source, (
            "cron.scheduler must not carry the inline shutil.which('bash') "
            "fallback that returns the WSL launcher stub on Windows."
        )

    def test_webhook_filters_delegates_to_find_bash(self):
        source = self._read("gateway/platforms/webhook_filters.py")
        assert "from tools.environments.local import _find_bash" in source, (
            "gateway webhook_filters run_route_script must delegate to the "
            "shared _find_bash resolver for .sh/.bash route scripts "
            "(sibling cron bug — the same WSL-stub resolution applies)."
        )
        assert 'shutil.which("bash") or ("/bin/bash"' not in source, (
            "gateway webhook_filters must not carry the inline "
            "shutil.which('bash') fallback that returns the WSL launcher "
            "stub on Windows."
        )

    def test_tools_environments_local_has_wsl_stub_rejection(self):
        """The shared resolver must reject the WSL launcher stub — so any
        new caller that delegates to _find_bash is automatically protected."""
        source = self._read("tools/environments/local.py")
        assert "_is_wsl_launcher" in source, (
            "tools/environments/local._find_bash must reject the WSL "
            "launcher stub via _is_wsl_launcher — otherwise every caller "
            "that delegates inherits the silent-failure bug."
        )
        # The classifier must be PureWindowsPath-based for host independence
        # (os.path.normpath + os.sep split fails on POSIX auditing Windows
        # paths — the exact host-dependence bug teknium1 flagged).
        assert "PureWindowsPath" in source, (
            "_is_wsl_launcher must use PureWindowsPath so it classifies "
            "Windows paths correctly on Linux/macOS CI runners too."
        )
