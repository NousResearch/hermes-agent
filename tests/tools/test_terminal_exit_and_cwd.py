"""Tests for exit-code interpretation, command-cwd resolution, and session-cwd helpers.

Covers ``_interpret_exit_code`` / ``_interpret_signal_exit`` (the human-readable
note added when a non-zero exit is expected), ``_resolve_command_cwd`` (the
``workdir`` > session-record > default resolution ladder, container-aware),
the ``_session_cwd`` record helpers, and the small env/cwd utility predicates
(``_parse_env_var``, ``_safe_getcwd``, ``_is_container_backend``,
``_is_unusable_container_cwd``).

Semantics asserted here are pinned against the live implementation, not a
spec: e.g. ``_parse_env_var`` *raises* on a malformed value (it does not fall
back to the default), and ``_safe_getcwd`` falls back to ``$HOME`` /
``TERMINAL_CWD`` (not ``/``).
"""

import json
import os

import pytest

import tools.terminal_tool as tt


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch):
    """Isolate each test from any shared session-cwd / override state."""
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})


class TestInterpretSignalExit:
    """_interpret_signal_exit: map signal terminations (negative & 128+signum)."""

    # -- negative codes: subprocess -signum semantics (definite signal death) --

    @pytest.mark.parametrize("code,expect", [
        (-3, "SIGQUIT"),
        (-4, "SIGILL"),
        (-6, "SIGABRT"),
        (-7, "SIGBUS"),
        (-8, "SIGFPE"),
        (-9, "SIGKILL"),
        (-11, "SIGSEGV"),
        (-13, "SIGPIPE"),
        (-15, "SIGTERM"),
        (-24, "SIGXCPU"),
        (-25, "SIGXFSZ"),
    ])
    def test_negative_curated_signals(self, code, expect):
        note = tt._interpret_signal_exit(code)
        assert note is not None
        assert expect in note
        assert "terminated by signal" in note

    def test_negative_oom_killer_note(self):
        # Killed by the kernel OOM killer (kill -9) must surface that hint.
        note = tt._interpret_signal_exit(-9)
        assert note is not None
        assert "OOM" in note

    def test_negative_unknown_signum_uses_name(self):
        # Signum without a curated note still yields a note; a platform-known
        # signal resolves to its name (e.g. 31 -> SIGUSR2 on macOS).
        note = tt._interpret_signal_exit(-31)
        assert note is not None
        assert "31" in note

    def test_negative_out_of_range_signum_generic(self):
        # Signum 65 is not a valid Signals member on common platforms: the
        # ValueError fallback yields a generic "signal 65" note, never raising.
        note = tt._interpret_signal_exit(-65)
        assert note is not None
        assert "65" in note

    def test_negative_sigint_excluded(self):
        # SIGINT has bespoke interrupt-marker handling in the executor.
        assert tt._interpret_signal_exit(-2) is None

    # -- 128+signum band: shell convention (hedged with "usually") --

    @pytest.mark.parametrize("code,expect", [
        (131, "SIGQUIT"),
        (132, "SIGILL"),
        (134, "SIGABRT"),
        (136, "SIGFPE"),
        (137, "SIGKILL"),
        (139, "SIGSEGV"),
        (141, "SIGPIPE"),
        (143, "SIGTERM"),
        (152, "SIGXCPU"),
        (153, "SIGXFSZ"),
    ])
    def test_shell_band_curated_signals(self, code, expect):
        note = tt._interpret_signal_exit(code)
        assert note is not None
        assert expect in note
        assert "usually" in note

    def test_shell_band_uncurated_signum_returns_none(self):
        # A signum outside the curated table is never guessed: an app could
        # legitimately exit with that code, so stay silent.
        assert tt._interpret_signal_exit(128 + 20) is None
        assert tt._interpret_signal_exit(128 + 65) is None

    def test_shell_band_sigint_excluded(self):
        assert tt._interpret_signal_exit(130) is None

    # -- normal / non-signal codes --

    @pytest.mark.parametrize("code", [0, 1, 2, 42, 100, 127, 128])
    def test_normal_codes_return_none(self, code):
        assert tt._interpret_signal_exit(code) is None


class TestInterpretExitCode:
    """_interpret_exit_code: non-erroneous non-zero exit notes (and None for errors)."""

    def test_exit_zero_returns_none(self):
        assert tt._interpret_exit_code("anything", 0) is None

    # -- command-specific semantics --

    @pytest.mark.parametrize("base", [
        "grep foo bar.txt", "egrep foo bar.txt", "fgrep foo bar.txt",
        "rg foo", "ag foo", "ack foo", "/usr/bin/grep foo", "/usr/local/bin/rg foo",
    ])
    def test_grep_family_exit_one_is_no_matches(self, base):
        assert tt._interpret_exit_code(base, 1) == "No matches found (not an error)"

    def test_grep_exit_two_is_error(self):
        # grep exit 2 is a real error (e.g. file not found) — no note.
        assert tt._interpret_exit_code("grep foo bar.txt", 2) is None

    @pytest.mark.parametrize("base", ["diff a b", "colordiff a b"])
    def test_diff_exit_one_is_files_differ(self, base):
        assert tt._interpret_exit_code(base, 1) == "Files differ (expected, not an error)"

    def test_diff_exit_two_is_error(self):
        assert tt._interpret_exit_code("diff a b", 2) is None

    def test_find_exit_one(self):
        assert tt._interpret_exit_code("find . -name x", 1) == (
            "Some directories were inaccessible (partial results may still be valid)"
        )

    @pytest.mark.parametrize("base", ["test -f foo", "[ -f foo ]"])
    def test_test_bracket_exit_one(self, base):
        assert tt._interpret_exit_code(base, 1) == (
            "Condition evaluated to false (expected, not an error)"
        )

    @pytest.mark.parametrize("code,expect", [
        (6, "Could not resolve host"),
        (7, "Failed to connect to host"),
        (22, "HTTP response code indicated error (e.g. 404, 500)"),
        (28, "Operation timed out"),
    ])
    def test_curl_codes(self, code, expect):
        assert tt._interpret_exit_code("curl -s http://example.com", code) == expect

    def test_curl_unlisted_code_is_none(self):
        assert tt._interpret_exit_code("curl -s http://example.com", 5) is None

    def test_git_exit_one(self):
        assert tt._interpret_exit_code("git diff", 1) == (
            "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"
        )

    # -- pipeline / chain: exit code comes from the LAST segment --

    def test_pipeline_last_segment_drives_semantics(self):
        assert tt._interpret_exit_code("find . -name x | grep foo", 1) == (
            "No matches found (not an error)"
        )

    def test_chain_and_then_last_segment(self):
        assert tt._interpret_exit_code("echo hi && grep foo", 1) == (
            "No matches found (not an error)"
        )

    def test_semicolon_last_segment_rules(self):
        # Last segment is `ls` — no semantics -> None even though grep ran first.
        assert tt._interpret_exit_code("grep foo a.txt; ls", 1) is None

    def test_env_assignment_prefix_still_resolves_to_command(self):
        assert tt._interpret_exit_code("VAR=val grep foo", 1) == (
            "No matches found (not an error)"
        )

    def test_only_env_assignment_has_no_command(self):
        assert tt._interpret_exit_code("FOO=bar", 1) is None

    # -- signal notes win over command semantics --

    def test_signal_note_wins_over_grep_semantics(self):
        note = tt._interpret_exit_code("grep foo huge.log", 137)
        assert note is not None
        assert "SIGKILL" in note

    # -- unknown command + non-zero -> None --

    @pytest.mark.parametrize("cmd", ["unknowncmd", "ls --weird", "./build.sh"])
    def test_unknown_command_nonzero_returns_none(self, cmd):
        assert tt._interpret_exit_code(cmd, 3) is None


def env_type_from(container):
    return "docker" if container else "local"


class TestResolveCommandCwd:
    """_resolve_command_cwd: workdir > session record > default (container-aware)."""

    def _resolve(self, monkeypatch, *, workdir, default_cwd, recorded="<none>",
                 container=False, unusable=False):
        """Call _resolve_command_cwd with the three seam functions mocked."""
        def fake_get(key):
            return None if recorded == "<none>" else recorded

        monkeypatch.setattr(tt, "get_session_cwd", fake_get)
        monkeypatch.setattr(tt, "_is_container_backend", lambda env: container)
        monkeypatch.setattr(tt, "_is_unusable_container_cwd", lambda cwd: unusable)
        return tt._resolve_command_cwd(
            workdir=workdir, default_cwd=default_cwd,
            session_key="sess-a", env_type=env_type_from(container),
        )

    def test_workdir_overrides_everything(self, monkeypatch):
        assert self._resolve(monkeypatch, workdir="/explicit", default_cwd="/def",
                             recorded="/recorded") == "/explicit"

    def test_record_beats_default(self, monkeypatch):
        assert self._resolve(monkeypatch, workdir=None, default_cwd="/def",
                             recorded="/recorded") == "/recorded"

    def test_no_record_uses_default(self, monkeypatch):
        assert self._resolve(monkeypatch, workdir=None, default_cwd="/def") == "/def"

    def test_container_unusable_record_falls_back_to_default(self, monkeypatch):
        # Host path recorded on a container backend must not be used (exit 126).
        assert self._resolve(monkeypatch, workdir=None, default_cwd="/def",
                             recorded="/home/user", container=True, unusable=True) == "/def"

    def test_container_usable_record_is_kept(self, monkeypatch):
        # A sandbox-native recorded cwd is fine even on a container backend.
        assert self._resolve(monkeypatch, workdir=None, default_cwd="/def",
                             recorded="/workspace", container=True, unusable=False) == "/workspace"

    def test_non_container_unusable_record_is_kept(self, monkeypatch):
        # On a non-container backend the container-unusability guard is moot.
        assert self._resolve(monkeypatch, workdir=None, default_cwd="/def",
                             recorded="/home/user", container=False, unusable=True) == "/home/user"


class TestSessionCwdStore:
    """record_session_cwd / get_session_cwd / clear_session_cwd round-trip."""

    def test_record_get_clear_cycle(self):
        tt.record_session_cwd("sess-a", "/start")
        assert tt.get_session_cwd("sess-a") == "/start"
        tt.record_session_cwd("sess-a", "/moved")
        assert tt.get_session_cwd("sess-a") == "/moved"
        tt.clear_session_cwd("sess-a")
        assert tt.get_session_cwd("sess-a") is None

    def test_none_key_collapses_to_default(self):
        tt.record_session_cwd(None, "/defaulted")
        assert tt.get_session_cwd(None) == "/defaulted"
        assert tt.get_session_cwd("default") == "/defaulted"

    def test_sessions_are_isolated(self):
        tt.record_session_cwd("sess-a", "/a")
        tt.record_session_cwd("sess-b", "/b")
        assert tt.get_session_cwd("sess-a") == "/a"
        assert tt.get_session_cwd("sess-b") == "/b"
        assert tt.get_session_cwd("sess-c") is None

    @pytest.mark.parametrize("bad", ["", "   ", None, 0, 123])
    def test_non_string_or_empty_cwd_is_ignored(self, bad):
        tt.record_session_cwd("sess-a", "/keep")
        tt.record_session_cwd("sess-a", bad)
        assert tt.get_session_cwd("sess-a") == "/keep"

    def test_clear_only_drops_named_session(self):
        tt.record_session_cwd("sess-a", "/a")
        tt.record_session_cwd("sess-b", "/b")
        tt.clear_session_cwd("sess-a")
        assert tt.get_session_cwd("sess-a") is None
        assert tt.get_session_cwd("sess-b") == "/b"


class TestParseEnvVar:
    """_parse_env_var: converts cleanly, raises on malformed values."""

    def test_valid_int(self):
        assert tt._parse_env_var("TT_TEST", "180") == 180

    def test_valid_converter(self, monkeypatch):
        monkeypatch.setenv("TT_TEST", '["a", "b"]')
        assert tt._parse_env_var("TT_TEST", "[]", converter=json.loads,
                                 type_label="valid JSON") == ["a", "b"]

    def test_not_set_uses_default(self, monkeypatch):
        monkeypatch.delenv("TT_TEST", raising=False)
        assert tt._parse_env_var("TT_TEST", "180") == 180

    def test_invalid_raises_with_name_and_label(self, monkeypatch):
        monkeypatch.setenv("TT_TEST", "5m")
        with pytest.raises(ValueError, match="TT_TEST"):
            tt._parse_env_var("TT_TEST", "180")

    def test_empty_string_int_raises(self, monkeypatch):
        # An empty-but-set value is NOT treated as absent: int("") raises.
        monkeypatch.setenv("TT_TEST", "")
        with pytest.raises(ValueError):
            tt._parse_env_var("TT_TEST", "180")

    def test_empty_string_str_converter_is_empty(self, monkeypatch):
        monkeypatch.setenv("TT_TEST", "")
        assert tt._parse_env_var("TT_TEST", "180", converter=str, type_label="string") == ""

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setenv("TT_TEST", "not json")
        with pytest.raises(ValueError, match="valid JSON"):
            tt._parse_env_var("TT_TEST", "[]", converter=json.loads, type_label="valid JSON")


class TestSafeGetcwd:
    """_safe_getcwd: tolerate deleted / TCC-blocked CWD, propagate unrelated OSError."""

    def test_normal_getcwd_passthrough(self):
        assert tt._safe_getcwd() == os.getcwd()

    def test_file_not_found_uses_home(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(
            FileNotFoundError(2, "No such file or directory")))
        assert tt._safe_getcwd() == os.path.expanduser("~")

    def test_permission_error_uses_home(self, monkeypatch):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(
            PermissionError(1, "Operation not permitted")))
        assert tt._safe_getcwd() == os.path.expanduser("~")

    def test_terminal_cwd_beats_home(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "/custom/from/env")
        monkeypatch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(
            PermissionError(1, "Operation not permitted")))
        assert tt._safe_getcwd() == "/custom/from/env"

    def test_unrelated_oserror_propagates(self, monkeypatch):
        # NotADirectoryError is an OSError but NOT caught deliberately — good
        # callers must see the real problem rather than a silent fallback.
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(
            NotADirectoryError(20, "not a directory")))
        with pytest.raises(OSError):
            tt._safe_getcwd()


class TestIsContainerBackend:
    """_is_container_backend: built-in sandbox backends, local/ssh excluded."""

    @pytest.mark.parametrize("env_type", ["docker", "singularity", "modal",
                                          "daytona", "vercel_sandbox"])
    def test_container_backends_true(self, env_type):
        assert tt._is_container_backend(env_type) is True

    @pytest.mark.parametrize("env_type", ["local", "ssh", "managed_modal", "", None])
    def test_non_container_backends_false(self, env_type):
        assert tt._is_container_backend(env_type) is False


class TestIsUnusableContainerCwd:
    """_is_unusable_container_cwd: host/relative paths can't be a sandbox workdir."""

    @pytest.mark.parametrize("cwd", ["/workspace", "/workspace/proj", "/root",
                                     "/root/sub", "/data", "/tmp/scratch"])
    def test_sandbox_paths_are_usable(self, cwd):
        assert tt._is_unusable_container_cwd(cwd) is False

    @pytest.mark.parametrize("cwd", [
        "/home/user", "/Users/alice/work", "C:\\Users\\me", "C:/Users/me", ".",
        "src/", "../up", "relative/path",
    ])
    def test_host_or_relative_paths_are_unusable(self, cwd):
        assert tt._is_unusable_container_cwd(cwd) is True

    def test_empty_cwd_is_not_unusable(self):
        assert tt._is_unusable_container_cwd("") is False
