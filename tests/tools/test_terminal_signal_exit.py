"""Tests for signal-termination exit code interpretation.

Ported from Kilo-Org/kilocode#12698 ("settle signal-terminated shell
commands as 128 + signum"): the model must see a human-readable note for
signal deaths instead of a bare exit_code=-9 / 137 it burns turns
mis-diagnosing.
"""

import pytest

from tools.terminal_tool import _interpret_exit_code, _interpret_signal_exit


class TestInterpretSignalExit:
    # ---- negative codes: subprocess -signum semantics (definite) ----

    @pytest.mark.parametrize("code,expect", [
        (-9, "SIGKILL"),
        (-11, "SIGSEGV"),
        (-15, "SIGTERM"),
        (-6, "SIGABRT"),
        (-8, "SIGFPE"),
        (-13, "SIGPIPE"),
    ])
    def test_negative_known_signals(self, code, expect):
        note = _interpret_signal_exit(code)
        assert note is not None
        assert expect in note
        assert "terminated by" in note.lower()

    def test_negative_oom_mentions_oom(self):
        note = _interpret_signal_exit(-9)
        assert "OOM" in note

    def test_negative_unknown_signal_still_reports(self):
        # signum without a curated note still yields a generic note.
        note = _interpret_signal_exit(-31)
        assert note is not None
        assert "31" in note

    # ---- 128+signum band: shell convention (hedged) ----

    @pytest.mark.parametrize("code,expect", [
        (137, "SIGKILL"),
        (139, "SIGSEGV"),
        (143, "SIGTERM"),
        (134, "SIGABRT"),
        (141, "SIGPIPE"),
    ])
    def test_shell_band_known_signals(self, code, expect):
        note = _interpret_signal_exit(code)
        assert note is not None
        assert expect in note
        # Hedged: a program can legitimately exit with these codes.
        assert "usually" in note

    def test_shell_band_uncurated_signum_returns_none(self):
        # 128+signum for a signum outside the curated table must stay
        # silent — we never guess on ambiguous application exit codes.
        assert _interpret_signal_exit(128 + 40) is None

    # ---- exclusions ----

    def test_sigint_130_excluded(self):
        # rc=130 has bespoke interrupt-marker handling in the executor.
        assert _interpret_signal_exit(130) is None
        assert _interpret_signal_exit(-2) is None

    @pytest.mark.parametrize("code", [0, 1, 2, 42, 100, 127, 128])
    def test_normal_codes_return_none(self, code):
        assert _interpret_signal_exit(code) is None


class TestSignalExitWiring:
    """_interpret_exit_code surfaces signal notes and keeps old semantics."""

    def test_signal_note_via_interpret_exit_code(self):
        note = _interpret_exit_code("python3 crash.py", -11)
        assert note is not None and "SIGSEGV" in note

    def test_shell_band_via_interpret_exit_code(self):
        note = _interpret_exit_code("./run_build.sh", 137)
        assert note is not None and "SIGKILL" in note

    def test_signal_note_wins_over_command_semantics(self):
        # A grep killed by SIGKILL must report the signal, not "no matches".
        note = _interpret_exit_code("grep foo huge.log", 137)
        assert "SIGKILL" in note

    def test_grep_exit_1_unchanged(self):
        note = _interpret_exit_code("grep foo bar.txt", 1)
        assert note is not None and "no matches" in note.lower()

    def test_success_unchanged(self):
        assert _interpret_exit_code("ls", 0) is None
