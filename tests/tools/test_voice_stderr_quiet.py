"""_StderrQuiet: PortAudio noise is captured, real errors still reach the terminal.

The bug this covers: a stream rebuilt while another holds the mic makes PortAudio print
``PaMacCore (AUHAL)|| Error on line 2523: err='-50'`` from C, straight onto fd 2, which truncates an
open clarify panel's border.

Assertions use ``capfd`` (fd-level), not ``capsys``: the noise is written with ``os.write(2, ...)``,
which bypasses ``sys.stderr`` entirely — the same reason no Python logging config can suppress it.
"""

import logging
import os
import sys

import pytest

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="fd 2 redirect is macOS-only")


def _quiet_class():
    from tools.voice_mode import _StderrQuiet

    return _StderrQuiet


def _write_noise():
    os.write(2, b"||PaMacCore (AUHAL)|| Error on line 2523: err='-50, msg=Unknown Error\n")


def _write_real_error():
    os.write(2, b"RuntimeError: something real\n")


@darwin_only
def test_portaudio_noise_never_reaches_the_terminal(capfd, caplog):
    """The whole point: the AUHAL line must not be written to fd 2."""
    with caplog.at_level(logging.DEBUG, logger="tools.voice_mode"):
        with _quiet_class()("audio input stream"):
            _write_noise()
    assert "PaMacCore" not in capfd.readouterr().err, "PortAudio noise still hit the terminal"


@darwin_only
def test_noise_is_kept_as_a_debug_record(capfd, caplog):
    """Suppressed from the UI, not thrown away: the diagnosis stays recoverable."""
    with caplog.at_level(logging.DEBUG, logger="tools.voice_mode"):
        with _quiet_class()("audio input stream"):
            _write_noise()
    assert any("PaMacCore" in r.getMessage() for r in caplog.records), "noise was silently dropped"


@darwin_only
def test_real_errors_still_pass_through(capfd):
    """The blind spot that made the naive version unsafe: a genuine error must survive."""
    with _quiet_class()("audio stream close"):
        _write_real_error()
    err = capfd.readouterr().err
    assert "RuntimeError: something real" in err, f"a real error was swallowed: {err!r}"


@darwin_only
def test_mixed_window_keeps_only_the_noise_quiet(capfd):
    """Both kinds in one window: PortAudio goes quiet, the real error comes through."""
    with _quiet_class()("probe"):
        _write_real_error()
        _write_noise()
    err = capfd.readouterr().err
    assert "something real" in err
    assert "PaMacCore" not in err


@darwin_only
def test_fd2_is_writable_afterwards(capfd):
    """A leaked redirect would send every later error into a closed temp file — audited not assumed."""
    with _quiet_class()("probe"):
        _write_noise()
    os.write(2, b"AFTER_THE_BLOCK\n")
    assert "AFTER_THE_BLOCK" in capfd.readouterr().err, "fd 2 was not restored"


@darwin_only
def test_unusable_fds_do_not_break_audio(monkeypatch, capfd):
    """A failure to redirect must degrade to 'noisy but working', never raise into the recorder."""
    real_dup = os.dup

    def fail_only_for_stderr(fd, *args, **kwargs):
        # Only fd 2: pytest's own capture calls os.dup(1), which must keep working.
        if fd == 2:
            raise OSError("no fds")
        return real_dup(fd, *args, **kwargs)

    monkeypatch.setattr(os, "dup", fail_only_for_stderr)
    quiet = _quiet_class()("probe")
    with quiet:  # must not raise
        _write_noise()
    assert quiet._tmp is None
    assert "PaMacCore" in capfd.readouterr().err, "redirect failed but the noise vanished too"
    # Undo before the test ends: pytest's capture also dups fd 2 during teardown.
    monkeypatch.undo()


def test_non_macos_is_a_no_op(monkeypatch, capfd):
    """Linux/ALSA does not print this way, so the redirect must not run there."""
    monkeypatch.setattr(sys, "platform", "linux")
    with _quiet_class()("probe"):
        _write_noise()
    assert "PaMacCore" in capfd.readouterr().err, "redirect ran off macOS"
