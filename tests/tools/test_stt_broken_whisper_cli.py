"""A discovered `whisper` CLI that cannot run must not count as a local STT backend.

Regression for #122981: on macOS with a tools-managed Python 3.14, openai-whisper
installs but cannot import torch (no prebuilt wheel), so the `whisper` binary dies with
``ModuleNotFoundError: No module named 'torch'``. Resolution trusted mere existence and
committed every voice message to the broken binary instead of falling through to a
working backend.
"""

import os
import stat
import subprocess

import pytest

from tools import transcription_local, transcription_tools


@pytest.fixture
def isolated(monkeypatch):
    """No faster-whisper, no explicit template, empty probe cache."""
    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", False)
    monkeypatch.delenv("HERMES_LOCAL_STT_COMMAND", raising=False)
    cache = getattr(transcription_local, "_LOCAL_COMMAND_PROBE_CACHE", None)
    if cache is not None:
        cache.clear()
    return monkeypatch


def _fake_run(monkeypatch, *, returncode, stderr=""):
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(transcription_local.subprocess, "run", _run)
    monkeypatch.setattr(transcription_local, "_find_whisper_binary",
                        lambda: "/fake/bin/whisper")
    return calls


def test_broken_whisper_cli_is_not_a_backend(isolated):
    """The reporter's exact signature: openai-whisper CLI dying on missing torch."""
    _fake_run(isolated, returncode=1,
              stderr="ModuleNotFoundError: No module named 'torch'")
    assert transcription_local._has_local_command() is False
    assert transcription_tools._detect_local_backend() is None
    assert transcription_tools._resolve_explicit_local() == "none"


def test_working_whisper_cli_is_a_backend(isolated):
    _fake_run(isolated, returncode=0)
    assert transcription_local._has_local_command() is True
    assert transcription_tools._detect_local_backend() == "local_command"


def test_probe_crash_is_unavailable(isolated):
    def _run(cmd, **kwargs):
        raise OSError("exec failed")

    isolated.setattr(transcription_local.subprocess, "run", _run)
    isolated.setattr(transcription_local, "_find_whisper_binary",
                     lambda: "/fake/bin/whisper")
    assert transcription_local._has_local_command() is False


def test_explicit_template_bypasses_probe(isolated):
    """A user-configured HERMES_LOCAL_STT_COMMAND is an explicit choice: never probed."""
    calls = _fake_run(isolated, returncode=1, stderr="boom")
    isolated.setenv("HERMES_LOCAL_STT_COMMAND", "my-stt {input_path}")
    assert transcription_local._has_local_command() is True
    assert calls == []


@pytest.mark.skipif(os.name != "posix", reason="needs a POSIX executable bit")
def test_live_broken_whisper_binary_on_path(isolated, tmp_path):
    """End to end with a real executable that fails like the reporter's openai-whisper."""
    script = tmp_path / "whisper"
    script.write_text("#!/bin/sh\necho \"ModuleNotFoundError: No module named 'torch'\" >&2\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    isolated.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    assert transcription_local._find_whisper_binary() is not None
    assert transcription_local._has_local_command() is False
    assert transcription_tools._detect_local_backend() is None


@pytest.mark.skipif(os.name != "posix", reason="needs a POSIX executable bit")
def test_live_working_whisper_binary_on_path(isolated, tmp_path):
    script = tmp_path / "whisper"
    script.write_text("#!/bin/sh\necho \"whisper 1.0\"\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    isolated.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    assert transcription_local._has_local_command() is True
    assert transcription_tools._detect_local_backend() == "local_command"
