"""
Regression tests for command-provider interpreter isolation.

Command TTS/STT providers (``tts.providers.<name>.type: "command"`` and
the STT equivalent) run user-configured shell templates that pin their own
interpreter (e.g. ``~/.chatterbox-venv/bin/python wrapper.py``). These
child interpreters are frequently a DIFFERENT CPython version than the
gateway's, so a ``PYTHONPATH`` entry built for the gateway interpreter
(launcher-injected site-packages shims, layered Docker filesystems) is a
foreign-version hazard: the child imports the gateway-version compiled
extension (``PIL._imaging``, ``numpy._core``, ``cryptography``) and
crashes at startup.

The existing Hermes-owned PYTHONPATH scrub
(``_strip_hermes_owned_pythonpath``) deliberately preserves such entries
(user-owned provenance, #74817 follow-up) — correct for generic children,
wrong for pinned-interpreter command providers where the template names
its interpreter explicitly and PYTHONPATH can only hurt.

Regression for the local-whisper STT crash reported against 3.13-shim
PYTHONPATH + 3.11 provider venv (ImportError: cannot import name
'_imaging' from 'PIL').
"""

import os
import sys
from unittest.mock import patch

from tools.tts_tool import _run_command_tts


class _Stream:
    def read(self, size):
        return ""


class _Proc:
    returncode = 0
    stdout = _Stream()
    stderr = _Stream()

    def wait(self, timeout=None):
        return 0


class TestCommandTtsStripsForeignPythonpath:
    def test_command_tts_strips_pythonpath_from_child_env(self, monkeypatch):
        """Command TTS providers must not inherit PYTHONPATH at all.

        The template pins its own interpreter; a launcher-injected
        PYTHONPATH built for the gateway interpreter makes a different-
        version child import foreign compiled extensions and crash.
        """
        monkeypatch.setenv("PYTHONPATH", "/opt/hermes-shims/py313/site-packages")

        captured = {}

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return _Proc()

        with patch("tools.tts_tool.subprocess.Popen", fake_popen):
            result = _run_command_tts("echo hi", timeout=1)

        assert result.returncode == 0
        assert "PYTHONPATH" not in captured["env"]

    def test_command_tts_env_scrub_is_unaffected(self, monkeypatch):
        """The PYTHONPATH strip must not weaken the #56332 secret scrub."""
        monkeypatch.setenv("PYTHONPATH", "/some/foreign/shims")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("MY_SAFE_TTS_VAR", "keep")

        captured = {}

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return _Proc()

        with patch("tools.tts_tool.subprocess.Popen", fake_popen):
            _run_command_tts("echo hi", timeout=1)

        env = captured["env"]
        assert "OPENAI_API_KEY" not in env
        assert env["MY_SAFE_TTS_VAR"] == "keep"

    def test_command_tts_without_pythonpath_unchanged(self, monkeypatch):
        """No PYTHONPATH set: child env is built exactly as before."""
        monkeypatch.delenv("PYTHONPATH", raising=False)
        monkeypatch.setenv("MY_SAFE_TTS_VAR", "keep")

        captured = {}

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return _Proc()

        with patch("tools.tts_tool.subprocess.Popen", fake_popen):
            _run_command_tts("echo hi", timeout=1)

        env = captured["env"]
        assert "PYTHONPATH" not in env
        assert env["MY_SAFE_TTS_VAR"] == "keep"

    def test_env_passthrough_still_works_alongside_strip(self, monkeypatch):
        """env_passthrough allowlist entries survive the PYTHONPATH strip."""
        monkeypatch.setenv("PYTHONPATH", "/foreign/shims")
        monkeypatch.setenv("MY_TTS_KEY", "tts-key-value")

        captured = {}

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return _Proc()

        with patch("tools.tts_tool.subprocess.Popen", fake_popen):
            _run_command_tts("echo hi", timeout=1, env_passthrough=["MY_TTS_KEY"])

        assert captured["env"]["MY_TTS_KEY"] == "tts-key-value"
