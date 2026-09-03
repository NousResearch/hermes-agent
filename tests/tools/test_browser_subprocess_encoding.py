"""Regression tests for UTF-8 subprocess decoding across browser tools (issue #102500).

On Windows with a non-UTF-8 code page (e.g. cp1252), subprocess.run(..., text=True)
without explicit encoding decodes child stdout using the locale ANSI code page.
When browser-use, agent-browser, or lightpanda output UTF-8 non-ASCII characters
(emojis, accents, symbols), the reader thread crashes with UnicodeDecodeError,
causing subprocess.run to silently return stdout=None.
Adding encoding="utf-8", errors="replace" ensures robust decoding and prevents
silent null output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

import hermes_cli.browser_connect as bc
import tools.browser_lightpanda as lp
import tools.browser_tool as bt
import tools.browser_use_cli as bu_cli


class TestBrowserExecEncoding:
    def test_browser_exec_passes_utf8_and_replace_to_subprocess(self, monkeypatch):
        """browser_exec must explicitly set encoding='utf-8' and errors='replace'."""
        captured_kwargs = {}

        def _mock_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(
                args=args[0] if args else kwargs.get("args"),
                returncode=0,
                stdout="PAGE_OUTPUT \U0001F434 \u00b7 caf\u00e9\n",
                stderr="",
            )

        monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["browser-use"])
        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", lambda *a, **kw: None)
        monkeypatch.setattr(subprocess, "run", _mock_run)

        res_str = bu_cli.browser_exec("print(1)")
        result = json.loads(res_str)

        assert captured_kwargs.get("text") is True
        assert captured_kwargs.get("encoding") == "utf-8"
        assert captured_kwargs.get("errors") == "replace"
        assert result["success"] is True
        assert "\U0001F434" in result["output"]
        assert "caf\u00e9" in result["output"]

    def test_browser_exec_real_subprocess_non_ascii_roundtrip(self, monkeypatch):
        """End-to-end execution of a child process emitting UTF-8 non-ASCII output."""
        payload = "TITLE \U0001F434 \u00b7 caf\u00e9"
        child_cmd = [
            sys.executable,
            "-c",
            "import sys, io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'); print("
            + repr(payload)
            + ")",
        ]

        monkeypatch.setattr(bu_cli, "_find_cli", lambda: child_cmd)
        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", lambda *a, **kw: None)

        res_str = bu_cli.browser_exec("ignored_by_mock_child")
        result = json.loads(res_str)

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["output"] is not None
        assert "TITLE \U0001F434 \u00b7 caf\u00e9" in result["output"]

    def test_browser_exec_real_subprocess_invalid_bytes_replaced(self, monkeypatch):
        """Invalid UTF-8 bytes are cleanly replaced with replacement character without crashing."""
        child_cmd = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'HELLO \\xff\\xfe WORLD\\n'); sys.stdout.buffer.flush()",
        ]

        monkeypatch.setattr(bu_cli, "_find_cli", lambda: child_cmd)
        monkeypatch.setattr(bu_cli, "_resolve_backend_cdp", lambda *a, **kw: None)

        res_str = bu_cli.browser_exec("ignored")
        result = json.loads(res_str)

        assert result["success"] is True
        assert result["output"] is not None
        assert "HELLO" in result["output"]
        assert "WORLD" in result["output"]
        assert "\ufffd" in result["output"]


class TestBrowserToolEncoding:
    def test_agent_browser_get_cdp_passes_utf8_and_replace(self, monkeypatch):
        captured_kwargs = {}

        def _mock_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(
                args=args[0] if args else kwargs.get("args"),
                returncode=0,
                stdout="ws://127.0.0.1:9222/devtools/browser/abc \U0001F434\n",
                stderr="",
            )

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_agent_browser_argv", lambda cmd: [cmd])
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(subprocess, "run", _mock_run)

        url = bt._agent_browser_get_cdp("sess1")
        assert url == "http://127.0.0.1:9222"
        assert captured_kwargs.get("text") is True
        assert captured_kwargs.get("encoding") == "utf-8"
        assert captured_kwargs.get("errors") == "replace"

    def test_agent_browser_close_session_passes_utf8_and_replace(self, monkeypatch):
        captured_kwargs = {}

        def _mock_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(
                args=args[0] if args else kwargs.get("args"),
                returncode=0,
                stdout="closed\n",
                stderr="",
            )

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_agent_browser_argv", lambda cmd: [cmd])
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(subprocess, "run", _mock_run)

        bt._agent_browser_close_session("sess1")
        assert captured_kwargs.get("text") is True
        assert captured_kwargs.get("encoding") == "utf-8"
        assert captured_kwargs.get("errors") == "replace"

    def test_real_profile_cdp_launch_passes_utf8_and_replace(self, monkeypatch):
        captured_kwargs = {}

        def _mock_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(
                args=args[0] if args else kwargs.get("args"),
                returncode=0,
                stdout="ok \U0001F434\n",
                stderr="",
            )

        monkeypatch.setattr(bt, "_use_real_profile", lambda: True)
        monkeypatch.setattr(bt, "_using_lightpanda_engine", lambda: False)
        cdp_calls = []
        def _mock_get_cdp(sess):
            if not cdp_calls:
                cdp_calls.append(1)
                return None
            return "http://127.0.0.1:9222"
        monkeypatch.setattr(bt, "_agent_browser_get_cdp", _mock_get_cdp)
        monkeypatch.setattr(bt, "_cdp_http_ready", lambda cdp: True)
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_agent_browser_argv", lambda cmd: [cmd])
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(subprocess, "run", _mock_run)

        monkeypatch.setattr(bc, "detect_default_chromium", lambda: "chrome")
        monkeypatch.setattr(bc, "real_profile_copy_dir", lambda b: "/tmp/hermes-profile")
        monkeypatch.setattr(bc, "snapshot_real_profile", lambda b: ("/tmp/hermes-profile", None))
        monkeypatch.setattr(bc, "chromium_executable", lambda b: "/bin/chrome")

        fake_popen = MagicMock()
        fake_popen.poll.return_value = None
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: fake_popen)

        with patch("builtins.open", mock_open(read_data="9222\nws://127.0.0.1:9222/devtools/browser\n")):
            cdp_url, err = bt._real_profile_cdp()

        assert err is None
        assert captured_kwargs.get("text") is True
        assert captured_kwargs.get("encoding") == "utf-8"
        assert captured_kwargs.get("errors") == "replace"


class TestBrowserLightpandaEncoding:
    def test_binary_supports_http_cache_passes_utf8_and_replace(self, monkeypatch):
        captured_kwargs = {}

        def _mock_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return subprocess.CompletedProcess(
                args=args[0] if args else kwargs.get("args"),
                returncode=0,
                stdout="lightpanda 0.3.0 --http-cache-dir \U0001F434\n",
                stderr="",
            )

        lp._binary_supports_http_cache.cache_clear()
        monkeypatch.setattr(subprocess, "run", _mock_run)

        supported = lp._binary_supports_http_cache("/usr/local/bin/lightpanda")
        assert supported is True
        assert captured_kwargs.get("text") is True
        assert captured_kwargs.get("encoding") == "utf-8"
        assert captured_kwargs.get("errors") == "replace"
