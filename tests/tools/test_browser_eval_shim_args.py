"""agent-browser CLI arguments must survive the Windows ``.cmd`` shim hop (#113838): cmd.exe
re-parses the child command line, so a newline truncates an argument to its first line and
``%VAR%`` expands. Host-specific bug — these are code-path invariants that run on every host."""

from __future__ import annotations

import base64
import importlib
import json
import pkgutil

import tools
from tools import browser_tool_session as bt_session

_SCRIPT = "JSON.stringify(\n  [...document.images].map(i => i.src)\n)"
_TEXT = "line one\nline two %PATH% 100%"
_SHIM = r"C:\Users\u\AppData\Roaming\npm\agent-browser.CMD"


def _run_through_seam(monkeypatch, browser_cmd: str, command: str, args: list, stdout: str) -> tuple:
    """Drive ``_run_browser_command`` up to the spawn with ``_spawn_and_collect`` captured; returns
    ``(cmd_parts, stdin_payload, result)`` so the wiring — not just the helper — is under test."""
    captured: dict = {}

    def fake_spawn(task_id, session_info, cmd_parts, cmd, engine, timeout, stdin_payload=None):
        captured.update(cmd_parts=cmd_parts, stdin=stdin_payload)
        return bt_session._interpret_browser_command_output(cmd, stdout, "", 0)

    monkeypatch.setattr(bt_session, "_browser_command_preflight", lambda: {"browser_cmd": browser_cmd})
    monkeypatch.setattr(bt_session, "_get_session_info", lambda task_id: {"session_name": "s", "cdp_url": None})
    monkeypatch.setattr(bt_session._cloud, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(bt_session._cloud, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(bt_session, "_spawn_and_collect", fake_spawn)
    result = bt_session._run_browser_command("t", command, args, timeout=5)
    return captured["cmd_parts"], captured["stdin"], result


def test_cmd_shim_routes_eval_and_multiline_text_losslessly(monkeypatch):
    """Through ``_run_browser_command``: a ``.cmd`` argv[0] sends ``eval`` as ``--base64 <script>``
    and any other command whose argv carries a newline or ``%`` as ``batch`` with the command as
    JSON on stdin (cmd.exe never parses the text), with the batch entry unwrapped to the usual
    ``{success, data, error}`` shape; plain arguments and non-shim spawns keep the raw argv."""
    parts, stdin, _ = _run_through_seam(monkeypatch, _SHIM, "eval", [_SCRIPT], '{"success":true,"data":{"result":1}}')
    assert parts[-3:-1] == ["eval", "--base64"] and stdin is None
    assert base64.b64decode(parts[-1]).decode("utf-8") == _SCRIPT  # byte-identical round trip

    batch_out = json.dumps([{"command": ["fill", "@e3", _TEXT], "error": None,
                             "result": {"filled": "@e3"}, "success": True}])
    parts, stdin, result = _run_through_seam(monkeypatch, _SHIM, "fill", ["@e3", _TEXT], batch_out)
    assert parts[-1] == "batch" and not any(("\n" in p or "%" in p) for p in parts)
    assert json.loads(stdin) == [["fill", "@e3", _TEXT]]
    assert result == {"success": True, "data": {"filled": "@e3"}, "error": None}

    parts, stdin, _ = _run_through_seam(monkeypatch, _SHIM, "fill", ["@e3", "one line"], '{"success":true,"data":{}}')
    assert parts[-3:] == ["fill", "@e3", "one line"] and stdin is None
    for argv0 in (r"C:\tools\agent-browser.exe", "/home/u/.local/bin/agent-browser"):
        parts, stdin, _ = _run_through_seam(monkeypatch, argv0, "fill", ["@e3", _TEXT], '{"success":true,"data":{}}')
        assert parts[-3:] == ["fill", "@e3", _TEXT] and stdin is None


def test_bundled_js_eval_payload_constants_are_single_line():
    """Every ``*_JS`` string constant in ``tools/browser_*`` is one line — the fixed
    ``_GET_IMAGES_JS`` and any future payload handed to the CLI ``eval`` as one argv element."""
    seen = []
    for mod_info in pkgutil.iter_modules(tools.__path__):
        if not mod_info.name.startswith("browser_"):
            continue
        module = importlib.import_module(f"tools.{mod_info.name}")
        for name, value in vars(module).items():
            if name.endswith("_JS") and isinstance(value, str):
                seen.append((mod_info.name, name))
                assert "\n" not in value, f"tools/{mod_info.name}.py::{name} spans multiple lines"
    assert ("browser_tool", "_GET_IMAGES_JS") in seen
