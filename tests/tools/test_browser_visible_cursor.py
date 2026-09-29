"""Tests for the VISIBLE-desktop browser contract.

Covers: headed-by-default resolution (``browser.headed`` / ``AGENT_BROWSER_HEADED``), the
in-page agent cursor overlay (shape of the injected script, enable/disable gating, injection
after an action command, the ``browser_exec`` preamble), window raising, and artifact-path
integrity (ASCII aliases + unverified-claim reporting).
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools import browser_cursor_overlay as overlay
from tools import browser_window_focus as focus


# ---------------------------------------------------------------------------
# headed-by-default resolution
# ---------------------------------------------------------------------------

def _reset_headed_cache():
    import tools.browser_tool as bt
    bt._cached_headed_mode = None
    bt._headed_mode_resolved = False


def _reset_overlay_cache():
    import tools.browser_tool as bt
    bt._cached_cursor_overlay = None
    bt._cursor_overlay_resolved = False


@pytest.fixture(autouse=True)
def _clean_caches(monkeypatch):
    for var in ("AGENT_BROWSER_HEADED", "AGENT_BROWSER_CURSOR_OVERLAY"):
        monkeypatch.delenv(var, raising=False)
    _reset_headed_cache()
    _reset_overlay_cache()
    focus._last_raise.clear()
    yield
    _reset_headed_cache()
    _reset_overlay_cache()
    focus._last_raise.clear()


class TestHeadedDefault:
    def test_absent_key_is_headed(self):
        from tools.browser_tool_cloud import _is_headed_mode
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert _is_headed_mode() is True

    def test_config_false_is_headless(self):
        from tools.browser_tool_cloud import _is_headed_mode
        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"headed": False}}):
            assert _is_headed_mode() is False

    def test_env_false_beats_config_true(self):
        from tools.browser_tool_cloud import _is_headed_mode
        with patch.dict(os.environ, {"AGENT_BROWSER_HEADED": "false"}):
            with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"headed": True}}):
                assert _is_headed_mode() is False

    def test_env_true_beats_config_false(self):
        from tools.browser_tool_cloud import _is_headed_mode
        with patch.dict(os.environ, {"AGENT_BROWSER_HEADED": "1"}):
            with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"headed": False}}):
                assert _is_headed_mode() is True

    def test_resolution_is_cached(self):
        from tools.browser_tool_cloud import _is_headed_mode
        with patch.dict(os.environ, {"AGENT_BROWSER_HEADED": "true"}):
            with patch("hermes_cli.config.read_raw_config", return_value={}):
                assert _is_headed_mode() is True
        os.environ.pop("AGENT_BROWSER_HEADED", None)
        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"headed": False}}) as read:
            assert _is_headed_mode() is True  # cached from the first call
            assert read.call_count == 0


# ---------------------------------------------------------------------------
# cursor overlay: config + script shape
# ---------------------------------------------------------------------------

class TestOverlayConfig:
    def test_default_enabled(self):
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert overlay.cursor_overlay_enabled() is True

    def test_config_false_disables(self):
        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"cursor_overlay": False}}):
            assert overlay.cursor_overlay_enabled() is False

    def test_env_beats_config(self):
        with patch.dict(os.environ, {"AGENT_BROWSER_CURSOR_OVERLAY": "false"}):
            with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"cursor_overlay": True}}):
                assert overlay.cursor_overlay_enabled() is False


class TestOverlayScript:
    def test_is_single_line_and_ascii(self):
        source = overlay.overlay_js_source()
        assert "\n" not in source
        assert source.isascii()
        assert source.strip().endswith(")()")

    def test_never_intercepts_events_and_stays_out_of_the_a11y_tree(self):
        source = overlay.overlay_js_source()
        assert "pointer-events:none" in source
        assert 'setAttribute("aria-hidden", "true")' in source
        assert 'setAttribute("role", "presentation")' in source
        assert "z-index:2147483647" in source

    def test_animates_move_click_and_typing(self):
        source = overlay.overlay_js_source()
        assert "translate3d(" in source            # GPU-friendly movement
        assert "240ms cubic-bezier" in source      # smooth, eased move
        assert "requestAnimationFrame" in source   # coalesced frames
        assert "PointerEvents" not in source       # no obvious typo leftover
        assert "prefers-reduced-motion" in source
        for event in ("mousemove", "mousedown", "mouseup", "keydown", "input", "focusin"):
            assert f'"{event}"' in source, event
        assert "function pulse" in source and "function typing" in source

    def test_install_is_idempotent(self):
        source = overlay.overlay_js_source()
        assert "hermes-cursor-present" in source        # early return when already installed
        assert 'document.getElementById(ROOT)' in source
        assert "hermes-cursor-installed" in source
        assert "DOMContentLoaded" in source             # document-start safety


class TestOverlayGating:
    def _session(self, **extra):
        return {"session_name": "s1", **extra}

    def test_injects_after_action_command(self):
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True), \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=True), \
             patch.object(overlay._origin(), "_is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_session._run_browser_command") as run:
            overlay.after_browser_command("task1", "click", self._session(), "auto")
        assert run.call_count == 1
        args, kwargs = run.call_args
        assert args[0] == "task1" and args[1] == "eval"
        assert args[2] == [overlay.overlay_js_source()]
        assert kwargs["_skip_overlay"] is True

    @pytest.mark.parametrize("command", ["snapshot", "get", "screenshot", "close", "console"])
    def test_read_only_commands_skip_injection(self, command):
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True), \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=True), \
             patch.object(overlay._origin(), "_is_camofox_mode", return_value=False), \
             patch("tools.browser_tool_session._run_browser_command") as run:
            overlay.after_browser_command("task1", command, self._session(), "auto")
        assert run.call_count == 0

    def test_disabled_by_config(self):
        with patch.object(overlay, "cursor_overlay_enabled", return_value=False), \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=True), \
             patch("tools.browser_tool_session._run_browser_command") as run:
            overlay.after_browser_command("task1", "click", self._session(), "auto")
        assert run.call_count == 0

    def test_headless_has_nothing_to_animate(self):
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True), \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=False), \
             patch("tools.browser_tool_session._run_browser_command") as run:
            overlay.after_browser_command("task1", "click", self._session(), "auto")
        assert run.call_count == 0

    def test_lightpanda_has_no_renderer(self):
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True), \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=True), \
             patch("tools.browser_tool_session._run_browser_command") as run:
            overlay.after_browser_command("task1", "click", self._session(), "lightpanda")
        assert run.call_count == 0


# ---------------------------------------------------------------------------
# injection actually reaches the agent-browser argv
# ---------------------------------------------------------------------------

_SNAPSHOT_STDOUT = ('{"success": true, "data": {"snapshot": "- heading \\"Hi\\" [ref=e1]", '
                    '"refs": {"e1": {}}}}')


def _run_and_capture(command):
    """Run one command with Popen mocked; return the list of spawned argv lists."""
    captured = []
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    def capture_popen(cmd, **kwargs):
        captured.append(cmd)
        return mock_proc

    with patch("subprocess.Popen", side_effect=capture_popen), \
         patch("os.open", return_value=99), \
         patch("os.close"), \
         patch("os.unlink"), \
         patch("os.makedirs"), \
         patch("builtins.open", MagicMock(return_value=MagicMock(
             __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=_SNAPSHOT_STDOUT))),
             __exit__=MagicMock(return_value=False),
         ))), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("tools.browser_tool_lifecycle._write_owner_pid"):
        from tools import browser_tool_session as bt_session
        bt_session._run_browser_command("task1", command, [], _engine_override="auto")
    return captured


@patch("tools.browser_tool_session._get_session_info")
@patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser")
@patch("tools.browser_tool_cloud._is_local_mode", return_value=True)
@patch("tools.browser_tool_install._chromium_installed", return_value=True)
@patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None)
@patch("tools.browser_tool_cdp._get_cdp_override", return_value="")
@patch("tools.browser_tool._is_camofox_mode", return_value=False)
class TestArgvWiring:
    def _prime(self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session):
        import tools.browser_tool as bt
        bt._cached_headed_mode = True
        bt._headed_mode_resolved = True
        _session.return_value = {"session_name": "test-sess"}

    def test_headed_flag_is_default(self, *_mocks):
        self._prime(*_mocks)
        captured = _run_and_capture("snapshot")
        assert "--headed" in captured[0]

    def test_action_command_adds_the_overlay_eval(self, *_mocks):
        self._prime(*_mocks)
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True), \
             patch("tools.browser_window_focus.bring_session_window_to_front"):
            captured = _run_and_capture("open")
        assert len(captured) == 2, captured
        assert "--headed" in captured[0]
        overlay_call = captured[1]
        assert overlay_call[-2] == "eval"
        assert overlay.overlay_js_source() in overlay_call

    def test_read_only_command_does_not_pay_for_the_overlay(self, *_mocks):
        self._prime(*_mocks)
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True):
            captured = _run_and_capture("snapshot")
        assert len(captured) == 1

    def test_overlay_injection_does_not_recurse(self, *_mocks):
        self._prime(*_mocks)
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True), \
             patch.object(overlay, "should_inject", return_value=True), \
             patch("tools.browser_window_focus.bring_session_window_to_front"):
            captured = _run_and_capture("click")
        # one clicked command + exactly one injection, never a third (the eval's own hook)
        assert len(captured) == 2, captured
        assert captured[1][-2] == "eval"


# ---------------------------------------------------------------------------
# Browser Use (browser_exec) path
# ---------------------------------------------------------------------------

class TestBrowserExecOverlayPreamble:
    def test_enabled_preamble_registers_and_installs(self):
        from tools import browser_use_cli as bu
        with patch.object(overlay, "cursor_overlay_enabled", return_value=True):
            preamble = bu._cursor_overlay_preamble()
        assert "Page.addScriptToEvaluateOnNewDocument" in preamble
        assert "js(_js)" in preamble
        assert overlay.overlay_js_source() in preamble
        assert "hermes-cursor-overlay-%s-%s" in preamble  # once-per-daemon registration guard
        compile(preamble, "<preamble>", "exec")           # must be valid Python

    def test_disabled_preamble_is_empty(self):
        from tools import browser_use_cli as bu
        with patch.object(overlay, "cursor_overlay_enabled", return_value=False):
            assert bu._cursor_overlay_preamble() == ""


# ---------------------------------------------------------------------------
# window raising
# ---------------------------------------------------------------------------

_WIN_NETSTAT = (
    "  Proto  Local Address          Foreign Address        State           PID\r\n"
    "  TCP    127.0.0.1:9222         0.0.0.0:0              LISTENING       12345\r\n"
    "  TCP    [::]:9222              [::]:0                 LISTENING       12345\r\n"
    "  TCP    127.0.0.1:9223         0.0.0.0:0              LISTENING       999\r\n"
)


class TestWindowFocus:
    def test_netstat_pid_parsing(self):
        import sys
        with patch.object(sys, "platform", "win32"):
            pids = focus._listening_pids_for_port(9222, runner=lambda argv: _WIN_NETSTAT)
        assert pids == [12345]

    def test_lsof_pid_parsing(self):
        import sys
        with patch.object(sys, "platform", "linux"):
            pids = focus._listening_pids_for_port(9222, runner=lambda argv: "4242\n4242\n")
        assert pids == [4242]

    def test_raise_is_throttled_per_session(self):
        with patch.object(focus, "_session_cdp_port", return_value=9222), \
             patch.object(focus, "_listening_pids_for_port", return_value=[12345]), \
             patch.object(focus, "_window_handles_for_pid", return_value=[777]), \
             patch.object(focus, "_raise_window", return_value=True):
            assert focus.bring_session_window_to_front("s1") is True
            assert focus.bring_session_window_to_front("s1") is False  # throttled
            assert focus.bring_session_window_to_front("s2") is True

    def test_unknown_port_is_a_noop(self):
        with patch.object(focus, "_session_cdp_port", return_value=0):
            assert focus.bring_session_window_to_front("s1") is False

    def test_only_screen_changing_commands_raise(self):
        session = {"session_name": "s1"}
        with patch.object(focus, "bring_session_window_to_front") as raise_win, \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=True), \
             patch("tools.browser_tool_cloud._get_browser_engine", return_value="auto"):
            focus.after_browser_command("open", session)
            focus.after_browser_command("snapshot", session)
            raise_win.assert_called_once_with("s1")

    def test_cloud_and_cdp_sessions_are_never_raised(self):
        with patch.object(focus, "bring_session_window_to_front") as raise_win, \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=True):
            focus.after_browser_command("open", {"session_name": "s1", "cdp_url": "wss://x/y"})
        raise_win.assert_not_called()

    def test_headless_never_raises(self):
        with patch.object(focus, "bring_session_window_to_front") as raise_win, \
             patch("tools.browser_tool_cloud._is_headed_mode", return_value=False):
            focus.after_browser_command("open", {"session_name": "s1"})
        raise_win.assert_not_called()


# ---------------------------------------------------------------------------
# artifact integrity (reported screenshot paths)
# ---------------------------------------------------------------------------

class TestArtifactIntegrity:
    def test_non_ascii_name_gets_ascii_alias(self, tmp_path):
        from tools import browser_use_cli as bu
        src = tmp_path / "podgl\u0105d_przegl\u0105darki.png"
        src.write_bytes(b"png")
        alias = bu._ascii_alias(str(src))
        assert alias is not None
        assert alias.isascii()
        assert Path(alias).is_file()
        assert Path(alias).name == "podglad_przegladarki.png"
        assert src.is_file()  # the original is kept

    def test_ascii_name_is_left_alone(self, tmp_path):
        from tools import browser_use_cli as bu
        src = tmp_path / "shot.png"
        src.write_bytes(b"png")
        assert bu._ascii_alias(str(src)) is None

    def test_missing_file_is_not_aliased(self, tmp_path):
        from tools import browser_use_cli as bu
        assert bu._ascii_alias(str(tmp_path / "podgl\u0105d.png")) is None

    def test_missing_artifacts_flags_unverified_claims(self, tmp_path):
        from tools import browser_use_cli as bu
        present = tmp_path / "real.png"
        present.write_bytes(b"png")
        gone = tmp_path / "podgl\u0105d_przegl\u0105darki.png"
        stdout = f"Saved {present}\nSaved {gone}\n"
        missing = bu._missing_artifacts(stdout)
        assert missing == [str(gone)]

    def test_browser_exec_reports_missing_artifact(self, tmp_path):
        from tools import browser_use_cli as bu
        gone = str(tmp_path / "podgl\u0105d_przegl\u0105darki.png")
        proc = subprocess.CompletedProcess(["browser-use"], 0, f"print('{gone}')\n{gone}\n", "")
        with patch.object(bu, "_find_cli", return_value=["browser-use"]), \
             patch.object(bu, "_base_subprocess_env", return_value={}), \
             patch.object(bu, "_route_backend", return_value=None), \
             patch.object(bu, "_cursor_overlay_preamble", return_value=""), \
             patch.object(bu, "_workspace_dir", return_value=None), \
             patch.object(bu, "_run_cli_killing_process_group", return_value=proc):
            raw = bu.browser_exec("print('ok')")
        payload = json.loads(raw)
        assert payload["missing_artifacts"] == [gone]
        assert "do not exist on disk" in payload["warning"]
        assert "screenshot_path" not in payload

    def test_browser_exec_reports_verified_screenshot_only(self, tmp_path):
        from tools import browser_use_cli as bu
        shot = tmp_path / "podgl\u0105d_przegl\u0105darki.png"
        shot.write_bytes(b"png")
        proc = subprocess.CompletedProcess(["browser-use"], 0, f"{shot}\n", "")
        with patch.object(bu, "_find_cli", return_value=["browser-use"]), \
             patch.object(bu, "_base_subprocess_env", return_value={}), \
             patch.object(bu, "_route_backend", return_value=None), \
             patch.object(bu, "_cursor_overlay_preamble", return_value=""), \
             patch.object(bu, "_workspace_dir", return_value=None), \
             patch.object(bu, "_native_screenshot_result", return_value=None), \
             patch.object(bu, "_run_cli_killing_process_group", return_value=proc):
            raw = bu.browser_exec("print('shot')")
        payload = json.loads(raw)
        reported = payload["screenshot_path"]
        assert reported.isascii() and Path(reported).is_file()
        assert payload["screenshot_source_path"] == str(shot)
        assert "missing_artifacts" not in payload


# ---------------------------------------------------------------------------
# the overlay must never alter what the agent "sees"
# ---------------------------------------------------------------------------

def test_overlay_root_is_hidden_from_snapshots():
    """An ``aria-hidden`` fixed element is dropped from Chromium's a11y tree."""
    source = overlay.overlay_js_source()
    assert 'aria-hidden", "true"' in source
    assert "role" in source and "presentation" in source
    # and it is never part of the document flow the snapshot walks
    assert "position:fixed" in source


def test_decorate_hook_swallows_everything():
    from tools import browser_tool_session as bt_session
    with patch.object(overlay, "after_browser_command", side_effect=RuntimeError("boom")), \
         patch.object(focus, "after_browser_command", side_effect=RuntimeError("boom")):
        bt_session._decorate_visible_desktop("t", "open", SimpleNamespace(), "auto")  # must not raise