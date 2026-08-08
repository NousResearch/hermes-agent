"""Tests for the Linux/X11 uinput XInput-leak refusal guard on cua-driver
input actions.

Protects against trycua/cua#2618 (cua-driver <0.13.1 can leak XInput master
pointers when /dev/uinput is not read+write accessible on Linux/X11), fixed
upstream by trycua/cua#2631 (cua-driver-rs-v0.13.1). Hermes #74148.

Input actions (click/drag/scroll/type_text/key/set_value) must refuse BEFORE
declaring an input session or invoking the driver whenever the known-unsafe
combination is detected, returning a structured ActionResult with a stable
code and actionable message. Capture/list/inspection must remain unaffected,
and every other combination (fixed driver, non-Linux, no DISPLAY, accessible
uinput, unparseable version) must preserve current behavior.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


class _RecordingSession:
    """Session stub. Input actions must refuse before ever touching this —
    tests assert ``call_tool_calls`` stays empty for a refused action."""

    def __init__(self, out: Optional[Dict[str, Any]] = None):
        self._out = out or {"isError": False, "data": {}, "structuredContent": {}}
        self.call_tool_calls = []

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.call_tool_calls.append((name, dict(args)))
        return self._out

    def supports_capability(self, capability: str, tool: Optional[str] = None) -> bool:
        return False


def _make_backend(session=None):
    from tools.computer_use.cua_backend import CuaDriverBackend

    be = CuaDriverBackend.__new__(CuaDriverBackend)
    be._session = session if session is not None else _RecordingSession()
    be._session_id = "test-run"
    be._snapshot_tokens = {}
    be._active_pid = 4242
    be._active_window_id = 7
    return be


def _unsafe_combo_patches(driver_version="0.12.6", uinput_accessible=False,
                          platform="linux"):
    return (
        patch("tools.computer_use.cua_backend.sys.platform", platform),
        patch(
            "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
            return_value=driver_version,
        ),
        patch(
            "tools.computer_use.cua_backend.uinput_read_write_accessible",
            return_value=uinput_accessible,
        ),
    )


INPUT_ACTIONS = [
    ("click", lambda be: be.click(element=1)),
    ("drag", lambda be: be.drag(from_element=1, to_element=2)),
    ("scroll", lambda be: be.scroll(direction="down", element=1)),
    ("type_text", lambda be: be.type_text("hi")),
    ("key", lambda be: be.key("return")),
    ("set_value", lambda be: be.set_value("x", element=1)),
]


class TestInputActionsRefuseOnUnsafeCombo:
    @pytest.mark.parametrize("name,call", INPUT_ACTIONS, ids=[n for n, _ in INPUT_ACTIONS])
    def test_refuses_before_touching_the_driver(self, name, call):
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            result = call(be)

        assert result.ok is False
        assert result.action == name
        assert result.code == "linux_uinput_xinput_leak_risk"
        assert "2618" in result.message
        assert "2631" in result.message
        assert "74148" in result.message
        for forbidden in ("chmod", "sudo"):
            assert forbidden not in result.message.lower()
        # The driver must never be touched by a refused input action.
        assert session.call_tool_calls == []

    @pytest.mark.parametrize("name,call", INPUT_ACTIONS, ids=[n for n, _ in INPUT_ACTIONS])
    def test_proceeds_normally_when_driver_is_fixed(self, name, call):
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.13.1",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            call(be)

        assert len(session.call_tool_calls) == 1

    @pytest.mark.parametrize("name,call", INPUT_ACTIONS, ids=[n for n, _ in INPUT_ACTIONS])
    def test_proceeds_normally_on_non_linux(self, name, call):
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "darwin"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            call(be)

        assert len(session.call_tool_calls) == 1

    @pytest.mark.parametrize("name,call", INPUT_ACTIONS, ids=[n for n, _ in INPUT_ACTIONS])
    def test_proceeds_normally_without_display(self, name, call):
        session = _RecordingSession()
        be = _make_backend(session)

        env = dict(os.environ)
        env.pop("DISPLAY", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            call(be)

        assert len(session.call_tool_calls) == 1

    @pytest.mark.parametrize("name,call", INPUT_ACTIONS, ids=[n for n, _ in INPUT_ACTIONS])
    def test_proceeds_normally_when_uinput_accessible(self, name, call):
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=True,
             ):
            call(be)

        assert len(session.call_tool_calls) == 1

    @pytest.mark.parametrize("name,call", INPUT_ACTIONS, ids=[n for n, _ in INPUT_ACTIONS])
    def test_proceeds_normally_for_unparseable_version(self, name, call):
        """Unknown/malformed versions must preserve current (no-guard) behavior."""
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value=None,
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            call(be)

        assert len(session.call_tool_calls) == 1


class TestCaptureAndInspectionRemainAvailable:
    def test_list_apps_is_not_guarded(self):
        session = _RecordingSession({
            "isError": False, "data": {}, "structuredContent": {"apps": [{"name": "x", "pid": 1}]},
        })
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            apps = be.list_apps()

        assert apps == [{"name": "x", "pid": 1}]
        assert len(session.call_tool_calls) == 1

    def test_capture_exact_target_is_not_guarded(self):
        from tools.computer_use.cua_backend import CuaDriverBackend

        session = _RecordingSession({
            "data": "✅ Chrome — 0 elements",
            "images": [],
            "structuredContent": {"elements": []},
            "isError": False,
        })
        be = CuaDriverBackend.__new__(CuaDriverBackend)
        be._session = session
        be._session_id = "test-run"
        be._snapshot_tokens = {}
        be._active_pid = None
        be._active_window_id = None
        be._last_app = None
        be._last_target = None

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            be.capture(mode="ax", pid=1816017, window_id=60817412)

        assert be._active_pid == 1816017
        assert len(session.call_tool_calls) >= 1


class TestUinputDeviceIsNotOpenedWhenIrrelevant:
    """``/dev/uinput`` must only be opened when platform/DISPLAY/version alone
    leave the risk open. A fixed (or non-Linux, display-less, unparseable)
    setup has nothing to learn from the device, so it must not touch it."""

    def test_device_not_opened_for_fixed_driver_version(self):
        be = _make_backend()

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.13.1",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
             ) as accessible:
            be.click(element=1)

        accessible.assert_not_called()

    def test_device_not_opened_for_unparseable_driver_version(self):
        be = _make_backend()

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value=None,
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
             ) as accessible:
            be.click(element=1)

        accessible.assert_not_called()

    def test_device_not_opened_off_linux(self):
        be = _make_backend()

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "darwin"), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
             ) as accessible:
            be.click(element=1)

        accessible.assert_not_called()

    def test_device_not_opened_without_display(self):
        be = _make_backend()

        env = dict(os.environ)
        env.pop("DISPLAY", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
             ) as accessible:
            be.click(element=1)

        accessible.assert_not_called()

    def test_device_is_still_opened_for_a_vulnerable_version(self):
        """The short-circuit must not disable the guard it fronts."""
        be = _make_backend()

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.12.6",
             ), \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ) as accessible:
            result = be.click(element=1)

        accessible.assert_called()
        assert result.ok is False
        assert result.code == "linux_uinput_xinput_leak_risk"


class TestDriverVersionProbeIsCached:
    def test_probe_runs_at_most_once_per_backend_instance(self):
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
                 return_value="0.13.1",
             ) as probe, \
             patch(
                 "tools.computer_use.cua_backend.uinput_read_write_accessible",
                 return_value=False,
             ):
            be.click(element=1)
            be.click(element=1)
            be.type_text("hi")

        assert probe.call_count == 1

    def test_probe_never_runs_off_linux(self):
        session = _RecordingSession()
        be = _make_backend(session)

        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "darwin"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
             ) as probe:
            be.click(element=1)

        probe.assert_not_called()

    def test_probe_never_runs_without_display(self):
        session = _RecordingSession()
        be = _make_backend(session)

        env = dict(os.environ)
        env.pop("DISPLAY", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend._probe_cua_driver_version_for_uinput_guard",
             ) as probe:
            be.click(element=1)

        probe.assert_not_called()


class TestGuardProbesTheActuallyLaunchedCommand:
    """Regression test (independent-review finding, #74148): the guard must
    probe ``--version`` on the exact command ``_CuaDriverSession.start()``
    launched, not a separately re-resolved default/PATH binary.

    ``_resolve_mcp_invocation`` can return a manifest-relocated executable
    (trycua/cua#1961) that differs from what ``resolve_cua_driver_cmd()``
    would resolve on its own — e.g. a thin-PATH GUI session resolving a
    generic wrapper while the manifest points at the real relocated
    binary. If the guard probes the wrong binary, a fixed wrapper can mask
    a vulnerable actual MCP executable and let unsafe input through.
    """

    def test_probes_the_manifest_relocated_command_actually_launched(self):
        from tools.computer_use.cua_backend import CuaDriverBackend, _CuaDriverSession, _AsyncBridge

        relocated_cmd = "/opt/cua/relocated-cua-driver-bin"
        default_resolved_cmd = "/usr/local/bin/cua-driver"

        session = _CuaDriverSession(_AsyncBridge())
        # Simulate what a completed start() recorded: the manifest hop
        # resolved a DIFFERENT binary than the generic default one.
        session._resolved_mcp_command = relocated_cmd

        be = CuaDriverBackend.__new__(CuaDriverBackend)
        be._session = session
        be._session_id = "test-run"

        version_proc = MagicMock(stdout="0.12.6", stderr="", returncode=0)
        with patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch("tools.computer_use.cua_backend.sys.platform", "linux"), \
             patch(
                 "tools.computer_use.cua_backend.resolve_cua_driver_cmd",
                 return_value=default_resolved_cmd,
             ), \
             patch("subprocess.run", return_value=version_proc) as run:
            be._uinput_leak_guard("click")

        assert run.call_args is not None, "guard never probed --version"
        probed_cmd = run.call_args.args[0][0]
        assert probed_cmd == relocated_cmd, (
            f"guard probed {probed_cmd!r} (the default-resolved binary) "
            f"instead of {relocated_cmd!r} (the manifest-relocated binary "
            "actually launched by _CuaDriverSession.start()) — a fixed "
            "wrapper could mask a vulnerable actual MCP executable"
        )


class TestSessionRecordsTheResolvedMcpCommand:
    """``_CuaDriverSession._lifecycle_coro`` must record the command it
    actually spawns so guard callers can probe that exact binary instead
    of independently re-resolving (and potentially picking a different
    wrapper/default)."""

    def test_lifecycle_records_manifest_relocated_command(self):
        from unittest.mock import AsyncMock
        import asyncio
        from tools.computer_use.cua_backend import _AsyncBridge, _CuaDriverSession

        bridge = _AsyncBridge()
        session = _CuaDriverSession(bridge)
        relocated_cmd = "/opt/cua/relocated-cua-driver-bin"

        async def drive_lifecycle():
            with patch("tools.computer_use.cua_backend.resolve_cua_driver_cmd",
                       return_value="cua-driver"), \
                 patch("tools.computer_use.cua_backend._resolve_mcp_invocation",
                       return_value=(relocated_cmd, ["mcp"])), \
                 patch("mcp.StdioServerParameters", return_value=MagicMock()), \
                 patch("mcp.client.stdio.stdio_client") as mock_stdio, \
                 patch("mcp.ClientSession") as mock_session_class:

                mock_stdio.return_value.__aenter__ = AsyncMock(
                    return_value=(MagicMock(), MagicMock()))
                mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

                fake_session = MagicMock()
                fake_session.initialize = AsyncMock()
                fake_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=fake_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                async def _signal_shutdown_when_ready():
                    for _ in range(200):
                        if session._shutdown_event is not None:
                            session._shutdown_event.set()
                            return
                        await asyncio.sleep(0.005)

                signal_task = asyncio.create_task(_signal_shutdown_when_ready())
                try:
                    await session._lifecycle_coro()
                except BaseException:
                    pass
                finally:
                    signal_task.cancel()
                    try:
                        await signal_task
                    except (asyncio.CancelledError, BaseException):
                        pass

        asyncio.run(drive_lifecycle())

        assert session._resolved_mcp_command == relocated_cmd
