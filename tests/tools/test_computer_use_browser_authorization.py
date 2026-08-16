"""Authorization plumbing for the cua-driver typed browser route.

Covers the authorization modes that let ``existing_profile`` attachment (and
bounded automation generally) work from Hermes:

* ``bounded`` permission mode — a private embedded daemon launched with a
  user-reviewed capability manifest (``--capability-manifest`` +
  ``--approve-capability-manifest``), failing loudly when the manifest is
  missing.
* mode resolution — config supplies standard/bounded only; explicit session
  YOLO still (and exclusively) selects unrestricted.
"""

from typing import Any, Dict

import pytest

from tools.computer_use import cua_backend as cb
from tools.computer_use.browser_route import CuaTypedBrowserRoute
from tools.computer_use.cua_backend import _EmbeddedCuaDaemon


def _driver_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"structuredContent": dict(payload)}


class _PrepareDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return True

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((name, dict(args)))
        return _driver_result({"status": "ok"})


def _route(driver: _PrepareDriver) -> CuaTypedBrowserRoute:
    return CuaTypedBrowserRoute(
        session_id="hermes-a",
        call_tool=driver.call,
        has_tool=driver.has_tool,
    )


# ── existing-profile authorization ownership ───────────────────────────


def test_existing_profile_prepare_delegates_authorization_to_driver():
    driver = _PrepareDriver()
    result = _route(driver).prepare(
        pid=101,
        window_id=202,
        profile_mode="existing_profile",
    )

    assert result["status"] == "ok"
    assert driver.calls == [
        (
            "browser_prepare",
            {
                "pid": 101,
                "window_id": 202,
                "strategy": {"kind": "existing_profile"},
                "session": "hermes-a",
            },
        )
    ]


def test_dispatch_does_not_forward_removed_approval_token():
    from unittest.mock import Mock

    from tools.computer_use.tool import _dispatch

    backend = Mock()
    backend.typed_browser_prepare.return_value = {"status": "ok"}

    _dispatch(
        backend,
        "cua_browser_prepare",
        {
            "pid": 101,
            "window_id": 202,
            "profile_mode": "existing_profile",
        },
    )

    kwargs = backend.typed_browser_prepare.call_args.kwargs
    assert "approval_token" not in kwargs
    assert kwargs["profile_mode"] == "existing_profile"


def test_schema_does_not_expose_approval_token():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA

    assert "approval_token" not in COMPUTER_USE_SCHEMA["parameters"]["properties"]


# ── bounded embedded daemon ─────────────────────────────────────────────


def test_bounded_daemon_requires_a_manifest():
    with pytest.raises(ValueError, match="capability_manifest"):
        _EmbeddedCuaDaemon("cua-driver", "bounded")


def test_bounded_daemon_requires_manifest_file_to_exist(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        _EmbeddedCuaDaemon(
            "cua-driver", "bounded",
            capability_manifest=str(tmp_path / "missing.yaml"),
        )


def test_bounded_daemon_env_does_not_bypass_approvals(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("version: 3\n", encoding="utf-8")
    daemon = _EmbeddedCuaDaemon(
        "cua-driver", "bounded", capability_manifest=str(manifest)
    )

    env = daemon.child_env()
    assert env["CUA_DRIVER_PERMISSION_MODE"] == "bounded"
    assert "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS" not in env


def test_unrestricted_daemon_env_keeps_explicit_bypass():
    daemon = _EmbeddedCuaDaemon("cua-driver", "unrestricted")
    env = daemon.child_env()
    assert env["CUA_DRIVER_PERMISSION_MODE"] == "unrestricted"
    assert env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] == "1"


def test_bounded_daemon_serves_with_approved_manifest(tmp_path, monkeypatch):
    """The spawn command carries the manifest + launch-time approval flags."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("version: 3\n", encoding="utf-8")
    daemon = _EmbeddedCuaDaemon(
        "cua-driver", "bounded", capability_manifest=str(manifest)
    )

    captured: Dict[str, Any] = {}

    class _FakeProc:
        stderr = None

        def poll(self):
            return None

    def _fake_popen(command, **kwargs):
        captured["command"] = list(command)
        return _FakeProc()

    def _fake_run(command, **kwargs):
        class _Probe:
            returncode = 0
        return _Probe()

    monkeypatch.setattr(cb.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cb.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        cb, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"])
    )

    daemon.start()

    command = captured["command"]
    assert "--permission-mode" in command
    assert command[command.index("--permission-mode") + 1] == "bounded"
    assert "--capability-manifest" in command
    assert (
        command[command.index("--capability-manifest") + 1]
        == str(manifest)
    )
    assert "--approve-capability-manifest" in command
    assert "--dangerously-bypass-approvals" not in command


def test_unrestricted_daemon_serve_command_unchanged(monkeypatch):
    daemon = _EmbeddedCuaDaemon("cua-driver", "unrestricted")

    captured: Dict[str, Any] = {}

    class _FakeProc:
        stderr = None

        def poll(self):
            return None

    monkeypatch.setattr(
        cb.subprocess, "Popen",
        lambda command, **kw: captured.update(command=list(command)) or _FakeProc(),
    )

    def _fake_run(command, **kwargs):
        class _Probe:
            returncode = 0
        return _Probe()

    monkeypatch.setattr(cb.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        cb, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"])
    )

    daemon.start()

    command = captured["command"]
    assert "--dangerously-bypass-approvals" in command
    assert "--capability-manifest" not in command


# ── standard-mode --grant existing-profile ──────────────────────────────


def test_grant_existing_profile_defaults_off(monkeypatch):
    monkeypatch.setattr(cb, "_computer_use_cfg", dict)
    assert cb._cua_grant_existing_profile() is False


def test_grant_existing_profile_reads_config(monkeypatch):
    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"grant_existing_profile": True}
    )
    assert cb._cua_grant_existing_profile() is True


# ── permission-mode resolution ──────────────────────────────────────────


def test_configured_mode_defaults_to_standard(monkeypatch):
    monkeypatch.setattr(cb, "_computer_use_cfg", dict)
    assert cb._cua_configured_permission_mode() == "standard"


def test_configured_mode_honors_bounded(monkeypatch):
    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"permission_mode": "Bounded"}
    )
    assert cb._cua_configured_permission_mode() == "bounded"


@pytest.mark.parametrize("value", ["unrestricted", "yolo", "off", 3, None])
def test_configured_mode_never_yields_unrestricted(monkeypatch, value):
    """A config line must never silently bypass approvals."""
    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"permission_mode": value}
    )
    assert cb._cua_configured_permission_mode() == "standard"


def test_capability_manifest_reads_config(monkeypatch):
    monkeypatch.setattr(
        cb, "_computer_use_cfg",
        lambda: {"capability_manifest": "  ~/manifests/cua.yaml  "},
    )
    assert cb._cua_capability_manifest() == "~/manifests/cua.yaml"
    monkeypatch.setattr(cb, "_computer_use_cfg", dict)
    assert cb._cua_capability_manifest() is None


def test_session_yolo_overrides_configured_bounded(monkeypatch):
    import tools.computer_use.tool as cu_tool

    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"permission_mode": "bounded"}
    )
    import tools.approval as approval

    monkeypatch.setattr(
        approval, "is_approval_bypass_active_for_session", lambda sid: True
    )
    assert cu_tool._cua_permission_mode("sess-1") == "unrestricted"


def test_no_yolo_uses_configured_bounded(monkeypatch):
    import tools.computer_use.tool as cu_tool

    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"permission_mode": "bounded"}
    )
    import tools.approval as approval

    monkeypatch.setattr(
        approval, "is_approval_bypass_active_for_session", lambda sid: False
    )
    monkeypatch.setattr(
        approval, "get_current_session_key", lambda default="": ""
    )
    assert cu_tool._cua_permission_mode("sess-1") == "bounded"


def test_no_yolo_no_config_stays_standard(monkeypatch):
    import tools.computer_use.tool as cu_tool

    monkeypatch.setattr(cb, "_computer_use_cfg", dict)
    import tools.approval as approval

    monkeypatch.setattr(
        approval, "is_approval_bypass_active_for_session", lambda sid: False
    )
    monkeypatch.setattr(
        approval, "get_current_session_key", lambda default="": ""
    )
    assert cu_tool._cua_permission_mode("sess-1") == "standard"


def test_backend_accepts_bounded_with_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("version: 3\n", encoding="utf-8")
    monkeypatch.setattr(
        cb, "_cua_capability_manifest", lambda: str(manifest)
    )
    monkeypatch.setattr(cb, "resolve_cua_driver_cmd", lambda override=None: "cua-driver")

    backend = cb.CuaDriverBackend(permission_mode="bounded")
    assert backend.permission_mode == "bounded"
    assert backend._embedded_daemon is not None
    assert backend._embedded_daemon.capability_manifest == str(manifest)


def test_backend_bounded_without_manifest_fails_loudly(monkeypatch):
    monkeypatch.setattr(cb, "_cua_capability_manifest", lambda: None)
    monkeypatch.setattr(cb, "resolve_cua_driver_cmd", lambda override=None: "cua-driver")

    with pytest.raises(ValueError, match="capability_manifest"):
        cb.CuaDriverBackend(permission_mode="bounded")


def test_backend_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported"):
        cb.CuaDriverBackend(permission_mode="wide-open")
