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
        grant_existing_profile=True,
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


def test_existing_profile_prepare_refused_without_config_grant():
    driver = _PrepareDriver()
    result = _route(driver).prepare(
        pid=101,
        window_id=202,
        profile_mode="existing_profile",
    )

    assert result["status"] == "refused"
    assert result["code"] == "browser_existing_profile_not_granted"
    assert "computer_use.grant_existing_profile" in result["message"]
    # Never reached the driver: the host refuses before the transport.
    assert driver.calls == []


def test_existing_profile_prepare_refused_in_unrestricted_without_grant():
    """An approval bypass must not stand in for the config grant.

    ``--yolo`` / ``-z`` give the session a private unrestricted daemon that
    answers every prepare, so without this host-side floor the documented
    ``grant_existing_profile: false`` default silently stopped protecting the
    live profile's pages, cookies, and storage.
    """
    driver = _PrepareDriver()
    result = _route(driver).prepare(
        pid=101,
        window_id=202,
        profile_mode="existing_profile",
        grant_existing_profile=False,
        permission_mode="unrestricted",
    )

    assert result["code"] == "browser_existing_profile_not_granted"
    assert driver.calls == []


def test_existing_profile_prepare_bounded_mode_exempt_from_grant():
    """bounded's reviewed capability manifest is the authorization boundary."""
    driver = _PrepareDriver()
    result = _route(driver).prepare(
        pid=101,
        window_id=202,
        profile_mode="existing_profile",
        grant_existing_profile=False,
        permission_mode="bounded",
    )

    assert result["status"] == "ok"
    assert [name for name, _ in driver.calls] == ["browser_prepare"]


def test_isolated_prepare_unaffected_by_the_grant():
    """The floor is scoped to existing_profile; isolated launches still work."""
    driver = _PrepareDriver()
    result = _route(driver).prepare(
        pid=101,
        profile_mode="isolated_new",
        allow_launch=True,
        grant_existing_profile=False,
    )

    assert result["status"] == "ok"
    assert [name for name, _ in driver.calls] == ["browser_prepare"]


def test_backend_resolves_authorization_and_ignores_model_supplied_values(monkeypatch):
    """pid/window_id come from the model; the grant never does."""
    captured: Dict[str, Any] = {}

    class _Route:
        def prepare(self, **kwargs: Any) -> Dict[str, Any]:
            captured.update(kwargs)
            return {"status": "ok"}

    backend = cb.CuaDriverBackend.__new__(cb.CuaDriverBackend)
    backend.permission_mode = "unrestricted"
    monkeypatch.setattr(cb, "_cua_grant_existing_profile", lambda: False)
    monkeypatch.setattr(
        cb.CuaDriverBackend, "_browser_route", lambda self: _Route()
    )

    backend.typed_browser_prepare(
        pid=101,
        window_id=202,
        profile_mode="existing_profile",
        # A model that tries to grant itself access must be ignored.
        grant_existing_profile=True,
        permission_mode="bounded",
    )

    assert captured["grant_existing_profile"] is False
    assert captured["permission_mode"] == "unrestricted"


# ── config grant stands in for the approval prompt ──────────────────────


def _preauth(**cfg: Any):
    from tools.computer_use import tool as cu_tool

    return cu_tool._config_preauthorized


def test_config_grant_preauthorizes_existing_profile_prepare(monkeypatch):
    """The durable opt-in is the authorization; re-prompting is redundant.

    It also made the documented opt-in unusable on non-interactive runs,
    where the prompt has nobody to answer it and the call dies on approval
    timeout rather than attaching.
    """
    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"grant_existing_profile": True}
    )
    assert _preauth()(
        "cua_browser_prepare", {"profile_mode": "existing_profile"}
    ) is True


def test_no_preauthorization_without_the_grant(monkeypatch):
    monkeypatch.setattr(cb, "_computer_use_cfg", dict)
    assert _preauth()(
        "cua_browser_prepare", {"profile_mode": "existing_profile"}
    ) is False


def test_preauthorization_scoped_to_existing_profile(monkeypatch):
    """Isolated launches keep prompting even when the grant is on."""
    monkeypatch.setattr(
        cb, "_computer_use_cfg", lambda: {"grant_existing_profile": True}
    )
    assert _preauth()(
        "cua_browser_prepare", {"profile_mode": "isolated_new"}
    ) is False
    assert _preauth()("click", {"profile_mode": "existing_profile"}) is False


def test_preauthorization_fails_closed_on_config_error(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(cb, "_computer_use_cfg", _boom)
    assert _preauth()(
        "cua_browser_prepare", {"profile_mode": "existing_profile"}
    ) is False


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


# ── manifest is a ceiling, not a mode ───────────────────────────────────


def _captured_serve_command(monkeypatch, daemon):
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
    monkeypatch.setattr(cb, "_resolve_mcp_invocation", lambda cmd: (cmd, ["mcp"]))
    daemon.start()
    return captured["command"]


def test_unrestricted_daemon_carries_a_v3_manifest(monkeypatch, tmp_path):
    """An approval bypass must not silently discard a v3 ceiling.

    cua-driver accepts a v3 manifest alongside any permission mode and it can
    only narrow a profile, never widen it. Pairing it with unrestricted is
    what bounds an approval-bypassed run to declared scope; dropping it meant
    the most carefully configured run became the least constrained one.
    """
    manifest = tmp_path / "cua-capabilities.yaml"
    manifest.write_text("version: 3\nallow:\n  tools:\n    - list_windows\n", encoding="utf-8")
    daemon = _EmbeddedCuaDaemon(
        "cua-driver", "unrestricted", capability_manifest=str(manifest)
    )

    command = _captured_serve_command(monkeypatch, daemon)

    assert "--dangerously-bypass-approvals" in command
    assert "--capability-manifest" in command
    assert command[command.index("--capability-manifest") + 1] == str(manifest)
    assert "--approve-capability-manifest" in command
    assert command[command.index("--permission-mode") + 1] == "unrestricted"


def test_unrestricted_daemon_does_not_forward_a_legacy_manifest(monkeypatch, tmp_path):
    """v1/v2 manifests must declare mode: bounded, so they cannot ride along.

    Forwarding one anyway aborts driver startup with "legacy capability
    manifest mode must be bounded" — turning a working session into a hard
    failure. Verified against cua-driver 0.20.0.
    """
    manifest = tmp_path / "legacy.yaml"
    manifest.write_text(
        "version: 1\nmode: bounded\nexpires_after: 1h\nidle_timeout: 5m\n",
        encoding="utf-8",
    )
    daemon = _EmbeddedCuaDaemon(
        "cua-driver", "unrestricted", capability_manifest=str(manifest)
    )

    command = _captured_serve_command(monkeypatch, daemon)

    assert "--dangerously-bypass-approvals" in command
    assert "--capability-manifest" not in command


def test_bounded_forwards_a_legacy_manifest_unchanged(monkeypatch, tmp_path):
    """bounded is exactly where legacy manifests belong; nothing changes."""
    manifest = tmp_path / "legacy.yaml"
    manifest.write_text(
        "version: 1\nmode: bounded\nexpires_after: 1h\nidle_timeout: 5m\n",
        encoding="utf-8",
    )
    daemon = _EmbeddedCuaDaemon(
        "cua-driver", "bounded", capability_manifest=str(manifest)
    )

    command = _captured_serve_command(monkeypatch, daemon)

    assert command[command.index("--permission-mode") + 1] == "bounded"
    assert command[command.index("--capability-manifest") + 1] == str(manifest)
    assert "--approve-capability-manifest" in command


def test_unparseable_manifest_is_not_forwarded_to_unrestricted(monkeypatch, tmp_path):
    """Fail safe: never turn a working bypassed run into a startup abort."""
    manifest = tmp_path / "broken.yaml"
    manifest.write_text("{{{ not yaml", encoding="utf-8")
    daemon = _EmbeddedCuaDaemon(
        "cua-driver", "unrestricted", capability_manifest=str(manifest)
    )

    command = _captured_serve_command(monkeypatch, daemon)

    assert "--capability-manifest" not in command


def test_unrestricted_daemon_without_a_manifest_is_unchanged(monkeypatch):
    """No manifest configured -> nothing new on the command line."""
    daemon = _EmbeddedCuaDaemon("cua-driver", "unrestricted")

    command = _captured_serve_command(monkeypatch, daemon)

    assert "--dangerously-bypass-approvals" in command
    assert "--capability-manifest" not in command


def test_unrestricted_daemon_rejects_a_missing_manifest_path(tmp_path):
    """A declared-but-absent ceiling fails loudly rather than silently opening up."""
    with pytest.raises(ValueError, match="capability manifest not found"):
        _EmbeddedCuaDaemon(
            "cua-driver",
            "unrestricted",
            capability_manifest=str(tmp_path / "does-not-exist.yaml"),
        )


def test_bounded_still_requires_a_manifest():
    with pytest.raises(ValueError, match="requires computer_use.capability_manifest"):
        _EmbeddedCuaDaemon("cua-driver", "bounded")
