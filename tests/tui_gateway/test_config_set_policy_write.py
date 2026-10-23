"""``config.set`` writes to approval policy from the TUI RPC surface.

Pins the YOLO global toggle and the ``approvals.mode`` setter against the real
config store: both route through ``_tui_policy_write``, which requires the live
gateway RPC transport as provenance (#104697). Split out of
``test_tui_gateway_server.py`` (file-size gate) — coverage is unchanged.
"""

import contextlib
from unittest.mock import Mock

from tui_gateway import server
from tui_gateway.transport import _POLICY_WRITE_TRANSPORTS, bind_transport, reset_transport

# A stub registered as this process's live gateway channel, the way ``entry.main()``
# registers the stdio transport in production.
_RPC_TRANSPORT = Mock()


@contextlib.contextmanager
def _policy_write_transport(transport):
    """Register *transport* as this process's live gateway channel for one RPC."""
    _POLICY_WRITE_TRANSPORTS.add(transport)
    try:
        yield
    finally:
        _POLICY_WRITE_TRANSPORTS.discard(transport)


def _dispatch_sync(req: dict, transport=None) -> dict | None:
    """Run one RPC to completion synchronously, with ``dispatch``'s transport binding."""
    token = bind_transport(transport)
    try:
        return server.handle_request(req)
    finally:
        reset_transport(token)


def test_config_set_yolo_global_scope_writes_approvals_mode(tmp_path, monkeypatch):
    """Shift+click the desktop zap -> scope="global" flips persistent approvals.mode."""
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # the policy writer resolves HERMES_HOME

    with _policy_write_transport(_RPC_TRANSPORT):
        resp_on = _dispatch_sync(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "yolo", "scope": "global"},
            },
            _RPC_TRANSPORT,
        )
        assert resp_on["result"]["value"] == "1"
        assert resp_on["result"]["scope"] == "global"
        assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "off"

        resp_off = _dispatch_sync(
            {
                "id": "2",
                "method": "config.set",
                "params": {"key": "yolo", "scope": "global"},
            },
            _RPC_TRANSPORT,
        )
    assert resp_off["result"]["value"] == "0"
    assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "manual"


def test_config_set_approval_mode_persists_three_way_value_and_emits_live_status(
    tmp_path, monkeypatch
):
    import hermes_yaml as yaml

    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # config.set writes via server._hermes_home, but the post-write
    # session.info emit resolves the effective mode through the canonical
    # tools.approval resolver (load_config → env HERMES_HOME).
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    server._sessions["sid"] = {"agent": object(), "session_key": "profile-session"}

    try:
        with _policy_write_transport(_RPC_TRANSPORT):
            resp = _dispatch_sync(
                {
                    "id": "1",
                    "method": "config.set",
                    "params": {"key": "approvals.mode", "value": "manual"},
                },
                _RPC_TRANSPORT,
            )
    finally:
        server._sessions.clear()

    assert resp["result"] == {"key": "approvals.mode", "value": "manual"}
    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "manual"
    assert emitted and emitted[0][0:2] == ("session.info", "sid")
    assert emitted[0][2]["approval_mode"] == "manual"


def test_config_set_yolo_global_scope_honors_explicit_value(tmp_path, monkeypatch):
    """An explicit value pins global approvals.mode regardless of prior state."""
    import hermes_yaml as yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"approvals": {"mode": "manual"}}), encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    # set_config_value resolves the config path from HERMES_HOME, not server._hermes_home.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with _policy_write_transport(_RPC_TRANSPORT):
        resp = _dispatch_sync(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "yolo", "scope": "global", "value": "1"},
            },
            _RPC_TRANSPORT,
        )
        assert resp["result"]["value"] == "1"
        assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "off"

        # Setting it on again is idempotent — stays off.
        resp_again = _dispatch_sync(
            {
                "id": "2",
                "method": "config.set",
                "params": {"key": "yolo", "scope": "global", "value": "1"},
            },
            _RPC_TRANSPORT,
        )
    assert resp_again["result"]["value"] == "1"
    assert yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))["approvals"]["mode"] == "off"


def test_managed_install_reports_rpc_error_not_success(tmp_path, monkeypatch):
    """A package-manager-managed install must not answer ``config.set`` with success.

    ``set_config_value`` returns early under managed mode without saving, so the
    pre-fix funnel reported the requested value as applied while config.yaml on
    disk never changed (review finding: managed writes break TUI replies).
    """
    import hermes_cli.config as cfg

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "is_managed", lambda: True)

    with _policy_write_transport(_RPC_TRANSPORT):
        resp = _dispatch_sync(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "approvals.mode", "value": "off"},
            },
            _RPC_TRANSPORT,
        )

    assert "error" in resp, resp
    assert "result" not in resp
    assert cfg_path.read_text(encoding="utf-8") == "approvals:\n  mode: manual\n"


def test_pinned_key_survives_as_rpc_error(tmp_path, monkeypatch):
    """A pinned (managed-scope) key raises SystemExit in the writer; the RPC must
    turn that into an error envelope instead of exiting the stdio backend."""
    import hermes_cli.config as cfg

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _pinned(key, action):
        raise SystemExit(1)

    def _boom(key, value):
        raise SystemExit(1)

    monkeypatch.setattr(cfg, "_exit_if_key_managed", _pinned)
    monkeypatch.setattr(cfg, "set_config_value", _boom)

    with _policy_write_transport(_RPC_TRANSPORT):
        resp = _dispatch_sync(
            {
                "id": "1",
                "method": "config.set",
                "params": {"key": "approvals.mode", "value": "off"},
            },
            _RPC_TRANSPORT,
        )

    assert "error" in resp, resp
    assert cfg_path.read_text(encoding="utf-8") == "approvals:\n  mode: manual\n"
