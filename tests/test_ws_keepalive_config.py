"""Config propagation tests for the WS keepalive + orphan-reap grace knobs (#79635)."""

import textwrap

import pytest


@pytest.fixture()
def _temp_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI_WS_ORPHAN_REAP_GRACE_S", raising=False)
    monkeypatch.delenv("HERMES_TUI_LEASE_RECLAIM_IDLE_S", raising=False)
    return home


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body))


def test_dashboard_ws_defaults_present(_temp_home):
    from hermes_cli.config import load_config

    _write_config(_temp_home, "model:\n  default: test-model\n")
    cfg = load_config()
    dash = cfg.get("dashboard") or {}
    assert dash.get("ws_ping_interval") == 20.0
    assert dash.get("ws_ping_timeout") == 20.0
    assert dash.get("ws_orphan_reap_grace_s") == 20.0
    assert dash.get("lease_reclaim_idle_s") == 300.0


def test_dashboard_ws_values_propagate_from_yaml(_temp_home):
    from hermes_cli.config import load_config

    _write_config(
        _temp_home,
        """
        dashboard:
          ws_ping_interval: 45.5
          ws_ping_timeout: 10
          ws_orphan_reap_grace_s: 90
          lease_reclaim_idle_s: 120
        """,
    )
    cfg = load_config()
    dash = cfg["dashboard"]
    assert dash["ws_ping_interval"] == 45.5
    assert dash["ws_ping_timeout"] == 10
    assert dash["ws_orphan_reap_grace_s"] == 90
    assert dash["lease_reclaim_idle_s"] == 120
    # Deep-merge: sibling defaults survive a partial user section.
    assert dash.get("theme") == "default"


def test_ws_orphan_reap_grace_reads_config(_temp_home):
    from tui_gateway import server

    _write_config(
        _temp_home,
        """
        dashboard:
          ws_orphan_reap_grace_s: 33
        """,
    )
    assert server._resolve_ws_orphan_reap_grace() == 33.0


def test_ws_orphan_reap_grace_env_var_overrides_config(_temp_home, monkeypatch):
    from tui_gateway import server

    _write_config(
        _temp_home,
        """
        dashboard:
          ws_orphan_reap_grace_s: 33
        """,
    )
    monkeypatch.setenv("HERMES_TUI_WS_ORPHAN_REAP_GRACE_S", "7")
    assert server._resolve_ws_orphan_reap_grace() == 7.0


def test_ws_orphan_reap_grace_invalid_values_fall_back(_temp_home, monkeypatch):
    from tui_gateway import server

    _write_config(
        _temp_home,
        """
        dashboard:
          ws_orphan_reap_grace_s: not-a-number
        """,
    )
    assert server._resolve_ws_orphan_reap_grace() == 20.0
    monkeypatch.setenv("HERMES_TUI_WS_ORPHAN_REAP_GRACE_S", "-5")
    assert server._resolve_ws_orphan_reap_grace() == 0.0


def test_lease_reclaim_idle_reads_config_and_env(_temp_home, monkeypatch):
    from tui_gateway import server

    _read = lambda: server._ws_orphan_setting(  # noqa: E731
        "HERMES_TUI_LEASE_RECLAIM_IDLE_S", "lease_reclaim_idle_s", 300.0
    )
    _write_config(
        _temp_home,
        """
        dashboard:
          lease_reclaim_idle_s: 45
        """,
    )
    assert _read() == 45.0
    monkeypatch.setenv("HERMES_TUI_LEASE_RECLAIM_IDLE_S", "7")
    assert _read() == 7.0


def test_lease_reclaim_idle_invalid_values_fall_back(_temp_home, monkeypatch):
    from tui_gateway import server

    _read = lambda: server._ws_orphan_setting(  # noqa: E731
        "HERMES_TUI_LEASE_RECLAIM_IDLE_S", "lease_reclaim_idle_s", 300.0
    )
    _write_config(
        _temp_home,
        """
        dashboard:
          lease_reclaim_idle_s: not-a-number
        """,
    )
    assert _read() == 300.0
    monkeypatch.setenv("HERMES_TUI_LEASE_RECLAIM_IDLE_S", "-5")
    assert _read() == 0.0
