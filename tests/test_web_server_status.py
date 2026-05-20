import pytest


@pytest.mark.asyncio
async def test_dashboard_status_ignores_stale_local_runtime_platform_snapshot(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "check_config_version", lambda: ("current", "latest"))
    monkeypatch.setattr(web_server, "get_running_pid", lambda: 12345)
    monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", None)
    monkeypatch.setattr(
        web_server,
        "read_runtime_status",
        lambda: {
            "pid": 99999,
            "gateway_state": "running",
            "platforms": {
                "discord": {
                    "state": "connected",
                    "updated_at": "2000-01-01T00:00:00+00:00",
                }
            },
            "updated_at": "2000-01-01T00:00:00+00:00",
        },
    )

    data = await web_server.get_status()

    assert data["gateway_running"] is True
    assert data["gateway_pid"] == 12345
    assert data["gateway_state"] == "running"
    assert data["gateway_platforms"] == {}
    assert data["gateway_updated_at"] is None
