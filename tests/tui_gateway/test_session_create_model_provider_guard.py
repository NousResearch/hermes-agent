"""Session creation must reject corrupt model/provider pairs before side effects."""

from unittest.mock import patch

from tui_gateway import server as srv


def test_invalid_pair_creates_no_session_and_schedules_no_build():
    before = set(srv._sessions)

    with (
        patch("tui_gateway.server._profile_home", return_value=None),
        patch("tui_gateway.server._load_cfg", return_value={}),
        patch("tui_gateway.server._schedule_agent_build") as schedule_build,
        patch("tui_gateway.server._schedule_session_cap_enforcement") as schedule_cap,
    ):
        response = srv._methods["session.create"](
            "rid-1",
            {
                "model": "deepseek-v4-pro",
                "provider": "xiaomi",
                "source": "desktop",
            },
        )

    assert response["error"]["code"] == 4002
    assert "belongs to provider 'deepseek'" in response["error"]["message"]
    assert set(srv._sessions) == before
    schedule_build.assert_not_called()
    schedule_cap.assert_not_called()
