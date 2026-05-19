from types import SimpleNamespace
from unittest.mock import patch


def test_session_info_includes_active_profile_name():
    from tui_gateway.server import _session_info

    agent = SimpleNamespace(
        model="gpt-test",
        reasoning_config={"enabled": True, "effort": "high"},
        service_tier=None,
        tools=[],
        total_tokens=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )

    with (
        patch("hermes_cli.profiles.get_active_profile_name", return_value="guilddali"),
        patch("hermes_cli.banner.get_update_result", return_value=None),
        patch("hermes_cli.config.recommended_update_command", return_value=""),
    ):
        info = _session_info(agent)

    assert info["profile"] == "guilddali"
