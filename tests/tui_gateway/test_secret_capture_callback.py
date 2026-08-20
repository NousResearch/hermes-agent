from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server():
    with patch.dict(
        "sys.modules",
        {
            "hermes_constants": MagicMock(
                get_hermes_home=MagicMock(
                    return_value=Path("/tmp/hermes_test_secret_callback")
                )
            ),
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        },
    ):
        import importlib

        yield importlib.import_module("tui_gateway.server")


def test_secret_capture_callback_stays_with_its_ui_session(server, monkeypatch):
    import tools.skills_tool as skills
    from gateway.session_context import clear_session_vars, set_session_vars

    emitted = []
    monkeypatch.setattr(
        server,
        "_block",
        lambda event, sid, payload, timeout=300: emitted.append(
            (event, sid, payload)
        ) or "",
    )

    server._wire_callbacks("session-A")
    server._wire_callbacks("session-B")

    def capture_for(session_id):
        tokens = set_session_vars(ui_session_id=session_id)
        try:
            return skills._capture_required_environment_variables(
                "demo-skill", [{"name": "DEMO_TOKEN", "prompt": "Token"}]
            )
        finally:
            clear_session_vars(tokens)

    result = capture_for("session-A")
    second_result = capture_for("session-B")

    assert result["setup_skipped"] is True
    assert second_result["setup_skipped"] is True
    assert emitted == [
        (
            "secret.request",
            "session-A",
            {
                "prompt": "Token",
                "env_var": "DEMO_TOKEN",
                "metadata": {"skill_name": "demo-skill"},
            },
        ),
        (
            "secret.request",
            "session-B",
            {
                "prompt": "Token",
                "env_var": "DEMO_TOKEN",
                "metadata": {"skill_name": "demo-skill"},
            },
        ),
    ]
