"""Server-request params are built from what the TOOL side supplies, and the tools strip ``None``
values before handing the dict over (``drive_preview_tool``, ``tour_tool``, the skill-secret capture).
A nullable field without a default therefore rejected every legitimately sparse producer at the
model boundary — the request never reached the UI. Each case here goes through the real callback."""

from __future__ import annotations

import json

import pytest

from tools import drive_preview_tool, tour_tool
from tui_gateway import server


@pytest.fixture
def asked(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_ask", lambda method, sid, params, timeout=None: calls.append((method, params)) or json.dumps({"ok": True}))
    return calls


def test_drive_preview_elements_reaches_the_renderer_with_only_its_action(asked):
    callback = server._agent_cbs("sid")["drive_preview_callback"]

    result = json.loads(drive_preview_tool.drive_preview_tool(action="elements", callback=callback))

    assert result == {"ok": True}
    (method, params), = asked
    assert (method, params.action, params.ref, params.max) == ("preview.act", "elements", None, None)


def test_tour_start_accepts_title_only_steps(asked, monkeypatch):
    monkeypatch.setitem(server._sessions, "sid", {"tour_bridge": "answered"})
    callback = server._agent_cbs("sid")["tour_callback"]

    result = json.loads(tour_tool.tour_tool(action="start", steps=[{"title": "Welcome"}], callback=callback))

    assert result == {"ok": True}
    (method, params), = asked
    assert (method, params.steps[0].title, params.steps[0].selector) == ("tour", "Welcome", None)


def test_skill_secret_metadata_may_omit_help_and_required_for(asked, monkeypatch):
    import tools.skills_tool as skills_tool

    captured = {}
    monkeypatch.setattr(skills_tool, "set_secret_capture_callback", lambda cb: captured.setdefault("cb", cb))
    monkeypatch.setattr("tools.terminal_tool.set_sudo_password_callback", lambda cb: None)
    monkeypatch.setattr("tools.project_tools.set_project_workspace_callback", lambda cb: None)
    server._wire_callbacks("sid")
    asked.clear()
    monkeypatch.setattr(server, "_ask", lambda *a, **k: asked.append(a) or "")

    outcome = captured["cb"]("API_KEY", "Paste the key", {"skill_name": "weather"})

    assert outcome["skipped"] is True
    (_method, _sid, params), = asked
    assert (params.metadata.skill_name, params.metadata.help, params.metadata.required_for) == ("weather", None, None)
