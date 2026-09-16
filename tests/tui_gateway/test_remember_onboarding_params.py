"""``profiles.remember_onboarding`` accepts what a current-main Desktop sends and keeps wizard state out of memory."""

from __future__ import annotations

import tui_gateway.onboarding_personalization as personalization
from tui_gateway import server


def test_remember_onboarding_accepts_committed_and_never_writes_it_as_focus(monkeypatch):
    """main's Desktop spreads its whole onboarding store into the call, ``committed`` (completed wizard-step ids)
    included; the model must accept it and the writer must never see it, least of all as ``focus``."""
    seen = {}
    monkeypatch.setattr(personalization, "remember_onboarding",
                        lambda answers: seen.update(answers) or {"saved": True, "profile": "default", "target": "user"})

    response = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "profiles.remember_onboarding", "params": {
        "answers": {"accent": None, "committed": ["name", "context"], "connectors": ["GitHub"], "context": "docs",
                    "name": "Kay", "layout": "basic"}}})

    assert response["result"]["saved"] is True, response
    assert "committed" not in seen
    assert seen["focus"] == [] and seen["connectors"] == ["GitHub"] and seen["name"] == "Kay"
