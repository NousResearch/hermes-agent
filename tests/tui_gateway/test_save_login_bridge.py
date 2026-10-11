"""The gateway's save-login bridge separates "nobody could ask" from "asked and declined".

``_wire_callbacks`` installs the bridge as the process's save-login surface. A client that cannot
render the card (older build, no handler for ``vault.save_login``) never answers the server request
and ``server_requests.send`` returns None; folding that into the same None the card's decline answers
made ``browser_vault_save_login`` report "the user chose not to save" on hosted TUIs where no card
ever appeared (#135420). The bridge now raises SaveLoginPromptUnavailable for the unanswered case;
an answered empty value stays a decline.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.vault_backends.unlock import SaveLoginPromptUnavailable  # noqa: E402
from tui_gateway import agent_callbacks, server_requests  # noqa: E402

bridge = agent_callbacks._save_login_prompt


def test_unanswered_request_raises_unavailable(monkeypatch):
    """None from send() = no renderer answered (old client / no handler / cancelled): NOT a decline."""
    monkeypatch.setattr(server_requests, "send", lambda *a, **k: None)
    with pytest.raises(SaveLoginPromptUnavailable):
        bridge("sid-1", "https://acme.test", "acme.test")


def test_answered_empty_value_is_a_decline(monkeypatch):
    """The card's decline path answers an empty value; the bridge maps it to None, not an error."""
    monkeypatch.setattr(server_requests, "send", lambda *a, **k: {"value": ""})
    assert bridge("sid-1", "https://acme.test", "acme.test") is None


def test_answered_login_json_is_returned(monkeypatch):
    sent = {}

    def fake_send(method, sid, params, *, timeout, **kw):
        sent.update(method=method, sid=sid, params=params, timeout=timeout)
        return {"value": json.dumps({"identifier": "tek@acme.test", "password": "pw"})}

    monkeypatch.setattr(server_requests, "send", fake_send)
    answer = bridge("sid-1", "https://acme.test", "acme.test")
    assert answer == {"identifier": "tek@acme.test", "password": "pw"}
    assert sent["method"] == "vault.save_login"
    assert sent["sid"] == "sid-1"
    assert sent["params"] == {"origin": "https://acme.test", "site": "acme.test"}
    assert sent["timeout"] == 180


def test_answered_garbage_is_a_decline_not_a_crash(monkeypatch):
    monkeypatch.setattr(server_requests, "send", lambda *a, **k: {"value": "not json {"})
    assert bridge("sid-1", "https://acme.test", "acme.test") is None
    monkeypatch.setattr(
        server_requests, "send", lambda *a, **k: {"value": json.dumps({"identifier": "x"})}
    )
    assert bridge("sid-1", "https://acme.test", "acme.test") is None
