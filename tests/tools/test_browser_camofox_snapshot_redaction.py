"""Regression for #122174: snapshot-derived Camofox results share the secret boundary."""
import json
from unittest.mock import Mock

import pytest

from agent import redact
from tools import browser_camofox as camofox
from tools import browser_tool as browser
from tools.registry import registry


SECRET = "synthetic-vault-value"


@pytest.fixture
def page(monkeypatch):
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    redact.register_vault_redaction_value(SECRET)
    monkeypatch.setattr(browser, "_is_camofox_mode", lambda: True)
    monkeypatch.setattr(camofox, "_sessions", {
        "redaction": {"user_id": "fixture-user", "tab_id": "fixture-tab", "session_key": "fixture"}
    })
    monkeypatch.setattr(camofox, "get_vnc_url", lambda: None)
    monkeypatch.setattr(camofox, "get_camofox_url", lambda: "http://127.0.0.1:9377")
    monkeypatch.setattr(browser, "get_browser_snapshot_threshold", lambda: 50000)
    data = {"snapshot": f'- textbox "Password" [e1]: {SECRET}\n- button "Submit" [e2]', "refsCount": 2}
    response = Mock()
    response.json.side_effect = lambda: dict(data)
    monkeypatch.setattr(camofox.requests, "get", lambda *a, **kw: response)
    posted = Mock()
    posted.json.return_value = {"url": "https://example.com", "title": "Fixture"}
    monkeypatch.setattr(camofox.requests, "post", lambda *a, **kw: posted)
    return data


def dispatch(name, **kwargs):
    return json.loads(registry.dispatch(name, kwargs, task_id="redaction"))


def test_snapshot_derived_outputs_share_vault_redaction(page, monkeypatch):
    result = dispatch("browser_navigate", url="https://example.com")
    assert result["success"] is True
    assert SECRET not in result["snapshot"]
    assert "redacted-vault-secret" in result["snapshot"]
    assert '[e1]' in result["snapshot"] and 'Submit' in result["snapshot"]
    assert result["element_count"] == 2

    from hermes_constants import get_hermes_dir
    monkeypatch.setattr(browser, "get_browser_snapshot_threshold", lambda: 500)
    page["snapshot"] += '\n' + '\n'.join(f'- text "row {i} {SECRET}"' for i in range(40))
    result = dispatch("browser_navigate", url="https://example.com")
    assert result["success"] is True
    assert SECRET not in result["snapshot"]
    assert "truncated" in result["snapshot"]
    files = list(get_hermes_dir("cache/web", "web_cache").glob("browser-snapshot-*.txt"))
    assert files
    assert all(SECRET not in f.read_text(encoding="utf-8") for f in files)

    page["snapshot"] = f'- img "{SECRET}" [e1]\n  /url: https://example.com/{SECRET}.png'
    result = dispatch("browser_get_images")
    assert result["success"] is True
    assert result["count"] == 1
    assert SECRET not in json.dumps(result["images"])
    assert "redacted-vault-secret" in result["images"][0]["alt"]


def test_snapshot_public_content_and_error_contracts_preserved(page, monkeypatch):
    for text in ["", '- heading "Public" [e1]']:
        page["snapshot"] = text
        assert dispatch("browser_snapshot")["snapshot"] == text
    assert dispatch("browser_get_images")["images"] == []

    def fail(*args, **kwargs):
        raise RuntimeError("fixture snapshot unavailable")
    monkeypatch.setattr(camofox.requests, "get", fail)
    assert dispatch("browser_snapshot")["success"] is False
    result = dispatch("browser_navigate", url="https://example.com")
    assert result["success"] is True
    assert "snapshot" not in result
