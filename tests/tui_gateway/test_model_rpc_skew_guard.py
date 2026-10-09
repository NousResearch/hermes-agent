"""#99859 (R2): the TUI gateway's model-serving JSON-RPCs refuse on code skew.

The Desktop's Models page reaches the same inventory through ``model.options`` /
``model.save_key`` that the browser dashboard reaches through the guarded
``/api/model/options``; a backend kept alive across ``hermes update`` serves stale
``sys.modules`` and would resolve a post-update model string against them (the
reporter's ``agent_init_failed``). Both RPCs must answer the same "restart" error
the gateway's ``/model`` switch already returns, and never reach the payload build.
"""

from __future__ import annotations

from unittest.mock import Mock

import tui_gateway.server as server


def _call(method: str, params: dict | None = None) -> dict:
    return server._methods[method]("rid", params or {})


def test_stale_model_options_refuses_instead_of_building(tmp_path, monkeypatch):
    import gateway.code_skew as code_skew

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
    builds = []
    monkeypatch.setattr("hermes_cli.inventory.build_model_options_payload",
                        Mock(side_effect=lambda *a, **k: builds.append(1)))

    resp = _call("model.options")
    assert resp.get("error") is not None, resp
    assert resp["error"]["code"] == 5098
    assert "abc1234567" in resp["error"]["message"]
    assert "def4567890" in resp["error"]["message"]
    assert "restart" in resp["error"]["message"].lower()
    assert builds == []


def test_stale_model_save_key_refuses_instead_of_writing(tmp_path, monkeypatch):
    import gateway.code_skew as code_skew

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    resp = _call("model.save_key", {"slug": "zai", "api_key": "sk-canary"})
    assert resp.get("error") is not None, resp
    assert resp["error"]["code"] == 5098
    # The stale process never touches the key store.
    assert "sk-canary" not in (tmp_path / ".env").read_text(encoding="utf-8") if (tmp_path / ".env").exists() else True


def test_fresh_model_options_builds_payload_unchanged(tmp_path, monkeypatch):
    import gateway.code_skew as code_skew

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_sig", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    expected = {"providers": []}
    monkeypatch.setattr("hermes_cli.inventory.build_model_options_payload",
                        Mock(return_value=expected))

    resp = _call("model.options")
    assert "result" in resp, resp
    assert resp["result"] == expected
