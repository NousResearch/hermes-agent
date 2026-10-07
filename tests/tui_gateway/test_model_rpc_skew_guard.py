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


# The switch itself goes through ``config.set`` key=model (Desktop picker, Desktop /model, the TUI):
# it must refuse on skew too, whether the pick would apply live or be stashed for the next turn.
def _skewed(monkeypatch):
    import gateway.code_skew as code_skew

    monkeypatch.setattr(code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890"))


def _no_live_switch(monkeypatch):
    import tui_gateway.methods_config_set as methods_config_set

    applied = []
    for module in (methods_config_set, server):
        if hasattr(module, "_apply_model_switch"):
            monkeypatch.setattr(module, "_apply_model_switch", lambda sid, *a, **k: applied.append(sid) or {})
    return applied


def test_stale_config_set_model_refuses_the_live_switch(monkeypatch):
    import threading

    _skewed(monkeypatch)
    applied = _no_live_switch(monkeypatch)
    monkeypatch.setitem(server._sessions, "idle", {"running": False, "agent": object(),
                                                   "agent_ready": threading.Event()})

    resp = _call("config.set", {"key": "model", "value": "gpt-5.5 --provider openrouter", "session_id": "idle"})

    assert resp.get("error", {}).get("code") == 5098, resp
    assert applied == []


def test_stale_config_set_model_refuses_to_stash_for_the_next_turn(monkeypatch):
    _skewed(monkeypatch)
    monkeypatch.setitem(server._sessions, "busy", {"running": True, "agent": object()})

    resp = _call("config.set", {"key": "model", "value": "gpt-5.5 --provider openrouter", "session_id": "busy"})

    assert resp.get("error", {}).get("code") == 5098, resp
    assert "pending_model_switch" not in server._sessions["busy"]


def test_pick_stashed_before_the_skew_is_dropped_at_turn_start(monkeypatch):
    _skewed(monkeypatch)
    applied = []
    monkeypatch.setattr(server, "_apply_model_switch", lambda sid, *a, **k: applied.append(sid) or {})
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *a: emitted.append(a))
    session = {"agent": object(), "pending_model_switch": {"raw": "gpt-5.5 --provider openrouter"}}

    server._apply_pending_model_switch("s", session)

    assert applied == []
    assert "pending_model_switch" not in session
    assert emitted and emitted[0][0] == "error" and "restart" in emitted[0][2]["message"].lower()


def test_config_adoption_is_skipped_and_left_unseen(monkeypatch):
    # `hermes update` moves the checkout and often the configured model too; the turn-start adoption
    # must not apply it on stale code, and must stay unseen so it lands after the restart.
    _skewed(monkeypatch)
    applied = []
    monkeypatch.setattr(server, "_apply_model_switch", lambda sid, *a, **k: applied.append(sid) or {})
    monkeypatch.setattr(server, "_config_model_target", lambda: ("config-model-x", "configprov"))
    session = {"agent": Mock(model="old-model", provider="oldprov")}

    server._sync_agent_model_with_config("s", session)

    assert applied == []
    assert "config_model_seen" not in session


def test_apply_model_switch_itself_refuses(monkeypatch):
    # Covers every caller that has no envelope of its own: the /model slash mirror, config adoption, /moa.
    import pytest

    _skewed(monkeypatch)
    with pytest.raises(RuntimeError, match="(?i)restart"):
        server._apply_model_switch("s", {"agent": Mock()}, "gpt-5.5 --provider openrouter")


def test_stale_slash_exec_model_refuses_before_the_worker(monkeypatch):
    # Desktop sends a typed `/model <name>` through slash.exec (runExec), not config.set.
    _skewed(monkeypatch)
    worker = Mock()
    monkeypatch.setitem(server._sessions, "desk", {"agent": object(), "slash_worker": worker,
                                                   "session_key": "k", "running": False})

    resp = _call("slash.exec", {"command": "/model gpt-5.5 --provider openrouter", "session_id": "desk"})

    assert resp.get("error", {}).get("code") == 5098, resp
    worker.run.assert_not_called()


def test_stale_moa_one_shot_refuses_on_a_lazy_session(monkeypatch):
    # A session with no agent yet takes /moa as an override the first build consumes, without ever
    # reaching _apply_model_switch; it must refuse up front too.
    _skewed(monkeypatch)
    monkeypatch.setitem(server._sessions, "lazy", {"agent": None, "session_key": "k", "running": False})

    resp = _call("command.dispatch", {"name": "moa", "arg": "compare these", "session_id": "lazy"})

    assert resp.get("error", {}).get("code") == 5098, resp
    assert "model_override" not in server._sessions["lazy"]
    assert "moa_one_shot_restore" not in server._sessions["lazy"]
