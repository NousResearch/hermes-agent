"""Config-sync must not rewrite a chat that already persisted a different model."""

import types

from tui_gateway import server


def _patch_config_model(monkeypatch, model, provider=""):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    cfg_model = {"default": model}
    if provider:
        cfg_model["provider"] = provider
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": cfg_model})


def test_config_sync_skips_session_with_different_persisted_model(monkeypatch):
    _patch_config_model(monkeypatch, "antigravity/gemini-3.8-flash-tiered", provider="omniroute")
    db = types.SimpleNamespace(get_session=lambda key: {"model": "id/xai/grok-4.7"})
    session = {
        "agent": types.SimpleNamespace(
            model="id/xai/grok-4.7",
            provider="omniroute",
            _session_db=db,
            session_id="session-key",
        ),
        "session_key": "session-key",
        "config_model_seen": ("antigravity/gemini-3.8-flash-tiered", "custom"),
    }
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append((sid, raw, kw)),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == []
    assert session["config_model_seen"] == ("antigravity/gemini-3.8-flash-tiered", "omniroute")


def test_config_sync_still_adopts_when_persisted_model_matches_default(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="nous")
    db = types.SimpleNamespace(get_session=lambda key: {"model": "new/model"})
    session = {
        "agent": types.SimpleNamespace(model="old/model", _session_db=db),
        "session_key": "session-key",
        "config_model_seen": ("old/model", "nous"),
    }
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model --provider nous"]


def test_config_sync_still_adopts_when_db_unreadable(monkeypatch):
    _patch_config_model(monkeypatch, "new/model", provider="nous")
    session = {
        "agent": types.SimpleNamespace(model="old/model"),
        "session_key": "session-key",
        "config_model_seen": ("old/model", "nous"),
    }
    calls = []
    monkeypatch.setattr(
        server,
        "_apply_model_switch",
        lambda sid, sess, raw, **kw: calls.append(raw),
    )

    server._sync_agent_model_with_config("sid", session)

    assert calls == ["new/model --provider nous"]
