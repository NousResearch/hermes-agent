"""``config.set`` refuses a key the administrator's managed scope pins, like ``hermes config set``.

Before, both TUI paths wrote the pinned key and answered success while the managed value kept
winning on every read. A managed scope that pins something else must not block other keys.
"""
from types import SimpleNamespace

import pytest
import hermes_agent.hermes_yaml as yaml

from hermes_agent.hermes_cli import managed_scope
from hermes_agent.tui_gateway import server


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import managed_scope
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: corp\n  sections:\n    tools: collapsed\n"
                                         "custom_prompt: policy\nmodel:\n  default: acme/pinned\n")
    (home / "config.yaml").write_text("display:\n  tui_theme: dark\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr(server, "_hermes_home", home)
    managed_scope.invalidate_managed_cache()
    server._cfg_cache = server._cfg_sig = server._cfg_path = None
    yield home
    managed_scope.invalidate_managed_cache()
    server._cfg_cache = server._cfg_sig = server._cfg_path = None


def _set(**params):
    return server._methods["config.set"](1, params)


@pytest.mark.parametrize(("key", "value", "pinned"), [
    ("skin", "mono", "display.skin"),  # one-key funnel (_write_config_keys)
    ("details_mode", "expanded", "display.sections.tools"),  # fan-out to every section leaf
    ("prompt", "mine", "custom_prompt"),  # its own raw save
])
def test_tui_config_set_refuses_a_managed_key_and_writes_nothing(homes, key, value, pinned):
    before = (homes / "config.yaml").read_text()
    answer = _set(key=key, value=value)
    assert answer["error"]["code"] == 4030, answer
    assert pinned in answer["error"]["message"] and "managed by your administrator" in answer["error"]["message"]
    assert (homes / "config.yaml").read_text() == before


_DEFINITE = [
    ("other/model --global", "model.default", None),
    ("other/model --global --reasoning high", "agent.reasoning_effort", "agent:\n  reasoning_effort: low\n"),
    ("other/model --global", "model.api_mode", "model:\n  api_mode: codex_responses\n"),  # always set or cleared
]
_CLEARED = ("other/model --global", "model.context_length", "model:\n  context_length: 8192\n")  # route changed


@pytest.mark.parametrize(("running", "pick", "pinned", "managed"), [
    *((running, *row) for running in (True, False) for row in _DEFINITE), (False, *_CLEARED)])
def test_tui_persisting_model_pick_is_refused_before_anything_changes(homes, monkeypatch, running, pick, pinned, managed):
    """Refused before the live switch, the model persist or the mid-turn queue, not after a partial success."""
    from hermes_agent.hermes_cli.model_switch import ModelSwitchResult
    if managed:
        (homes.parent / "managed" / "config.yaml").write_text(managed)
        managed_scope.invalidate_managed_cache()
    (homes / "config.yaml").write_text("model:\n  default: old/model\n  provider: nous\n  context_length: 4096\n")
    monkeypatch.setenv("HERMES_HOME", str(homes))
    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="other/model", target_provider="openrouter", is_global=True))
    agent = SimpleNamespace(model="old/model", provider="nous", reasoning_config=None)  # no switch_model: never swapped
    server._sessions["managed-pick"] = session = {"session_key": "managed-pick", "agent": agent, "running": running}
    before = (homes / "config.yaml").read_text()
    try:
        answer = _set(key="model", value=pick, session_id="managed-pick")
    finally:
        server._sessions.pop("managed-pick", None)
    assert answer["error"]["code"] == 4030, answer
    assert pinned in answer["error"]["message"]
    assert session == {"session_key": "managed-pick", "agent": agent, "running": running}  # nothing queued
    assert vars(agent) == {"model": "old/model", "provider": "nous", "reasoning_config": None}
    assert (homes / "config.yaml").read_text() == before


def test_tui_config_set_still_writes_a_key_the_scope_does_not_pin(homes):
    assert _set(key="theme", value="light")["result"] == {"key": "theme", "value": "light"}
    assert yaml.safe_load((homes / "config.yaml").read_text())["display"]["tui_theme"] == "light"


def _queue_then_apply(homes, monkeypatch, raw_model, pick, resolved):
    """Queue ``pick`` on a busy session (live agent on another model), then apply it at turn start."""
    from hermes_agent.hermes_cli.model_switch import ModelSwitchResult
    (homes.parent / "managed" / "config.yaml").write_text("model:\n  context_length: 8192\n")
    managed_scope.invalidate_managed_cache()
    (homes / "config.yaml").write_text(raw_model)
    monkeypatch.setenv("HERMES_HOME", str(homes))
    monkeypatch.setattr("hermes_agent.hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, is_global=True, **resolved))
    swaps, notices = [], []
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt",
                 "_append_model_switch_marker", "_emit_session_info", "_merge_preflight_warning"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: notices.append((event, payload)))
    agent = SimpleNamespace(model="live/model", provider="nous", base_url="", api_key="", reasoning_config=None,
                            switch_model=lambda **kw: swaps.append(kw))
    server._sessions["queued-pick"] = session = {"session_key": "queued-pick", "agent": agent, "running": True}
    try:
        answer = _set(key="model", value=pick, session_id="queued-pick", confirm_expensive_model=True)
        session["running"] = False
        server._apply_pending_model_switch("queued-pick", session)
    finally:
        server._sessions.pop("queued-pick", None)
    return answer, session, swaps, notices


def test_tui_queued_same_route_pick_keeps_a_pinned_context_length(homes, monkeypatch):
    """Only model.context_length is pinned and the same route keeps it: queued (as on main), then applied
    without touching it. The queue-time check must not count a clear the resolved plan doesn't make."""
    raw = "model:\n  default: same/model\n  provider: custom\n  base_url: http://localhost:1234/v1\n  context_length: 4096\n"
    answer, session, swaps, notices = _queue_then_apply(homes, monkeypatch, raw, "same/model --global", {
        "new_model": "same/model", "target_provider": "custom", "base_url": "http://localhost:1234/v1"})
    assert answer["result"]["deferred"] is True, answer
    assert [kw["new_model"] for kw in swaps] == ["same/model"]
    assert not [p for e, p in notices if e == "notification.show"]
    model = yaml.safe_load((homes / "config.yaml").read_text())["model"]
    assert (model["default"], model["provider"], model["context_length"]) == ("same/model", "custom", 4096)
    assert session["model_override"]["model"] == "same/model"


def test_tui_queued_route_change_that_clears_a_pinned_context_length_is_refused_at_apply(homes, monkeypatch):
    """The resolved plan clears the pinned model.context_length: refused when the queued pick is applied,
    shown as the dropped-switch notice with the CLI's refusal text; nothing switched or saved."""
    raw = "model:\n  default: old/model\n  provider: nous\n  context_length: 4096\n"
    answer, session, swaps, notices = _queue_then_apply(homes, monkeypatch, raw, "other/model --global", {
        "new_model": "other/model", "target_provider": "openrouter"})
    assert answer["result"]["deferred"] is True, answer
    assert swaps == [] and "model_override" not in session and "pending_model_switch" not in session
    assert (homes / "config.yaml").read_text() == raw
    [text] = [p["text"] for e, p in notices if e == "notification.show" and p["key"] == "model_switch.dropped"]
    assert text.startswith("Could not switch model: Cannot set 'model.context_length': it is managed by your administrator")
