"""Saving a Bot profile model updates its idle chat without submitting a prompt."""
from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest
import tui_gateway.server as srv
from hermes_cli.model_switch import ModelSwitchResult


def test_profile_save_switches_idle_followers_and_publishes_model(tmp_path, monkeypatch):
    home = tmp_path / "bot"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home.joinpath("config.yaml").write_text("model:\n  default: before\n  provider: custom:test\n", encoding="utf8")
    monkeypatch.setattr("hermes_cli.model_selection_guards.combined_selection_warning", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda *, raw_input, explicit_provider, **kw:
                        ModelSwitchResult(success=True, new_model=raw_input, target_provider=explicit_provider))
    for name in ("_restart_slash_worker", "_persist_live_session_runtime", "_persist_live_session_system_prompt", "_append_model_switch_marker"):
        monkeypatch.setattr(srv, name, lambda *a, **kw: None)
    emitted = []
    monkeypatch.setattr(srv, "_emit_session_info", lambda sid, session: emitted.append((sid, session["agent"].model)))

    def session(**extra):
        agent = SimpleNamespace(model="before", provider="custom:test", base_url="", api_key="")
        def switch_model(**kw):
            agent.model, agent.provider = kw["new_model"], kw["new_provider"]
        agent.switch_model = switch_model
        return dict(agent=agent, profile_home=str(home), follow_profile_config=True,
                    config_model_seen=("before", "custom:test"), **extra)

    idle = session()
    busy = session(running=True)
    other = session()
    other["profile_home"] = str(tmp_path / "other")
    ordinary = session()
    ordinary["follow_profile_config"] = False
    monkeypatch.setattr(srv, "_sessions", {"idle": idle, "busy": busy, "other": other, "ordinary": ordinary})
    result = srv._methods["profiles.configure"]("save", {"name": "default", "model": "after", "provider": "custom:test"})["result"]
    assert result["applied"]["model"] is True
    assert yaml.safe_load(home.joinpath("config.yaml").read_text(encoding="utf8"))["model"]["default"] == "after"
    assert emitted == [("idle", "after")]
    assert idle["agent"].model == "after"
    assert all(s["agent"].model == "before" for s in (busy, other, ordinary))


def test_guarded_profile_save_does_not_sync_before_confirmation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.model_selection_guards.combined_selection_warning",
                        lambda *a, **kw: SimpleNamespace(message="Confirm selection"))
    switched = []
    monkeypatch.setattr(srv, "_sync_agent_model_with_config", lambda *a: switched.append(a))
    monkeypatch.setattr(srv, "_sessions", {"idle": {"agent": object(), "profile_home": str(tmp_path), "follow_profile_config": True}})
    result = srv._methods["profiles.configure"]("save", {"name": "default", "model": "after", "provider": "custom:test"})["result"]
    assert result["confirm_required"] is True
    assert not switched
    assert not tmp_path.joinpath("config.yaml").exists()


@pytest.mark.parametrize("failure", ["scope_enter", "scope_exit", "runtime_persist"])
def test_saved_model_survives_one_chat_sync_failure(tmp_path, monkeypatch, caplog, failure):
    home = tmp_path / "bot"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = home / "config.yaml"
    config.write_text("model:\n  default: before\n  provider: custom:test\n", encoding="utf8")
    monkeypatch.setattr("hermes_cli.model_selection_guards.combined_selection_warning", lambda *a, **kw: None)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda *, raw_input, explicit_provider, **kw:
                        ModelSwitchResult(success=True, new_model=raw_input, target_provider=explicit_provider))
    for name in ("_restart_slash_worker", "_persist_live_session_system_prompt", "_append_model_switch_marker"):
        monkeypatch.setattr(srv, name, lambda *a, **kw: None)

    def session(model="before"):
        agent = SimpleNamespace(model=model, provider="custom:test", base_url="", api_key="")
        def switch_model(**kw):
            agent.model, agent.provider = kw["new_model"], kw["new_provider"]
        agent.switch_model = switch_model
        return dict(agent=agent, profile_home=str(home), follow_profile_config=True,
                    config_model_seen=("before", "custom:test"))

    broken = session("after" if failure == "runtime_persist" else "before")
    healthy = session()
    if failure == "runtime_persist":
        # A superseded composer pin on an already-matching runtime persists
        # outside the model-switch transaction's own exception handler.
        broken["model_override"] = {"model": "after", "provider": "custom:test"}
        broken["composer_override_profile"] = {"model": "before", "provider": "custom:test"}
    original_scope = srv._session_profile_runtime_scope

    class Scope:
        def __init__(self, session, **kwargs):
            self.session = session
            self.inner = original_scope(session, **kwargs)

        def __enter__(self):
            if self.session is broken and failure == "scope_enter":
                raise RuntimeError("test scope entry failed")
            return self.inner.__enter__()

        def __exit__(self, *exc):
            result = self.inner.__exit__(*exc)
            if self.session is broken and failure == "scope_exit":
                raise RuntimeError("test scope exit failed")
            return result

    def persist(session):
        if session is broken and failure == "runtime_persist":
            raise OSError("test persistence failed")

    monkeypatch.setattr(srv, "_session_profile_runtime_scope", Scope)
    monkeypatch.setattr(srv, "_persist_live_session_runtime", persist)
    monkeypatch.setattr(srv, "_sessions", {"broken": broken, "healthy": healthy})
    emitted = []
    monkeypatch.setattr(srv, "_emit_session_info", lambda sid, session: emitted.append((sid, session["agent"].model)))

    try:
        response = srv._methods["profiles.configure"]("save", {"name": "default", "model": "after", "provider": "custom:test"})
    finally:
        monkeypatch.setattr(srv, "_session_profile_runtime_scope", original_scope)

    assert "error" not in response
    assert response["result"]["applied"]["model"] is True
    assert yaml.safe_load(config.read_text(encoding="utf8"))["model"]["default"] == "after"
    assert healthy["agent"].model == "after"
    assert ("healthy", "after") in emitted
    assert any("broken" in record.getMessage() and record.exc_info for record in caplog.records)
