"""Resumed and pinned TUI sessions follow config.yaml model changes.

Config-driven model routing (e.g. a quota router rewriting ``model.default`` to avoid
overages) only works if open and resumed sessions actually adopt the new config model.
Before this contract, resume converted the stored row's model into a session pin and the
per-turn sync skipped every pinned session, so a resumed chat stayed on the model it was
saved with no matter how many turns ran or how config changed.

Contract:
* An explicit ``/model`` pick records the config model it diverged from
  (``composer_override_profile``), for every session, not only Bot Chats.
* Resume restores a stored model only while that marker still equals the config model.
  An unmarked row (its model came from config routing) or a stale marker follows config.
* The per-turn sync supersedes a pin once config moves away from the pick's marker
  (or, for a legacy unmarked pin, away from the session's config baseline).
* A failed adoption is retried on later turns (notified once per config target), so a
  transient failure cannot strand the session on the old model.
"""

from __future__ import annotations

import json
import types
from unittest.mock import patch

import pytest

from hermes_state import SessionDB

import tui_gateway.server as server

OLD = ("gpt-old", "openai-codex")
NEW = ("claude-new", "anthropic")


def _target(monkeypatch, model, provider):
    monkeypatch.setattr(server, "_config_model_target", lambda: (model, provider))


def _row(model_config_extra=None, **row_extra):
    model_config = {"model": OLD[0], "provider": OLD[1], **(model_config_extra or {})}
    return {"model": OLD[0], "billing_provider": OLD[1], "model_config": json.dumps(model_config), **row_extra}


class TestStoredOverridesFollowConfig:
    def test_unmarked_row_follows_config_model(self, monkeypatch):
        """The kanban incident: a row whose model came from config routing must not pin on resume."""
        _target(monkeypatch, *NEW)
        assert server._stored_session_runtime_overrides(_row()) == {}

    def test_explicit_pick_restored_while_config_unchanged(self, monkeypatch):
        _target(monkeypatch, *NEW)
        row = _row({"composer_override_profile": {"model": NEW[0], "provider": NEW[1]}})
        overrides = server._stored_session_runtime_overrides(row)
        assert overrides["model_override"]["model"] == OLD[0]
        assert overrides["provider_override"] == OLD[1]

    def test_explicit_pick_dropped_once_config_moves(self, monkeypatch):
        _target(monkeypatch, *NEW)
        row = _row({"composer_override_profile": {"model": "older-default", "provider": "nous"}})
        assert server._stored_session_runtime_overrides(row) == {}

    def test_unmarked_row_restored_when_config_has_no_model(self, monkeypatch):
        """No config model = nothing to follow; keep the stored runtime rather than an arbitrary default."""
        _target(monkeypatch, "", "")
        assert server._stored_session_runtime_overrides(_row())["model_override"]["model"] == OLD[0]


class TestExplicitPickRecordsMarker:
    def test_model_command_on_normal_session_records_config_marker(self, monkeypatch):
        _target(monkeypatch, *NEW)
        agent = types.SimpleNamespace(model=NEW[0], provider=NEW[1], base_url="", api_key="", api_mode="")
        session = {"agent": agent, "session_key": "k", "model_override": None}
        result = types.SimpleNamespace(
            success=True, error_message="", new_model=OLD[0], target_provider=OLD[1], api_key="k",
            base_url="", api_mode="", warning_message="", runtime_capabilities=None)
        monkeypatch.setattr(server, "_commit_agent_switch", lambda *a, **k: None)
        with (
            patch("hermes_cli.model_switch.resolve_persist_behavior", return_value=False),
            patch("hermes_cli.model_switch.switch_model", return_value=result),
        ):
            server._apply_model_switch("sid", session, f"{OLD[0]} --provider {OLD[1]}")
        assert session["model_override"]["model"] == OLD[0]
        assert session["composer_override_profile"] == {"model": NEW[0], "provider": NEW[1]}


def _live(**extra):
    session = {"agent": types.SimpleNamespace(model=OLD[0], provider=OLD[1]), "session_key": "k"}
    session.update(extra)
    return session


def _record_switches(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "_apply_model_switch", lambda sid, sess, raw, **kw: calls.append(raw))
    return calls


class TestLiveSyncSupersedesPins:
    def test_pin_superseded_when_config_moves_from_marker(self, monkeypatch):
        _target(monkeypatch, *NEW)
        calls = _record_switches(monkeypatch)
        session = _live(model_override={"model": OLD[0], "provider": OLD[1]},
                        composer_override_profile={"model": "older-default", "provider": "nous"},
                        config_model_seen=("older-default", "nous"))
        server._sync_agent_model_with_config("sid", session)
        assert calls == [f"{NEW[0]} --provider {NEW[1]}"]
        assert "model_override" not in session
        assert session["composer_override_profile"] is None

    def test_pin_kept_while_config_equals_marker(self, monkeypatch):
        _target(monkeypatch, *NEW)
        calls = _record_switches(monkeypatch)
        session = _live(model_override={"model": OLD[0], "provider": OLD[1]},
                        composer_override_profile={"model": NEW[0], "provider": NEW[1]},
                        config_model_seen=("older-default", "nous"))
        server._sync_agent_model_with_config("sid", session)
        assert calls == []
        assert session["model_override"]["model"] == OLD[0]

    def test_unmarked_pin_superseded_when_config_moves_from_baseline(self, monkeypatch):
        _target(monkeypatch, *NEW)
        calls = _record_switches(monkeypatch)
        session = _live(model_override={"model": OLD[0], "provider": OLD[1]}, config_model_seen=OLD)
        server._sync_agent_model_with_config("sid", session)
        assert calls == [f"{NEW[0]} --provider {NEW[1]}"]
        assert "model_override" not in session

    def test_unmarked_pin_without_baseline_is_kept(self, monkeypatch):
        """No baseline = no evidence config moved since the pick; never yank a fresh pin."""
        _target(monkeypatch, *NEW)
        calls = _record_switches(monkeypatch)
        session = _live(model_override={"model": OLD[0], "provider": OLD[1]})
        server._sync_agent_model_with_config("sid", session)
        assert calls == []
        assert session["model_override"]["model"] == OLD[0]


class TestFailedAdoptionRetries:
    def test_failed_switch_retried_next_turn_and_notified_once(self, monkeypatch):
        _target(monkeypatch, *NEW)
        attempts, emits = [], []

        def fail_then_succeed(sid, sess, raw, **kw):
            attempts.append(raw)
            if len(attempts) < 3:
                raise ValueError("anthropic package missing")

        monkeypatch.setattr(server, "_apply_model_switch", fail_then_succeed)
        monkeypatch.setattr(server, "_emit", lambda ev, sid, payload: emits.append(ev))
        session = _live(config_model_seen=OLD)
        for _turn in range(4):
            server._sync_agent_model_with_config("sid", session)
        assert len(attempts) == 3  # two failures retried, then success stops further attempts
        assert emits == ["error"]
        assert session["config_model_seen"] == NEW


class TestResumeEndToEnd:
    def test_normal_session_pick_survives_resume_until_config_moves(self, monkeypatch, tmp_path):
        """Production path on a normal (non Bot Chat) session: /model records the marker into the real
        SessionDB row; deferred session.resume restores the pick while config.yaml is unchanged and
        drops it once config moves (the quota-router case)."""
        home = tmp_path / "home"
        home.mkdir()
        cfg = home / "config.yaml"
        cfg.write_text("model:\n  default: claude-new\n  provider: anthropic\n", encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
        stored = "20261008-000000-norm"
        db = SessionDB(db_path=home / "state.db")
        db.create_session(stored, "tui", model=NEW[0], model_config={"model": NEW[0], "provider": NEW[1]})
        db.append_message(stored, "user", "hi")
        db.append_message(stored, "assistant", "hello")

        class _FakeAgent:
            model, provider, base_url, api_key, api_mode = NEW[0], NEW[1], "", "", ""
            _session_db = db

            def switch_model(self, **kw):
                self.model, self.provider = kw["new_model"], kw["new_provider"]

        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(server, "_hermes_home", str(home))
        monkeypatch.setattr(server, "_get_db", lambda: SessionDB(db_path=home / "state.db"))
        for name in ("_enable_gateway_prompts", "_schedule_resume_hydration", "_schedule_session_cap_enforcement",
                     "_emit", "_restart_slash_worker"):
            monkeypatch.setattr(server, name, lambda *a, **k: None)
        monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
        monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
        live = {"agent": _FakeAgent(), "session_key": stored, "model_override": None}
        result = types.SimpleNamespace(
            success=True, error_message="", new_model=OLD[0], target_provider=OLD[1], api_key="k",
            base_url="", api_mode="", warning_message="", runtime_capabilities=None)
        known = set(server._sessions)
        try:
            with (
                patch("hermes_cli.model_switch.resolve_persist_behavior", return_value=False),
                patch("hermes_cli.model_switch.switch_model", return_value=result),
            ):
                server._apply_model_switch("sid-live", live, OLD[0])

            row = json.loads(db.get_session(stored)["model_config"])
            assert row["composer_override_profile"] == {"model": NEW[0], "provider": NEW[1]}

            def resume():
                resp = server.handle_request({"id": "1", "method": "session.resume", "params": {
                    "session_id": stored, "source": "tui", "defer_history": True, "omit_messages": True}})
                assert "error" not in resp, resp
                with server._sessions_lock:
                    return server._sessions.pop(resp["result"]["session_id"])

            record = resume()
            assert record["model_override"]["model"] == OLD[0]
            assert record["composer_override_profile"] == {"model": NEW[0], "provider": NEW[1]}

            cfg.write_text("model:\n  default: cheaper-model\n  provider: nous\n", encoding="utf-8")
            assert resume().get("model_override") is None
        finally:
            db.close()
            with server._sessions_lock:
                for sid in [s for s in server._sessions if s not in known]:
                    server._sessions.pop(sid, None)
