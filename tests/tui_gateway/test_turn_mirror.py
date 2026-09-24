"""Turn mirroring (``tui_gateway/turn_mirror.py``): gateway-owned sessions continued from a local
surface (Desktop chat, TUI, dashboard) optionally forward each completed exchange to the session's
own platform chat.

Contract pinned here:

* ``compose_mirror_message`` renders marker + quoted user message + reply, multi-line, keeping
  blank lines inside the quote;
* ``build_mirror_plan`` is the single decision point: synthetic turns (``display_kind``,
  ``__``-prefixed rids), silent/empty replies, empty user text, missing targets and disabled
  config all yield ``None``; the split switch maps to page-indicator retention;
* ``resolve_mirror_target`` reads the session row fail-closed — missing pieces skip rather than
  guess a delivery target;
* ``mirror_turn`` records a durable obligation before spawning delivery, deduplicates the same
  turn, and only spawns delivery for complete turns of gateway-owned sessions;
* ``_deliver`` forwards the silent/split options to the standalone sender, while profile-scoped
  ledger writes leave the launch environment unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import time
from types import SimpleNamespace
from typing import Any

import pytest

from tui_gateway import turn_mirror


def _cfg(**platform_overrides):
    return {"display": {"platforms": {"telegram": platform_overrides}}}


def _patch_db(monkeypatch, db_or_exc):
    """Patch the session-DB context manager ``turn_mirror`` resolves at call time."""

    @contextlib.contextmanager
    def fake_db(session):
        if isinstance(db_or_exc, Exception):
            raise db_or_exc
        yield db_or_exc

    monkeypatch.setattr("tui_gateway.session_workdir._session_db", fake_db)


class _FakeDB:
    def __init__(self, row):
        self._row = row

    def get_session(self, session_id):
        return self._row


class TestComposeMirrorMessage:
    def test_quotes_multiline_and_keeps_blank_lines(self):
        msg = turn_mirror.compose_mirror_message("line1\n\nline3", "reply", "📲")
        assert msg == "📲\n\n> line1\n>\n> line3\n\nreply"

    def test_empty_label_hides_the_marker(self):
        assert turn_mirror.compose_mirror_message("hi", "yo", "") == "> hi\n\nyo"

    def test_reply_is_right_trimmed(self):
        assert turn_mirror.compose_mirror_message("q", "reply\n\n\n", "") == "> q\n\nreply"


class TestMirrorSettings:
    def test_disabled_by_default(self):
        assert turn_mirror.mirror_settings({}, "telegram") is None

    def test_enabled_resolution_and_defaults(self):
        assert turn_mirror.mirror_settings(_cfg(mirror_local_turns=True), "telegram") == {
            "silent": True, "chunk_indicators": True, "label": "📲",
        }

    def test_truthy_tokens_normalise(self):
        assert turn_mirror.mirror_settings(_cfg(mirror_local_turns="off"), "telegram") is None
        assert turn_mirror.mirror_settings(_cfg(mirror_local_turns="on"), "telegram") is not None
        settings = turn_mirror.mirror_settings(
            _cfg(mirror_local_turns=True, mirror_local_turns_silent=False), "telegram")
        assert settings is not None
        assert settings["silent"] is False

    def test_split_switch_maps_to_chunk_indicators(self):
        plain = turn_mirror.mirror_settings(_cfg(mirror_local_turns=True), "telegram")
        split = turn_mirror.mirror_settings(
            _cfg(mirror_local_turns=True, mirror_local_turns_4096_split=True), "telegram")
        assert plain is not None and split is not None
        assert plain["chunk_indicators"] is True
        assert split["chunk_indicators"] is False


def _plan(**overrides: Any):
    kwargs: dict[str, Any] = dict(
        platform="telegram", chat_id="123", thread_id=None,
        user_text="hello", reply_text="world", display_kind=None, rid="rid-1",
        cfg=_cfg(mirror_local_turns=True),
    )
    kwargs.update(overrides)
    return turn_mirror.build_mirror_plan(**kwargs)


class TestBuildMirrorPlan:
    def test_basic_plan_fields(self):
        plan = _plan(thread_id="77")
        assert plan is not None
        assert plan.platform == "telegram"
        assert plan.chat_id == "123"
        assert plan.thread_id == "77"
        assert plan.message == "📲\n\n> hello\n\nworld"
        assert plan.silent is True
        assert plan.chunk_indicators is True

    def test_synthetic_turns_blocked(self):
        assert _plan(display_kind="auto_continue") is None
        assert _plan(rid="__auto_continue__123") is None

    def test_silent_and_empty_replies_blocked(self):
        assert _plan(reply_text="[SILENT]") is None
        assert _plan(reply_text=" [SILENT] ") is None
        assert _plan(reply_text="") is None
        assert _plan(reply_text=None) is None

    def test_empty_or_nonstring_user_text_blocked(self):
        assert _plan(user_text="") is None
        assert _plan(user_text="   ") is None
        assert _plan(user_text=None) is None
        assert _plan(user_text=123) is None
        assert _plan(user_text=[{"type": "text", "text": "hi"}]) is None

    def test_missing_target_and_disabled_config_blocked(self):
        assert _plan(platform="") is None
        assert _plan(chat_id="") is None
        assert _plan(cfg={}) is None


class TestResolveMirrorTarget:
    def test_reads_row_facts(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDB({"source": "telegram", "chat_id": "860", "thread_id": None,
                                       "transport_profile": "ops"}))
        session = {"session_key": "agent:main:telegram:dm:860"}
        assert turn_mirror.resolve_mirror_target(session, SimpleNamespace(session_id="sid1")) == (
            "telegram", "860", None, "ops")

    def test_thread_id_passthrough(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDB({"source": "telegram", "chat_id": "860", "thread_id": "77"}))
        session = {"session_key": "s1"}
        assert turn_mirror.resolve_mirror_target(session, SimpleNamespace(session_id="sid1")) == (
            "telegram", "860", "77", None)

    def test_missing_fields_fail_closed(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDB({}))
        session = {"session_key": "s1"}
        assert turn_mirror.resolve_mirror_target(session, SimpleNamespace(session_id="sid1")) == (
            "", "", None, None)

    def test_db_failure_fails_closed(self, monkeypatch):
        _patch_db(monkeypatch, RuntimeError("db unavailable"))
        session = {"session_key": "s1"}
        assert turn_mirror.resolve_mirror_target(session, SimpleNamespace(session_id="sid1")) == (
            "", "", None, None)

    def test_no_session_id_skips_without_db(self, monkeypatch):
        def boom(session):  # pragma: no cover - must not be reached
            raise AssertionError("db must not be opened without a session id")

        monkeypatch.setattr("tui_gateway.session_workdir._session_db", boom)
        assert turn_mirror.resolve_mirror_target({}, SimpleNamespace(session_id="")) == (
            "", "", None, None)


class TestMirrorTurn:
    @pytest.fixture
    def delivered(self, monkeypatch):
        plans = []
        monkeypatch.setattr(turn_mirror, "_start_delivery", plans.append)
        monkeypatch.setattr(
            "tui_gateway.server._load_cfg", lambda: _cfg(mirror_local_turns=True))
        return plans

    def test_spawns_delivery_for_gateway_owned_session(self, monkeypatch, delivered):
        _patch_db(monkeypatch, _FakeDB({"source": "telegram", "chat_id": "860", "thread_id": None}))
        session = {"session_key": "agent:main:telegram:dm:860"}
        turn_mirror.mirror_turn(session, SimpleNamespace(session_id="sid1"), "hello", "world")
        assert len(delivered) == 1
        assert delivered[0].platform == "telegram"
        assert delivered[0].chat_id == "860"
        assert "> hello" in delivered[0].message

    def test_records_before_delivery_and_keeps_transport_profile(self, monkeypatch, delivered):
        _patch_db(monkeypatch, _FakeDB({"source": "telegram", "chat_id": "860", "thread_id": "77",
                                       "transport_profile": "ops"}))
        order = []

        def record(plan, obligation_id):
            order.append(("record", plan.adapter_profile, obligation_id))
            return True

        monkeypatch.setattr(turn_mirror, "_record_mirror_obligation", record)
        session = {"session_key": "agent:main:telegram:dm:860"}
        turn_mirror.mirror_turn(
            session, SimpleNamespace(session_id="sid1"), "hello", "world", rid="rid-1")
        assert order[0][0] == "record"
        assert order[0][1] == "ops"
        assert order[0][2] == delivered[0].obligation_id
        assert delivered[0].adapter_profile == "ops"

    def test_duplicate_obligation_does_not_spawn_second_delivery(self, monkeypatch, delivered):
        _patch_db(monkeypatch, _FakeDB({"source": "telegram", "chat_id": "860", "thread_id": None}))
        monkeypatch.setattr(turn_mirror, "_record_mirror_obligation", lambda plan, oid: False)
        turn_mirror.mirror_turn(
            {"session_key": "agent:main:telegram:dm:860"},
            SimpleNamespace(session_id="sid1"), "hello", "world", rid="rid-1")
        assert delivered == []

    def test_records_in_shared_ledger_and_restores_launch_scope(self, monkeypatch, tmp_path):
        launch_home = tmp_path / "launch"
        transport_home = launch_home / "profiles" / "ops"
        transport_home.mkdir(parents=True)
        (launch_home / "config.yaml").write_text("{}\n", encoding="utf-8")
        (transport_home / "config.yaml").write_text(
            "gateway:\n  delivery_ledger: true\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(launch_home))

        plan = turn_mirror.MirrorPlan(
            platform="telegram", chat_id="860", message="mirror", session_key="session-1",
            adapter_profile="ops", silent=True)
        assert turn_mirror._record_mirror_obligation(plan, "ob-1") is True

        with sqlite3.connect(launch_home / "state.db") as conn:
            row = conn.execute(
                "SELECT state, metadata_json, adapter_profile FROM delivery_obligations "
                "WHERE obligation_id='ob-1'"
            ).fetchone()
        assert row == ("pending", '{"notify":false}', "ops")
        assert not (transport_home / "state.db").exists()
        assert os.environ["HERMES_HOME"] == str(launch_home)

    def test_skips_non_gateway_sources(self, monkeypatch, delivered):
        _patch_db(monkeypatch, _FakeDB({"source": "tui", "chat_id": "860", "thread_id": None}))
        turn_mirror.mirror_turn({"session_key": "s1"}, SimpleNamespace(session_id="sid1"), "hello", "world")
        assert delivered == []

    def test_skips_disabled_config(self, monkeypatch, delivered):
        monkeypatch.setattr("tui_gateway.server._load_cfg", lambda: {})
        _patch_db(monkeypatch, _FakeDB({"source": "telegram", "chat_id": "860", "thread_id": None}))
        turn_mirror.mirror_turn({"session_key": "s1"}, SimpleNamespace(session_id="sid1"), "hello", "world")
        assert delivered == []

    def test_never_raises_on_db_failure(self, monkeypatch, delivered):
        _patch_db(monkeypatch, RuntimeError("db unavailable"))
        turn_mirror.mirror_turn({"session_key": "s1"}, SimpleNamespace(session_id="sid1"), "hello", "world")
        assert delivered == []


class TestDeliver:
    def test_forwards_options_to_the_standalone_sender(self, monkeypatch):
        calls = []

        async def fake_send(platform, pconfig, chat_id, message, thread_id=None,
                            silent=False, chunk_indicators=True):
            calls.append(dict(platform=platform, chat_id=chat_id, message=message,
                              thread_id=thread_id, silent=silent, chunk_indicators=chunk_indicators))
            return {"success": True}

        monkeypatch.setattr("tools.send_message_senders._registry_standalone_send", fake_send)
        turn_mirror._deliver(turn_mirror.MirrorPlan(
            platform="telegram", chat_id="860", message="m", thread_id="5",
            silent=True, chunk_indicators=False))
        assert calls == [dict(platform="telegram", chat_id="860", message="m",
                              thread_id="5", silent=True, chunk_indicators=False)]

    def test_success_marks_attempting_and_delivered_with_message_id(self, monkeypatch):
        calls = []

        async def fake_send(*args, **kwargs):
            return {"success": True, "message_id": "42"}

        monkeypatch.setattr("tools.send_message_senders._registry_standalone_send", fake_send)
        monkeypatch.setattr("gateway.delivery_ledger.mark_attempting",
                            lambda oid: calls.append(("attempting", oid)))
        monkeypatch.setattr("gateway.delivery_ledger.mark_delivered",
                            lambda oid, message_id=None: calls.append(("delivered", oid, message_id)))
        turn_mirror._deliver(turn_mirror.MirrorPlan(
            platform="telegram", chat_id="860", message="m", obligation_id="ob-1"))
        assert calls == [("attempting", "ob-1"), ("delivered", "ob-1", "42")]

    def test_failure_marks_failed_without_raising(self, monkeypatch):
        calls = []

        async def fake_send(*args, **kwargs):
            return {"error": "temporary failure"}

        monkeypatch.setattr("tools.send_message_senders._registry_standalone_send", fake_send)
        monkeypatch.setattr("gateway.delivery_ledger.mark_attempting",
                            lambda oid: calls.append(("attempting", oid)))
        monkeypatch.setattr("gateway.delivery_ledger.mark_failed",
                            lambda oid, error="": calls.append(("failed", oid, error)))
        turn_mirror._deliver(turn_mirror.MirrorPlan(
            platform="telegram", chat_id="860", message="m", obligation_id="ob-1"))
        assert calls == [("attempting", "ob-1"), ("failed", "ob-1", "temporary failure")]

    def test_delivery_failure_is_contained(self, monkeypatch, caplog):
        async def boom(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr("tools.send_message_senders._registry_standalone_send", boom)
        with caplog.at_level(logging.WARNING):
            turn_mirror._deliver(turn_mirror.MirrorPlan(
                platform="telegram", chat_id="860", message="m"))
        assert any("turn mirror send failed" in rec.message for rec in caplog.records)


class TestDeliveryThread:
    def test_start_delivery_spawns_daemon_thread(self, monkeypatch):
        ran = []
        monkeypatch.setattr(turn_mirror, "_deliver", ran.append)
        plan = turn_mirror.MirrorPlan(platform="telegram", chat_id="860", message="m")
        turn_mirror._start_delivery(plan)
        for _ in range(50):
            if ran:
                break
            time.sleep(0.01)
        assert ran == [plan]
