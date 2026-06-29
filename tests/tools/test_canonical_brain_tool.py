from __future__ import annotations

import json

import pytest

from tools import canonical_brain_tool as cbt


def test_route_back_sent_requires_receipt_message_id():
    out = cbt.route_back_tool(
        case_id="case:test",
        target_ref={"id": "emil"},
        message_summary="summary",
        source_refs={"platform": "discord", "message_id": "m1"},
        mode="record_sent_receipt",
        receipt={},
    )
    data = json.loads(out)
    assert "requires receipt.message_id" in data["error"]


@pytest.mark.parametrize(
    "target_ref",
    [
        {"id": "1521098105041191104", "dm_channel_id": "1521098105041191104"},
        {"id": "plamenka", "recipient_id": "1282940574533423125"},
        {"id": "plamenka", "lane": "plamenka_direct_dm"},
        {"id": "plamenka", "role": "plamenka_direct_dm"},
        {"id": "plamenka", "channel_type": "dm"},
    ],
)
def test_route_back_blocks_dm_targets_before_helper(monkeypatch, target_ref):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after forbidden DM target")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    out = cbt.route_back_tool(
        case_id="case:test",
        target_ref=target_ref,
        message_summary="forward this to Plamenka",
        source_refs={"platform": "discord", "message_id": "m1"},
        mode="queue_intent",
    )
    data = json.loads(out)

    assert "forbids direct-message/DM targets" in data["error"]
    assert called["helper"] is False


def test_route_back_sent_blocks_dm_receipt_before_helper(monkeypatch):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after forbidden DM receipt")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    out = cbt.canonical_event_append_tool(
        event_type="route_back.sent",
        case_id="case:test",
        summary="DM receipt must not be accepted",
        source_refs={"platform": "discord", "message_id": "m1"},
        payload={
            "route_back": {
                "target_ref": {"id": "1521098105041191104"},
                "receipt": {
                    "message_id": "delivered-1",
                    "dm_channel_id": "1521098105041191104",
                    "recipient_id": "1282940574533423125",
                },
            },
            "receipt": {
                "message_id": "delivered-1",
                "dm_channel_id": "1521098105041191104",
            },
        },
    )
    data = json.loads(out)

    assert "route_back.sent forbids direct-message/DM delivery receipts" in data["error"]
    assert called["helper"] is False


@pytest.mark.parametrize("mode", ["record_required_only", "queue_intent"])
def test_route_back_pending_modes_return_non_terminal_guard(monkeypatch, mode):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    out = cbt.route_back_tool(
        case_id="case:test",
        target_ref={"id": "plamenka-thread"},
        message_summary="forward this status to Plamenka",
        source_refs={"platform": "discord", "message_id": "m1"},
        mode=mode,
        idempotency_key=f"idem-{mode}",
    )
    data = json.loads(out)

    assert data["success"] is True
    assert data["route_back"]["mode"] == mode
    assert data["route_back"]["terminal_outcome"] is False
    assert data["route_back"]["required_next_step"] == "deliver_route_back_or_record_blocked"
    assert "Do not present this as delivered or complete" in data["route_back"]["final_answer_guard"]


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("record_sent_receipt", {"receipt": {"message_id": "delivered-1"}}),
        ("record_blocked", {"blocker_reason": "missing target channel permission"}),
    ],
)
def test_route_back_terminal_modes_return_terminal_outcome(monkeypatch, mode, kwargs):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    out = cbt.route_back_tool(
        case_id="case:test",
        target_ref={"id": "plamenka-thread"},
        message_summary="forward this status to Plamenka",
        source_refs={"platform": "discord", "message_id": "m1"},
        mode=mode,
        idempotency_key=f"idem-{mode}",
        **kwargs,
    )
    data = json.loads(out)

    assert data["success"] is True
    assert data["route_back"]["mode"] == mode
    assert data["route_back"]["terminal_outcome"] is True
    assert data["route_back"]["required_next_step"] == "none"
    assert "final_answer_guard" not in data["route_back"]


def test_canonical_event_append_blocks_keyword_authority_secret_like_payload():
    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:test",
        summary="summary",
        source_refs={"platform": "discord", "message_id": "m1"},
        payload={"note": "token=abc"},
    )
    data = json.loads(out)
    assert "secret_like_content_blocked:payload" in data["error"]


@pytest.mark.parametrize(
    ("kwargs", "blocked_field"),
    [
        ({"summary": "token=abc123456789012345"}, "summary"),
        ({"actors": {"requester": {"note": "password=abc123456789012345"}}}, "actors"),
        ({"safety": {"operator_note": "secret=abc123456789012345"}}, "safety"),
    ],
)
def test_append_blocks_secret_like_fields_before_helper(monkeypatch, kwargs, blocked_field):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after secret-like input")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    params = {
        "event_type": "case.note",
        "case_id": "case:test",
        "summary": "safe summary",
        "source_refs": {"platform": "discord", "message_id": "m1"},
    }
    params.update(kwargs)

    out = cbt.canonical_event_append_tool(**params)
    data = json.loads(out)

    assert f"secret_like_content_blocked:{blocked_field}" in data["error"]
    assert called["helper"] is False


@pytest.mark.parametrize(
    ("kwargs", "blocked_field"),
    [
        ({"payload": {"token": "abc"}}, "payload"),
        ({"source_refs": {"platform": "discord", "message_id": "m1", "authorization": "Bearer abc"}}, "source_refs"),
        ({"actors": {"credentials": {"password": "abc"}}}, "actors"),
        ({"payload": {"items": [{"safe": "ok"}, {"nested": {"access_token": "abc"}}]}}, "payload"),
        ({"payload": {"receipt": {"payment_credential": "abc"}}}, "payload"),
    ],
)
def test_append_blocks_structured_secret_keys_before_helper(monkeypatch, kwargs, blocked_field):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after structured secret input")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    params = {
        "event_type": "case.note",
        "case_id": "case:test",
        "summary": "safe summary",
        "source_refs": {"platform": "discord", "message_id": "m1"},
    }
    params.update(kwargs)

    out = cbt.canonical_event_append_tool(**params)
    data = json.loads(out)

    assert f"secret_like_content_blocked:{blocked_field}" in data["error"]
    assert called["helper"] is False


@pytest.mark.parametrize(
    ("kwargs", "blocked_field"),
    [
        ({"message_summary": "token=abc123456789012345"}, "message_summary"),
        ({"target_ref": {"id": "emil", "note": "secret=abc123456789012345"}}, "target_ref"),
        ({"mode": "record_sent_receipt", "receipt": {"message_id": "m1", "audit": "password=abc123456789012345"}}, "receipt"),
    ],
)
def test_route_back_blocks_secret_like_fields_before_helper(monkeypatch, kwargs, blocked_field):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after secret-like input")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    params = {
        "case_id": "case:test",
        "target_ref": {"id": "emil"},
        "message_summary": "safe summary",
        "source_refs": {"platform": "discord", "message_id": "m1"},
    }
    params.update(kwargs)

    out = cbt.route_back_tool(**params)
    data = json.loads(out)

    assert f"secret_like_content_blocked:{blocked_field}" in data["error"]
    assert called["helper"] is False


@pytest.mark.parametrize(
    ("kwargs", "blocked_field"),
    [
        ({"target_ref": {"id": "emil", "token": "abc"}}, "target_ref"),
        ({"receipt": {"message_id": "m1", "authorization": "Bearer abc"}, "mode": "record_sent_receipt"}, "receipt"),
        ({"target_ref": {"id": "emil", "credentials": {"password": "abc"}}}, "target_ref"),
        ({"receipt": {"message_id": "m1", "trail": [{"private_key": "abc"}]}, "mode": "record_sent_receipt"}, "receipt"),
        ({"blocker_reason": {"payment_credential": "abc"}, "mode": "record_blocked"}, "blocker_reason"),
    ],
)
def test_route_back_blocks_structured_secret_keys_before_helper(monkeypatch, kwargs, blocked_field):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after structured secret input")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    params = {
        "case_id": "case:test",
        "target_ref": {"id": "emil"},
        "message_summary": "safe summary",
        "source_refs": {"platform": "discord", "message_id": "m1"},
    }
    params.update(kwargs)

    out = cbt.route_back_tool(**params)
    data = json.loads(out)

    assert f"secret_like_content_blocked:{blocked_field}" in data["error"]
    assert called["helper"] is False


def test_event_uuid_is_deterministic_from_idempotency_key():
    assert cbt._event_uuid("same-key") == cbt._event_uuid("same-key")
    assert cbt._event_uuid("same-key") != cbt._event_uuid("different-key")


class _FakeSock:
    def close(self):
        pass


class _FakeHelper:
    def __init__(self):
        self.queries = []

    @staticmethod
    def get_secret_value():
        return "not-printed"

    @staticmethod
    def connect(password):
        assert password == "not-printed"
        return _FakeSock()

    @staticmethod
    def sql_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def json_sql(value):
        return _FakeHelper.sql_quote(json.dumps(value, sort_keys=True, separators=(",", ":"))) + "::jsonb"

    def query(self, sock, sql):
        self.queries.append(sql)
        if sql.lstrip().upper().startswith("SELECT"):
            return {"rows": [["event-id", "case.note", "case:test", "2026-01-01", "idem"]], "command_tag": "SELECT 1"}
        return {"rows": [], "command_tag": "INSERT 0 1"}


def _session_env(values):
    def getter(name, default=""):
        return values.get(name, default)

    return getter


def test_append_uses_helper_and_returns_readback(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:test",
        summary="summary",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="idem",
    )
    data = json.loads(out)
    assert data["success"] is True
    assert data["status"] == "CANONICAL_EVENT_APPEND_PASS"
    assert data["idempotency_key"] == "idem"
    assert data["inserted"] is True
    assert any("INSERT INTO canonical_event_log" in q for q in fake.queries)


def test_append_allows_negative_secret_safety_flags(monkeypatch):
    """Boolean safety flags should not be mistaken for recorded secret values."""
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:test",
        summary="safe operational note",
        source_refs={"platform": "discord", "message_id": "m1"},
        safety={"secret": False, "payment_credential": False},
        idempotency_key="idem-negative-safety-flags",
    )
    data = json.loads(out)

    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"secret":false' in sql
    assert '"payment_credential":false' in sql


def test_append_still_blocks_positive_secret_safety_field_before_helper(monkeypatch):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded after structured secret input")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:test",
        summary="safe operational note",
        source_refs={"platform": "discord", "message_id": "m1"},
        safety={"secret": "abc"},
    )
    data = json.loads(out)

    assert "secret_like_content_blocked:safety" in data["error"]
    assert called["helper"] is False


def test_append_fills_missing_source_refs_from_session_context(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env(
            {
                "HERMES_SESSION_PLATFORM": "discord",
                "HERMES_SESSION_CHAT_ID": "1518976443168854146",
                "HERMES_SESSION_THREAD_ID": "1518976443168854146",
                "HERMES_SESSION_MESSAGE_ID": "msg-123",
                "HERMES_SESSION_ID": "sess-abc",
                "HERMES_SESSION_USER_NAME": "Plamenka",
            }
        ),
    )

    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:video-mp4",
        summary="Пламенка asks route-back to Emil's Home channel",
        source_refs={},
        idempotency_key="idem-session-ref",
    )
    data = json.loads(out)

    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"platform":"discord"' in sql
    assert '"chat_id":"1518976443168854146"' in sql
    assert '"thread_id":"1518976443168854146"' in sql
    assert '"message_id":"msg-123"' in sql
    assert '"source_ref_source":"hermes_session_context"' in sql


def test_append_uses_manual_session_ref_when_message_id_missing(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env(
            {
                "HERMES_SESSION_PLATFORM": "discord",
                "HERMES_SESSION_CHAT_ID": "1504852355588423801",
                "HERMES_SESSION_THREAD_ID": "1518976443168854146",
                "HERMES_SESSION_KEY": "session-key-abc",
            }
        ),
    )

    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:video-mp4",
        summary="manual session source fallback",
        source_refs={},
        idempotency_key="idem-manual-ref",
    )
    data = json.loads(out)

    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"manual_ref":"hermes_session:discord:1504852355588423801:1518976443168854146:session-key-abc"' in sql


def test_append_missing_source_refs_without_context_fails_before_helper(monkeypatch):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded when source refs are unresolved")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    monkeypatch.setattr(cbt, "_get_session_env", lambda name, default="": default)

    out = cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:test",
        summary="summary",
        source_refs={},
    )
    data = json.loads(out)

    assert "source_refs.platform is required" in data["error"]
    assert called["helper"] is False


def test_check_requirements_false_when_private_helper_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(cbt, "CLOUD_SQL_HELPER", tmp_path / "missing.py")

    assert cbt.check_canonical_brain_requirements() is False


def test_check_requirements_requires_explicit_profile_enablement(monkeypatch, tmp_path):
    helper = tmp_path / "cloud_sql_synthetic_write_gate.py"
    helper.write_text("# helper")
    monkeypatch.setattr(cbt, "CLOUD_SQL_HELPER", helper)
    monkeypatch.setattr(cbt, "load_config", lambda: {"canonical_brain": {"audit_bridge": {"enabled": False}}})

    assert cbt.check_canonical_brain_requirements() is False

    monkeypatch.setattr(cbt, "load_config", lambda: {"canonical_brain": {"tools_enabled": True}})
    assert cbt.check_canonical_brain_requirements() is True
