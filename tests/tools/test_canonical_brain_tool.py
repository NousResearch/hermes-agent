from __future__ import annotations

import hashlib
import json
import re

import pytest

from tools import canonical_brain_tool as cbt


_REAL_DISCORD_VERIFY_MESSAGE_RECEIPT = cbt._discord_verify_message_receipt


@pytest.fixture(autouse=True)
def _verified_discord_receipt(monkeypatch):
    monkeypatch.setattr(
        cbt,
        "_discord_expected_content_sha256",
        lambda content: hashlib.sha256(str(content).encode("utf-8")).hexdigest(),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_verify_message_receipt",
        lambda **kwargs: {
            "verified": True,
            "channel_id": kwargs["channel_id"],
            "message_id": kwargs["message_id"],
            "content_sha256": kwargs.get("expected_content_sha256") or "a" * 64,
        },
    )


def test_route_back_sent_requires_live_adapter_readback(monkeypatch):
    called = {"helper": False}
    monkeypatch.setattr(
        cbt,
        "_discord_verify_message_receipt",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("live_discord_receipt_verifier_unavailable")
        ),
    )
    monkeypatch.setattr(
        cbt,
        "_load_helper",
        lambda: called.update(helper=True),
    )
    out = cbt._route_back_state_impl(
        case_id="case:test",
        target_ref={"id": "public-channel", "channel_id": "public-channel"},
        message_summary="summary",
        source_refs={"platform": "discord", "message_id": "m1"},
        mode="record_sent_receipt",
        receipt={
            "message_id": "discord-message",
            "channel_id": "public-channel",
            "content_sha256": "a" * 64,
        },
        _internal_sent=True,
        _execution_binding={
            "target_channel_id": "public-channel",
            "content_sha256": "a" * 64,
        },
    )
    data = json.loads(out)
    assert "live_discord_receipt_verifier_unavailable" in data["error"]
    assert called["helper"] is False


def test_route_back_state_registry_cannot_dispatch_internal_sent_mode():
    out = cbt.registry.dispatch(
        "route_back_state",
        {
            "case_id": "case:test",
            "target_ref": {"id": "public", "channel_id": "public"},
            "message_summary": "must stay internal",
            "source_refs": {"platform": "discord", "message_id": "m1"},
            "mode": "record_sent_receipt",
            "receipt": {
                "message_id": "m2",
                "channel_id": "public",
                "content_sha256": "a" * 64,
            },
            "_internal_sent": True,
            "_execution_binding": {
                "target_channel_id": "public",
                "content_sha256": "a" * 64,
            },
        },
    )
    assert "mode_not_allowed:record_sent_receipt" in json.loads(out)["error"]


def test_route_back_public_wrapper_has_no_internal_sent_arguments():
    with pytest.raises(TypeError, match="_internal_sent"):
        cbt.route_back_tool(
            case_id="case:test",
            target_ref={"id": "public", "channel_id": "public"},
            message_summary="must stay internal",
            source_refs={"platform": "discord", "message_id": "m1"},
            mode="record_sent_receipt",
            receipt={
                "message_id": "m2",
                "channel_id": "public",
                "content_sha256": "a" * 64,
            },
            _internal_sent=True,
        )


def test_route_back_sent_rejects_target_receipt_channel_mismatch():
    with pytest.raises(ValueError, match="target channel must exactly match"):
        cbt._validate_append_request(
            event_type="route_back.sent",
            case_id="case:target-mismatch",
            summary="mismatch",
            source_refs={"platform": "discord", "message_id": "source"},
            actors={},
            payload={
                "route_back": {
                    "target_ref": {
                        "id": "target-a",
                        "channel_id": "channel-a",
                        "channel_type": "public_channel",
                    },
                    "execution_binding": {
                        "target_channel_id": "channel-a",
                        "content_sha256": "a" * 64,
                    },
                },
                "receipt": {
                    "message_id": "message-b",
                    "channel_id": "channel-b",
                    "content_sha256": "a" * 64,
                },
            },
            safety={},
        )


def test_route_back_sent_binding_must_match_durable_execution_intent():
    fake = _FakeHelper()
    idempotency_key = "routeback:intent-binding:1"
    intent_id = cbt._event_uuid(
        idempotency_key,
        "route_back.intent.created",
        "case:test",
    )
    fake.event_payloads[intent_id] = {
        "route_back": {
            "execution_binding": {
                "target_channel_id": "channel-1",
                "content_sha256": "a" * 64,
            },
        },
    }

    with pytest.raises(ValueError, match="does not match the durable execution intent"):
        cbt._validate_route_back_sent_against_intent(
            fake,
            _FakeSock(),
            case_id="case:test",
            idempotency_key=idempotency_key,
            payload={
                "route_back": {
                    "execution_binding": {
                        "target_channel_id": "channel-1",
                        "content_sha256": "b" * 64,
                    },
                },
            },
        )


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
                "channel_id": "1521098105041191104",
                "content_sha256": "a" * 64,
                "dm_channel_id": "1521098105041191104",
            },
        },
    )
    data = json.loads(out)

    assert "forbids direct-message/DM" in data["error"]
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


def test_route_back_blocked_returns_terminal_outcome(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    out = cbt.route_back_tool(
        case_id="case:test",
        target_ref={"id": "plamenka-thread"},
        message_summary="forward this status to Plamenka",
        source_refs={"platform": "discord", "message_id": "m1"},
        mode="record_blocked",
        blocker_reason="missing target channel permission",
        idempotency_key="idem-record-blocked",
    )
    data = json.loads(out)

    assert data["success"] is True
    assert data["route_back"]["mode"] == "record_blocked"
    assert data["route_back"]["terminal_outcome"] is True
    assert data["route_back"]["required_next_step"] == "none"
    assert "final_answer_guard" not in data["route_back"]


def test_route_back_execute_owner_target_sends_then_records_sent(monkeypatch):
    fake = _FakeHelper()
    sent = {}
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    def fake_post(channel_id, content, *, timeout=15):
        sent["channel_id"] = channel_id
        sent["content"] = content
        return {"id": "discord-msg-1", "channel_id": channel_id}

    monkeypatch.setattr(cbt, "_discord_post_message", fake_post)

    out = cbt.route_back_execute_tool(
        case_id="case:promo-product-api-1522505332318932992",
        target_ref={"id": "1279454038731264061", "mention": "<@1279454038731264061>"},
        message="<@1279454038731264061> Алекс предава: product id 95435.",
        message_summary="Alex route-back to Emil with promo product API id",
        source_refs={"platform": "discord", "thread_id": "1522505332318932992", "message_id": "1522505336614027304"},
        idempotency_key="routeback:promo-product-api:1522505336614027304:emo:v2",
    )
    data = json.loads(out)

    assert data["success"] is True
    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT"
    assert sent == {
        "channel_id": "1504852355588423801",
        "content": "<@1279454038731264061> Алекс предава: product id 95435.",
    }
    assert data["receipt"]["message_id"] == "discord-msg-1"
    assert data["receipt"]["channel_id"] == "1504852355588423801"
    assert data["route_back_record"]["route_back"]["mode"] == "record_sent_receipt"
    sql = "\n".join(fake.queries)
    assert "route_back.sent" in sql
    assert "discord-msg-1" in sql
    assert '"channel_type":"public_channel"' in sql


def test_route_back_execute_send_failure_records_blocked(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    def fail_post(channel_id, content, *, timeout=15):
        raise RuntimeError("network down")

    monkeypatch.setattr(cbt, "_discord_post_message", fail_post)

    out = cbt.route_back_execute_tool(
        case_id="case:test",
        target_ref={"lane": "emil_lomliev"},
        message="<@1279454038731264061> route-back message",
        message_summary="route-back to owner",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="idem-route-back-execute-blocked",
    )
    data = json.loads(out)

    assert data["success"] is True
    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert data["blocker_reason"] == "discord_send_failed:RuntimeError"
    assert data["route_back_record"]["route_back"]["mode"] == "record_blocked"
    sql = "\n".join(fake.queries)
    assert "route_back.blocked" in sql
    assert "discord_send_failed:RuntimeError" in sql


def test_route_back_execute_registered_teammate_uses_public_default_lane(monkeypatch):
    fake = _FakeHelper()
    called = {"send": False}

    def fake_post(channel_id, content, *, timeout=15):
        called["send"] = True
        called["channel_id"] = channel_id
        return {"id": "discord-msg-alex", "channel_id": channel_id}

    monkeypatch.setattr(cbt, "_discord_post_message", fake_post)
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    out = cbt.route_back_execute_tool(
        case_id="case:test",
        target_ref={"id": "1282940511962791959"},
        message="forward this",
        message_summary="route-back to Alex",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="routeback:alex:public:1",
    )
    data = json.loads(out)

    assert data["success"] is True
    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT"
    assert called["send"] is True
    assert called["channel_id"] == "1504852408227069993"
    assert data["route_back_record"]["route_back"]["mode"] == "record_sent_receipt"
    assert "route_back.sent" in "\n".join(fake.queries)


def test_route_back_execute_normalizes_conflicting_target_channel_fields(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda target_ref: {
            "channel_id": "canonical-public-channel",
            "channel_type": "public_channel",
            "target_kind": "exact_public_directory_target",
            "target_member_key": None,
            "target_member_id": "member-1",
            "target_mention": "<@member-1>",
        },
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(
        cbt,
        "_discord_post_message",
        lambda channel_id, content, timeout=15: {
            "id": "discord-normalized",
            "channel_id": channel_id,
        },
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:test",
        target_ref={
            "id": "member-1",
            "channel_id": "stale-channel",
            "thread_id": "stale-thread",
            "chat_id": "stale-chat",
        },
        message="normalized target",
        message_summary="normalize conflicting channel surfaces",
        source_refs={"platform": "discord", "message_id": "m-normalized"},
        idempotency_key="routeback:normalize-target:1",
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT"
    sql = "\n".join(fake.queries)
    assert "canonical-public-channel" in sql
    assert "stale-channel" not in sql
    assert "stale-thread" not in sql
    assert "stale-chat" not in sql


def test_fresh_claimed_route_back_intent_does_not_send_or_record_terminal(
    monkeypatch,
):
    fake = _FakeHelper()
    sends = []
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_discord_post_message",
        lambda channel_id, content, timeout=15: (
            sends.append((channel_id, content))
            or {"id": "discord-msg-once", "channel_id": channel_id}
        ),
    )
    kwargs = {
        "case_id": "case:test",
        "target_ref": {"id": "1282940511962791959"},
        "message": "send once",
        "message_summary": "exactly once route-back",
        "source_refs": {"platform": "discord", "message_id": "m-once"},
        "idempotency_key": "routeback:exactly-once:1",
    }
    first = json.loads(cbt.route_back_execute_tool(**kwargs))
    assert first["status"] == "ROUTE_BACK_EXECUTE_SENT"

    fake.insert_tags_by_event_type["route_back.intent.created"] = "INSERT 0 0"
    retry = json.loads(cbt.route_back_execute_tool(**kwargs))

    assert retry["status"] == "ROUTE_BACK_EXECUTE_ALREADY_CLAIMED_PENDING"
    assert retry["lease_seconds"] == cbt.ROUTE_BACK_INTENT_LEASE_SECONDS
    assert "Do not send" in retry["final_answer_guard"]
    assert len(sends) == 1


def test_stale_claimed_route_back_intent_records_uncertain_block_without_send(
    monkeypatch,
):
    sends = []
    blocked = []
    old = (
        cbt.dt.datetime.now(cbt.dt.timezone.utc)
        - cbt.dt.timedelta(seconds=cbt.ROUTE_BACK_INTENT_LEASE_SECONDS + 1)
    ).isoformat()
    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": False,
            "deduped": True,
            "readback_verified": True,
            "readback": [["event", "route_back.intent.created", "case:test", old]],
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_route_back_record_blocked",
        lambda **kwargs: blocked.append(kwargs) or {"success": True},
    )
    monkeypatch.setattr(
        cbt,
        "_discord_post_message",
        lambda *args, **kwargs: sends.append((args, kwargs)),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:test",
        target_ref={"id": "1282940511962791959"},
        message="do not resend stale uncertain intent",
        message_summary="stale intent",
        source_refs={"platform": "discord", "message_id": "m-stale"},
        idempotency_key="routeback:stale-intent:1",
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert data["blocker_reason"] == (
        "route_back_execution_intent_lease_expired_outcome_uncertain"
    )
    assert sends == []
    assert blocked[0]["blocker_reason"] == data["blocker_reason"]


def test_sent_receipt_record_failure_records_route_back_blocked(monkeypatch):
    calls = []

    def _record(**kwargs):
        calls.append(kwargs["mode"])
        if kwargs["mode"] == "queue_intent":
            return json.dumps({
                "success": True,
                "inserted": True,
                "readback_verified": True,
            })
        if kwargs["mode"] == "record_sent_receipt":
            return json.dumps({"success": False, "error": "database unavailable"})
        assert kwargs["mode"] == "record_blocked"
        assert kwargs["blocker_reason"] == "route_back_sent_receipt_persistence_failed"
        return json.dumps({"success": True, "inserted": True, "readback_verified": True})

    monkeypatch.setattr(cbt, "_route_back_state_impl", _record)
    monkeypatch.setattr(
        cbt,
        "_discord_post_message",
        lambda *args, **kwargs: {"id": "discord-receipt", "channel_id": "channel"},
    )
    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:test",
        target_ref={"id": "1282940511962791959"},
        message="delivered once",
        message_summary="receipt persistence failure",
        source_refs={"platform": "discord", "message_id": "m-receipt"},
        idempotency_key="routeback:receipt-failure:1",
    ))

    assert calls == ["queue_intent", "record_sent_receipt", "record_blocked"]
    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECEIPT_RECORD_BLOCKED"
    assert data["receipt"]["message_id"] == "discord-receipt"
    assert data["blocked_record"]["success"] is True
    assert "Do not resend" in data["final_answer_guard"]


def test_discord_post_message_uses_gateway_loop_and_cancels_timeout(monkeypatch):
    import concurrent.futures
    from types import SimpleNamespace

    from gateway.config import Platform

    class _Loop:
        @staticmethod
        def is_running():
            return True

    class _Adapter:
        MAX_MESSAGE_LENGTH = 2000

        @staticmethod
        def format_message(content):
            return content

        @staticmethod
        def truncate_message(content, max_length):
            return [content]

        async def send(self, channel_id, content, metadata=None):
            return SimpleNamespace(success=True, message_id="never")

    class _Future:
        cancelled = False

        def result(self, timeout):
            assert timeout == 3
            raise concurrent.futures.TimeoutError

        def cancel(self):
            self.cancelled = True

    future = _Future()

    def _schedule(coro, loop):
        assert isinstance(loop, _Loop)
        coro.close()
        return future

    runner = SimpleNamespace(
        adapters={Platform.DISCORD: _Adapter()},
        _gateway_loop=_Loop(),
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(cbt.asyncio, "run_coroutine_threadsafe", _schedule)

    with pytest.raises(RuntimeError, match="discord_adapter_send_timeout"):
        cbt._discord_post_message("channel-1", "hello", timeout=3)
    assert future.cancelled is True


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("channel_id", "wrong-channel"),
        ("message_id", "wrong-message"),
        ("content_sha256", "b" * 64),
    ],
)
def test_discord_receipt_verifier_requires_exact_returned_fields(
    monkeypatch,
    field,
    wrong_value,
):
    from types import SimpleNamespace

    from gateway.config import Platform

    class _Loop:
        @staticmethod
        def is_running():
            return True

    class _Adapter:
        async def verify_public_message_receipt(self, **kwargs):
            return kwargs

    returned = {
        "verified": True,
        "channel_id": "channel-1",
        "message_id": "message-1",
        "content_sha256": "a" * 64,
    }
    returned[field] = wrong_value

    class _Future:
        @staticmethod
        def result(timeout):
            return returned

    def _schedule(coro, loop):
        coro.close()
        return _Future()

    runner = SimpleNamespace(
        adapters={Platform.DISCORD: _Adapter()},
        _gateway_loop=_Loop(),
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(cbt.asyncio, "run_coroutine_threadsafe", _schedule)

    with pytest.raises(RuntimeError, match=f"{field}_mismatch"):
        _REAL_DISCORD_VERIFY_MESSAGE_RECEIPT(
            channel_id="channel-1",
            message_id="message-1",
            expected_content_sha256="a" * 64,
        )


def test_discord_post_message_binds_live_readback_to_rendered_content(monkeypatch):
    from types import SimpleNamespace

    from gateway.config import Platform

    expected = hashlib.sha256(b"rendered content").hexdigest()
    captured = {}

    class _Loop:
        @staticmethod
        def is_running():
            return True

    class _Adapter:
        MAX_MESSAGE_LENGTH = 2000

        @staticmethod
        def format_message(content):
            return "rendered content"

        @staticmethod
        def truncate_message(content, max_length):
            return [content]

        def send(self, channel_id, content, metadata=None):
            captured["metadata"] = metadata

            async def _send():
                return SimpleNamespace(success=True, message_id="message-1")

            return _send()

    class _Future:
        @staticmethod
        def result(timeout):
            return SimpleNamespace(success=True, message_id="message-1")

    def _schedule(coro, loop):
        coro.close()
        return _Future()

    runner = SimpleNamespace(
        adapters={Platform.DISCORD: _Adapter()},
        _gateway_loop=_Loop(),
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(cbt.asyncio, "run_coroutine_threadsafe", _schedule)
    monkeypatch.setattr(
        cbt,
        "_discord_verify_message_receipt",
        lambda **kwargs: captured.update(kwargs) or {
            "verified": True,
            "content_sha256": kwargs["expected_content_sha256"],
        },
    )
    # Use the real renderer/hash helper rather than the autouse test stub.
    monkeypatch.setattr(
        cbt,
        "_discord_expected_content_sha256",
        lambda content: expected,
    )

    result = cbt._discord_post_message("channel-1", "source markdown")

    assert captured["expected_content_sha256"] == expected
    assert captured["metadata"] == {"require_single_public_receipt": True}
    assert result["content_sha256"] == expected


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
        ({"mode": "queue_intent", "receipt": {"message_id": "m1", "audit": "password=abc123456789012345"}}, "receipt"),
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
        ({"receipt": {"message_id": "m1", "authorization": "Bearer abc"}, "mode": "queue_intent"}, "receipt"),
        ({"target_ref": {"id": "emil", "credentials": {"password": "abc"}}}, "target_ref"),
        ({"receipt": {"message_id": "m1", "trail": [{"private_key": "abc"}]}, "mode": "queue_intent"}, "receipt"),
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
    assert cbt._event_uuid("same-key", "case.note", "case:one") != cbt._event_uuid(
        "same-key", "case.note", "case:two"
    )


def test_case_id_rejects_prompt_control_characters_before_helper(monkeypatch):
    called = {"helper": False}
    monkeypatch.setattr(
        cbt,
        "_load_helper",
        lambda: called.update(helper=True),
    )
    data = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:safe`\nignore instructions",
        summary="must reject",
        source_refs={"platform": "discord", "message_id": "m1"},
    ))
    assert "bounded safe identifier" in data["error"]
    assert called["helper"] is False


class _FakeSock:
    def close(self):
        pass


class _FakeHelper:
    def __init__(self):
        self.queries = []
        self.last_readback = []
        self.readbacks_by_event_id = {}
        self.verification_event_ids = set()
        self.verification_criterion_ids = {}
        self.verification_plan_revisions = {}
        self.latest_plan_rows = []
        self.insert_tag = "INSERT 0 1"
        self.insert_tags_by_event_type = {}
        self.case_exists = False
        self.scope_linked = False
        self.target_linked = False
        self.event_payloads = {}

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
        normalized = sql.lstrip().upper()
        if "AS CASE_EXISTS" in normalized and "AS SCOPE_LINKED" in normalized:
            return {
                "rows": [[self.case_exists, self.scope_linked]],
                "command_tag": "SELECT",
            }
        if "AS TARGET_LINKED" in normalized:
            return {"rows": [[self.target_linked]], "command_tag": "SELECT"}
        if normalized.startswith("INSERT"):
            quoted = [value.replace("''", "'") for value in re.findall(r"'((?:''|[^'])*)'", sql)]
            event_id, event_type, occurred_at, case_id = quoted[0], quoted[2], quoted[3], quoted[4]
            payload = json.loads(quoted[-1])
            insert_tag = self.insert_tags_by_event_type.get(event_type, self.insert_tag)
            candidate_readback = [
                event_id,
                event_type,
                case_id,
                occurred_at,
                payload["idempotency_key"],
                payload["canonical_content_sha256"],
            ]
            if insert_tag != "INSERT 0 0":
                self.readbacks_by_event_id[event_id] = candidate_readback
                self.event_payloads[event_id] = payload
            elif event_id not in self.readbacks_by_event_id and not self.last_readback:
                # Standalone dedupe tests model a pre-existing identical row.
                self.readbacks_by_event_id[event_id] = candidate_readback
            self.last_readback = [
                self.readbacks_by_event_id.get(
                    event_id,
                    self.last_readback[0] if self.last_readback else candidate_readback,
                )
            ]
            if event_type == "task.plan.updated":
                plan = payload["plan"]
                if insert_tag != "INSERT 0 0":
                    self.latest_plan_rows = [[
                        event_id,
                        plan["plan_id"],
                        str(plan["revision"]),
                        json.dumps(plan),
                    ]]
            return {"rows": [], "command_tag": insert_tag}
        if "PAYLOAD->'PLAN'->>'REVISION'" in normalized:
            return {"rows": self.latest_plan_rows, "command_tag": "SELECT"}
        if "AS INTENT_PAYLOAD" in normalized:
            quoted = [
                value.replace("''", "'")
                for value in re.findall(r"'((?:''|[^'])*)'", sql)
            ]
            intent_event_id = quoted[1] if len(quoted) > 1 else ""
            intent_payload = self.event_payloads.get(intent_event_id)
            return {
                "rows": [[json.dumps(intent_payload)]] if intent_payload else [],
                "command_tag": "SELECT",
            }
        if "PAYLOAD->'VERIFICATION'->>'OUTCOME' = 'PASSED'" in normalized:
            return {
                "rows": [
                    [
                        event_id,
                        json.dumps(self.verification_criterion_ids.get(event_id, ["tests"])),
                        str(self.verification_plan_revisions.get(event_id, 1)),
                    ]
                    for event_id in sorted(self.verification_event_ids)
                ],
                "command_tag": "SELECT",
            }
        if sql.lstrip().upper().startswith("SELECT"):
            return {"rows": self.last_readback, "command_tag": "SELECT 1"}
        return {"rows": [], "command_tag": "UNKNOWN"}


def _session_env(values):
    def getter(name, default=""):
        return values.get(name, default)

    return getter


def test_observed_gateway_context_forces_scope_when_requirements_probe_fails(
    monkeypatch,
):
    fake = _FakeHelper()
    fake.case_exists = True
    fake.scope_linked = False
    monkeypatch.setattr(cbt, "check_canonical_brain_requirements", lambda: False)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env({
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_THREAD_ID": "thread-current",
            "HERMES_SESSION_USER_ID": "not-owner",
        }),
    )
    monkeypatch.setattr(cbt, "_configured_plan_owner_ids", lambda: {"owner"})

    with pytest.raises(PermissionError, match="outside the current observed"):
        cbt._authorize_append_scope(fake, _FakeSock(), case_id="case:foreign")


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


def _plan_payload(*, state="active", revision=1, verification_event_ids=None):
    steps = [
        {"id": "orient", "content": "Inspect exact current state", "status": "completed", "depends_on": []},
        {
            "id": "implement",
            "content": "Implement the approved change",
            "status": "completed" if state == "completed" else "in_progress",
            "depends_on": ["orient"],
        },
    ]
    plan = {
        "plan_id": "plan:p0",
        "revision": revision,
        "objective": "Complete the approved P0 task workspace gate",
        "state": state,
        "success_criteria": [
            {"id": "tests", "content": "Receipt-backed tests pass"},
        ],
        "steps": steps,
        "current_step_id": None if state == "completed" else "implement",
        "resume_cursor": {
            "next_step_id": None if state == "completed" else "implement",
            "summary": "Resume from implementation" if state != "completed" else "No remaining work",
        },
        "attempts": [],
        "decisions": [],
        "artifacts": [],
    }
    if verification_event_ids is not None:
        plan["verification_event_ids"] = verification_event_ids
    return {"plan": plan}


def test_task_plan_append_is_bounded_model_authored_workspace(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="P0 plan revision one",
        source_refs={"platform": "discord", "message_id": "m-plan"},
        payload=_plan_payload(),
        idempotency_key="plan:p0:r1",
    ))

    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"plan_id":"plan:p0"' in sql
    assert '"attestation":"model_authored"' in sql
    assert '"state":"active"' in sql


def test_narrow_authorized_active_plan_check_reads_one_latest_plan(monkeypatch):
    fake = _FakeHelper()
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(_plan_payload()["plan"]),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    assert cbt.canonical_active_plan_matches(
        case_id="case:p0",
        plan_id="plan:p0",
    ) is True
    assert cbt.canonical_active_plan_matches(
        case_id="case:p0",
        plan_id="plan:other",
    ) is False
    assert all("capability.check.recorded" not in sql for sql in fake.queries)


def test_task_plan_rejects_two_in_progress_steps_before_helper(monkeypatch):
    called = {"helper": False}
    payload = _plan_payload()
    payload["plan"]["steps"][0]["status"] = "in_progress"

    def boom():
        called["helper"] = True
        raise AssertionError("helper must not be loaded for invalid task plan")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="invalid plan",
        source_refs={"platform": "discord", "message_id": "m-plan"},
        payload=payload,
    ))

    assert "at most one in_progress" in data["error"]
    assert called["helper"] is False


def test_task_verification_then_completed_plan_requires_existing_pass_receipt(monkeypatch):
    fake = _FakeHelper()
    verification_event_id = "11111111-1111-4111-8111-111111111111"
    fake.verification_event_ids.add(verification_event_id)
    fake.verification_plan_revisions[verification_event_id] = 1
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(_plan_payload()["plan"]),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    verification = json.loads(cbt.canonical_event_append_tool(
        event_type="task.verification.recorded",
        case_id="case:p0",
        summary="Targeted tests passed",
        source_refs={"platform": "discord", "message_id": "m-verify"},
        payload={
            "verification": {
                "verification_id": "tests",
                "plan_id": "plan:p0",
                "plan_revision": 1,
                "criterion_ids": ["tests"],
                "outcome": "passed",
                "summary": "Targeted tests passed",
                "receipt": {"kind": "pytest", "ref": "pytest:46-passed"},
            },
        },
        idempotency_key="verify:p0:tests",
    ))
    assert verification["success"] is True

    completed = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="P0 plan completed",
        source_refs={"platform": "discord", "message_id": "m-complete"},
        payload=_plan_payload(
            state="completed",
            revision=2,
            verification_event_ids=[verification_event_id],
        ),
        idempotency_key="plan:p0:r2",
    ))
    assert completed["success"] is True
    assert any("PAYLOAD->'VERIFICATION'->>'OUTCOME' = 'PASSED'" in sql.upper() for sql in fake.queries)


def test_completed_plan_rejects_missing_verification_event(monkeypatch):
    fake = _FakeHelper()
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(_plan_payload()["plan"]),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    missing = "22222222-2222-4222-8222-222222222222"

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="must not complete",
        source_refs={"platform": "discord", "message_id": "m-complete"},
        payload=_plan_payload(state="completed", revision=2, verification_event_ids=[missing]),
        idempotency_key="plan:p0:missing",
    ))

    assert "missing or non-passing verification" in data["error"]
    assert not any(sql.lstrip().upper().startswith("INSERT") for sql in fake.queries)


def test_completed_plan_requires_receipts_for_every_success_criterion(monkeypatch):
    fake = _FakeHelper()
    verification_event_id = "33333333-3333-4333-8333-333333333333"
    fake.verification_event_ids.add(verification_event_id)
    fake.verification_criterion_ids[verification_event_id] = ["different-criterion"]
    fake.verification_plan_revisions[verification_event_id] = 1
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(_plan_payload()["plan"]),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="must not complete with uncovered criteria",
        source_refs={"platform": "discord", "message_id": "m-complete"},
        payload=_plan_payload(
            state="completed",
            revision=2,
            verification_event_ids=[verification_event_id],
        ),
        idempotency_key="plan:p0:uncovered",
    ))

    assert "success criteria without passing verification" in data["error"]
    assert "tests" in data["error"]


def test_task_plan_revision_cannot_regress_or_reuse_revision(monkeypatch):
    fake = _FakeHelper()
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "2",
        json.dumps({**_plan_payload()["plan"], "revision": 2}),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    payload = _plan_payload()
    payload["plan"]["revision"] = 1
    regressed = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="regressed",
        source_refs={"platform": "discord", "message_id": "m1"},
        payload=payload,
        idempotency_key="plan:p0:regressed",
    ))
    assert "revision regressed" in regressed["error"]

    payload["plan"]["revision"] = 2
    reused = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="reused revision",
        source_refs={"platform": "discord", "message_id": "m2"},
        payload=payload,
        idempotency_key="plan:p0:reused",
    ))
    assert "revision must increase" in reused["error"]


def test_competing_successors_of_same_plan_revision_share_one_cas_identity(monkeypatch):
    fake = _FakeHelper()
    predecessor = _plan_payload()["plan"]
    predecessor_row = [
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(predecessor),
    ]
    fake.latest_plan_rows = [predecessor_row]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    continuation = _plan_payload(revision=2)
    first = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="continue existing plan",
        source_refs={"platform": "discord", "message_id": "m-cont"},
        payload=continuation,
        idempotency_key="plan:p0:continue",
    ))
    assert first["success"] is True

    fake.latest_plan_rows = [predecessor_row]
    fake.insert_tag = "INSERT 0 0"
    replacement = _plan_payload()
    replacement["plan"].update({
        "plan_id": "plan:replacement",
        "supersedes_plan_id": "plan:p0",
        "supersedes_plan_revision": 1,
    })
    competing = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="competing replacement",
        source_refs={"platform": "discord", "message_id": "m-replace"},
        payload=replacement,
        idempotency_key="plan:replacement:r1",
    ))

    assert competing["status"] == "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT"
    assert competing["event_id"] == first["event_id"]


def test_sql_validation_selects_revision_graph_head_not_timestamp_uuid_fallback():
    fake = _FakeHelper()
    old = {**_plan_payload()["plan"], "plan_id": "plan:A3", "revision": 1}
    replacement = {
        **_plan_payload()["plan"],
        "plan_id": "plan:B3",
        "revision": 1,
        "supersedes_plan_id": "plan:A3",
        "supersedes_plan_revision": 1,
    }
    continuation = {**replacement, "revision": 2}
    fake.latest_plan_rows = [
        ["3d898e4f-36dc-5211-973d-61efedfe3654", "plan:B3", "2", json.dumps(continuation)],
        ["48e80467-ed6a-5966-88e2-0af5bb8571d6", "plan:A3", "1", json.dumps(old)],
        ["b1942d63-47c1-5648-9bbd-aa4aa741ad9b", "plan:B3", "1", json.dumps(replacement)],
    ]

    latest = cbt._latest_task_plan_record(fake, _FakeSock(), "case:3")

    assert latest["plan_id"] == "plan:B3"
    assert latest["revision"] == 2


def test_same_plan_revision_cannot_change_supersession_metadata(monkeypatch):
    fake = _FakeHelper()
    prior = {
        **_plan_payload()["plan"],
        "plan_id": "plan:B",
        "supersedes_plan_id": "plan:A",
        "supersedes_plan_revision": 4,
    }
    fake.latest_plan_rows = [["prior", "plan:B", "1", json.dumps(prior)]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    candidate = _plan_payload(revision=2)
    candidate["plan"].update({
        "plan_id": "plan:B",
        "supersedes_plan_id": "plan:other",
        "supersedes_plan_revision": 9,
    })

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="invalid lineage rewrite",
        source_refs={"platform": "discord", "message_id": "m-lineage"},
        payload=candidate,
    ))

    assert "supersession metadata is immutable" in data["error"]


@pytest.mark.parametrize("terminal_state", ["completed", "cancelled"])
def test_same_plan_cannot_advance_after_terminal_state(monkeypatch, terminal_state):
    fake = _FakeHelper()
    prior = _plan_payload()["plan"]
    prior["state"] = terminal_state
    fake.latest_plan_rows = [["prior", "plan:p0", "1", json.dumps(prior)]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="invalid terminal continuation",
        source_refs={"platform": "discord", "message_id": "m-terminal"},
        payload=_plan_payload(revision=2),
    ))

    assert "cannot advance under the same plan_id" in data["error"]


def test_completed_revision_cannot_shrink_prior_steps_or_criteria(monkeypatch):
    fake = _FakeHelper()
    prior = _plan_payload()["plan"]
    prior["success_criteria"].append({"id": "security", "content": "Security passes"})
    prior["steps"].append({
        "id": "security-review",
        "content": "Review security",
        "status": "pending",
        "depends_on": ["implement"],
    })
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(prior),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    completed = _plan_payload(
        state="completed",
        revision=2,
        verification_event_ids=["11111111-1111-4111-8111-111111111111"],
    )

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="invalid shrink",
        source_refs={"platform": "discord", "message_id": "m-shrink"},
        payload=completed,
        idempotency_key="plan:p0:shrink",
    ))

    assert "must preserve prior success criteria" in data["error"]
    assert not any(sql.lstrip().upper().startswith("INSERT") for sql in fake.queries)


@pytest.mark.parametrize(
    "scope_change",
    [
        "change_criterion_content",
        "change_step_content",
        "change_step_dependencies",
    ],
)
def test_active_same_plan_revision_cannot_change_structural_scope(
    monkeypatch,
    scope_change,
):
    fake = _FakeHelper()
    prior = _plan_payload()["plan"]
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(prior),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    updated = _plan_payload(revision=2)
    plan = updated["plan"]
    if scope_change == "change_criterion_content":
        plan["success_criteria"][0]["content"] = "A weaker replacement criterion"
    elif scope_change == "change_step_content":
        plan["steps"][1]["content"] = "Implement only the easy subset"
    else:
        plan["steps"][1]["depends_on"] = []

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary=f"invalid active scope change: {scope_change}",
        source_refs={"platform": "discord", "message_id": f"m-{scope_change}"},
        payload=updated,
        idempotency_key=f"plan:p0:{scope_change}",
    ))

    assert "same plan_id revision must preserve" in data["error"]
    assert "explicit plan supersession" in data["error"]
    assert not any(sql.lstrip().upper().startswith("INSERT") for sql in fake.queries)


@pytest.mark.parametrize("removed_obligation", ["criterion", "step"])
def test_active_same_plan_revision_cannot_remove_prior_obligations(
    monkeypatch,
    removed_obligation,
):
    fake = _FakeHelper()
    prior = _plan_payload()["plan"]
    if removed_obligation == "criterion":
        prior["success_criteria"].append({
            "id": "security",
            "content": "Security review passes",
        })
    else:
        prior["steps"].append({
            "id": "security-review",
            "content": "Review security",
            "status": "pending",
            "depends_on": ["implement"],
        })
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(prior),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary=f"invalid active {removed_obligation} removal",
        source_refs={
            "platform": "discord",
            "message_id": f"m-remove-active-{removed_obligation}",
        },
        payload=_plan_payload(revision=2),
        idempotency_key=f"plan:p0:remove-active-{removed_obligation}",
    ))

    assert (
        "must preserve prior success criteria"
        if removed_obligation == "criterion"
        else "must preserve prior steps and dependencies"
    ) in data["error"]
    assert not any(sql.lstrip().upper().startswith("INSERT") for sql in fake.queries)


def test_active_same_plan_revision_allows_monotonic_additive_refinement(monkeypatch):
    fake = _FakeHelper()
    prior = _plan_payload()["plan"]
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(prior),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    refined = _plan_payload(revision=2)
    refined["plan"]["success_criteria"].append({
        "id": "security",
        "content": "Security review passes",
    })
    refined["plan"]["steps"].append({
        "id": "security-review",
        "content": "Review security",
        "status": "pending",
        "depends_on": ["implement"],
    })

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="monotonic active-plan refinement",
        source_refs={"platform": "discord", "message_id": "m-additive-refinement"},
        payload=refined,
        idempotency_key="plan:p0:additive-refinement",
    ))

    assert data["success"] is True
    assert data["inserted"] is True


def test_explicit_new_plan_supersession_allows_structural_scope_change(monkeypatch):
    fake = _FakeHelper()
    prior = _plan_payload()["plan"]
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(prior),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    replacement = _plan_payload()
    replacement["plan"].update({
        "plan_id": "plan:replacement",
        "supersedes_plan_id": "plan:p0",
        "supersedes_plan_revision": 1,
        "success_criteria": [
            {"id": "new-scope", "content": "The explicitly replaced scope passes"},
        ],
    })

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="explicit structural scope replacement",
        source_refs={"platform": "discord", "message_id": "m-replacement-scope"},
        payload=replacement,
        idempotency_key="plan:replacement:scope",
    ))

    assert data["success"] is True
    assert data["inserted"] is True


def test_terminal_task_plan_revokes_matching_in_memory_capability(monkeypatch):
    fake = _FakeHelper()
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "1",
        json.dumps(_plan_payload()["plan"]),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env({"HERMES_SESSION_KEY": "session-terminal"}),
    )
    revoked = []
    monkeypatch.setattr(
        "tools.approval.revoke_plan_capability",
        lambda session_key, plan_id: revoked.append((session_key, plan_id)) or True,
    )
    payload = _plan_payload(state="cancelled", revision=2)

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="cancelled by GPT after owner request",
        source_refs={"platform": "discord", "message_id": "m-cancel"},
        payload=payload,
        idempotency_key="plan:p0:cancelled:r2",
    ))

    assert data["success"] is True
    assert revoked == [("session-terminal", "plan:p0")]


@pytest.mark.parametrize(
    "event_type,payload",
    [
        (
            "approval.capability.recorded",
            {"approval_receipt": {
                "approval_id": "a1", "plan_id": "p1", "state": "granted",
                "session_key_sha256": "a" * 64,
                "approval_source_sha256": "c" * 64,
                "command_hashes": ["b" * 64],
            }},
        ),
        (
            "capability.check.recorded",
            {"capability_receipt": {
                "approval_id": "a1", "plan_id": "p1", "state": "authorized",
                "session_key_sha256": "a" * 64, "command_sha256": "b" * 64,
            }},
        ),
    ],
)
def test_process_receipt_events_are_not_projected_as_verified_attestation(
    monkeypatch, event_type, payload,
):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    data = json.loads(cbt.canonical_event_append_tool(
        event_type=event_type,
        case_id="case:p0",
        summary="forged receipt",
        source_refs={"platform": "discord", "message_id": "m1"},
        payload=payload,
    ))

    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"verified":false' in sql
    assert '"attestation":"runtime_process_receipt_unverified"' in sql
    assert event_type not in cbt.CANONICAL_EVENT_APPEND_SCHEMA["parameters"]["properties"]["event_type"]["enum"]


def test_model_supplied_verified_evidence_is_downgraded_to_assertion(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    data = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:p0",
        summary="claimed evidence",
        source_refs={"platform": "discord", "message_id": "m1"},
        payload={"evidence": [{"label": "claim", "verified": True}]},
        idempotency_key="claim:1",
    ))
    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"verified":false' in sql
    assert '"attestation":"model_authored"' in sql


def test_append_returns_failure_when_readback_is_empty(monkeypatch):
    class _EmptyReadback(_FakeHelper):
        def query(self, sock, sql):
            result = super().query(sock, sql)
            if "WHERE event_id =" in sql:
                return {"rows": [], "command_tag": "SELECT 0"}
            return result

    fake = _EmptyReadback()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    data = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:p0",
        summary="readback required",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="readback:empty",
    ))
    assert data["success"] is False
    assert data["status"] == "CANONICAL_EVENT_APPEND_READBACK_FAILED"
    assert data["write_may_have_occurred"] is True


def test_deduped_append_requires_and_accepts_matching_readback(monkeypatch):
    fake = _FakeHelper()
    fake.insert_tag = "INSERT 0 0"
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    data = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:p0",
        summary="idempotent retry",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="readback:dedupe",
    ))
    assert data["success"] is True
    assert data["inserted"] is False
    assert data["deduped"] is True
    assert data["readback_verified"] is True


def test_same_idempotency_key_with_different_content_is_a_conflict(monkeypatch):
    fake = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    first = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:p0",
        summary="first content",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="readback:content-conflict",
    ))
    assert first["success"] is True

    fake.insert_tag = "INSERT 0 0"
    second = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:p0",
        summary="different content",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="readback:content-conflict",
    ))
    assert second["success"] is False
    assert second["status"] == "CANONICAL_EVENT_APPEND_IDEMPOTENCY_CONFLICT"
    assert "different canonical content" in second["error"]


def test_stale_task_verification_revision_is_rejected(monkeypatch):
    fake = _FakeHelper()
    latest_plan = _plan_payload()["plan"]
    latest_plan["revision"] = 2
    fake.latest_plan_rows = [[
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "plan:p0",
        "2",
        json.dumps(latest_plan),
    ]]
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)

    data = json.loads(cbt.canonical_event_append_tool(
        event_type="task.verification.recorded",
        case_id="case:p0",
        summary="stale tests",
        source_refs={"platform": "discord", "message_id": "m-stale"},
        payload={
            "verification": {
                "verification_id": "tests",
                "plan_id": "plan:p0",
                "plan_revision": 1,
                "criterion_ids": ["tests"],
                "outcome": "passed",
                "summary": "stale tests",
                "receipt": {"kind": "pytest", "ref": "pytest:stale"},
            },
        },
        idempotency_key="verify:p0:stale",
    ))
    assert "plan_revision is stale" in data["error"]
    assert not any(sql.lstrip().upper().startswith("INSERT") for sql in fake.queries)


def test_task_plan_rejects_dependency_cycle_and_invalid_active_cursor(monkeypatch):
    called = {"helper": False}

    def boom():
        called["helper"] = True
        raise AssertionError("invalid plan must fail before helper load")

    monkeypatch.setattr(cbt, "_load_helper", boom)
    cyclic = _plan_payload()
    cyclic["plan"]["steps"][0]["depends_on"] = ["implement"]
    cycle = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="cycle",
        source_refs={"platform": "discord", "message_id": "m-cycle"},
        payload=cyclic,
    ))
    assert "contains a cycle" in cycle["error"]

    invalid_cursor = _plan_payload()
    invalid_cursor["plan"]["steps"][1]["status"] = "pending"
    invalid_cursor["plan"]["current_step_id"] = "orient"
    invalid_cursor["plan"]["resume_cursor"]["next_step_id"] = "orient"
    cursor = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="invalid cursor",
        source_refs={"platform": "discord", "message_id": "m-cursor"},
        payload=invalid_cursor,
    ))
    assert "pending or in_progress" in cursor["error"]

    blocked_cursor = _plan_payload()
    blocked_cursor["plan"]["steps"][0]["status"] = "pending"
    blocked_cursor["plan"]["steps"][1]["status"] = "pending"
    blocked_cursor["plan"]["current_step_id"] = "implement"
    blocked_cursor["plan"]["resume_cursor"]["next_step_id"] = "implement"
    dependency = json.loads(cbt.canonical_event_append_tool(
        event_type="task.plan.updated",
        case_id="case:p0",
        summary="cursor dependency unfinished",
        source_refs={"platform": "discord", "message_id": "m-dependency"},
        payload=blocked_cursor,
    ))
    assert "cursor has non-terminal dependencies" in dependency["error"]
    assert called["helper"] is False


def test_route_back_dm_block_is_sanitized_and_durably_verified(monkeypatch):
    fake = _FakeHelper()
    sent = {"called": False}
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_discord_post_message",
        lambda *args, **kwargs: sent.update(called=True),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:p0",
        target_ref={"id": "user-1", "dm_channel_id": "dm-123", "channel_type": "dm"},
        message="Never send this DM",
        message_summary="DM must be blocked",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="routeback:dm:block",
    ))

    assert data["success"] is True
    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert sent["called"] is False
    sql = "\n".join(fake.queries)
    assert "route_back.blocked" in sql
    assert "dm_channel_id" not in sql
    assert "original_target_ref_sha256" in sql


def test_route_back_block_record_failure_is_not_reported_as_terminal_success(monkeypatch):
    monkeypatch.setattr(
        cbt,
        "_route_back_record_blocked",
        lambda **kwargs: {"success": False, "error": "brain unavailable"},
    )
    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:p0",
        target_ref={"id": "unknown-target"},
        message="route back",
        message_summary="unresolved",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="routeback:unknown:block:1",
    ))
    assert data["success"] is False
    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED_RECORD_FAILED"
    assert "durable" in data["final_answer_guard"]


def test_discord_query_is_exactly_thread_scoped_for_non_owner(monkeypatch):
    helper = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env({
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_THREAD_ID": "thread-1",
            "HERMES_SESSION_USER_ID": "teammate",
        }),
    )
    monkeypatch.setattr(
        cbt,
        "load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner"]}},
    )

    foreign = json.loads(cbt.canonical_brain_query_tool(thread_id="thread-2"))
    assert "outside the current Discord thread scope" in foreign["error"]

    own_case = json.loads(cbt.canonical_brain_query_tool(case_id="case:p0"))
    assert own_case["success"] is True
    assert "EXISTS" in helper.queries[-1]
    assert "thread-1" in helper.queries[-1]


def test_existing_case_append_cannot_bootstrap_scope_from_model_refs(monkeypatch):
    helper = _FakeHelper()
    helper.case_exists = True
    helper.scope_linked = False
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    monkeypatch.setattr(cbt, "_canonical_scope_enforced", lambda: True)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env({
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_THREAD_ID": "thread-attacker",
            "HERMES_SESSION_USER_ID": "teammate",
        }),
    )
    monkeypatch.setattr(
        cbt,
        "load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner"]}},
    )

    forged = json.loads(cbt.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:foreign",
        summary="try to forge linkage",
        source_refs={
            "platform": "discord",
            "thread_id": "thread-attacker",
            "message_id": "m-forged",
        },
        idempotency_key="forged:scope",
    ))
    assert "outside the current observed Discord case scope" in forged["error"]
    assert not any(sql.lstrip().upper().startswith("INSERT") for sql in helper.queries)
    scope_sql = "\n".join(helper.queries)
    assert "observed_session" in scope_sql
    assert "deterministic_runtime_receipt" in scope_sql


def test_non_discord_query_fails_closed_without_authenticated_owner(monkeypatch):
    helper = _FakeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    monkeypatch.setattr(cbt, "_canonical_scope_enforced", lambda: True)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env({
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "chat-1",
            "HERMES_SESSION_USER_ID": "teammate",
        }),
    )
    monkeypatch.setattr(
        cbt,
        "load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner"]}},
    )
    data = json.loads(cbt.canonical_brain_query_tool(case_id="case:p0"))
    assert "authenticated owner outside Discord" in data["error"]


def test_route_back_direct_target_requires_config_or_runtime_case_link(monkeypatch):
    helper = _FakeHelper()
    helper.case_exists = True
    helper.scope_linked = True
    helper.target_linked = False
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)
    monkeypatch.setattr(cbt, "_canonical_scope_enforced", lambda: True)
    monkeypatch.setattr(
        cbt,
        "_get_session_env",
        _session_env({
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_THREAD_ID": "requester-thread",
            "HERMES_SESSION_USER_ID": "teammate",
        }),
    )
    monkeypatch.setattr(
        cbt,
        "load_config",
        lambda: {"approvals": {"plan_owner_user_ids": ["owner"]}},
    )
    monkeypatch.delenv("DISCORD_ALLOWED_CHANNELS", raising=False)
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda target_ref: {
            "channel_id": "unlinked-public-channel",
            "channel_type": "public_channel",
            "target_kind": "exact_public_directory_target",
            "target_member_key": None,
            "target_member_id": None,
            "target_mention": None,
        },
    )
    sent = {"called": False}
    monkeypatch.setattr(
        cbt,
        "_discord_post_message",
        lambda *args, **kwargs: sent.update(called=True),
    )
    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:p0",
        target_ref={"channel_id": "unlinked-public-channel"},
        message="Do not send",
        message_summary="unlinked target",
        source_refs={"platform": "discord", "message_id": "m1"},
        idempotency_key="routeback:unlinked",
    ))
    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert sent["called"] is False
    assert data["blocker_reason"].startswith("target_not_approved_or_unresolved")


def test_resume_bundle_requires_exact_case_id():
    data = json.loads(cbt.canonical_brain_query_tool(
        thread_id="thread-1",
        view="resume_bundle",
    ))
    assert "resume_bundle requires an exact case_id" in data["error"]


def test_resume_bundle_fetches_completed_plan_verification_ids_exactly(monkeypatch):
    referenced_id = "11111111-1111-4111-8111-111111111111"
    completed_plan = _plan_payload(
        state="completed",
        revision=2,
        verification_event_ids=[referenced_id],
    )["plan"]

    def event_row(event_id, event_type, occurred_at, payload):
        return {
            "event_id": event_id,
            "schema_version": "canonical_event.v1",
            "event_type": event_type,
            "case_id": "case:p0",
            "occurred_at": occurred_at,
            "source": {},
            "actor": {},
            "subject": {},
            "evidence": [],
            "decision": {},
            "status": {},
            "next_action": {},
            "safety": {},
            "payload": payload,
        }

    plan_row = event_row(
        "plan-event",
        "task.plan.updated",
        "2026-01-02T00:00:00Z",
        {"plan": completed_plan},
    )
    exact_row = event_row(
        referenced_id,
        "task.verification.recorded",
        "2026-01-01T00:00:00Z",
        {"verification": {
            "plan_id": "plan:p0",
            "plan_revision": 1,
            "criterion_ids": ["tests"],
            "outcome": "passed",
        }},
    )

    class _ResumeHelper(_FakeHelper):
        def query(self, sock, sql):
            self.queries.append(sql)
            normalized = sql.upper()
            if "WITH PLAN_EVENTS AS" in normalized:
                return {"rows": [{
                    "event_id": plan_row["event_id"],
                    "plan_id": "plan:p0",
                    "revision": "2",
                    "plan": completed_plan,
                    "occurred_at": plan_row["occurred_at"],
                }]}
            if "E.EVENT_ID IN" in normalized:
                return {"rows": [exact_row]}
            if "E.EVENT_TYPE = 'TASK.VERIFICATION.RECORDED'" in normalized:
                return {"rows": [
                    event_row(
                        f"new-{index:03d}",
                        "task.verification.recorded",
                        f"2026-01-03T00:{index // 60:02d}:{index % 60:02d}Z",
                        {"verification": {"outcome": "failed"}},
                    )
                    for index in range(80)
                ]}
            if "E.EVENT_TYPE = 'APPROVAL.CAPABILITY.RECORDED'" in normalized:
                return {"rows": []}
            if "E.EVENT_TYPE = 'CAPABILITY.CHECK.RECORDED'" in normalized:
                return {"rows": []}
            if normalized.lstrip().startswith("SELECT E.EVENT_ID::TEXT"):
                return {"rows": [plan_row]}
            return {"rows": []}

    helper = _ResumeHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)

    data = json.loads(cbt.canonical_brain_query_tool(
        case_id="case:p0",
        view="resume_bundle",
        limit=80,
    ))

    assert data["success"] is True
    assert data["support_incomplete"] is False
    assert data["support"]["missing_verification_event_ids"] == []
    assert data["cases"][0]["workspace"]["completion_receipts_satisfied"] is True
    assert any("E.EVENT_ID IN" in sql.upper() for sql in helper.queries)


def test_resume_bundle_marks_missing_exact_verification_support_incomplete(monkeypatch):
    referenced_id = "11111111-1111-4111-8111-111111111111"
    completed_plan = _plan_payload(
        state="completed",
        revision=2,
        verification_event_ids=[referenced_id],
    )["plan"]

    class _MissingHelper(_FakeHelper):
        def query(self, sock, sql):
            self.queries.append(sql)
            normalized = sql.upper()
            if "WITH PLAN_EVENTS AS" in normalized:
                return {"rows": [["plan-event", "plan:p0", "2", json.dumps(completed_plan)]]}
            if "E.EVENT_ID IN" in normalized:
                return {"rows": []}
            if "E.EVENT_TYPE =" in normalized:
                return {"rows": []}
            if normalized.lstrip().startswith("SELECT E.EVENT_ID::TEXT"):
                return {"rows": []}
            return {"rows": []}

    helper = _MissingHelper()
    monkeypatch.setattr(cbt, "_load_helper", lambda: helper)

    data = json.loads(cbt.canonical_brain_query_tool(
        case_id="case:p0",
        view="resume_bundle",
    ))

    assert data["support_incomplete"] is True
    assert data["support"]["missing_verification_event_ids"] == [referenced_id]
    assert "completed_plan_verification_support_missing" in data["support"]["reasons"]
