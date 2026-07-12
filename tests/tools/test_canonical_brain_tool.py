from __future__ import annotations

import hashlib
import json
import re

import pytest

from gateway.discord_edge_writer_authority import (
    derive_routeback_edge_idempotency_key,
)
from tools import canonical_brain_tool as cbt


_REAL_DISCORD_VERIFY_MESSAGE_RECEIPT = cbt._discord_verify_message_receipt
_REAL_DISCORD_EDGE_RECONCILE = cbt._discord_edge_reconcile

_DISCORD_GUILD_ID = "1282725267068157972"
_DISCORD_CHANNEL_ID = "1504852408227069993"
_EDGE_REQUEST_ID = "11111111-1111-4111-8111-111111111111"
_EDGE_CAPABILITY_ID = "22222222-2222-4222-8222-222222222222"


class _NoEdgeRecord(RuntimeError):
    code = "discord_edge_reconciliation_not_available"
    dispatch_uncertain = False


def _public_edge_target(
    *,
    channel_id: str = _DISCORD_CHANNEL_ID,
    target_type: str = "public_guild_channel",
    parent_channel_id: str | None = None,
) -> dict:
    target = {
        "channel_id": channel_id,
        "channel_type": (
            "public_thread"
            if target_type == "public_guild_thread"
            else "public_channel"
        ),
        "target_type": target_type,
        "guild_id": _DISCORD_GUILD_ID,
        "target_kind": "exact_public_directory_target",
        "target_member_key": None,
        "target_member_id": None,
        "target_mention": None,
    }
    if parent_channel_id is not None:
        target["parent_channel_id"] = parent_channel_id
    return target


def _signed_edge_request(
    *,
    content: str,
    idempotency_key: str,
    channel_id: str = _DISCORD_CHANNEL_ID,
    target_type: str = "public_guild_channel",
    parent_channel_id: str | None = None,
) -> dict:
    target = {
        "target_type": target_type,
        "guild_id": _DISCORD_GUILD_ID,
        "channel_id": channel_id,
    }
    if parent_channel_id is not None:
        target["parent_channel_id"] = parent_channel_id
    return {
        "protocol": "discord-edge.v1",
        "request_id": _EDGE_REQUEST_ID,
        "sequence": 1,
        "deadline_unix_ms": 4_000_000_000_000,
        "operation": "public.message.send",
        "target": target,
        "payload": {"content": content},
        "idempotency_key": idempotency_key,
        "capability": {
            "key_id": "1" * 64,
            "payload": {
                "protocol": "discord-edge-capability.v1",
                "capability_id": _EDGE_CAPABILITY_ID,
                "operation": "public.message.send",
                "target": target,
                "idempotency_key": idempotency_key,
            },
            "signature": "A" * 86,
        },
    }


def _signed_edge_receipt(
    *,
    outcome: str,
    content: str,
    idempotency_key: str,
    channel_id: str = _DISCORD_CHANNEL_ID,
    target_type: str = "public_guild_channel",
    blocker_code: str | None = None,
) -> dict:
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return {
        "key_id": "2" * 64,
        "payload": {
            "protocol": "discord-edge-receipt.v1",
            "receipt_id": "33333333-3333-4333-8333-333333333333",
            "edge_request_id": _EDGE_REQUEST_ID,
            "capability_id": _EDGE_CAPABILITY_ID,
            "operation": "public.message.send",
            "target": {
                "target_type": target_type,
                "guild_id": _DISCORD_GUILD_ID,
                "channel_id": channel_id,
            },
            "idempotency_key": idempotency_key,
            "request_sha256": "3" * 64,
            "content_sha256": content_sha256,
            "outcome": outcome,
            "discord_object_id": (
                "1522505336614027304" if outcome == "verified" else None
            ),
            "bot_user_id": "1521098105041191104",
            "adapter_accepted": outcome == "verified",
            "readback_verified": outcome == "verified",
            "readback_content_sha256": (
                content_sha256 if outcome == "verified" else None
            ),
            "readback_thread": None,
            "blocker_code": blocker_code,
            "occurred_at_unix_ms": 2_000_000_000_000,
        },
        "signature": "B" * 86,
    }


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

    def _no_edge_record(_client, _exact_intent):
        raise _NoEdgeRecord("no exact durable edge journal record")

    # Route-back execution must now perform an authenticated mutation-free
    # edge lookup before claiming.  Individual recovery/race tests override
    # this default with exact durable evidence.
    monkeypatch.setattr(cbt, "_discord_edge_reconcile", _no_edge_record)


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
            _writer_owned_event=True,
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
    out = cbt._canonical_event_append_impl(
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
        _writer_owned_event=True,
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


def test_configured_writer_rejects_generic_queue_intent_before_append(monkeypatch):
    monkeypatch.setattr(
        "gateway.canonical_writer_boundary.writer_boundary_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        "gateway.canonical_writer_boundary.in_writer_service",
        lambda: False,
    )
    monkeypatch.setattr(
        cbt,
        "_canonical_event_append_impl",
        lambda **kwargs: pytest.fail("queue_intent must not generic-append"),
    )

    data = json.loads(cbt.route_back_tool(
        case_id="case:test",
        target_ref={"id": "public-channel"},
        message_summary="must use exact execute claim",
        source_refs={"platform": "discord", "message_id": "m-queue"},
        mode="queue_intent",
        idempotency_key="queue-intent:configured-writer:1",
    ))

    assert "queue_intent is writer-owned" in data["error"]
    assert "route_back_execute" in data["error"]


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


@pytest.mark.parametrize(
    "target_ref",
    [
        {"id": "alex", "mention": "<@1282940574533423125>"},
        {"id": "alex", "mention": "unknown-person-alias"},
        {"id": "alex", "channel_id": "1504852553031221391"},
        {"channel_id": "public-channel-a", "thread_id": "public-thread-b"},
    ],
)
def test_route_back_target_contradictions_require_clarification(target_ref):
    with pytest.raises(ValueError, match="clarify"):
        cbt._resolve_route_back_public_target(target_ref)


def test_route_back_target_conflict_uses_typed_preclaim_blocker(monkeypatch):
    writer_calls = []
    edge_calls = []

    def _writer(operation, payload, *, idempotency_key=None):
        writer_calls.append((operation, payload, idempotency_key))
        return {"success": True, "outcome": "blocked", "preclaim": True}

    monkeypatch.setattr(cbt, "_writer_proxy_result", _writer)
    monkeypatch.setattr(
        cbt,
        "route_back_tool",
        lambda **kwargs: pytest.fail("typed preclaim must not use generic append"),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_preconnect",
        lambda: edge_calls.append("preconnect"),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:target-conflict",
        target_ref={"id": "alex", "mention": "<@1282940574533423125>"},
        message="must not guess target",
        message_summary="conflicting teammate identities",
        source_refs={"platform": "discord", "message_id": "m-conflict"},
        idempotency_key="routeback:target-conflict:1",
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert data["clarification_required"] is True
    assert "conflicting people" in data["clarification"]
    assert "Do not guess" in data["final_answer_guard"]
    assert edge_calls == []
    assert len(writer_calls) == 1
    operation, payload, writer_idempotency_key = writer_calls[0]
    assert operation == "routeback.finalize_blocked"
    assert payload["preclaim"] is True
    assert "execution_binding" not in payload
    assert payload["blocker_reason"] == "target_not_approved_or_unresolved:ValueError"
    assert writer_idempotency_key == "routeback:target-conflict:1"


def test_route_back_preconnects_then_claims_exact_edge_intent_and_finalizes_sent(
    monkeypatch,
):
    content = "A verified public route-back"
    idempotency_key = "routeback:edge:verified:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_client = object()
    order = []
    claim_calls = []
    terminal_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(
        cbt,
        "_discord_edge_preconnect",
        lambda: order.append("preconnect") or edge_client,
    )

    class _NoEdgeRecord(RuntimeError):
        code = "discord_edge_reconciliation_not_available"

    monkeypatch.setattr(
        cbt,
        "_discord_edge_reconcile",
        lambda client, exact_intent: order.append("reconcile")
        or (_ for _ in ()).throw(_NoEdgeRecord("no edge record")),
    )

    def _claim(**kwargs):
        order.append("claim")
        claim_calls.append(kwargs)
        return json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        })

    def _execute(client, request):
        order.append("execute")
        assert client is edge_client
        assert request == edge_request
        return {
            "state": "verified",
            "blocker": None,
            "replayed": False,
            "receipt": edge_receipt,
        }

    def _finalize(**kwargs):
        order.append("finalize")
        terminal_calls.append(kwargs)
        return json.dumps({"success": True, "outcome": "sent"})

    monkeypatch.setattr(cbt, "_record_route_back_execution_intent", _claim)
    monkeypatch.setattr(cbt, "_discord_edge_execute", _execute)
    monkeypatch.setattr(cbt, "_record_route_back_edge_terminal", _finalize)

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-verified",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="verified public route-back",
        source_refs={"platform": "discord", "message_id": "source-1"},
        idempotency_key=idempotency_key,
    ))

    assert data["success"] is True
    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT"
    assert data["edge_receipt"] == edge_receipt
    assert order == ["preconnect", "reconcile", "claim", "execute", "finalize"]
    assert claim_calls[0]["discord_edge_intent"] == {
        "operation": "public.message.send",
        "target": {
            "target_type": "public_guild_channel",
            "guild_id": _DISCORD_GUILD_ID,
            "channel_id": _DISCORD_CHANNEL_ID,
        },
        "payload": {"content": content},
        "idempotency_key": derive_routeback_edge_idempotency_key(
            case_id="case:edge-verified",
            canonical_idempotency_key=idempotency_key,
        ),
    }
    assert claim_calls[0]["target_ref"]["guild_id"] == _DISCORD_GUILD_ID
    assert claim_calls[0]["target_ref"]["target_type"] == "public_guild_channel"
    assert terminal_calls == [{
        "outcome": "sent",
        "case_id": "case:edge-verified",
        "target_ref": claim_calls[0]["target_ref"],
        "message_summary": "verified public route-back",
        "source_refs": {"platform": "discord", "message_id": "source-1"},
        "idempotency_key": idempotency_key,
        "execution_binding": claim_calls[0]["execution_binding"],
        "discord_edge_request": edge_request,
        "discord_edge_receipt": edge_receipt,
    }]


def test_discord_edge_execute_returns_only_exact_signed_receipt_mapping():
    request = {"signed": "writer-request"}
    receipt = {"signed": "edge-receipt"}

    class _Receipt:
        @staticmethod
        def to_message():
            return receipt

    class _Result:
        state = "verified"
        blocker = None
        replayed = False
        receipt = _Receipt()

    class _Client:
        @staticmethod
        def execute(discord_edge_request, *, require_preconnected):
            assert discord_edge_request is request
            assert require_preconnected is True
            return _Result()

    assert cbt._discord_edge_execute(_Client(), request) == {
        "state": "verified",
        "blocker": None,
        "replayed": False,
        "receipt": receipt,
    }


def test_discord_edge_reconcile_builds_read_only_exact_binding():
    from gateway.discord_edge_protocol import (
        DiscordEdgeErrorCode,
        DiscordEdgeReconciliationQuery,
        SignedDiscordEdgeEnvelope,
        parse_request_for_reconciliation,
    )

    content = "Recover only exact durable evidence"
    idempotency_key = "routeback:edge:reconcile-helper:1"
    request_message = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    receipt_message = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    request = parse_request_for_reconciliation(request_message)
    receipt = SignedDiscordEdgeEnvelope.from_mapping(
        receipt_message,
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        label="receipt",
    )

    class _Result:
        state = "verified"
        blocker = None
        replayed = True

        def __init__(self):
            self.request = request
            self.receipt = receipt

    class _Client:
        @staticmethod
        def reconcile(query, *, require_preconnected):
            assert isinstance(query, DiscordEdgeReconciliationQuery)
            assert require_preconnected is False
            assert query.idempotency_key == idempotency_key
            assert query.operation.value == "public.message.send"
            assert query.target.channel_id == _DISCORD_CHANNEL_ID
            assert query.request_sha256 == request.intent.request_sha256
            assert query.content_sha256 == request.intent.content_sha256
            return _Result()

    result = _REAL_DISCORD_EDGE_RECONCILE(
        _Client(),
        {
            "operation": "public.message.send",
            "target": request_message["target"],
            "payload": request_message["payload"],
            "idempotency_key": idempotency_key,
        },
    )

    assert result == {
        "request": request_message,
        "state": "verified",
        "blocker": None,
        "replayed": True,
        "receipt": receipt_message,
    }


def test_route_back_deduped_claim_reconciles_signed_edge_evidence_without_send(
    monkeypatch,
):
    content = "Recover the exact verified edge receipt"
    idempotency_key = "routeback:edge:deduped-reconcile:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_client = object()
    reconciliation_calls = []
    recovery_calls = []
    terminal_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", lambda: edge_client)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: pytest.fail(
            "durable edge evidence must recover before ordinary claim"
        ),
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_recovery",
        lambda **kwargs: recovery_calls.append(kwargs)
        or json.dumps({
            "success": True,
            "authorization_id": "routeauth:recovered",
            "recovery_kind": "edge_evidence",
            "recovered": True,
            "inserted": False,
            "deduped": True,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda *args: pytest.fail("deduped reconciliation must never dispatch"),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_reconcile",
        lambda client, exact_intent: reconciliation_calls.append(
            (client, exact_intent)
        )
        or {
            "request": edge_request,
            "state": "verified",
            "blocker": None,
            "replayed": True,
            "receipt": edge_receipt,
        },
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: terminal_calls.append(kwargs)
        or json.dumps({"success": True, "outcome": "sent"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-deduped-reconcile",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="reconcile after process loss",
        source_refs={"platform": "discord", "message_id": "source-reconcile"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECONCILED"
    assert data["success"] is True
    assert data["edge_reconciled"] is True
    assert len(reconciliation_calls) == 1
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["recovery_kind"] == "edge_evidence"
    assert recovery_calls[0]["discord_edge_request"] == edge_request
    assert recovery_calls[0]["discord_edge_receipt"] == edge_receipt
    assert terminal_calls[0]["discord_edge_request"] == edge_request
    assert terminal_calls[0]["discord_edge_receipt"] == edge_receipt


def test_route_back_deduped_claim_dispatches_fresh_request_only_after_edge_no_record(
    monkeypatch,
):
    case_id = "case:edge-deduped-no-record"
    content = "Recover a claim that never reached the edge"
    canonical_key = "routeback:edge:deduped-no-record:1"
    edge_key = derive_routeback_edge_idempotency_key(
        case_id=case_id,
        canonical_idempotency_key=canonical_key,
    )
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=edge_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=edge_key,
    )
    execute_calls = []
    reconcile_calls = []

    class _NoEdgeRecord(RuntimeError):
        code = "discord_edge_reconciliation_not_available"
        dispatch_uncertain = False

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": False,
            "deduped": True,
            "discord_edge_request": edge_request,
        }),
    )

    def _no_record(client, exact_intent):
        reconcile_calls.append((client, exact_intent))
        raise _NoEdgeRecord("no exact journal row")

    monkeypatch.setattr(cbt, "_discord_edge_reconcile", _no_record)
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: execute_calls.append((client, request))
        or {
            "state": "verified",
            "blocker": None,
            "replayed": False,
            "receipt": edge_receipt,
        },
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: json.dumps({"success": True, "outcome": "sent"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id=case_id,
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="claim-before-edge recovery",
        source_refs={"platform": "discord", "message_id": "source-no-record"},
        idempotency_key=canonical_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECONCILED"
    assert len(reconcile_calls) == 1
    assert len(execute_calls) == 1
    assert execute_calls[0][1] == edge_request


@pytest.mark.parametrize("scope_failure", ["blocked_result", "typed_exception"])
def test_route_back_restart_no_record_uses_exact_recovery_takeover(
    monkeypatch,
    scope_failure,
):
    case_id = "case:edge-restart-no-record"
    content = "Continue the exact claim after an epoch-only restart"
    canonical_key = "routeback:edge:restart-no-record:1"
    edge_key = derive_routeback_edge_idempotency_key(
        case_id=case_id,
        canonical_idempotency_key=canonical_key,
    )
    fresh_request = _signed_edge_request(
        content=content,
        idempotency_key=edge_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=edge_key,
    )
    claim_calls = []
    recovery_calls = []
    execute_calls = []

    class _ScopeMismatch(RuntimeError):
        code = "scope_mismatch"

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)

    def _claim(**kwargs):
        claim_calls.append(kwargs)
        if scope_failure == "typed_exception":
            raise _ScopeMismatch("old capability epoch")
        return json.dumps({
            "success": False,
            "error_code": "scope_mismatch",
        })

    monkeypatch.setattr(cbt, "_record_route_back_execution_intent", _claim)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_recovery",
        lambda **kwargs: recovery_calls.append(kwargs)
        or json.dumps({
            "success": True,
            "inserted": False,
            "deduped": True,
            "recovered": True,
            "recovery_kind": "edge_no_record",
            "discord_edge_request": fresh_request,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: execute_calls.append((client, request))
        or {
            "state": "verified",
            "blocker": None,
            "replayed": False,
            "receipt": edge_receipt,
        },
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: json.dumps({"success": True, "outcome": "sent"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id=case_id,
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="restart recovery",
        source_refs={"platform": "discord", "message_id": "source-restart"},
        idempotency_key=canonical_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECONCILED"
    assert len(claim_calls) == 1
    assert len(recovery_calls) == 1
    assert recovery_calls[0]["recovery_kind"] == "edge_no_record"
    assert "discord_edge_request" not in recovery_calls[0]
    assert len(execute_calls) == 1
    assert execute_calls[0][1] == fresh_request


def test_route_back_nonverified_edge_evidence_finalizes_blocked(
    monkeypatch,
):
    content = "Not independently verified"
    idempotency_key = "routeback:edge:blocked:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="blocked_before_dispatch",
        content=content,
        idempotency_key=idempotency_key,
        blocker_code="discord_readback_unverified",
    )
    terminal_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: {
            "state": "blocked",
            "blocker": "discord_readback_unverified",
            "replayed": False,
            "receipt": edge_receipt,
        },
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: terminal_calls.append(kwargs)
        or json.dumps({"success": True, "outcome": "blocked"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-blocked",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="edge could not verify delivery",
        source_refs={"platform": "discord", "message_id": "source-blocked"},
        idempotency_key=idempotency_key,
    ))

    assert data["success"] is False
    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert terminal_calls[0]["outcome"] == "blocked"
    assert terminal_calls[0]["discord_edge_request"] == edge_request
    assert terminal_calls[0]["discord_edge_receipt"] == edge_receipt


def test_route_back_delayed_accepted_receipt_cannot_beat_current_verified(
    monkeypatch,
):
    content = "Current verified truth wins over a delayed accepted receipt"
    idempotency_key = "routeback:edge:accepted-then-verified:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    delayed_receipt = _signed_edge_receipt(
        outcome="accepted_unverified",
        content=content,
        idempotency_key=idempotency_key,
        blocker_code="readback_timeout",
    )
    verified_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    order = []
    terminal_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: order.append("execute")
        or {
            "state": "dispatching",
            "blocker": "readback_timeout",
            "replayed": False,
            "receipt": delayed_receipt,
        },
    )

    def _reconcile_after_dispatch(client, exact_intent):
        order.append("reconcile")
        if order.count("reconcile") == 1:
            raise _NoEdgeRecord("preclaim journal is empty")
        return {
            "request": edge_request,
            "state": "verified",
            "blocker": None,
            "replayed": True,
            "receipt": verified_receipt,
        }

    monkeypatch.setattr(
        cbt,
        "_discord_edge_reconcile",
        _reconcile_after_dispatch,
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: order.append("finalize")
        or terminal_calls.append(kwargs)
        or json.dumps({"success": True, "outcome": "sent"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-accepted-then-verified",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="monotonic edge receipt",
        source_refs={"platform": "discord", "message_id": "source-race"},
        idempotency_key=idempotency_key,
    ))

    assert data["success"] is True
    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECONCILED"
    assert order == ["reconcile", "execute", "reconcile", "finalize"]
    assert len(terminal_calls) == 1
    assert terminal_calls[0]["outcome"] == "sent"
    assert terminal_calls[0]["discord_edge_receipt"] == verified_receipt
    assert terminal_calls[0]["discord_edge_receipt"] != delayed_receipt


def test_route_back_current_accepted_receipt_keeps_claim_pending(monkeypatch):
    content = "Accepted mutation still awaits exact readback"
    idempotency_key = "routeback:edge:accepted-pending:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    accepted_receipt = _signed_edge_receipt(
        outcome="accepted_unverified",
        content=content,
        idempotency_key=idempotency_key,
        blocker_code="readback_timeout",
    )
    terminal_calls = []
    reconcile_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: {
            "state": "dispatching",
            "blocker": "readback_timeout",
            "replayed": False,
            "receipt": accepted_receipt,
        },
    )

    def _reconcile_accepted(client, exact_intent):
        reconcile_calls.append(exact_intent)
        if len(reconcile_calls) == 1:
            raise _NoEdgeRecord("preclaim journal is empty")
        return {
            "request": edge_request,
            "state": "dispatching",
            "blocker": "readback_timeout",
            "replayed": True,
            "receipt": accepted_receipt,
        }

    monkeypatch.setattr(
        cbt,
        "_discord_edge_reconcile",
        _reconcile_accepted,
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: terminal_calls.append(kwargs)
        or pytest.fail("accepted_unverified must not become Canonical terminal"),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-accepted-pending",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="readback remains pending",
        source_refs={"platform": "discord", "message_id": "source-pending"},
        idempotency_key=idempotency_key,
    ))

    assert data["success"] is False
    assert data["status"] == (
        "ROUTE_BACK_EXECUTE_EDGE_ACCEPTED_PENDING_VERIFICATION"
    )
    assert data["edge_receipt"] == accepted_receipt
    assert data["edge_reconciled"] is True
    assert data["resend_forbidden"] is True
    assert len(reconcile_calls) == 2
    assert terminal_calls == []


def test_route_back_dispatch_uncertain_reconciles_then_finalizes_blocked(monkeypatch):
    content = "Dispatch outcome remains durably uncertain"
    idempotency_key = "routeback:edge:dispatch-uncertain:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    uncertain_receipt = _signed_edge_receipt(
        outcome="dispatch_uncertain",
        content=content,
        idempotency_key=idempotency_key,
        blocker_code="transport_closed",
    )
    terminal_calls = []
    reconcile_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: {
            "state": "dispatching",
            "blocker": "transport_closed",
            "replayed": False,
            "receipt": uncertain_receipt,
        },
    )

    def _reconcile_uncertain(client, exact_intent):
        reconcile_calls.append(exact_intent)
        if len(reconcile_calls) == 1:
            raise _NoEdgeRecord("preclaim journal is empty")
        return {
            "request": edge_request,
            "state": "dispatching",
            "blocker": "transport_closed",
            "replayed": True,
            "receipt": uncertain_receipt,
        }

    monkeypatch.setattr(
        cbt,
        "_discord_edge_reconcile",
        _reconcile_uncertain,
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: terminal_calls.append(kwargs)
        or json.dumps({"success": True, "outcome": "blocked"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-dispatch-uncertain",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="durably uncertain dispatch",
        source_refs={"platform": "discord", "message_id": "source-uncertain"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED_RECONCILED"
    assert terminal_calls[0]["outcome"] == "blocked"
    assert terminal_calls[0]["discord_edge_receipt"] == uncertain_receipt


def test_route_back_preconnect_failure_records_only_safe_preclaim_blocker(monkeypatch):
    blocked_calls = []
    claim_calls = []
    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(
        cbt,
        "_discord_edge_preconnect",
        lambda: (_ for _ in ()).throw(ConnectionError("edge unavailable")),
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: claim_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cbt,
        "_route_back_record_blocked",
        lambda **kwargs: blocked_calls.append(kwargs)
        or {"success": True, "outcome": "blocked", "preclaim": True},
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-preconnect",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message="No claim without an authenticated edge",
        message_summary="edge unavailable before claim",
        source_refs={"platform": "discord", "message_id": "source-preconnect"},
        idempotency_key="routeback:edge:preconnect:1",
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert data["blocker_reason"] == "discord_edge_preconnect_failed:ConnectionError"
    assert claim_calls == []
    assert "execution_binding" not in blocked_calls[0]


def test_route_back_post_claim_transport_loss_stays_pending_without_terminal_or_retry(
    monkeypatch,
):
    content = "Transport loss has an uncertain delivery outcome"
    idempotency_key = "routeback:edge:transport-loss:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    execute_calls = []
    terminal_calls = []
    blocked_calls = []

    class _TransportLoss(RuntimeError):
        dispatch_uncertain = True
        code = "response_lost_after_dispatch"

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )

    def _lose_transport(client, request):
        execute_calls.append((client, request))
        raise _TransportLoss("edge response was lost")

    monkeypatch.setattr(cbt, "_discord_edge_execute", _lose_transport)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: terminal_calls.append(kwargs),
    )
    monkeypatch.setattr(
        cbt,
        "_route_back_record_blocked",
        lambda **kwargs: blocked_calls.append(kwargs),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-transport-loss",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="transport outcome uncertain",
        source_refs={"platform": "discord", "message_id": "source-loss"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == (
        "ROUTE_BACK_EXECUTE_EDGE_RECEIPT_PENDING_RECONCILIATION"
    )
    assert data["delivery_outcome_uncertain"] is True
    assert data["resend_forbidden"] is True
    assert data["edge_error"] == "response_lost_after_dispatch"
    assert len(execute_calls) == 1
    assert terminal_calls == []
    assert blocked_calls == []


def test_route_back_post_claim_transport_loss_reconciles_without_resend(monkeypatch):
    content = "The edge persisted the receipt before transport loss"
    idempotency_key = "routeback:edge:transport-reconcile:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    execute_calls = []
    reconcile_calls = []
    terminal_calls = []

    class _TransportLoss(RuntimeError):
        dispatch_uncertain = True
        code = "response_lost_after_dispatch"

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )

    def _lose_transport(client, request):
        execute_calls.append((client, request))
        raise _TransportLoss("response lost")

    monkeypatch.setattr(cbt, "_discord_edge_execute", _lose_transport)

    def _reconcile_after_transport(client, exact_intent):
        reconcile_calls.append((client, exact_intent))
        if len(reconcile_calls) == 1:
            raise _NoEdgeRecord("preclaim journal is empty")
        return {
            "request": edge_request,
            "state": "verified",
            "blocker": None,
            "replayed": True,
            "receipt": edge_receipt,
        }

    monkeypatch.setattr(
        cbt,
        "_discord_edge_reconcile",
        _reconcile_after_transport,
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: terminal_calls.append(kwargs)
        or json.dumps({"success": True, "outcome": "sent"}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:edge-transport-reconcile",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="transport loss recovery",
        source_refs={"platform": "discord", "message_id": "source-recovery"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECONCILED"
    assert data["edge_reconciled"] is True
    assert len(execute_calls) == 1
    assert len(reconcile_calls) == 2
    assert len(terminal_calls) == 1


@pytest.mark.parametrize(
    ("claim", "expected_status"),
    [
        (
            {"success": False, "inserted": False, "error": "writer unavailable"},
            "ROUTE_BACK_EXECUTE_INTENT_FAILED",
        ),
        (
            {"success": True, "inserted": False, "deduped": True},
            "ROUTE_BACK_EXECUTE_OUTCOME_UNCERTAIN_PENDING_RECONCILIATION",
        ),
    ],
)
def test_route_back_never_dispatches_without_fresh_writer_authority(
    monkeypatch,
    claim,
    expected_status,
):
    edge_calls = []
    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps(claim),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda *args: edge_calls.append(args),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:no-new-claim",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message="Never dispatch without fresh writer authority",
        message_summary="claim gate",
        source_refs={"platform": "discord", "message_id": "source-claim"},
        idempotency_key="routeback:edge:no-new-claim:1",
    ))

    assert data["status"] == expected_status
    assert edge_calls == []


def test_route_back_thread_target_binds_exact_guild_parent_and_target_type(
    monkeypatch,
):
    thread_id = "1522505332318932992"
    parent_id = "1504852408227069993"
    content = "Reply in the exact public guild thread"
    idempotency_key = "routeback:edge:thread:1"
    claim_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(
            channel_id=thread_id,
            target_type="public_guild_thread",
            parent_channel_id=parent_id,
        ),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: claim_calls.append(kwargs)
        or json.dumps({"success": False, "inserted": False}),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:thread-binding",
        target_ref={"thread_id": thread_id},
        message=content,
        message_summary="exact public thread",
        source_refs={"platform": "discord", "message_id": "source-thread"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_INTENT_FAILED"
    assert claim_calls[0]["discord_edge_intent"]["target"] == {
        "target_type": "public_guild_thread",
        "guild_id": _DISCORD_GUILD_ID,
        "channel_id": thread_id,
        "parent_channel_id": parent_id,
    }
    assert claim_calls[0]["target_ref"]["target_type"] == "public_guild_thread"
    assert claim_calls[0]["target_ref"]["parent_channel_id"] == parent_id


def test_route_back_finalizer_retries_signed_evidence_without_resending(monkeypatch):
    content = "Edge delivered exactly once"
    idempotency_key = "routeback:edge:finalize-retry:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    execute_calls = []
    finalize_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)
    monkeypatch.setattr(
        cbt,
        "_record_route_back_execution_intent",
        lambda **kwargs: json.dumps({
            "success": True,
            "inserted": True,
            "discord_edge_request": edge_request,
        }),
    )
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: execute_calls.append((client, request))
        or {
            "state": "verified",
            "blocker": None,
            "replayed": False,
            "receipt": edge_receipt,
        },
    )

    def _finalize(**kwargs):
        finalize_calls.append(kwargs)
        if len(finalize_calls) == 1:
            raise TimeoutError("writer response lost")
        return json.dumps({"success": True, "outcome": "sent"})

    monkeypatch.setattr(cbt, "_record_route_back_edge_terminal", _finalize)

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:finalize-retry",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="bounded terminal retry",
        source_refs={"platform": "discord", "message_id": "source-finalize"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_SENT_RECONCILED"
    assert len(execute_calls) == 1
    assert len(finalize_calls) == 2
    assert finalize_calls[0]["discord_edge_request"] == edge_request
    assert finalize_calls[1]["discord_edge_receipt"] == edge_receipt


def test_route_back_terminal_persistence_failure_never_invents_blocked(monkeypatch):
    content = "Verified edge evidence awaits Canonical persistence"
    idempotency_key = "routeback:edge:terminal-pending:1"
    edge_request = _signed_edge_request(
        content=content,
        idempotency_key=idempotency_key,
    )
    edge_receipt = _signed_edge_receipt(
        outcome="verified",
        content=content,
        idempotency_key=idempotency_key,
    )
    claim_calls = []
    blocked_calls = []

    monkeypatch.setattr(cbt, "_existing_route_back_terminal", lambda **kwargs: {})
    monkeypatch.setattr(
        cbt,
        "_resolve_route_back_public_target",
        lambda _target_ref: _public_edge_target(),
    )
    monkeypatch.setattr(cbt, "_authorize_route_back_execution", lambda **kwargs: None)
    monkeypatch.setattr(cbt, "_discord_edge_preconnect", object)

    def _claim(**kwargs):
        claim_calls.append(kwargs)
        if len(claim_calls) == 1:
            return json.dumps({
                "success": True,
                "inserted": True,
                "discord_edge_request": edge_request,
            })
        return json.dumps({"success": True, "inserted": False})

    monkeypatch.setattr(cbt, "_record_route_back_execution_intent", _claim)
    monkeypatch.setattr(
        cbt,
        "_discord_edge_execute",
        lambda client, request: {
            "state": "verified",
            "blocker": None,
            "replayed": False,
            "receipt": edge_receipt,
        },
    )
    monkeypatch.setattr(
        cbt,
        "_record_route_back_edge_terminal",
        lambda **kwargs: json.dumps({"success": False, "error": "writer unavailable"}),
    )
    monkeypatch.setattr(
        cbt,
        "_route_back_record_blocked",
        lambda **kwargs: blocked_calls.append(kwargs),
    )

    data = json.loads(cbt.route_back_execute_tool(
        case_id="case:terminal-pending",
        target_ref={"channel_id": _DISCORD_CHANNEL_ID},
        message=content,
        message_summary="Canonical terminal pending",
        source_refs={"platform": "discord", "message_id": "source-pending"},
        idempotency_key=idempotency_key,
    ))

    assert data["status"] == "ROUTE_BACK_EXECUTE_CANONICAL_TERMINAL_PENDING"
    assert data["delivery_outcome_verified"] is True
    assert data["resend_forbidden"] is True
    assert len(claim_calls) == 2
    assert blocked_calls == []


def test_legacy_gateway_discord_receipt_verifier_is_fail_closed():
    with pytest.raises(
        RuntimeError,
        match="discord_receipt_verification_requires_privileged_edge",
    ):
        _REAL_DISCORD_VERIFY_MESSAGE_RECEIPT(
            channel_id=_DISCORD_CHANNEL_ID,
            message_id="1522505336614027304",
            expected_content_sha256="a" * 64,
        )


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
    def open_connection():
        return _FakeSock()

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


def test_check_requirements_false_when_boundary_policy_absent(monkeypatch, request):
    from gateway import canonical_writer_boundary as writer_boundary

    writer_boundary._reset_frozen_writer_boundary_config_for_tests()
    request.addfinalizer(
        writer_boundary._reset_frozen_writer_boundary_config_for_tests
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    assert cbt.check_canonical_brain_requirements() is False


def test_check_requirements_requires_explicit_writer_boundary(monkeypatch, request):
    from gateway import canonical_writer_boundary as writer_boundary

    writer_boundary._reset_frozen_writer_boundary_config_for_tests()
    request.addfinalizer(
        writer_boundary._reset_frozen_writer_boundary_config_for_tests
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"canonical_brain": {"audit_bridge": {"enabled": False}}})

    assert cbt.check_canonical_brain_requirements() is False

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"canonical_brain": {"tools_enabled": True}})
    assert cbt.check_canonical_brain_requirements() is False

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "canonical_brain": {
                "tools_enabled": True,
                "writer_boundary": {
                    "enabled": True,
                    "socket_path": "/run/muncho-canonical-writer/writer.sock",
                },
            }
        },
    )
    # Runtime edits are intentionally invisible until a process restart.
    assert cbt.check_canonical_brain_requirements() is False
    writer_boundary._reset_frozen_writer_boundary_config_for_tests()
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
    assert cbt.canonical_active_plan_revision(
        case_id="case:p0",
        plan_id="plan:p0",
    ) == 1
    assert cbt.canonical_active_plan_matches(
        case_id="case:p0",
        plan_id="plan:p0",
        plan_revision=1,
    ) is True
    assert cbt.canonical_active_plan_matches(
        case_id="case:p0",
        plan_id="plan:p0",
        plan_revision=2,
    ) is False
    assert cbt.canonical_active_plan_matches(
        case_id="case:p0",
        plan_id="plan:other",
    ) is False
    assert all("capability.check.recorded" not in sql for sql in fake.queries)


def test_active_plan_revision_requires_exact_positive_writer_revision(monkeypatch):
    responses = iter([
        {"matches": True, "plan_revision": 7},
        {"matches": True, "plan_revision": None},
        {"matches": False, "plan_revision": 7},
    ])
    monkeypatch.setattr(
        cbt,
        "_writer_proxy_result",
        lambda *args, **kwargs: next(responses),
    )

    assert cbt.canonical_active_plan_revision(
        case_id="case:p0",
        plan_id="plan:p0",
    ) == 7
    assert cbt.canonical_active_plan_revision(
        case_id="case:p0",
        plan_id="plan:p0",
    ) is None
    assert cbt.canonical_active_plan_revision(
        case_id="case:p0",
        plan_id="plan:p0",
    ) is None


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
                "plan_revision": 1,
                "session_key_sha256": "a" * 64,
                "approval_source_sha256": "c" * 64,
                "command_hashes": ["b" * 64],
            }},
        ),
        (
            "capability.check.recorded",
            {"capability_receipt": {
                "approval_id": "a1", "plan_id": "p1", "state": "authorized",
                "plan_revision": 1,
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
    data = json.loads(cbt._canonical_event_append_impl(
        event_type=event_type,
        case_id="case:p0",
        summary="forged receipt",
        source_refs={"platform": "discord", "message_id": "m1"},
        payload=payload,
        _writer_owned_event=True,
    ))

    assert data["success"] is True
    sql = "\n".join(fake.queries)
    assert '"verified":false' in sql
    assert '"attestation":"runtime_process_receipt_unverified"' in sql
    assert event_type not in cbt.CANONICAL_EVENT_APPEND_SCHEMA["parameters"]["properties"]["event_type"]["enum"]


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (
            "approval.capability.recorded",
            {"approval_receipt": {
                "approval_id": "a1",
                "plan_id": "p1",
                "state": "granted",
                "session_key_sha256": "a" * 64,
                "approval_source_sha256": "c" * 64,
                "command_hashes": ["b" * 64],
            }},
        ),
        (
            "capability.check.recorded",
            {"capability_receipt": {
                "approval_id": "a1",
                "plan_id": "p1",
                "state": "authorized",
                "session_key_sha256": "a" * 64,
                "command_sha256": "b" * 64,
            }},
        ),
    ],
)
def test_process_receipts_require_exact_plan_revision_before_database(
    monkeypatch,
    event_type,
    payload,
):
    called = {"helper": False}
    monkeypatch.setattr(
        cbt,
        "_load_helper",
        lambda: called.update(helper=True),
    )

    data = json.loads(cbt._canonical_event_append_impl(
        event_type=event_type,
        case_id="case:p0",
        summary="missing revision",
        source_refs={"platform": "discord", "message_id": "m-revision"},
        payload=payload,
        _writer_owned_event=True,
    ))

    assert "plan_revision must be a positive bounded integer" in data["error"]
    assert called["helper"] is False


@pytest.mark.parametrize(
    "event_type",
    sorted(cbt.WRITER_OWNED_EVENT_TYPES),
)
def test_generic_model_append_excludes_writer_owned_events_before_database(
    monkeypatch,
    event_type,
):
    called = {"helper": False}
    monkeypatch.setattr(
        cbt,
        "_load_helper",
        lambda: called.update(helper=True),
    )

    data = json.loads(cbt.canonical_event_append_tool(
        event_type=event_type,
        case_id="case:p0",
        summary="must use typed writer operation",
        source_refs={"platform": "discord", "message_id": "m-writer-owned"},
    ))

    assert event_type not in cbt.ALLOWED_EVENT_TYPES
    assert event_type not in (
        cbt.CANONICAL_EVENT_APPEND_SCHEMA["parameters"]["properties"]
        ["event_type"]["enum"]
    )
    assert f"event_type_not_allowed:{event_type}" in data["error"]
    assert called["helper"] is False


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
    edge = {"called": False}
    monkeypatch.setattr(cbt, "_load_helper", lambda: fake)
    monkeypatch.setattr(
        cbt,
        "_discord_edge_preconnect",
        lambda: edge.update(called=True),
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
    assert edge["called"] is False
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
    edge = {"called": False}
    monkeypatch.setattr(
        cbt,
        "_discord_edge_preconnect",
        lambda: edge.update(called=True),
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
    assert edge["called"] is False
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
