"""Seeded contract matrix for opt-in tiered memory write approval.

All secret-shaped fixtures are deterministic synthetic strings. They are never
valid credentials.
"""

from __future__ import annotations

import importlib
import builtins
import json
import os
import subprocess
import sys
import threading
import time

import pytest


TIER_CASES = (
    ("memory", "Parser lives at src/tools/parser.py.", None, "Tier0", "operational_fact"),
    ("memory", "Retry count is 3.", None, "Tier0", "operational_fact"),
    ("memory", "Deployments run from main via `./run.sh`.", None, "Tier0", "operational_fact"),
    ("memory", "Service listens on port 8080.", None, "Tier0", "operational_fact"),
    ("memory", "Alice Smith is the project lead.", None, "Tier1", "proper_name"),
    ("memory", "Customer MRR is EUR 5,000.", None, "Tier1", "customer_financial_data"),
    ("memory", "Delete staging data after every deploy.", None, "Tier1", "imperative_future_instruction"),
    ("memory", "Please deploy src/app.py now.", None, "Tier1", "imperative_future_instruction"),
    ("memory", "Configure src/app.py for production.", None, "Tier1", "imperative_future_instruction"),
    ("memory", "Set timeout in config/app.yml.", None, "Tier1", "imperative_future_instruction"),
    ("memory", "Julio owns src/app.py.", None, "Tier1", "proper_name"),
    ("memory", "Build budget for src/app.py is $5000.", None, "Tier1", "customer_financial_data"),
    (
        "memory",
        "Database username is admin in `config/database.yml`.",
        None,
        "Tier1",
        "sensitive_identifier",
    ),
    (
        "memory",
        "Passport ID is `XK1234567`.",
        None,
        "Tier1",
        "pii",
    ),
    (
        "memory",
        "Customer gross margin is in reports/q2.md.",
        None,
        "Tier1",
        "customer_financial_data",
    ),
    (
        "memory",
        "Deploy from main every Friday.",
        None,
        "Tier1",
        "imperative_future_instruction",
    ),
    ("memory", "The project lead owns delivery.", None, "Tier1", "uncertain"),
    ("user", "Prefers compact status updates.", None, "Tier1", "user_profile"),
    ("memory", "Owner email is person@example.invalid.", None, "Tier1", "pii"),
    ("memory", "Customer balance is EUR 1,250.00.", None, "Tier1", "customer_financial_data"),
    ("memory", "Opaque reference: q7F3mK9vP2xL8cN4dR6tY1uW5zA0sB", None, "Tier1", "high_entropy_value"),
    ("memory", "Always deploy this service on Fridays.", None, "Tier1", "imperative_future_instruction"),
    ("memory", "Maybe the production queue uses workers.", None, "Tier1", "uncertain"),
    ("memory", "See https://example.invalid/customer/42.", None, "Tier1", "sensitive_identifier"),
    ("memory", "Build uses src/new.py.", "password=hunter2-synthetic", "Tier2", "credential_assignment"),
    ("memory", "AWS key " + "AKIA" + "ABCDEFGHIJKLMNOP", None, "Tier2", "aws_access_key"),
    ("memory", "GitHub token " + "ghp" + "_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", None, "Tier2", "github_token"),
    ("memory", "Slack token " + "xoxb" + "-" + "123456789012-123456789012-abcdefghijklmnopqrstuvwx", None, "Tier2", "slack_token"),
    ("memory", "-----BEGIN " + "PRIVATE KEY-----", None, "Tier2", "private_key"),
    ("memory", "JWT eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.c2lnbmF0dXJl", None, "Tier2", "jwt"),
    ("memory", "Authorization: Bearer synthetic-token-value-123456", None, "Tier2", "bearer_token"),
    ("memory", "Stripe key sk_test_1234567890ABCDEF", None, "Tier2", "deterministic_token"),
)

TIER0_AMBIGUOUS_OPERATIONAL_CASES = (
    "Request routing is configured in config/routes.yml.",
    "Card component lives at src/components/Card.tsx.",
    "Database transaction isolation is set in config/db.yml.",
    "Release date 2026-07-13 is recorded in CHANGELOG.md.",
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "matrix-profile")
    monkeypatch.setenv("HERMES_SESSION_ID", "matrix-session")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "matrix-task")
    return tmp_path


def _set_memory_config(**values):
    from hermes_cli.config import load_config, save_config

    config = load_config() or {}
    config.setdefault("memory", {}).update(values)
    save_config(config)


def _activate_fleet_drain_marker(hermes_home, monkeypatch):
    profile_home = hermes_home / "profiles" / "ares"
    profile_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    marker = (
        profile_home.parent
        / "audit"
        / "memory-drain"
        / "fleet-drain-active.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text('{"schema":"hermes-memory-fleet-drain/v1"}\n', encoding="utf-8")
    marker.chmod(0o600)
    return profile_home, marker


def _fail_write_approval_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools" and "write_approval" in fromlist:
            raise ImportError("synthetic write approval import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


FALSEY_WRITE_APPROVAL_VALUES = (
    "false",
    "off",
    "0",
)


def test_memory_fleet_drain_marker_is_durable_and_idempotent(
    hermes_home, monkeypatch,
):
    from tools import write_approval as wa

    profiles_root = hermes_home / "profiles"
    profile_home = profiles_root / "ares"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    wa.begin_memory_fleet_drain(profiles_root)
    marker = wa.memory_fleet_drain_marker_path()
    first = marker.read_bytes()
    wa.begin_memory_fleet_drain(profiles_root)

    assert marker == profiles_root / "audit" / "memory-drain" / "fleet-drain-active.json"
    assert first == b'{"schema":"hermes-memory-fleet-drain/v1"}\n'
    assert marker.read_bytes() == first
    assert marker.stat().st_mode & 0o777 == 0o600
    assert wa.memory_fleet_drain_active()
    assert wa.memory_fleet_drain_active(profiles_root)

    wa.clear_memory_fleet_drain(profiles_root)
    assert not marker.exists()
    assert not wa.memory_fleet_drain_active(profiles_root)


def test_fleet_drain_marker_blocks_builtin_single_write(
    hermes_home, monkeypatch,
):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    profile_home, _marker = _activate_fleet_drain_marker(hermes_home, monkeypatch)
    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()

    result = json.loads(memory_tool(
        action="add",
        target="memory",
        content="Parser lives at src/parser.py.",
        store=store,
    ))

    assert result == {
        "success": False,
        "marker": "MEMORY_FLEET_DRAIN_ACTIVE",
        "error": "MEMORY_FLEET_DRAIN_ACTIVE",
    }
    assert store.memory_entries == []
    assert wa.list_pending(wa.MEMORY) == []
    assert list((profile_home / "pending" / "memory").glob("*.json")) == []


def test_memory_fleet_drain_blocks_builtin_single_without_pending_artifact(
    hermes_home, monkeypatch,
):
    test_fleet_drain_marker_blocks_builtin_single_write(hermes_home, monkeypatch)


def test_fleet_drain_marker_blocks_builtin_batch_write(
    hermes_home, monkeypatch,
):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    profile_home, _marker = _activate_fleet_drain_marker(hermes_home, monkeypatch)
    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()

    result = json.loads(memory_tool(
        target="memory",
        operations=[
            {"action": "add", "content": "Parser lives at src/parser.py."},
            {"action": "add", "content": "Owner email is person@example.invalid."},
        ],
        store=store,
    ))

    assert result == {
        "success": False,
        "marker": "MEMORY_FLEET_DRAIN_ACTIVE",
        "error": "MEMORY_FLEET_DRAIN_ACTIVE",
    }
    assert store.memory_entries == []
    assert wa.list_pending(wa.MEMORY) == []
    assert list((profile_home / "pending" / "memory").glob("*.json")) == []


def test_memory_fleet_drain_blocks_builtin_batch_without_pending_artifact(
    hermes_home, monkeypatch,
):
    test_fleet_drain_marker_blocks_builtin_batch_write(hermes_home, monkeypatch)


@pytest.mark.parametrize("action", ["add", "update", "remove", "auto_extract"])
def test_fleet_drain_marker_blocks_holographic_write(
    hermes_home, monkeypatch, action,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    profile_home = hermes_home / "profiles" / "ares"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(profile_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        fact_id = provider._store.add_fact(
            "Parser lives at src/parser.py.", category="project"
        )
        before = provider._store.list_facts(limit=10)
        _profile_home, _marker = _activate_fleet_drain_marker(hermes_home, monkeypatch)

        if action == "add":
            response = provider.handle_tool_call("fact_store", {
                "action": "add",
                "content": "Owner email is person@example.invalid.",
            })
        elif action == "update":
            response = provider.handle_tool_call("fact_store", {
                "action": "update",
                "fact_id": fact_id,
                "content": "Owner email is person@example.invalid.",
            })
        elif action == "remove":
            response = provider.handle_tool_call("fact_store", {
                "action": "remove",
                "fact_id": fact_id,
            })
        else:
            responses = []
            original_handle = provider._handle_fact_store

            def observe_handle(args, **kwargs):
                response = original_handle(args, **kwargs)
                responses.append(response)
                return response

            monkeypatch.setattr(provider, "_handle_fact_store", observe_handle)
            provider._auto_extract_facts([{
                "role": "user",
                "content": "I prefer concise status updates for this project.",
            }])
            assert len(responses) == 1
            response = responses[0]

        assert json.loads(response) == {
            "success": False,
            "marker": "MEMORY_FLEET_DRAIN_ACTIVE",
            "error": "MEMORY_FLEET_DRAIN_ACTIVE",
        }
        assert provider._store.list_facts(limit=10) == before
        assert wa.list_pending(wa.MEMORY) == []
        assert list((profile_home / "pending" / "memory").glob("*.json")) == []
    finally:
        provider.shutdown()


def test_memory_fleet_drain_blocks_holographic_without_pending_artifact(
    hermes_home, monkeypatch,
):
    test_fleet_drain_marker_blocks_holographic_write(hermes_home, monkeypatch, "add")


@pytest.mark.parametrize("value", FALSEY_WRITE_APPROVAL_VALUES)
def test_falsey_string_gate_import_failure_preserves_single_write(
    hermes_home, monkeypatch, value,
):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=value, write_approval_tiered=True)
    store = MemoryStore()
    _fail_write_approval_import(monkeypatch)

    result = json.loads(memory_tool(
        action="add", target="memory", content="Parser lives at src/parser.py.", store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == ["Parser lives at src/parser.py."]


@pytest.mark.parametrize("value", FALSEY_WRITE_APPROVAL_VALUES)
def test_falsey_string_gate_import_failure_preserves_batch_write(
    hermes_home, monkeypatch, value,
):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=value, write_approval_tiered=True)
    store = MemoryStore()
    _fail_write_approval_import(monkeypatch)

    result = json.loads(memory_tool(
        target="memory",
        operations=[{"action": "add", "content": "Parser lives at src/parser.py."}],
        store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == ["Parser lives at src/parser.py."]


def test_import_failure_falsey_gate_preserves_single_write(
    hermes_home, monkeypatch,
):
    test_falsey_string_gate_import_failure_preserves_single_write(
        hermes_home, monkeypatch, "false",
    )


def test_import_failure_falsey_gate_preserves_batch_write(
    hermes_home, monkeypatch,
):
    test_falsey_string_gate_import_failure_preserves_batch_write(
        hermes_home, monkeypatch, "false",
    )


@pytest.mark.parametrize(
    "target,content,old_text,expected_tier,reason_code",
    TIER_CASES,
    ids=[f"{case[3]}-{case[4]}" for case in TIER_CASES],
)
def test_seeded_tier_matrix(target, content, old_text, expected_tier, reason_code):
    from tools.write_approval import classify_memory_write

    decision = classify_memory_write(target=target, content=content, old_text=old_text)

    assert decision.tier.value == expected_tier
    assert reason_code in decision.reason_codes


@pytest.mark.parametrize("content", TIER0_AMBIGUOUS_OPERATIONAL_CASES)
def test_ambiguous_operational_nouns_and_iso_dates_remain_tier0(content):
    from tools.write_approval import classify_memory_write

    decision = classify_memory_write(target="memory", content=content)

    assert decision.tier.value == "Tier0"
    assert decision.reason_codes == ("operational_fact",)


def test_all_classifier_reasons_match_canonical_contract():
    from tools.memory_tool import classify_memory_write_tier
    from tools.write_approval import MEMORY_REASON_CODES, classify_memory_write

    canonical = set(MEMORY_REASON_CODES)
    emitted = set()
    for target, content, old_text, _expected_tier, _reason_code in TIER_CASES:
        decision = classify_memory_write(
            target=target,
            content=content,
            old_text=old_text,
        )
        emitted.update(decision.reason_codes)

    assert emitted == canonical

    external = classify_memory_write_tier(
        target="memory",
        action="add",
        content="Alice Smith is the project lead.",
    )
    serialized = json.loads(json.dumps(external))
    assert "proper_name" in serialized["reason_codes"]
    assert "proper-name" not in serialized["reason_codes"]
    assert all(type(reason) is str for reason in serialized["reason_codes"])


def test_public_drain_classifier_returns_stable_mapping():
    from tools.memory_tool import classify_memory_write_tier

    single = classify_memory_write_tier(
        target="memory",
        action="add",
        content="AWS key " + "AKIA" + "ABCDEFGHIJKLMNOP",
    )
    batch = classify_memory_write_tier(
        target="memory",
        action="batch",
        operations=[
            {"action": "add", "content": "Parser lives at src/parser.py."},
            {"action": "add", "content": "Owner email is person@example.invalid."},
        ],
    )

    assert single == {
        "tier": "tier2",
        "operation_shape": "add",
        "reason_codes": ["aws_access_key"],
    }
    assert batch["tier"] == "tier1"
    assert batch["operation_shape"] == "batch"
    assert "pii" in batch["reason_codes"]


def test_native_replay_receipt_prevents_duplicate_apply_after_discard_crash(
    hermes_home, monkeypatch,
):
    from tools import memory_tool as mt
    from tools import write_approval as wa

    pending = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Parser lives at src/parser.py."},
        summary="Tier0 operational fact",
        origin="foreground",
    )
    effects = []

    def apply_once(_payload, _store):
        effects.append("applied")
        return {"success": True}

    original_discard = wa.discard_pending
    discard_calls = 0

    def crash_once(subsystem, pending_id):
        nonlocal discard_calls
        discard_calls += 1
        if discard_calls == 1:
            raise OSError("synthetic crash before pending unlink")
        return original_discard(subsystem, pending_id)

    monkeypatch.setattr(mt, "apply_memory_pending", apply_once)
    monkeypatch.setattr(wa, "discard_pending", crash_once)

    with pytest.raises(OSError, match="synthetic crash"):
        wa.replay_pending(wa.MEMORY, pending["id"])
    assert wa.get_pending(wa.MEMORY, pending["id"]) is not None

    assert wa.replay_pending(wa.MEMORY, pending["id"]) is True
    assert effects == ["applied"]
    assert wa.get_pending(wa.MEMORY, pending["id"]) is None


def test_memory_cli_approve_uses_crash_convergent_replay(hermes_home, monkeypatch):
    from hermes_cli.write_approval_commands import _approve
    from tools import write_approval as wa

    pending = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Parser lives at src/app.py."},
        summary="redacted",
        origin="foreground",
    )
    replayed = []

    def replay(subsystem, pending_id):
        replayed.append((subsystem, pending_id))
        return True

    monkeypatch.setattr(wa, "replay_pending", replay)
    monkeypatch.setattr(
        wa,
        "discard_pending",
        lambda *_args: pytest.fail("approve must let replay_pending own cleanup"),
    )

    result = _approve(wa.MEMORY, [pending["id"]], object())

    assert result == "Approved 1 memory write(s)."
    assert replayed == [(wa.MEMORY, pending["id"])]


def test_memory_cli_approve_surfaces_replay_failure(hermes_home, monkeypatch):
    from hermes_cli.write_approval_commands import _approve
    from tools import write_approval as wa

    pending = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Parser lives at src/app.py."},
        summary="redacted",
        origin="foreground",
    )
    monkeypatch.setattr(wa, "replay_pending", lambda *_args: False)

    result = _approve(wa.MEMORY, [pending["id"]], object())

    assert result.startswith("Approved 0 memory write(s).\nFailed:")
    assert f"{pending['id']}: replay failed; pending write retained" in result


def test_native_replay_converges_after_apply_before_receipt_crash(
    hermes_home, monkeypatch,
):
    from tools import memory_tool as mt
    from tools import write_approval as wa

    pending = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Parser lives at src/parser.py."},
        summary="Tier0 operational fact",
        origin="foreground",
    )
    original_persist = wa._persist_replay_receipt
    persist_calls = 0

    def crash_once(receipt):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise OSError("synthetic crash after target mutation")
        return original_persist(receipt)

    monkeypatch.setattr(wa, "_persist_replay_receipt", crash_once)

    with pytest.raises(OSError, match="synthetic crash"):
        wa.replay_pending(wa.MEMORY, pending["id"])
    assert wa.get_pending(wa.MEMORY, pending["id"]) is not None

    assert wa.replay_pending(wa.MEMORY, pending["id"]) is True
    store = mt.load_on_disk_store()
    assert store.memory_entries == ["Parser lives at src/parser.py."]
    assert wa.get_pending(wa.MEMORY, pending["id"]) is None


def test_native_replace_replay_converges_after_apply_before_receipt_crash(
    hermes_home, monkeypatch,
):
    from tools import memory_tool as mt
    from tools import write_approval as wa

    store = mt.load_on_disk_store()
    assert store.add("memory", "Parser lives at src/old_parser.py.")["success"] is True
    pending = wa.stage_write(
        wa.MEMORY,
        {
            "action": "replace",
            "target": "memory",
            "old_text": "old_parser.py",
            "content": "Parser lives at src/parser.py.",
        },
        summary="Tier0 operational fact",
        origin="foreground",
    )
    original_persist = wa._persist_replay_receipt
    persist_calls = 0

    def crash_once(receipt):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise OSError("synthetic crash after target mutation")
        return original_persist(receipt)

    monkeypatch.setattr(wa, "_persist_replay_receipt", crash_once)

    with pytest.raises(OSError, match="synthetic crash"):
        wa.replay_pending(wa.MEMORY, pending["id"])
    assert wa.replay_pending(wa.MEMORY, pending["id"]) is True
    assert mt.load_on_disk_store().memory_entries == ["Parser lives at src/parser.py."]
    assert wa.get_pending(wa.MEMORY, pending["id"]) is None


def test_tiered_audit_is_redacted_and_carries_available_provenance(hermes_home):
    from tools import write_approval as wa

    raw = "password=audit-only-synthetic"
    decision = wa.classify_memory_write(target="memory", content=raw)
    record = wa.audit_memory_decision(
        decision,
        action="add",
        target="memory",
        store="builtin",
        content=raw,
    )

    audit_path = hermes_home / "logs" / "memory-write-audit.jsonl"
    persisted = audit_path.read_text(encoding="utf-8")
    assert raw not in persisted
    assert "MEMORY_SECRET_REJECT" in persisted
    assert record["timestamp"]
    assert record["profile"] == "matrix-profile"
    assert record["session"] == "matrix-session"
    assert record["task"] == "matrix-task"
    assert record["origin"] == "foreground"
    assert record["action"] == "add"
    assert record["target"] == "memory"
    assert record["store"] == "builtin"
    assert record["tier"] == "Tier2"
    assert len(record["content_sha256"]) == 64


def test_tier0_writes_inline_and_tier1_stages(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()

    allowed = json.loads(memory_tool(
        action="add", target="memory", content="Parser lives at src/parser.py.", store=store,
    ))
    staged = json.loads(memory_tool(
        action="add", target="memory", content="Owner email is person@example.invalid.", store=store,
    ))

    assert allowed["success"] is True
    assert store.memory_entries == ["Parser lives at src/parser.py."]
    assert staged["staged"] is True
    assert staged["tier"] == "Tier1"
    pending = wa.get_pending(wa.MEMORY, staged["pending_id"])
    assert pending["payload"]["content"] == "Owner email is person@example.invalid."
    assert "person@example.invalid" not in pending["summary"]


def test_stage_write_persistence_failure_has_no_pending_success_or_artifact(
    hermes_home, monkeypatch,
):
    from tools import write_approval as wa

    decision = wa.classify_memory_write(
        target="memory", content="Owner email is person@example.invalid.",
    )
    audit = wa.audit_memory_decision(
        decision,
        action="add",
        target="memory",
        store="builtin",
        content="Owner email is person@example.invalid.",
    )

    def fail_persistence(*_args, **_kwargs):
        raise OSError("synthetic pending disk failure")

    monkeypatch.setattr(wa, "_atomic_write", fail_persistence, raising=False)

    with pytest.raises(RuntimeError, match="Failed to persist pending memory write"):
        wa.stage_write(
            wa.MEMORY,
            {
                "action": "add",
                "target": "memory",
                "content": "Owner email is person@example.invalid.",
                "_memory_audit": wa.memory_audit_context(audit),
            },
            summary="redacted",
            origin="foreground",
        )

    pending_dir = hermes_home / "pending" / "memory"
    assert wa.list_pending(wa.MEMORY) == []
    assert list(pending_dir.glob("*.json*")) == []
    records = [
        json.loads(line)
        for line in (hermes_home / "logs" / "memory-write-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event_type"] for record in records] == ["decision", "failed"]
    assert records[-1]["failure_code"] == "pending_store_unavailable"


def test_atomic_pending_write_is_private_complete_and_durable(
    hermes_home, monkeypatch,
):
    import stat

    from tools import write_approval as wa

    pending_dir = hermes_home / "pending" / "memory"
    pending_dir.mkdir(parents=True)
    path = pending_dir / "durable.json"
    record = {"payload": "x" * 8192}
    original_open = wa.os.open
    original_write = wa.os.write
    original_fsync = wa.os.fsync
    original_replace = wa.os.replace
    events = []
    temp_open = {}
    write_calls = 0

    def tracked_open(target, flags, mode=0o777):
        descriptor = original_open(target, flags, mode)
        if str(target).endswith(".tmp"):
            temp_open.update(flags=flags, mode=mode)
        return descriptor

    def short_write(descriptor, data):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            prefix = data[: max(1, len(data) // 3)]
            return original_write(descriptor, prefix)
        return original_write(descriptor, data)

    def tracked_fsync(descriptor):
        kind = "dir_fsync" if stat.S_ISDIR(wa.os.fstat(descriptor).st_mode) else "file_fsync"
        events.append(kind)
        return original_fsync(descriptor)

    def tracked_replace(source, destination):
        events.append("replace")
        return original_replace(source, destination)

    monkeypatch.setattr(wa.os, "open", tracked_open)
    monkeypatch.setattr(wa.os, "write", short_write)
    monkeypatch.setattr(wa.os, "fsync", tracked_fsync)
    monkeypatch.setattr(wa.os, "replace", tracked_replace)

    wa._atomic_write(path, record)

    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert temp_open["flags"] & wa.os.O_EXCL
    assert temp_open["mode"] == 0o600
    assert write_calls > 1
    assert events == ["file_fsync", "replace", "dir_fsync"]
    assert list(pending_dir.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "boundary",
    ["write", "file_fsync", "replace", "dir_fsync"],
)
def test_atomic_pending_write_boundary_failure_never_stages_or_leaves_artifact(
    hermes_home, monkeypatch, boundary,
):
    import stat

    from tools import write_approval as wa

    def injected_write(*_args, **_kwargs):
        raise OSError("synthetic write failure")

    original_fsync = wa.os.fsync

    def injected_fsync(descriptor):
        is_dir = stat.S_ISDIR(wa.os.fstat(descriptor).st_mode)
        if (boundary == "dir_fsync" and is_dir) or (
            boundary == "file_fsync" and not is_dir
        ):
            raise OSError(f"synthetic {boundary} failure")
        return original_fsync(descriptor)

    def injected_replace(*_args, **_kwargs):
        raise OSError("synthetic replace failure")

    if boundary == "write":
        monkeypatch.setattr(wa.os, "write", injected_write)
    elif boundary in {"file_fsync", "dir_fsync"}:
        monkeypatch.setattr(wa.os, "fsync", injected_fsync)
    else:
        monkeypatch.setattr(wa.os, "replace", injected_replace)

    with pytest.raises(wa.PendingWriteError):
        wa.stage_write(
            wa.MEMORY,
            {
                "action": "add",
                "target": "memory",
                "content": "Owner email is person@example.invalid.",
            },
            summary="redacted",
            origin="foreground",
        )

    pending_dir = hermes_home / "pending" / "memory"
    assert wa.list_pending(wa.MEMORY) == []
    assert list(pending_dir.glob("*.json*")) == []


def test_builtin_stage_persistence_failure_surfaces_error_without_pending_id(
    hermes_home, monkeypatch,
):
    from tools import memory_tool as mt
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)

    def fail_persistence(*_args, **_kwargs):
        raise OSError("synthetic pending disk failure")

    monkeypatch.setattr(wa, "_atomic_write", fail_persistence, raising=False)
    result = json.loads(mt.memory_tool(
        action="add",
        target="memory",
        content="Owner email is person@example.invalid.",
        store=mt.MemoryStore(),
    ))

    assert result["success"] is False
    assert "Failed to persist pending memory write" in result["error"]
    assert "staged" not in result
    assert "pending_id" not in result
    assert wa.list_pending(wa.MEMORY) == []


def test_holographic_stage_persistence_failure_surfaces_error_without_pending_id(
    hermes_home, monkeypatch,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")

    def fail_persistence(*_args, **_kwargs):
        raise OSError("synthetic pending disk failure")

    monkeypatch.setattr(wa, "_atomic_write", fail_persistence, raising=False)
    try:
        result = json.loads(provider.handle_tool_call("fact_store", {
            "action": "add",
            "content": "Owner email is person@example.invalid.",
        }))
    finally:
        provider.shutdown()

    assert result["success"] is False
    assert "Failed to persist pending memory write" in result["error"]
    assert "staged" not in result
    assert "pending_id" not in result
    assert wa.list_pending(wa.MEMORY) == []


def test_tier2_rejects_without_staging_or_storage(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    secret = "AWS key " + "AKIA" + "ABCDEFGHIJKLMNOP"

    result = json.loads(memory_tool(
        action="add", target="memory", content=secret, store=store,
    ))

    assert result["success"] is False
    assert result["tier"] == "Tier2"
    assert result["marker"] == "MEMORY_SECRET_REJECT"
    assert secret not in json.dumps(result)
    assert store.memory_entries == []
    assert wa.list_pending(wa.MEMORY) == []


def test_public_pending_replay_applies_and_discards_only_on_success(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import load_on_disk_store

    applied = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": "Parser lives at src/parser.py."},
        summary="synthetic",
        origin="foreground",
    )
    stale = wa.stage_write(
        wa.MEMORY,
        {"action": "replace", "target": "memory", "old_text": "missing", "content": "New fact."},
        summary="synthetic",
        origin="foreground",
    )

    assert wa.replay_pending(wa.MEMORY, applied["id"]) is True
    assert wa.get_pending(wa.MEMORY, applied["id"]) is None
    assert load_on_disk_store().memory_entries == ["Parser lives at src/parser.py."]
    assert wa.replay_pending(wa.MEMORY, stale["id"]) is False
    assert wa.get_pending(wa.MEMORY, stale["id"]) is not None


def test_batch_uses_highest_tier_and_remains_atomic(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    tier1_ops = [
        {"action": "add", "content": "Parser lives at src/parser.py."},
        {"action": "add", "content": "Owner email is person@example.invalid."},
    ]
    tier2_ops = [
        {"action": "add", "content": "Parser lives at src/parser.py."},
        {"action": "add", "content": "password=batch-secret-synthetic"},
    ]

    staged = json.loads(memory_tool(target="memory", operations=tier1_ops, store=store))
    rejected = json.loads(memory_tool(target="memory", operations=tier2_ops, store=store))

    assert staged["staged"] is True
    assert staged["tier"] == "Tier1"
    assert rejected["tier"] == "Tier2"
    assert rejected["marker"] == "MEMORY_SECRET_REJECT"
    assert store.memory_entries == []
    assert len(wa.list_pending(wa.MEMORY)) == 1


def test_classifier_failure_defers_to_ordinary_gate(hermes_home, monkeypatch):
    from tools import memory_tool as mt

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = mt.MemoryStore()

    def fail_classifier(**_kwargs):
        raise RuntimeError("synthetic classifier failure")

    monkeypatch.setattr("tools.write_approval.classify_memory_write", fail_classifier)
    result = json.loads(mt.memory_tool(
        action="add", target="memory", content="Parser lives at src/parser.py.", store=store,
    ))

    assert result["staged"] is True
    assert store.memory_entries == []


def test_tier2_audit_failure_still_rejects_without_staging(hermes_home, monkeypatch):
    from tools import memory_tool as mt
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = mt.MemoryStore()

    def fail_audit(*_args, **_kwargs):
        raise OSError("synthetic audit sink failure")

    monkeypatch.setattr(wa, "audit_memory_decision", fail_audit)
    result = json.loads(mt.memory_tool(
        action="add",
        target="memory",
        content="AWS key " + "AKIA" + "ABCDEFGHIJKLMNOP",
        store=store,
    ))

    assert result["tier"] == "Tier2"
    assert result["marker"] == "MEMORY_SECRET_REJECT"
    assert store.memory_entries == []
    assert wa.list_pending(wa.MEMORY) == []


@pytest.mark.parametrize(
    "content,expected_tier",
    [
        ("Parser lives at src/parser.py.", "Tier0"),
        ("Owner email is person@example.invalid.", "Tier1"),
    ],
)
def test_audit_sink_failure_never_applies_memory_inline(
    hermes_home, monkeypatch, content, expected_tier,
):
    from tools import memory_tool as mt
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = mt.MemoryStore()

    def fail_audit(*_args, **_kwargs):
        raise OSError("synthetic audit sink failure")

    monkeypatch.setattr(wa, "audit_memory_decision", fail_audit)
    result = json.loads(mt.memory_tool(
        action="add", target="memory", content=content, store=store,
    ))

    assert result["staged"] is True
    assert result["tier"] == expected_tier
    assert store.memory_entries == []
    assert len(wa.list_pending(wa.MEMORY)) == 1


def test_default_off_is_native_compatible(hermes_home):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=False, write_approval_tiered=False)
    store = MemoryStore()
    result = json.loads(memory_tool(
        action="add", target="memory", content="Owner email is person@example.invalid.", store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == ["Owner email is person@example.invalid."]


def test_configured_gate_import_failure_blocks_single_write(hermes_home, monkeypatch):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    _fail_write_approval_import(monkeypatch)

    result = json.loads(memory_tool(
        action="add", target="memory", content="Parser lives at src/parser.py.", store=store,
    ))

    assert result == {
        "success": False,
        "marker": "MEMORY_WRITE_GATE_UNAVAILABLE",
        "error": "Memory write approval is configured but unavailable.",
    }
    assert store.memory_entries == []


def test_configured_gate_import_failure_blocks_batch_write(hermes_home, monkeypatch):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    _fail_write_approval_import(monkeypatch)

    result = json.loads(memory_tool(
        target="memory",
        operations=[{"action": "add", "content": "Parser lives at src/parser.py."}],
        store=store,
    ))

    assert result == {
        "success": False,
        "marker": "MEMORY_WRITE_GATE_UNAVAILABLE",
        "error": "Memory write approval is configured but unavailable.",
    }
    assert store.memory_entries == []


def test_disabled_gate_import_failure_preserves_native_write(hermes_home, monkeypatch):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=False, write_approval_tiered=True)
    store = MemoryStore()
    _fail_write_approval_import(monkeypatch)

    result = json.loads(memory_tool(
        action="add", target="memory", content="Parser lives at src/parser.py.", store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == ["Parser lives at src/parser.py."]


@pytest.mark.parametrize("value", FALSEY_WRITE_APPROVAL_VALUES)
def test_holographic_falsey_gate_import_failure_uses_native_fact_tools(
    hermes_home, monkeypatch, value,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider

    _set_memory_config(write_approval=value, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        fact_id = provider._store.add_fact("Parser lives at src/parser.py.")
        _fail_write_approval_import(monkeypatch)

        added = json.loads(provider.handle_tool_call("fact_store", {
            "action": "add",
            "content": "Worker lives at src/worker.py.",
        }))
        feedback = json.loads(provider.handle_tool_call("fact_feedback", {
            "action": "helpful",
            "fact_id": fact_id,
        }))

        assert added["status"] == "added"
        assert feedback["helpful_count"] == 1
        assert {fact["content"] for fact in provider._store.list_facts(limit=10)} == {
            "Parser lives at src/parser.py.",
            "Worker lives at src/worker.py.",
        }
    finally:
        provider.shutdown()


@pytest.mark.parametrize("tool_name,args", [
    ("fact_store", {"action": "add", "content": "Worker lives at src/worker.py."}),
    ("fact_feedback", {"action": "helpful", "fact_id": 1}),
])
def test_holographic_configured_gate_import_failure_is_fail_closed(
    hermes_home, monkeypatch, tool_name, args,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        fact_id = provider._store.add_fact("Parser lives at src/parser.py.")
        if tool_name == "fact_feedback":
            args = {**args, "fact_id": fact_id}
        before = provider._store.list_facts(limit=10)
        _fail_write_approval_import(monkeypatch)

        result = json.loads(provider.handle_tool_call(tool_name, args))

        assert result == {
            "success": False,
            "marker": "MEMORY_WRITE_GATE_UNAVAILABLE",
            "error": "Memory write approval is configured but unavailable.",
        }
        assert provider._store.list_facts(limit=10) == before
    finally:
        provider.shutdown()


def test_tiered_flag_alone_does_not_change_native_behavior(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=False, write_approval_tiered=True)
    store = MemoryStore()
    secret = "AWS key " + "AKIA" + "ABCDEFGHIJKLMNOP"
    result = json.loads(memory_tool(
        action="add", target="memory", content=secret, store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == [secret]
    assert wa.list_pending(wa.MEMORY) == []
    assert not (hermes_home / "logs" / "memory-write-audit.jsonl").exists()


def test_holographic_mutations_share_tiers_and_native_pending_replay(hermes_home):
    pytest.importorskip("numpy")
    from hermes_cli.write_approval_commands import _approve
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        allowed = json.loads(provider.handle_tool_call(
            "fact_store", {"action": "add", "content": "Parser lives at src/parser.py."}
        ))
        staged = json.loads(provider.handle_tool_call(
            "fact_store", {"action": "add", "content": "Owner email is person@example.invalid."}
        ))
        rejected = json.loads(provider.handle_tool_call(
            "fact_store", {"action": "add", "content": "password=fact-secret-synthetic"}
        ))
        recalled = json.loads(provider.handle_tool_call(
            "fact_store", {"action": "search", "query": "parser"}
        ))

        assert allowed["status"] == "added"
        assert staged["staged"] is True and staged["tier"] == "Tier1"
        assert rejected["tier"] == "Tier2"
        assert rejected["marker"] == "MEMORY_SECRET_REJECT"
        assert recalled["count"] == 1
        assert len(wa.list_pending(wa.MEMORY)) == 1

        approval = _approve(wa.MEMORY, [staged["pending_id"]], MemoryStore())
        assert approval == "Approved 1 memory write(s)."
        facts = provider._store.list_facts(limit=10)
        assert {fact["content"] for fact in facts} == {
            "Parser lives at src/parser.py.",
            "Owner email is person@example.invalid.",
        }
    finally:
        provider.shutdown()


def test_holographic_classifies_secret_in_tags_before_persisting(hermes_home):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        secret = "password=tag-secret-synthetic"
        result = json.loads(provider.handle_tool_call("fact_store", {
            "action": "add",
            "content": "Parser lives at src/parser.py.",
            "category": "project",
            "tags": secret,
        }))

        assert result["tier"] == "Tier2"
        assert result["marker"] == "MEMORY_SECRET_REJECT"
        assert secret not in json.dumps(result)
        assert provider._store.list_facts(limit=10) == []
        assert wa.list_pending(wa.MEMORY) == []
    finally:
        provider.shutdown()


@pytest.mark.parametrize("field", ["content", "category", "tags"])
def test_holographic_classifies_every_new_persisted_text_field(hermes_home, field):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        secret = "password=new-field-secret-synthetic"
        args = {
            "action": "add",
            "content": "Parser lives at src/parser.py.",
            "category": "project",
            "tags": "parser",
        }
        args[field] = secret

        result = json.loads(provider.handle_tool_call("fact_store", args))

        assert result["tier"] == "Tier2"
        assert result["marker"] == "MEMORY_SECRET_REJECT"
        assert secret not in json.dumps(result)
        assert provider._store.list_facts(limit=10) == []
        assert wa.list_pending(wa.MEMORY) == []
    finally:
        provider.shutdown()


@pytest.mark.parametrize("field", ["content", "category", "tags"])
def test_holographic_classifies_every_existing_persisted_text_field(hermes_home, field):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        secret = "password=existing-field-secret-synthetic"
        values = {
            "content": "Parser lives at src/parser.py.",
            "category": "project",
            "tags": "parser",
        }
        values[field] = secret
        fact_id = provider._store.add_fact(**values)

        result = json.loads(provider.handle_tool_call("fact_store", {
            "action": "remove",
            "fact_id": fact_id,
        }))

        assert result["removed"] is True
        assert secret not in json.dumps(result)
        assert provider._store.list_facts(limit=10) == []
        assert wa.list_pending(wa.MEMORY) == []
    finally:
        provider.shutdown()


def test_existing_secret_can_be_removed_without_pending_retention(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    secret = "password=remove-existing-synthetic"
    store.add("memory", secret)

    result = json.loads(memory_tool(
        action="remove", target="memory", old_text=secret, store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == []
    assert wa.list_pending(wa.MEMORY) == []
    assert secret not in json.dumps(result)
    pending_dir = hermes_home / "pending" / "memory"
    assert secret not in "".join(
        path.read_text(encoding="utf-8") for path in pending_dir.glob("*.json")
    )


def test_existing_secret_can_be_replaced_with_sanitized_text_without_staging(
    hermes_home,
):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    secret = "password=replace-existing-synthetic"
    store.add("memory", f"Legacy credential: {secret}")

    result = json.loads(memory_tool(
        action="replace",
        target="memory",
        old_text=secret,
        content="credential removed",
        store=store,
    ))

    assert result["success"] is True
    assert store.memory_entries == ["credential removed"]
    assert wa.list_pending(wa.MEMORY) == []
    assert secret not in json.dumps(result)


def test_new_secret_write_rejects_without_staging(hermes_home):
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    secret = "password=new-write-synthetic"

    result = json.loads(memory_tool(
        action="add", target="memory", content=secret, store=store,
    ))

    assert result["success"] is False
    assert result["marker"] == wa.MEMORY_SECRET_REJECT
    assert store.memory_entries == []
    assert wa.list_pending(wa.MEMORY) == []
    assert secret not in json.dumps(result)


def test_holographic_existing_secret_remediation_never_stages_raw_payload(
    hermes_home,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    secret = "password=holographic-existing-synthetic"
    try:
        fact_id = provider._store.add_fact(secret, category="project")

        updated = json.loads(provider.handle_tool_call("fact_store", {
            "action": "update",
            "fact_id": fact_id,
            "content": "Credential removed from the project record.",
        }))

        assert updated["updated"] is True
        assert wa.list_pending(wa.MEMORY) == []
        assert secret not in json.dumps(updated)
        assert provider._store.list_facts(limit=1)[0]["content"] == (
            "Credential removed from the project record."
        )
    finally:
        provider.shutdown()


def test_holographic_auto_extraction_uses_tier_gate(hermes_home):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({
        "db_path": str(hermes_home / "facts.db"),
        "auto_extract": True,
    })
    provider.initialize("holographic-session")
    try:
        secret = "password=auto-extract-secret-synthetic"

        provider.on_session_end([
            {"role": "user", "content": f"I prefer {secret}"},
        ])

        assert provider._store.list_facts(limit=10) == []
        assert wa.list_pending(wa.MEMORY) == []
    finally:
        provider.shutdown()


def test_holographic_update_classifies_existing_content_and_feedback_stays_allowed(hermes_home):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        fact_id = provider._store.add_fact("Parser lives at src/old.py.")
        staged = json.loads(provider.handle_tool_call("fact_store", {
            "action": "update",
            "fact_id": fact_id,
            "content": "Owner email is person@example.invalid.",
        }))
        feedback = json.loads(provider.handle_tool_call("fact_feedback", {
            "action": "helpful", "fact_id": fact_id,
        }))

        assert staged["staged"] is True
        assert provider._store.list_facts(limit=1)[0]["content"] == "Parser lives at src/old.py."
        assert feedback["fact_id"] == fact_id
        assert feedback["new_trust"] > feedback["old_trust"]
    finally:
        provider.shutdown()


def test_holographic_pending_replay_reports_unapplied_mutation(hermes_home):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import apply_holographic_pending

    result = apply_holographic_pending(
        {"action": "update", "fact_id": 999, "content": "Parser lives at src/new.py."},
        config={"db_path": str(hermes_home / "facts.db")},
    )

    assert result["success"] is False
    assert "not applied" in result["error"]


@pytest.mark.parametrize("action", ["update", "remove"])
def test_holographic_pending_replay_uses_transactional_snapshot_cas(hermes_home, action):
    pytest.importorskip("numpy")
    from hermes_cli.write_approval_commands import _approve
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")
    try:
        fact_id = provider._store.add_fact(
            "Alice Smith is the project lead.", category="project", tags="lead",
        )
        args = {"action": action, "fact_id": fact_id}
        if action == "update":
            args["content"] = "Alice Smith is the delivery lead."
        staged = json.loads(provider.handle_tool_call("fact_store", args))
        pending = wa.get_pending(wa.MEMORY, staged["pending_id"])

        assert pending["payload"]["expected_fact_sha256"]
        provider._store.update_fact(fact_id, content="Concurrent fact change.")

        approval = _approve(wa.MEMORY, [staged["pending_id"]], MemoryStore())
        current = provider._store.list_facts(limit=10)
        assert "Approved 0 memory write(s)." in approval
        assert "replay failed; pending write retained" in approval
        assert wa.get_pending(wa.MEMORY, staged["pending_id"]) is not None
        assert current[0]["content"] == "Concurrent fact change."
    finally:
        provider.shutdown()


def test_holographic_replay_converges_after_apply_before_receipt_crash(
    hermes_home, monkeypatch,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    db_path = hermes_home / "facts.db"
    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(db_path)})
    provider.initialize("holographic-session")
    try:
        fact_id = provider._store.add_fact("Owner email is person@example.invalid.")
        staged = json.loads(provider.handle_tool_call("fact_store", {
            "action": "update", "fact_id": fact_id, "trust_delta": 0.2,
        }))
        assert staged["staged"] is True
    finally:
        provider.shutdown()

    original_persist = wa._persist_replay_receipt
    persist_calls = 0

    def crash_once(receipt):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise OSError("synthetic crash after target mutation")
        return original_persist(receipt)

    monkeypatch.setattr(wa, "_persist_replay_receipt", crash_once)
    with pytest.raises(OSError, match="synthetic crash"):
        wa.replay_pending(wa.MEMORY, staged["pending_id"])
    assert wa.replay_pending(wa.MEMORY, staged["pending_id"]) is True

    provider = HolographicMemoryProvider({"db_path": str(db_path)})
    provider.initialize("verify-replay")
    try:
        fact = provider._store.list_facts(limit=1)[0]
        assert fact["trust_score"] == pytest.approx(0.7)
    finally:
        provider.shutdown()
    assert wa.get_pending(wa.MEMORY, staged["pending_id"]) is None


@pytest.mark.parametrize("action", ["update", "remove"])
def test_holographic_replay_retries_after_secondary_maintenance_failure(
    hermes_home, monkeypatch, action,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider
    from tools import write_approval as wa

    db_path = hermes_home / "facts.db"
    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(db_path)})
    provider.initialize("holographic-session")
    try:
        original = "Owner email is person@example.invalid."
        fact_id = provider._store.add_fact(original, category="project")
        args = {"action": action, "fact_id": fact_id}
        if action == "update":
            args["content"] = "Parser lives at src/app.py."
        staged = json.loads(provider.handle_tool_call("fact_store", args))
    finally:
        provider.shutdown()

    from plugins.memory.holographic.store import MemoryStore as HolographicStore

    original_rebuild = HolographicStore._rebuild_bank
    failures = 0

    def fail_once(self, category):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise RuntimeError("synthetic secondary maintenance failure")
        return original_rebuild(self, category)

    monkeypatch.setattr(HolographicStore, "_rebuild_bank", fail_once)

    assert wa.replay_pending(wa.MEMORY, staged["pending_id"]) is False
    assert wa.get_pending(wa.MEMORY, staged["pending_id"]) is not None
    assert not wa._replay_receipt_path(wa._replay_receipt(
        wa.MEMORY,
        staged["pending_id"],
        wa.get_pending(wa.MEMORY, staged["pending_id"])["payload"],
    )).exists()

    verifier = HolographicMemoryProvider({"db_path": str(db_path)})
    verifier.initialize("verify-rollback")
    try:
        facts = verifier._store.list_facts(limit=10)
        assert len(facts) == 1
        assert facts[0]["content"] == original
    finally:
        verifier.shutdown()

    assert wa.replay_pending(wa.MEMORY, staged["pending_id"]) is True
    assert wa.get_pending(wa.MEMORY, staged["pending_id"]) is None


def test_langfuse_tags_recall_and_feedback_without_raw_fact_metadata(monkeypatch):
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")
    state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None)
    monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task", "session"), state)
    monkeypatch.setattr(mod, "_get_langfuse", lambda: object())

    started = {}

    def fake_start(_state, **kwargs):
        started.update(kwargs)
        return object()

    monkeypatch.setattr(mod, "_start_child_observation", fake_start)
    mod.on_pre_tool_call(
        tool_name="fact_store",
        args={"action": "search", "query": "private recalled phrase"},
        task_id="task",
        session_id="session",
        tool_call_id="recall-call",
    )

    assert started["metadata"]["memory_operation"] == "recall"
    assert started["metadata"]["memory_action"] == "search"
    assert "private recalled phrase" not in json.dumps(started["metadata"])
    assert "private recalled phrase" not in json.dumps(started["input_value"])

    feedback_observation = object()
    state.tools["feedback-call"] = feedback_observation
    ended = {}

    def fake_end(observation, **kwargs):
        ended["observation"] = observation
        ended.update(kwargs)

    monkeypatch.setattr(mod, "_end_observation", fake_end)
    mod.on_post_tool_call(
        tool_name="fact_feedback",
        args={"action": "helpful", "fact_id": 7},
        result={
            "fact_id": 7,
            "old_trust": 0.5,
            "new_trust": 0.55,
            "content": "private fact outcome",
        },
        task_id="task",
        session_id="session",
        tool_call_id="feedback-call",
    )

    assert ended["metadata"]["memory_operation"] == "feedback"
    assert ended["metadata"]["memory_feedback"] == "helpful"
    assert ended["metadata"]["memory_feedback_outcome"] == "recorded"
    assert "content" not in ended["metadata"]
    assert "private fact outcome" not in json.dumps(ended["output"])


@pytest.mark.parametrize(
    "tool_name,args,result,expected_outcome",
    [
        (
            "fact_store",
            {"action": "search", "query": "private query"},
            {"results": [{"content": "private recalled fact"}], "count": 1},
            "success",
        ),
        (
            "fact_store",
            {"action": "search", "query": "private query"},
            {"results": [], "count": 0},
            "empty",
        ),
        (
            "fact_store",
            {"action": "search", "query": "private query"},
            {"error": "private recall backend failure"},
            "failure",
        ),
        (
            "hindsight_recall",
            {"query": "private query"},
            {"result": "private recalled observation"},
            "success",
        ),
        (
            "hindsight_recall",
            {"query": "private query"},
            {"result": "No relevant memories found."},
            "empty",
        ),
        (
            "hindsight_recall",
            {"query": "private query"},
            {"error": "private recall backend failure"},
            "failure",
        ),
    ],
)
def test_langfuse_recall_observations_emit_content_free_outcome(
    monkeypatch, tool_name, args, result, expected_outcome,
):
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")
    state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None)
    observation = object()
    state.tools["recall-call"] = observation
    monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task", "session"), state)

    ended = {}
    monkeypatch.setattr(
        mod,
        "_end_observation",
        lambda _observation, **kwargs: ended.update(kwargs),
    )
    mod.on_post_tool_call(
        tool_name=tool_name,
        args=args,
        result=result,
        task_id="task",
        session_id="session",
        tool_call_id="recall-call",
    )

    assert ended["metadata"]["memory_operation"] == "recall"
    assert ended["metadata"]["memory_recall_outcome"] == expected_outcome
    encoded = json.dumps(ended)
    assert "private query" not in encoded
    assert "private recalled" not in encoded
    assert "private recall backend failure" not in encoded


def test_langfuse_recall_none_is_empty_only_after_tool_completion(monkeypatch):
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")
    args = {"query": "private query"}

    pre_metadata = mod._memory_tool_metadata("hindsight_recall", args)
    assert "memory_recall_outcome" not in pre_metadata

    state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None)
    state.tools["recall-call"] = object()
    monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task", "session"), state)
    ended = {}
    monkeypatch.setattr(
        mod,
        "_end_observation",
        lambda _observation, **kwargs: ended.update(kwargs),
    )

    mod.on_post_tool_call(
        tool_name="hindsight_recall",
        args=args,
        result=None,
        task_id="task",
        session_id="session",
        tool_call_id="recall-call",
    )

    assert ended["metadata"]["memory_recall_outcome"] == "empty"
    assert "private query" not in json.dumps(ended)


@pytest.mark.parametrize(
    "tool_name,result,expected_outcome",
    [
        ("hindsight_recall", {"result": "No relevant memories found."}, "empty"),
        ("hindsight_recall", {"error": "private Hindsight failure"}, "failure"),
        ("hindsight_recall", {"status": "error", "message": "private failure"}, "failure"),
        ("honcho_search", {"result": "No relevant context found."}, "empty"),
        ("honcho_search", {"error": "private Honcho failure"}, "failure"),
        ("mem0_search", {"result": "No relevant memories found."}, "empty"),
        ("mem0_search", {"error": "private Mem0 failure"}, "failure"),
        ("supermemory_search", {"results": [], "count": 0}, "empty"),
        ("supermemory_search", {"error": "private Supermemory failure"}, "failure"),
        ("viking_search", {"results": [], "total": 0}, "empty"),
        ("viking_search", {"error": "private OpenViking failure"}, "failure"),
        ("retaindb_search", {"results": []}, "empty"),
        ("retaindb_search", {"error": "private RetainDB failure"}, "failure"),
        ("brv_query", {"result": "No relevant memories found."}, "empty"),
        ("brv_query", {"error": "private ByteRover failure"}, "failure"),
    ],
)
def test_langfuse_recall_outcome_matches_native_adapter_shapes(
    tool_name, result, expected_outcome,
):
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")

    metadata = mod._memory_tool_metadata(
        tool_name,
        {"query": "private query"},
        result,
    )

    assert metadata["memory_operation"] == "recall"
    assert metadata["memory_recall_outcome"] == expected_outcome
    assert "private" not in json.dumps(metadata).lower()


def test_langfuse_redacts_fact_store_pre_input_and_post_output(monkeypatch):
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")
    state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None)
    monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task", "session"), state)
    monkeypatch.setattr(mod, "_get_langfuse", lambda: object())

    captured = {}
    observation = object()

    def fake_start(_state, **kwargs):
        captured["started"] = kwargs
        return observation

    def fake_end(_observation, **kwargs):
        captured["ended"] = kwargs

    monkeypatch.setattr(mod, "_start_child_observation", fake_start)
    monkeypatch.setattr(mod, "_end_observation", fake_end)
    category_secret = "password=category-secret-synthetic"
    args = {
        "action": "add",
        "content": "private fact input",
        "tags": "private-tag",
        "category": category_secret,
    }
    mod.on_pre_tool_call(
        tool_name="fact_store", args=args, task_id="task", session_id="session",
        tool_call_id="fact-call",
    )
    mod.on_post_tool_call(
        tool_name="fact_store", args=args,
        result={"status": "added", "content": "private fact output"},
        task_id="task", session_id="session", tool_call_id="fact-call",
    )

    encoded = json.dumps({
        "input": captured["started"]["input_value"],
        "pre_metadata": captured["started"]["metadata"],
        "output": captured["ended"]["output"],
        "post_metadata": captured["ended"]["metadata"],
    })
    assert "private fact input" not in encoded
    assert "private-tag" not in encoded
    assert category_secret not in encoded
    assert "private fact output" not in encoded
    assert captured["ended"]["metadata"]["memory_operation"] == "write"
    assert captured["ended"]["metadata"]["memory_outcome"] == "applied"


def test_langfuse_redacts_builtin_memory_observation_and_trace_duplicates(monkeypatch):
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")
    state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None)
    secret = "password=trace-duplicate-synthetic"
    state.turn_tool_calls = [{
        "id": "memory-call",
        "arguments": json.dumps({"action": "add", "content": secret}),
        "function": {
            "name": "memory",
            "arguments": json.dumps({"action": "add", "content": secret}),
        },
    }]
    monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task", "session"), state)
    monkeypatch.setattr(mod, "_get_langfuse", lambda: object())

    captured = {}
    observation = object()
    monkeypatch.setattr(
        mod,
        "_start_child_observation",
        lambda _state, **kwargs: captured.setdefault("started", kwargs) or observation,
    )
    monkeypatch.setattr(
        mod,
        "_end_observation",
        lambda _observation, **kwargs: captured.setdefault("ended", kwargs),
    )
    args = {"action": "add", "target": "memory", "content": secret}

    mod.on_pre_tool_call(
        tool_name="memory", args=args, task_id="task", session_id="session",
        tool_call_id="memory-call",
    )
    mod.on_post_tool_call(
        tool_name="memory", args=args,
        result={"success": False, "marker": "MEMORY_SECRET_REJECT", "content": secret},
        task_id="task", session_id="session", tool_call_id="memory-call",
    )

    encoded = json.dumps(
        {"captured": captured, "tool_calls": state.turn_tool_calls}, default=str,
    )
    assert secret not in encoded
    assert captured["started"]["metadata"]["memory_operation"] == "write"
    assert captured["started"]["metadata"]["memory_store"] == "builtin"
    assert captured["ended"]["metadata"]["memory_outcome"] == "rejected"


def test_langfuse_serializers_redact_memory_tool_call_history():
    sys.modules.pop("plugins.observability.langfuse", None)
    mod = importlib.import_module("plugins.observability.langfuse")
    secret = "password=serialized-history-synthetic"
    tool_call = {
        "id": "memory-call",
        "type": "function",
        "function": {
            "name": "memory",
            "arguments": json.dumps({"action": "add", "content": secret}),
        },
    }
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "name": "memory", "tool_call_id": "memory-call", "content": secret},
    ]

    encoded = json.dumps(mod._serialize_messages(messages))

    assert secret not in encoded
    assert "memory_payload_redacted" in encoded


def test_memory_pending_lock_is_reentrant_and_blocks_other_processes(hermes_home):
    from tools import write_approval as wa

    with wa.memory_pending_lock():
        with wa.memory_pending_lock():
            assert (hermes_home / "pending" / "memory" / ".drain.lock").exists()

    code = """
from tools.write_approval import memory_pending_lock
with memory_pending_lock():
    print('locked', flush=True)
    input()
"""
    env = os.environ.copy()
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        completed = threading.Event()
        thread = threading.Thread(
            target=lambda: (wa.list_pending(wa.MEMORY), completed.set()),
            daemon=True,
        )
        thread.start()
        time.sleep(0.2)
        assert not completed.is_set()
        process.stdin.write("\n")
        process.stdin.flush()
        process.wait(timeout=5)
        thread.join(timeout=5)
        assert completed.is_set()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_audit_records_cover_correlated_memory_lifecycle_without_raw_content(hermes_home):
    from hermes_cli.write_approval_commands import _approve
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()
    safe = "Parser lives at src/parser.py."
    sensitive = "Owner email is person@example.invalid."
    secret = "password=lifecycle-secret-synthetic"

    assert json.loads(memory_tool(
        action="add", target="memory", content=safe, store=store,
    ))["success"]
    staged = json.loads(memory_tool(
        action="add", target="memory", content=sensitive, store=store,
    ))
    rejected = json.loads(memory_tool(
        action="add", target="memory", content=secret, store=store,
    ))
    assert rejected["tier"] == "Tier2"
    assert _approve(wa.MEMORY, [staged["pending_id"]], store) == "Approved 1 memory write(s)."

    records = [
        json.loads(line)
        for line in (hermes_home / "logs" / "memory-write-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    common = {
        "event_type", "timestamp", "decision_id", "profile", "session", "task",
        "origin", "action", "target", "store", "tier", "reason_codes",
        "content_sha256",
    }
    assert all(common <= record.keys() for record in records)
    assert {record["event_type"] for record in records} >= {
        "decision", "staged", "applied", "rejected", "discarded",
    }
    assert all(record["decision_id"] for record in records)
    assert secret not in json.dumps(records)
    assert sensitive not in json.dumps(records)
    staged_ids = {
        record["decision_id"] for record in records
        if record["event_type"] == "staged" and record["tier"] == "Tier1"
    }
    assert len(staged_ids) == 1


def test_public_replay_emits_replayed_and_failed_lifecycle_events(hermes_home):
    from tools import write_approval as wa

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    decision = wa.classify_memory_write(
        target="memory", content="Owner email is person@example.invalid.",
    )
    audit = wa.audit_memory_decision(
        decision, action="add", target="memory", store="builtin",
        content="Owner email is person@example.invalid.",
    )
    applied = wa.stage_write(
        wa.MEMORY,
        {
            "action": "add", "target": "memory",
            "content": "Owner email is person@example.invalid.",
            "_memory_audit": audit,
        },
        summary="redacted", origin="foreground",
    )
    stale = wa.stage_write(
        wa.MEMORY,
        {
            "action": "replace", "target": "memory", "old_text": "missing",
            "content": "Owner email is other@example.invalid.",
            "_memory_audit": audit,
        },
        summary="redacted", origin="foreground",
    )

    assert wa.replay_pending(wa.MEMORY, applied["id"]) is True
    assert wa.replay_pending(wa.MEMORY, stale["id"]) is False
    records = [
        json.loads(line)
        for line in (hermes_home / "logs" / "memory-write-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "replayed" in {record["event_type"] for record in records}
    assert "failed" in {record["event_type"] for record in records}
    assert wa.get_pending(wa.MEMORY, stale["id"]) is not None


def test_builtin_direct_apply_exception_emits_failed_lifecycle(
    hermes_home, monkeypatch,
):
    from tools.memory_tool import MemoryStore, memory_tool

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    store = MemoryStore()

    def fail_add(_target, _content):
        raise RuntimeError("synthetic direct store failure")

    monkeypatch.setattr(store, "add", fail_add)
    with pytest.raises(RuntimeError, match="synthetic direct store failure"):
        memory_tool(
            action="add", target="memory", content="Parser lives at src/parser.py.",
            store=store,
        )

    records = [
        json.loads(line)
        for line in (hermes_home / "logs" / "memory-write-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event_type"] for record in records] == ["decision", "failed"]
    assert records[-1]["failure_code"] == "direct_apply_exception"


def test_holographic_direct_apply_exception_emits_failed_lifecycle(
    hermes_home, monkeypatch,
):
    pytest.importorskip("numpy")
    from plugins.memory.holographic import HolographicMemoryProvider

    _set_memory_config(write_approval=True, write_approval_tiered=True)
    provider = HolographicMemoryProvider({"db_path": str(hermes_home / "facts.db")})
    provider.initialize("holographic-session")

    def fail_add(*_args, **_kwargs):
        raise RuntimeError("synthetic direct fact failure")

    monkeypatch.setattr(provider._store, "add_fact", fail_add)
    try:
        result = json.loads(provider.handle_tool_call(
            "fact_store", {"action": "add", "content": "Parser lives at src/parser.py."},
        ))
        assert "synthetic direct fact failure" in result["error"]
        records = [
            json.loads(line)
            for line in (hermes_home / "logs" / "memory-write-audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [record["event_type"] for record in records] == ["decision", "failed"]
        assert records[-1]["failure_code"] == "direct_apply_exception"
    finally:
        provider.shutdown()
