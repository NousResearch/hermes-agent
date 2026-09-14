"""Real private-byte custody and owner-session RPC tests, isolated from live state."""

import base64
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from gateway.classic_output_exports import ClassicExports, PENDING_TTL
from gateway.hosted_room_artifacts import RoomArtifactError, RoomArtifactOutbox
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import hosted_room_artifact  # register the real handler
from tools.registry import registry
from tui_gateway import classic_exports, server
from tui_gateway.transport import bind_transport, reset_transport


REQUEST = {"request_id": "request-1", "group_id": "group-1", "thread_id": "thread-1", "issued_at": time.time(),
           "recipients": [{"installation": "other", "profile": "reviewer"}]}


@pytest.fixture
def custody(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    token = set_hermes_home_override(home)
    try:
        yield ClassicExports(home)
    finally:
        reset_hermes_home_override(token)


def share(store, row, path):
    session = {"_classic_export_admission": classic_exports.Admission(store, row)}
    token = server._current_runtime_session_record.set(session)
    classic_token = classic_exports.bind(session, session["_classic_export_admission"])
    try:
        return json.loads(registry.dispatch("share_group_file", {"path": str(path)}))
    finally:
        classic_exports.reset(classic_token)
        server._current_runtime_session_record.reset(token)


def publish(store, data=b"# welcome\n\x00\xff"):
    path = store.outbox.db_path.parent / "welcome.bin"
    path.write_bytes(data)
    row, fresh = store.admit("writer-session", REQUEST, "create and share")
    assert fresh
    result = share(store, row, path)
    assert result["ok"], result
    store.settle(row["export_id"], "Shared welcome.bin", True)
    return row, result, data


def test_explicit_share_retains_exact_bytes_across_restart_without_ack_or_ttl(custody):
    row, result, data = publish(custody)
    with custody.outbox._connect() as conn:
        conn.execute("UPDATE classic_output_exports SET expires=0")
        conn.execute("UPDATE hosted_room_output_artifacts SET created_at=0")
    reopened = ClassicExports(custody.home)
    metadata, copied = reopened.read(row["export_id"], result["artifact_id"])
    assert copied == data
    assert metadata["sha256"] == hashlib.sha256(data).hexdigest()
    assert "path" not in metadata
    assert reopened.status(row["export_id"])["state"] == "published"


def test_pending_output_private_then_retired_and_replay_cannot_reopen(custody):
    row, _ = custody.admit("writer-session", REQUEST, "share")
    path = custody.outbox.db_path.parent / "private.md"
    path.write_text("draft")
    result = share(custody, row, path)
    assert result["ok"]
    with pytest.raises(RoomArtifactError, match="not published"):
        custody.read(row["export_id"], result["artifact_id"])
    custody.retire(row["export_id"])
    assert share(custody, row, path)["ok"] is False
    replay, fresh = custody.admit("writer-session", REQUEST, "share")
    assert not fresh and replay["state"] == "retired"


def test_duplicate_admission_is_idempotent_but_changed_input_is_refused(custody):
    first, _ = custody.admit("writer-session", REQUEST, "share")
    again, fresh = custody.admit("writer-session", REQUEST, "share")
    assert not fresh and first == again
    with pytest.raises(RoomArtifactError, match="replay changed"):
        custody.admit("writer-session", REQUEST, "different")
    with pytest.raises(RoomArtifactError, match="replay changed"):
        custody.admit("writer-session", {**REQUEST, "recipients": [{"installation": "other", "profile": "new"}]}, "share")


def test_no_admission_or_forged_json_cannot_share(custody):
    path = custody.outbox.db_path.parent / "private.md"
    path.write_text("private")
    for context in ({}, {"_classic_export_admission": {"export_id": "forged"}}):
        token = server._current_runtime_session_record.set(context)
        try:
            assert json.loads(registry.dispatch("share_group_file", {"path": str(path)}))["ok"] is False
        finally:
            server._current_runtime_session_record.reset(token)


@pytest.mark.parametrize("name", [".env", "state.db", ".ssh/key"])
def test_private_paths_refused(custody, name):
    row, _ = custody.admit("writer-session", REQUEST, "share")
    path = custody.outbox.db_path.parent / name
    path.parent.mkdir(exist_ok=True)
    if name != "state.db":
        path.write_text("secret")
    assert share(custody, row, path)["ok"] is False


def test_published_quota_refuses_new_output_without_evicting(custody, monkeypatch):
    row, result, data = publish(custody)
    monkeypatch.setattr("gateway.classic_output_exports.MAX_EXPORTS", 1)
    with pytest.raises(RoomArtifactError, match="quota"):
        custody.admit("writer-session", {**REQUEST, "request_id": "second"}, "share")
    assert custody.read(row["export_id"], result["artifact_id"])[1] == data


def test_orphan_pending_cleanup_never_retires_other_profile(custody, tmp_path):
    row, _ = custody.admit("writer-session", REQUEST, "share")
    with custody.outbox._connect() as conn:
        conn.execute("UPDATE classic_output_exports SET expires=0")
    custody.prune()
    with pytest.raises(RoomArtifactError, match="not found"):
        custody.lookup(row["export_id"])


def test_gateway_cleanup_reclaims_expired_admission_from_inactive_profile(custody):
    other = ClassicExports(custody.outbox.db_path.parent / 'profiles' / 'inactive')
    row, _ = other.admit('inactive-session', REQUEST, 'share')
    with custody.outbox._connect() as conn:
        conn.execute('UPDATE classic_output_exports SET expires=0 WHERE export_id=?', (row['export_id'],))
    custody.prune()
    with pytest.raises(RoomArtifactError, match='not found'):
        other.lookup(row['export_id'])


def test_profile_and_group_retirement_are_exact(custody, tmp_path):
    row, result, data = publish(custody)
    other = ClassicExports(custody.outbox.db_path.parent / "profiles" / "other")
    with pytest.raises(RoomArtifactError, match="profile"):
        other.read(row["export_id"], result["artifact_id"])
    custody.retire_group("unrelated")
    assert custody.read(row["export_id"], result["artifact_id"])[1] == data
    custody.retire_group(REQUEST["group_id"])
    with pytest.raises(RoomArtifactError, match="not published"):
        custody.read(row["export_id"], result["artifact_id"])


def test_rpc_current_owned_session_can_read_after_transport_rotation(custody, monkeypatch):
    row, result, data = publish(custody)
    old, current = object(), object()
    session = {"transport": current, "profile_home": custody.home, "session_key": "new-reader-session"}
    monkeypatch.setitem(server._sessions, "reader", session)
    monkeypatch.setattr(classic_exports, "local_authority_gateway_id", lambda: "installation")
    params = {"session_id": "reader", "installation": "installation", "group_id": REQUEST["group_id"],
              "export_id": row["export_id"], "artifact_id": result["artifact_id"]}
    for caller, allowed in ((old, False), (current, True)):
        token = bind_transport(caller)
        try:
            response = server._methods["session.export.read"](1, params)
        finally:
            reset_transport(token)
        if allowed:
            assert base64.b64decode(response["result"]["content_base64"]) == data
        else:
            assert response["error"]["code"] == 4150


def test_schema_stable_across_turns_and_ordinary_chat_unchanged(monkeypatch):
    agent = SimpleNamespace(tools=[], valid_tool_names=set(), platform="desktop")
    session = {"agent": agent, "room_plumbing": True}
    classic_exports.install_schema(session)
    first = json.dumps(agent.tools)
    session.pop("room_plumbing")
    classic_exports.install_schema(session)
    assert json.dumps(agent.tools) == first
    assert [entry["function"]["name"] for entry in agent.tools] == ["share_group_file"]
    other = SimpleNamespace(tools=[], valid_tool_names=set(), platform="desktop")
    monkeypatch.setattr(classic_exports, "plumbing", lambda _: False)
    classic_exports.install_schema({"agent": other})
    assert other.tools == []


def test_ordinary_turn_cannot_inherit_stale_admission(custody):
    row, _ = custody.admit('writer', REQUEST, 'share')
    session = {'_classic_export_admission': classic_exports.Admission(custody, row)}
    token = classic_exports.bind(session, None)
    try:
        assert classic_exports.active_scope() is None
        classic_exports.settle(session, 'ordinary text', True)
        assert custody.status(row['export_id'])['state'] == 'running'
    finally:
        classic_exports.reset(token)


def test_delayed_tool_context_cannot_follow_a_new_generation(custody):
    first, _ = custody.admit('writer', REQUEST, 'share')
    session = {'_classic_export_admission': classic_exports.Admission(custody, first)}
    token = classic_exports.bind(session, session['_classic_export_admission'])
    try:
        second, _ = custody.admit('writer', {**REQUEST, 'request_id': 'next'}, 'share next')
        session['_classic_export_admission'] = classic_exports.Admission(custody, second)
        assert classic_exports.active_scope().export_id == first['export_id']
        session['_turn_cancel_requested'] = True
        assert classic_exports.active_scope() is None
    finally:
        classic_exports.reset(token)


def test_disband_tombstone_refuses_late_new_admissions(custody):
    custody.retire_group(REQUEST['group_id'])
    with pytest.raises(RoomArtifactError, match='retired'):
        custody.admit('writer', REQUEST, 'share')
    custody.retire_group(REQUEST['group_id'])


def test_expired_unknown_request_never_reexecutes(custody):
    with pytest.raises(RoomArtifactError, match='expired'):
        custody.admit('writer', {**REQUEST, 'issued_at': time.time() - PENDING_TTL - 1}, 'share')


def test_published_terminal_replay_cannot_retract_or_change_output(custody):
    row, result, data = publish(custody)
    custody.settle(row['export_id'], 'Shared welcome.bin', True)
    custody.settle(row['export_id'], '', False)
    with pytest.raises(RoomArtifactError, match='changed'):
        custody.settle(row['export_id'], 'changed', True)
    assert custody.read(row['export_id'], result['artifact_id'])[1] == data


def test_shared_byte_quota_and_symlink_refusal(custody, monkeypatch):
    row, _ = custody.admit('writer', REQUEST, 'share')
    path = custody.outbox.db_path.parent / 'file.txt'
    path.write_text('not secret')
    link = path.with_name('linked.txt')
    link.symlink_to(path)
    assert share(custody, row, link)['ok'] is False
    monkeypatch.setattr('gateway.hosted_room_artifacts.MAX_GATEWAY_BLOB_BYTES', 1)
    assert share(custody, row, path)['ok'] is False


def test_runtime_scope_cannot_borrow_another_profile_home(custody, tmp_path):
    row, _ = custody.admit('writer', REQUEST, 'share')
    session = {'_classic_export_admission': classic_exports.Admission(custody, row)}
    token = classic_exports.bind(session, session['_classic_export_admission'])
    home_token = set_hermes_home_override(tmp_path / 'other-profile')
    try:
        assert classic_exports.active_scope() is None
    finally:
        reset_hermes_home_override(home_token)
        classic_exports.reset(token)


def test_group_retirement_fences_every_export_before_cleanup_and_recovers(custody, monkeypatch):
    exports = []
    for index in range(3):
        row, _ = custody.admit('writer', {**REQUEST, 'request_id': f'retire-{index}'}, 'share')
        path = custody.outbox.db_path.parent / f'output-{index}.txt'
        path.write_text(f'file {index}')
        result = share(custody, row, path)
        assert result['ok']
        if index < 2:
            custody.settle(row['export_id'], 'shared', True)
        exports.append((row, result, path))
    blobs = list(custody.outbox.blob_root.glob('blob_*'))

    def fail_cleanup(_scope):
        raise OSError('forced cleanup failure after durable retirement')

    monkeypatch.setattr(custody.outbox, 'discard', fail_cleanup)
    with pytest.raises(OSError, match='forced cleanup'):
        custody.retire_group(REQUEST['group_id'])
    assert all(path.exists() for path in blobs)
    for row, result, path in exports:
        assert custody.lookup(row['export_id'])['state'] == 'retired'
        with pytest.raises(RoomArtifactError, match='not published'):
            custody.read(row['export_id'], result['artifact_id'])
        assert share(custody, row, path)['ok'] is False
    with pytest.raises(RoomArtifactError, match='active admitted'):
        custody.settle(exports[-1][0]['export_id'], 'late completion', True)

    recovered = ClassicExports(custody.home)
    assert not any(path.exists() for path in blobs)
    for row, result, path in exports:
        with pytest.raises(RoomArtifactError, match='not published'):
            recovered.read(row['export_id'], result['artifact_id'])
        assert share(recovered, row, path)['ok'] is False
    recovered.retire_group(REQUEST['group_id'])
    recovered.retire_group(REQUEST['group_id'])


def test_identical_tool_share_replays_at_full_count_without_allocating(custody, monkeypatch):
    monkeypatch.setattr('gateway.classic_output_exports.MAX_FILES', 1)
    row, _ = custody.admit('writer', REQUEST, 'share')
    path = custody.outbox.db_path.parent / 'same.txt'
    path.write_text('same bytes')
    first = share(custody, row, path)
    assert first['ok']
    blobs = set(custody.outbox.blob_root.iterdir())
    second = share(custody, row, path)
    assert second['ok'] and second['artifact_id'] == first['artifact_id']
    assert set(custody.outbox.blob_root.iterdir()) == blobs
    other = path.with_name('other.txt')
    other.write_text('new bytes')
    assert share(custody, row, other)['ok'] is False
    custody.settle(row['export_id'], 'shared', True)
    assert custody.read(row['export_id'], first['artifact_id'])[1] == b'same bytes'
