"""Peer routing uses current session identity, not stale display metadata.

Unit tests isolate SessionDB lookups to place deterministic changes between
browse and revalidation. Filesystem delivery uses the real mailbox implementation.
The existing test_peer_messaging.py owns full registry/SessionDB integration.
"""
import copy
import json
import time
from types import SimpleNamespace

import pytest

from tools import peer_messaging_tool as pm


@pytest.fixture
def scope_env(tmp_path, monkeypatch):
    now = time.time()
    rows = {
        sid: dict(id=sid, source="cli", profile_name="default", ended_at=None,
                  archived=False, cwd=str(tmp_path / "repo"), started_at=now,
                  parent_session_id=None, model_config={})
        for sid in ("sender", "recipient")
    }
    projected = copy.deepcopy(list(rows.values()))
    chains = {}
    queries = []

    def list_sessions_rich(**kwargs):
        queries.append(kwargs)
        # This seam models supported SQL filters; the integration fixture tests
        # the real store. Without sources=, managed rows can consume the page.
        candidates = [row for row in projected if not kwargs.get("sources")
                      or row["source"] in kwargs["sources"]]
        return copy.deepcopy(candidates[:kwargs["limit"]])

    db = SimpleNamespace(
        get_session=lambda sid: copy.deepcopy(rows.get(sid)),
        get_compression_tip=lambda sid: chains.get(sid, [sid])[-1],
        get_compression_chain=lambda sid: list(chains.get(sid, [sid])),
        list_sessions_rich=list_sessions_rich, close=lambda: None,
    )
    monkeypatch.setattr(pm, "_open_db", lambda: db)
    monkeypatch.setattr(pm, "_inbox_root", lambda: tmp_path / "runtime" / "peer_messages")
    return SimpleNamespace(rows=rows, projected=projected, chains=chains, queries=queries, db=db)


@pytest.mark.parametrize("field,value", [
    ("profile_name", "another-profile"), ("cwd", "/another/project"),
    ("git_repo_root", "/another/repo"),
])
def test_discovery_cannot_restore_old_routing_metadata(scope_env, field, value):
    scope_env.rows["recipient"][field] = value
    result = json.loads(pm.peer_sessions(current_session_id="sender"))
    assert result["success"] is True
    assert result["peers"] == []


def test_discovery_keeps_fresh_labels_and_transcript_activity(scope_env):
    fresh = scope_env.rows["recipient"]
    fresh.update(title="new title", last_activity_description="editing UI", started_at=1)
    for snapshot in scope_env.projected:
        if snapshot["id"] == "recipient":
            snapshot.update(title="old title", last_activity_description="editing schema",
                            started_at=1, last_active=time.time())
    peers = json.loads(pm.peer_sessions(current_session_id="sender"))["peers"]
    assert len(peers) == 1
    assert peers[0]["title"] == fresh["title"]
    assert peers[0]["last_activity_description"] == fresh["last_activity_description"]
    assert peers[0]["last_active"] > 1


def test_discovery_does_not_regress_a_fresh_heartbeat(scope_env):
    for row in scope_env.projected:
        if row["id"] == "recipient":
            row.update(started_at=1, last_active=1)
    scope_env.rows["recipient"]["last_activity_at"] = time.time()
    assert len(json.loads(pm.peer_sessions(current_session_id="sender"))["peers"]) == 1


def test_managed_history_cannot_exhaust_local_discovery_page(scope_env):
    scope_env.projected[:0] = [dict(id=f"managed-{n}", source="telegram") for n in range(100)]
    result = json.loads(pm.peer_sessions(current_session_id="sender"))
    assert [p["session_id"] for p in result["peers"]] == ["recipient"]
    assert scope_env.queries[-1]["compact_rows"] is True


def _rotate(env):
    env.rows["recipient"].update(ended_at=time.time(), end_reason="compression")
    env.rows["tip"] = {**env.rows["recipient"], "id": "tip", "parent_session_id": "recipient",
                       "ended_at": None, "end_reason": None}
    env.chains["recipient"] = ["recipient", "tip"]


@pytest.mark.parametrize("changes", [
    {"archived": True}, {"source": "telegram"}, {"profile_name": "foreign"},
    {"model_config": {"_delegate_from": "worker-owner"}},
])
def test_old_alias_cannot_bypass_its_own_boundary(scope_env, changes):
    _rotate(scope_env)
    scope_env.rows["recipient"].update(changes)
    result = json.loads(pm.peer_send("recipient", "must not be routed", from_session_id="sender"))
    assert "error" in result
    assert not pm._inbox_dir("tip").exists()


@pytest.mark.parametrize("changes", [
    {"archived": True}, {"source": "telegram"}, {"profile_name": "foreign"},
])
def test_receive_and_ack_cannot_cross_disallowed_ancestor(scope_env, changes):
    entry = pm.mailbox.enqueue(pm._inbox_dir("recipient"), sender="sender", target="recipient", text="old mail")
    _rotate(scope_env)
    scope_env.rows["recipient"].update(changes)
    result = json.loads(pm.peer_receive(current_session_id="tip"))
    assert result["messages"] == []
    receipt = json.loads(pm.peer_receive(action="ack", message_ids=[entry["message_id"]], current_session_id="tip"))
    assert receipt["acknowledged"] == []
    assert (pm._inbox_dir("recipient") / f"{entry['message_id']}.json").exists()


def test_compression_still_delivers_and_acknowledges_valid_ancestors(scope_env):
    entry = pm.mailbox.enqueue(pm._inbox_dir("recipient"), sender="sender", target="recipient", text="old mail")
    _rotate(scope_env)
    assert json.loads(pm.peer_send("recipient", "new mail", from_session_id="sender"))["target_session_id"] == "tip"
    messages = json.loads(pm.peer_receive(current_session_id="tip"))["messages"]
    assert {e["message"] for e in messages} == {"old mail", "new mail"}
    receipt = json.loads(pm.peer_receive(action="ack", message_ids=[entry["message_id"]], current_session_id="tip"))
    assert receipt["acknowledged"] == [entry["message_id"]]


def test_intermediate_archived_segment_cannot_be_skipped(scope_env):
    _rotate(scope_env)
    scope_env.rows["middle"] = {**scope_env.rows["recipient"], "id": "middle",
                                "parent_session_id": "recipient", "archived": True}
    scope_env.rows["tip"]["parent_session_id"] = "middle"
    scope_env.chains["recipient"] = ["recipient", "middle", "tip"]
    scope_env.chains["middle"] = ["middle", "tip"]
    result = json.loads(pm.peer_send("recipient", "must not skip archive", from_session_id="sender"))
    assert "error" in result


def test_foreign_profile_in_middle_is_not_hidden_by_matching_endpoints(scope_env):
    _rotate(scope_env)
    scope_env.rows["middle"] = {**scope_env.rows["recipient"], "id": "middle",
                                "parent_session_id": "recipient", "profile_name": "foreign"}
    scope_env.rows["tip"]["parent_session_id"] = "middle"
    scope_env.chains["recipient"] = ["recipient", "middle", "tip"]
    result = json.loads(pm.peer_send("recipient", "not a local route", from_session_id="sender"))
    assert "error" in result


@pytest.mark.parametrize("marker", ["_branched_from", "_reset_from"])
def test_changed_fork_marker_invalidates_a_selected_route(scope_env, marker):
    _rotate(scope_env)
    # Store selection happened before this edge became an independent branch.
    scope_env.rows["tip"]["model_config"] = {marker: "recipient"}
    result = json.loads(pm.peer_send("recipient", "not for the new branch", from_session_id="sender"))
    assert "error" in result
    assert not pm._inbox_dir("tip").exists()
