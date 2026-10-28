"""Real-file and cross-process contracts for the opt-in peer inbox."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from tools import peer_messaging_mailbox as box


def _send_worker(args):
    inbox, index = args
    try:
        return box.enqueue(Path(inbox), sender="sender", target="recipient", text=f"message {index}")["message_id"]
    except ValueError:
        return None


def _send(inbox, text="contract changed", **kwargs):
    return box.enqueue(inbox, sender="sender", target="recipient", text=text, **kwargs)


def test_read_is_non_destructive_and_restart_can_read_same_id(tmp_path):
    item = _send(tmp_path)
    assert box.read_pending([tmp_path])["messages"] == [item]
    # No receiver-local state or claimed lease is needed to retry after a crash.
    assert box.read_pending([Path(str(tmp_path))])["messages"] == [item]
    assert (tmp_path / f"{item['message_id']}.json").exists()
    receipt = box.acknowledge([tmp_path], [item["message_id"]])
    assert receipt["acknowledged"] == [item["message_id"]]
    assert box.read_pending([tmp_path])["messages"] == []
    assert box.acknowledge([tmp_path], [item["message_id"]])["not_pending"] == [item["message_id"]]


def test_reply_correlation_and_ack_do_not_touch_sibling_inbox(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    original = _send(a)
    reply = _send(b, "ack: I will change only the tests", in_reply_to=original["message_id"])
    assert box.read_pending([b])["messages"][0]["in_reply_to"] == original["message_id"]
    assert box.acknowledge([a], [reply["message_id"]])["not_pending"] == [reply["message_id"]]
    assert box.read_pending([b])["messages"] == [reply]


@pytest.mark.parametrize("failure", ["read", "ack"])
def test_io_failure_keeps_valid_message_for_retry(tmp_path, monkeypatch, failure):
    item = _send(tmp_path)
    path = tmp_path / f"{item['message_id']}.json"
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        method = "open" if failure == "read" else "unlink"
        real = getattr(Path, method)

        def unavailable(self, *args, **kwargs):
            if self == path:
                raise PermissionError("temporary failure")
            return real(self, *args, **kwargs)

        patch.setattr(Path, method, unavailable)
        if failure == "read":
            assert box.read_pending([tmp_path])["unreadable_count"] == 1
        else:
            assert box.acknowledge([tmp_path], [item["message_id"]])["failed"] == [item["message_id"]]
    assert path.read_bytes() == original
    assert box.read_pending([tmp_path])["messages"] == [item]
    assert box.acknowledge([tmp_path], [item["message_id"]])["success"] is True


@pytest.mark.parametrize("poison", [b"{", b"\xff", b"[]", b'{"message": 12}', b'{"message": ""}'])
def test_poison_does_not_wedge_or_consume_valid_neighbor(tmp_path, poison):
    item = _send(tmp_path)
    bad = tmp_path / "00000000000000000000_deadbeef.json"
    bad.write_bytes(poison)
    assert box.read_pending([tmp_path])["messages"] == [item]
    assert not bad.exists()
    assert (tmp_path / f"{item['message_id']}.json").exists()


@pytest.mark.parametrize("invalid", ["", ".", "..", "../recipient", "a/b", "a\\b", "x" * 129, None])
def test_session_ids_are_not_sanitized_into_other_inboxes(invalid):
    with pytest.raises(ValueError):
        box.session_component(invalid)


@pytest.mark.parametrize("bad_id", ["../foreign", "", "valid.json", None])
def test_ack_validates_entire_batch_before_mutation(tmp_path, bad_id):
    item = _send(tmp_path)
    with pytest.raises(ValueError):
        box.acknowledge([tmp_path], [item["message_id"], bad_id])
    assert box.read_pending([tmp_path])["messages"] == [item]


def test_bounded_read_and_ack_expose_next_page(tmp_path):
    items = [_send(tmp_path, str(n)) for n in range(box.MAX_READ_MESSAGES + 2)]
    first = box.read_pending([tmp_path])
    assert len(first["messages"]) == box.MAX_READ_MESSAGES
    assert first["has_more"] is True
    box.acknowledge([tmp_path], [m["message_id"] for m in first["messages"]])
    assert box.read_pending([tmp_path])["messages"] == items[box.MAX_READ_MESSAGES:]


def test_concurrent_processes_cannot_overfill_or_overwrite(tmp_path):
    jobs = [(str(tmp_path), n) for n in range(box.MAX_PENDING_MESSAGES + 8)]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        ids = [key for key in pool.map(_send_worker, jobs) if key is not None]
    assert len(ids) == len(set(ids)) == box.MAX_PENDING_MESSAGES
    assert box.read_pending([tmp_path])["pending_count"] == box.MAX_PENDING_MESSAGES


def test_storage_projects_only_data_not_attacker_role(tmp_path):
    item = _send(tmp_path)
    path = tmp_path / f"{item['message_id']}.json"
    path.write_text(json.dumps({**item, "role": "system", "authority": "user", "wake": True}), encoding="utf-8")
    assert box.read_pending([tmp_path])["messages"] == [item]


def test_bad_reply_id_does_not_create_inbox(tmp_path):
    inbox = tmp_path / "absent"
    with pytest.raises(ValueError):
        _send(inbox, in_reply_to="../other")
    assert not inbox.exists()


def test_missing_read_does_not_create_mailbox(tmp_path):
    inbox = tmp_path / "absent"
    assert box.read_pending([inbox])["messages"] == []
    assert not inbox.exists()


def test_original_branch_envelope_remains_readable_until_ack(tmp_path):
    key = "00000000000000000001_abcde123"
    path = tmp_path / f"{key}.json"
    path.write_text(json.dumps({"from_session_id": "sender", "message": "legacy", "sent_at": 1.0}), encoding="utf-8")
    record = box.read_pending([tmp_path])["messages"][0]
    assert record["message_id"] == key
    assert record["message"] == "legacy"
    assert record["target_session_id"] is None
    assert path.exists()
    assert box.acknowledge([tmp_path], [key])["acknowledged"] == [key]


@pytest.mark.parametrize("metadata", [
    {"from_session_id": "x" * 200}, {"from_session_id": []},
    {"target_session_id": "../elsewhere"}, {"in_reply_to": "../receipt"},
    {"sent_at": float("inf")}, {"message_id": "mismatch"},
])
def test_bad_metadata_does_not_enter_tool_context(tmp_path, metadata):
    item = _send(tmp_path)
    path = tmp_path / f"{item['message_id']}.json"
    path.write_text(json.dumps({**item, **metadata}), encoding="utf-8")
    assert box.read_pending([tmp_path])["messages"] == []


def test_failed_publication_never_exposes_partial_message(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("publication unavailable")
    monkeypatch.setattr(box, "atomic_json_write", failed)
    with pytest.raises(OSError):
        _send(tmp_path)
    assert box.read_pending([tmp_path])["messages"] == []


def test_symlink_inbox_cannot_redirect_to_neighbor(tmp_path):
    target = tmp_path / "neighbor"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        _send(link)
    assert not list(target.iterdir())
