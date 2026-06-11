"""Cloud-channel pusher (channels slice 4.0, hermes side).

Covers the pure row→batch mapping and the pusher's watermark/dedupe contract
against a real sqlite file — no network (the cloud call is monkeypatched).
"""

import sqlite3

import pytest

from tui_gateway import cloud_channels
from tui_gateway.cloud_channels import CloudChannelPusher, rows_to_batch


def test_rows_to_batch_maps_local_columns_to_the_wire_shape():
    rows = [{
        "id": 41, "role": "user", "content": "hello", "sender_device": "Omar iPhone",
        "tool_name": None, "tool_calls": None, "finish_reason": None,
        "token_count": 12, "timestamp": 1760000000.5,
    }]
    batch = rows_to_batch(rows, "ko-mac")
    assert batch == [{
        "origin_message_id": "41", "origin_device_id": "ko-mac",
        "role": "user", "content": "hello", "sender_device": "Omar iPhone",
        "tool_name": None, "tool_calls": None, "finish_reason": None,
        "token_count": 12, "origin_ts": 1760000000.5,
    }]


def test_rows_to_batch_stamps_this_device_when_a_user_row_has_no_sender():
    batch = rows_to_batch([{"id": 1, "role": "user", "content": "hi"}], "ko-mac")
    assert batch[0]["sender_device"] == "ko-mac"
    # Assistant rows stay unattributed — the agent isn't a "device".
    batch = rows_to_batch([{"id": 2, "role": "assistant", "content": "yo"}], "ko-mac")
    assert batch[0]["sender_device"] is None


def test_rows_to_batch_skips_roles_the_cloud_rejects():
    rows = [{"id": 1, "role": "developer", "content": "x"}, {"id": 2, "role": "user", "content": "ok"}]
    batch = rows_to_batch(rows, "ko-mac")
    assert [m["origin_message_id"] for m in batch] == ["2"]


def test_invite_member_posts_to_channel_invites(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"accept_token": "tok_1", "email": body["email"], "permission": body["permission"]}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.invite_member("chan/1", "ada@example.com", "admin")

    assert result["accept_token"] == "tok_1"
    assert sent == [(
        "POST",
        "/v1/channels/chan%2F1/invites",
        {"email": "ada@example.com", "permission": "admin"},
    )]


def test_invite_member_falls_back_to_read_for_unknown_permissions(monkeypatch):
    bodies = []

    def fake_request(method, path, body=None, timeout=15.0):
        bodies.append(body)
        return {}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    cloud_channels.invite_member("chan_1", "ada@example.com", "owner")

    assert bodies == [{"email": "ada@example.com", "permission": "read"}]


def test_list_members_gets_channel_members(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"members": [{"account_id": "acct_1", "permission": "read"}], "count": 1}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.list_members("chan/1")

    assert result["count"] == 1
    assert sent == [("GET", "/v1/channels/chan%2F1/members", None)]


def test_set_member_permission_patches_member(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"account_id": "acct/1", "permission": body["permission"]}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.set_member_permission("chan/1", "acct/1", "post")

    assert result["permission"] == "post"
    assert sent == [("PATCH", "/v1/channels/chan%2F1/members/acct%2F1", {"permission": "post"})]


def test_set_member_permission_falls_back_to_read(monkeypatch):
    bodies = []

    def fake_request(method, path, body=None, timeout=15.0):
        bodies.append(body)
        return {}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    cloud_channels.set_member_permission("chan_1", "acct_1", "owner")

    assert bodies == [{"permission": "read"}]


def test_remove_member_deletes_member(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"ok": True, "removed": "acct/1"}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.remove_member("chan/1", "acct/1")

    assert result["ok"] is True
    assert sent == [("DELETE", "/v1/channels/chan%2F1/members/acct%2F1", None)]


def test_delete_channel_deletes_channel(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"ok": True, "deleted": "chan/1"}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.delete_channel("chan/1")

    assert result["ok"] is True
    assert sent == [("DELETE", "/v1/channels/chan%2F1", None)]


def test_accept_invite_posts_token_to_accept_endpoint(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"ok": True, "channel_id": "chan_1", "permission": "read"}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.accept_invite("tok/1")

    assert result["channel_id"] == "chan_1"
    assert sent == [("POST", "/v1/channels/invites/accept?token=tok%2F1", None)]


def test_list_channels_gets_owned_and_joined_channels(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"channels": [{"id": "chan_1", "your_permission": "admin"}], "count": 1}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.list_channels()

    assert result["count"] == 1
    assert sent == [("GET", "/v1/channels", None)]


def test_list_messages_gets_channel_messages_with_cursor(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"messages": [{"seq": 42, "role": "user", "content": "hello"}], "count": 1}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.list_messages("chan/1", since_seq=41, limit=25)

    assert result["count"] == 1
    assert sent == [("GET", "/v1/channels/chan%2F1/messages?since_seq=41&limit=25", None)]


def test_list_participants_gets_channel_roster(monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"participants": [{"device": "ko-mac", "count": 1}], "host_connected": True}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)

    result = cloud_channels.list_participants("chan/1")

    assert result["host_connected"] is True
    assert sent == [("GET", "/v1/channels/chan%2F1/participants", None)]


class _FakeStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_stream_messages_parses_cloud_sse(monkeypatch):
    paths = []

    def fake_stream(path, timeout=310.0):
        paths.append(path)
        return _FakeStream([
            b": keepalive\n",
            b"\n",
            b"event: message\n",
            b'data: {"seq": 42, "role": "user", "content": "hello"}\n',
            b"\n",
            b"event: error\n",
            b'data: {"message": "boom"}\n',
            b"\n",
        ])

    monkeypatch.setattr(cloud_channels, "_stream_request", fake_stream)

    events = list(cloud_channels.stream_messages("chan/1", since_seq=41))

    assert paths == ["/v1/channels/chan%2F1/stream?since_seq=41"]
    assert events == [
        ("message", {"seq": 42, "role": "user", "content": "hello"}),
        ("error", {"message": "boom"}),
    ]


@pytest.fixture()
def message_db(tmp_path):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE messages (
             id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
             sender_device TEXT, tool_name TEXT, tool_calls TEXT,
             finish_reason TEXT, token_count INTEGER, timestamp REAL)"""
    )
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        [("s1", "user", "first", 1.0), ("s1", "assistant", "second", 2.0), ("OTHER", "user", "foreign", 3.0)],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_push_once_tails_past_the_watermark_and_advances_it(message_db, monkeypatch):
    sent = []

    def fake_request(method, path, body=None, timeout=15.0):
        sent.append((method, path, body))
        return {"accepted": len(body["messages"]), "last_seq": len(body["messages"])}

    monkeypatch.setattr(cloud_channels, "_request", fake_request)
    pusher = CloudChannelPusher(db_path=message_db, session_key="s1", channel_id="c1", device_name="ko-mac")

    assert pusher.push_once() == 2          # only s1's two rows, not the foreign session's
    assert pusher.watermark == 2            # advanced past everything read
    assert sent[0][1] == "/v1/channels/c1/messages"
    assert [m["content"] for m in sent[0][2]["messages"]] == ["first", "second"]

    assert pusher.push_once() == 0          # nothing new -> no network call
    assert len(sent) == 1


def test_push_once_survives_cloud_errors_without_moving_the_watermark(message_db, monkeypatch):
    def boom(method, path, body=None, timeout=15.0):
        raise RuntimeError("cloud POST /x -> 503: unavailable")

    monkeypatch.setattr(cloud_channels, "_request", boom)
    pusher = CloudChannelPusher(db_path=message_db, session_key="s1", channel_id="c1", device_name="ko-mac")
    with pytest.raises(RuntimeError):
        pusher.push_once()
    # Watermark untouched -> the rows are retried next cycle (cloud dedupes).
    assert pusher.watermark == 0


def test_cloud_enabled_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("HERMES_CLOUD_TOKEN", raising=False)
    assert cloud_channels.cloud_enabled() is False
    monkeypatch.setenv("HERMES_CLOUD_TOKEN", "mb_test")
    assert cloud_channels.cloud_enabled() is True
