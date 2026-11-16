"""Unit tests for slack_post_thread. _post_message and the reused token resolver
are mocked; these pin required-field validation, payload plumbing, error shaping,
and registration under the non-floor slack_post toolset."""
import json
import pytest

import tools.slack_post_thread_tool as spt


@pytest.fixture(autouse=True)
def _fake_post(monkeypatch):
    calls = []

    async def fake_post(token, channel, thread_ts, text):
        calls.append({"token": token, "channel": channel,
                      "thread_ts": thread_ts, "text": text})
        return {"ok": True, "ts": "1700000000.000999"}

    async def fake_permalink(token, channel, message_ts):
        return f"https://x.slack.com/archives/{channel}/p{message_ts.replace('.', '')}"

    monkeypatch.setattr(spt, "_post_message", fake_post)
    monkeypatch.setattr(spt, "_get_permalink", fake_permalink)
    monkeypatch.setattr(spt, "_resolve_slack_token", lambda: "xoxb-test")
    return calls


def _run(args):
    from model_tools import _run_async
    return json.loads(_run_async(spt._slack_post_thread_handler(args)))


def test_posts_to_explicit_thread(_fake_post):
    out = _run({"chat_id": "C1", "thread_ts": "111.222", "message": "hi"})
    assert out["success"] is True
    assert _fake_post[0] == {"token": "xoxb-test", "channel": "C1",
                             "thread_ts": "111.222", "text": "hi"}


def test_thread_reply_returns_receipt(_fake_post):
    out = _run({"chat_id": "C1", "thread_ts": "111.222", "message": "hi"})
    assert out["message_ts"] == "1700000000.000999"
    assert out["thread_ts"] == "111.222"
    assert out["permalink"].endswith("/C1/p1700000000000999")


def test_root_post_when_thread_ts_omitted(_fake_post):
    out = _run({"chat_id": "C1", "message": "hi"})
    assert out["success"] is True
    assert _fake_post[0]["thread_ts"] == ""
    # A root post starts its own thread: later replies thread under message_ts.
    assert out["thread_ts"] == out["message_ts"] == "1700000000.000999"
    assert out["permalink"]


def test_requires_chat_id_and_message(_fake_post):
    assert "error" in _run({"chat_id": "C1", "thread_ts": "1.2"})   # no message
    assert "error" in _run({"thread_ts": "1.2", "message": "x"})    # no chat_id
    assert not _fake_post


def test_error_when_no_token(monkeypatch, _fake_post):
    monkeypatch.setattr(spt, "_resolve_slack_token", lambda: "")
    out = _run({"chat_id": "C1", "thread_ts": "1.2", "message": "x"})
    assert "error" in out


def test_slack_api_error_surfaces(monkeypatch):
    async def fake(token, channel, thread_ts, text):
        return {"ok": False, "error": "channel_not_found"}
    monkeypatch.setattr(spt, "_post_message", fake)
    out = _run({"chat_id": "C1", "thread_ts": "1.2", "message": "x"})
    assert "error" in out and "channel_not_found" in out["error"]


def test_registered_in_slack_post_toolset():
    from tools.registry import registry
    import tools.slack_post_thread_tool  # noqa: F401
    assert registry.get_toolset_for_tool("slack_post_thread") == "slack_post"
    assert registry.get_entry("slack_post_thread").is_async is True


def test_slack_post_toolset_declared():
    import toolsets
    assert "slack_post" in toolsets.TOOLSETS
    assert toolsets.TOOLSETS["slack_post"]["tools"] == ["slack_post_thread"]
