"""Account ownership survives every real transport stage, including recovery."""

import json

import pytest

from tools import codex_web_audio as audio


@pytest.mark.parametrize("account_id", [None, "workspace-a"])
@pytest.mark.parametrize("poll", [False, True])
@pytest.mark.parametrize("outcome", ["audio", "empty", "unauthorized_poll"])
def test_subscription_requests_retain_account_identity(monkeypatch, account_id, poll, outcome):
    requests = []
    expected = "Speak this exact sentence."
    message = {"id": "assistant", "author": {"role": "assistant"},
               "content": {"content_type": "text", "parts": [expected]}}

    class Response:
        status_code = 200

        def __init__(self, body, content_type="application/json"):
            self.body = body
            self.headers = {"content-type": content_type}

        def close(self):
            pass

        def iter_content(self, _size):
            yield self.body

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get(self, *_args, **_kwargs):
            return Response(b"")

        def request(self, _method, url, **kwargs):
            path = url.removeprefix(audio._BASE_URL)
            requests.append((path, kwargs["headers"]))
            if path == audio._CONVERSATION:
                event = {"conversation_id": "conversation"} if poll else {
                    "v": {"conversation_id": "conversation", "message": message}}
                return Response(("data: " + json.dumps(event)).encode(), "text/event-stream")
            if path == audio._SYNTHESIZE:
                return Response(b"" if outcome == "empty" else b"ID3speech", "audio/mpeg")
            if path == "/backend-api/conversation/conversation" and outcome == "unauthorized_poll":
                response = Response(b'{}')
                response.status_code = 401
                return response
            bodies = {
                audio._CONVERSATION_PREPARE: {"conduit_token": "conduit"},
                audio._SENTINEL_PREPARE: {"prepare_token": "prepare"},
                audio._SENTINEL_FINALIZE: {"token": "requirements"},
                "/backend-api/conversation/conversation": {
                    "current_node": "assistant", "mapping": {"assistant": {"message": message}}},
            }
            return Response(json.dumps(bodies[path]).encode())

    monkeypatch.setattr(audio.requests, "Session", Session)
    if poll and outcome == "unauthorized_poll":
        with pytest.raises(RuntimeError, match="HTTP 401"):
            audio.synthesize_codex_speech(expected, "token", account_id=account_id)
        assert requests[-1][0] == "/backend-api/conversation/conversation"
        assert not any(path == audio._SYNTHESIZE for path, _ in requests)
    elif outcome == "empty":
        with pytest.raises(RuntimeError, match="empty audio"):
            audio.synthesize_codex_speech(expected, "token", account_id=account_id)
    else:
        speech = audio.synthesize_codex_speech(expected, "token", account_id=account_id)
        assert speech.audio == b"ID3speech"
        assert requests[-1][0] == audio._SYNTHESIZE
    assert len([path for path, _ in requests if path == audio._CONVERSATION_PREPARE]) == 5
    assert any(path == "/backend-api/conversation/conversation" for path, _ in requests) is poll
    for _path, headers in requests:
        assert headers["authorization"] == "Bearer token"
        assert headers.get("ChatGPT-Account-ID") == account_id
