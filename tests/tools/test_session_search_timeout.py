import asyncio
import json

from tools.session_search_tool import session_search


class DummyDB:
    def search_messages(self, query, role_filter=None, exclude_sources=None, limit=50, offset=0):
        return [{"session_id": "s1", "source": "telegram", "session_started": 1710000000, "model": "gpt-5.4"}]

    def get_session(self, session_id):
        return {"session_id": session_id, "source": "telegram", "started_at": 1710000000}

    def get_messages_as_conversation(self, session_id):
        return [{"role": "user", "content": "drift fixes github repo"}]


def test_session_search_returns_timeout_error_on_async_timeout(monkeypatch):
    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError()

    def fake_run_async(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr("tools.session_search_tool.asyncio.wait_for", fake_wait_for)
    monkeypatch.setattr("model_tools._run_async", fake_run_async)

    result = json.loads(session_search(query="drift OR fixes", limit=1, db=DummyDB(), current_session_id="current"))

    assert result["success"] is False
    assert "timed out" in result["error"].lower()
