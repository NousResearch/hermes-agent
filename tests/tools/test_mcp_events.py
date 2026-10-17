"""MCP Events client (webhook delivery): opt-in utility tools, subscribe/unsubscribe against the
dynamic webhook route store, and the refresh loop's re-subscription rules."""

import asyncio
import base64
import json
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from mcp.shared.exceptions import MCPError
from tools import mcp_tool_events as events
from tools.mcp_tool_registration import _select_utility_schemas
from tests.waiting import async_wait_until

PUBLIC_URL = "https://hermes.example.com"
ARGS = {"feed_id": "11111111-1111-4111-8111-111111111111"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"platforms:\n  webhook:\n    enabled: true\n    extra:\n      public_url: {PUBLIC_URL}/\n")
    return tmp_path


def _routes(home):
    path = home / "webhook_subscriptions.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _grant(seconds=86400):
    expires = datetime.now(UTC) + timedelta(seconds=seconds)
    return {"id": "sub_1", "refreshBefore": expires.isoformat().replace("+00:00", "Z"), "cursor": None,
            "truncated": False}


class ObservedLock(asyncio.Lock):
    """Records that someone queued for the session, so a test can act while they wait."""
    contended = False

    async def acquire(self):
        self.contended = self.locked()
        return await super().acquire()


class FakeSession:
    def __init__(self, *replies):
        self.sent, self._replies = [], list(replies)

    async def send_request(self, request, result_type):
        self.sent.append(request.model_dump(by_alias=True, mode="json", exclude_none=True))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _server(session):
    return SimpleNamespace(name="ambush", session=session, _rpc_lock=asyncio.Lock(), _config={"events": True})


def _subscribe(session, **extra):
    return asyncio.run(events._subscribe_event(
        session, {"name": "feed.emission.created", "arguments": ARGS, "deliver": "telegram", **extra}, "ambush"))


class TestUtilitySelection:
    def _keys(self, config):
        server = SimpleNamespace(initialize_result=SimpleNamespace(
            capabilities=SimpleNamespace(tools=SimpleNamespace(), resources=None, prompts=None)))
        return {entry["handler_key"] for entry in _select_utility_schemas("ambush", server, config)}

    def test_event_tools_need_the_opt_in(self):
        assert not self._keys({}) & set(events.EVENT_UTILITY_KEYS)
        assert set(events.EVENT_UTILITY_KEYS) <= self._keys({"events": True})


class TestSubscribe:
    def test_subscribe_registers_a_signed_route_then_the_subscription(self, home):
        session = FakeSession(_grant())
        result = _subscribe(session, instructions="Explain the deal.")

        (name, route), = _routes(home).items()
        sent = session.sent[0]
        assert sent["method"] == "events/subscribe"
        assert sent["params"]["delivery"] == {
            "mode": "webhook", "url": f"{PUBLIC_URL}/webhooks/{name}", "secret": route["secret"]}
        assert sent["params"]["arguments"] == ARGS and "cursor" not in sent["params"]
        assert route["secret"].startswith("whsec_")
        assert len(base64.b64decode(route["secret"].removeprefix("whsec_"))) == 32
        assert route["deliver"] == "telegram" and "Explain the deal." in route["prompt"]
        assert route["mcp_event"]["url"] == sent["params"]["delivery"]["url"]
        assert route["mcp_event"]["subscription_id"] == "sub_1"
        assert result["subscribed"]["refreshBefore"] == route["mcp_event"]["refresh_before"]

    def test_resubscribing_keeps_the_route_secret(self, home):
        _subscribe(FakeSession(_grant()))
        first = next(iter(_routes(home).values()))["secret"]
        _subscribe(FakeSession(_grant()))
        assert [r["secret"] for r in _routes(home).values()] == [first]

    def test_a_rejected_subscription_leaves_no_route(self, home):
        failed = MCPError(-32027, "CallbackEndpointError", {"reason": "challenge_failed"})
        result = _subscribe(FakeSession(failed))
        assert result == {"error": {"code": -32027, "message": "CallbackEndpointError",
                                    "data": {"reason": "challenge_failed"}}}
        assert _routes(home) == {}

    def test_subscribe_needs_an_https_public_url(self, home):
        (home / "config.yaml").write_text("platforms:\n  webhook:\n    extra:\n      public_url: http://x\n")
        session = FakeSession()
        assert "public_url" in _subscribe(session)["error"]
        assert session.sent == [] and _routes(home) == {}


class TestUnsubscribe:
    def test_unsubscribe_sends_the_granted_url_and_drops_the_route(self, home):
        _subscribe(FakeSession(_grant()))
        url = next(iter(_routes(home).values()))["mcp_event"]["url"]
        (home / "config.yaml").write_text("{}\n")  # the subscription key is the URL that was granted
        session = FakeSession({})
        result = asyncio.run(events._unsubscribe_event(
            session, {"name": "feed.emission.created", "arguments": ARGS}, "ambush"))
        assert session.sent[0]["params"] == {"name": "feed.emission.created", "arguments": ARGS,
                                             "delivery": {"url": url}}
        assert "unsubscribed" in result and _routes(home) == {}


class TestRefresh:
    def test_due_time_follows_the_grant(self):
        now = time.time()
        expires = datetime.fromtimestamp(now + 1000, UTC).isoformat()
        assert events._refresh_due_at({}) == 0.0
        assert events._refresh_due_at({"granted_at": now, "refresh_before": None}) == float("inf")
        assert events._refresh_due_at({"granted_at": now, "refresh_before": expires}) == pytest.approx(now + 800)

    def test_a_fresh_connection_resubscribes_every_route(self, home):
        _subscribe(FakeSession(_grant()))
        session = FakeSession(_grant(seconds=3600))
        asyncio.run(events.refresh_server_subscriptions(_server(session), force=True))
        assert [s["method"] for s in session.sent] == ["events/subscribe"]

    def test_a_routine_pass_skips_routes_that_are_not_due(self, home):
        _subscribe(FakeSession(_grant()))
        session = FakeSession()
        next_due = asyncio.run(events.refresh_server_subscriptions(_server(session), force=False))
        assert session.sent == [] and next_due > time.time()

    def test_a_forbidden_refresh_ends_the_subscription(self, home):
        _subscribe(FakeSession(_grant()))
        session = FakeSession(MCPError(-32024, "Forbidden"))
        asyncio.run(events.refresh_server_subscriptions(_server(session), force=True))
        assert _routes(home) == {}

    def test_an_unsubscribe_while_waiting_for_the_session_is_not_undone(self, home):
        _subscribe(FakeSession(_grant()))
        session = FakeSession(_grant())
        server = _server(session)

        async def scenario():
            server._rpc_lock = ObservedLock()
            async with server._rpc_lock:  # an unsubscribe holds the session...
                refresh = asyncio.create_task(events.refresh_server_subscriptions(server, force=True))
                await async_wait_until(lambda: server._rpc_lock.contended, what="refresh queued on the session")
                (home / "webhook_subscriptions.json").write_text("{}")  # ...and drops the route
            await refresh
        asyncio.run(scenario())
        assert session.sent == [] and _routes(home) == {}
