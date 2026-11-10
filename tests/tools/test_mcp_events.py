"""MCP Events client (webhook delivery): opt-in utility tools, subscribe/unsubscribe against the
dynamic webhook route store, and the refresh loop's re-subscription rules."""

import asyncio
import base64
import contextlib
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


def _state(home):
    path = home / "mcp_event_subscriptions.json"
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

    def test_a_native_list_events_tool_keeps_every_event_utility(self):
        # Calendar servers commonly ship a native ``list_events`` tool; a native tool shadows a utility
        # of the same name, which would leave the model no way to list event types.
        from tools import mcp_tool_registration as reg
        from tools.mcp_tool_schema import _build_utility_schemas
        native = SimpleNamespace(name="list_events", description="Calendar events", inputSchema={"type": "object"})
        utilities = [e for e in _build_utility_schemas("cal") if e["handler_key"] in events.EVENT_UTILITY_KEYS]
        candidates = reg._tool_candidates("cal", [native], lambda name: True, 30)
        candidates += reg._utility_candidates("cal", utilities, 30)
        kept = {c.origin for c in reg._resolve_name_collisions("cal", candidates)}
        assert "tool 'list_events'" in kept
        assert {f"generated utility {key!r}" for key in events.EVENT_UTILITY_KEYS} <= kept


class TestList:
    def test_listing_stops_at_the_page_cap(self, home):
        class Endless:
            calls = 0

            async def send_request(self, request, result_type):
                Endless.calls += 1
                await asyncio.sleep(0)
                return {"events": [{"name": f"e{Endless.calls}"}], "nextCursor": f"c{Endless.calls}"}

        result = asyncio.run(asyncio.wait_for(events._list_event_types(Endless(), {}, "ambush"), timeout=5))
        assert len(result["events"]) == Endless.calls == events._MAX_LIST_PAGES
        assert "truncated" in result


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
        state = _state(home)[name]
        assert state["url"] == sent["params"]["delivery"]["url"]
        assert state["subscription_id"] == "sub_1"
        assert result["subscribed"]["refreshBefore"] == state["refresh_before"]
        assert set(route["mcp_event"]) == {"server", "name", "arguments"}  # the route holds no changing state

    def test_resubscribing_keeps_the_route_secret(self, home):
        _subscribe(FakeSession(_grant()))
        data = _routes(home)
        next(iter(data.values()))["created_at"] = "2026-01-01T00:00:00Z"
        (home / "webhook_subscriptions.json").write_text(json.dumps(data))
        first = next(iter(data.values()))
        _subscribe(FakeSession(_grant()))
        assert list(_routes(home).values()) == [first]  # identical: in-flight runs keep their route digest

    def test_a_fresh_subscription_ignores_state_left_by_a_removed_route(self, home):
        name = events._route_name("ambush", "feed.emission.created", ARGS)  # e.g. after `hermes webhook remove`
        (home / "mcp_event_subscriptions.json").write_text(json.dumps({name: {"cursor": "stale"}}))
        session = FakeSession(_grant())
        _subscribe(session)
        assert "cursor" not in session.sent[0]["params"]

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

    def test_the_approval_shows_the_instructions_and_the_destination(self, monkeypatch):
        shown = {}

        def consent(message, description, **kwargs):
            shown.update(message=message, description=description)
            return "accept"
        monkeypatch.setattr("tools.approval_prompt.request_elicitation_consent", consent)
        args = {"name": "feed.emission.created", "arguments": ARGS, "instructions": "Post a two-line summary.",
                "deliver": "telegram", "deliver_chat_id": "-100123"}
        assert events._consent_error("ambush", args) is None
        text = f"{shown['message']}\n{shown['description']}"
        assert "Post a two-line summary." in text and "telegram" in text and "-100123" in text

    def test_overlong_instructions_are_refused_before_asking(self):
        assert events._subscribe_args_error({"name": "x", "instructions": "a" * events._MAX_INSTRUCTIONS_CHARS}) is None
        assert events._subscribe_args_error({"name": "x", "instructions": "a" * (events._MAX_INSTRUCTIONS_CHARS + 1)})
        assert events._subscribe_args_error({"name": "x", "deliver_chat_id": 123})

    def test_a_disabled_subscription_is_not_turned_back_on(self, home):
        _subscribe(FakeSession(_grant()))
        data = _routes(home)
        (name, route), = data.items()
        route["enabled"] = False
        (home / "webhook_subscriptions.json").write_text(json.dumps(data))
        session = FakeSession()
        assert "error" in _subscribe(session)
        assert session.sent == [] and _routes(home)[name]["enabled"] is False


class TestUnsubscribe:
    def test_unsubscribe_sends_the_granted_url_and_drops_the_route(self, home):
        _subscribe(FakeSession(_grant()))
        url = next(iter(_state(home).values()))["url"]
        (home / "config.yaml").write_text("{}\n")  # the subscription key is the URL that was granted
        session = FakeSession({})
        result = asyncio.run(events._unsubscribe_event(
            session, {"name": "feed.emission.created", "arguments": ARGS}, "ambush"))
        assert session.sent[0]["params"] == {"name": "feed.emission.created", "arguments": ARGS,
                                             "delivery": {"url": url}}
        assert "unsubscribed" in result and _routes(home) == {} and _state(home) == {}


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

    @pytest.mark.parametrize("code", [-32024, -32023, -32602, -32026],
                             ids=["forbidden", "not-found", "invalid-params", "unsupported"])
    def test_a_refusal_that_cannot_change_ends_the_subscription(self, home, code):
        _subscribe(FakeSession(_grant()))
        session = FakeSession(MCPError(code, "refused"))
        asyncio.run(events.refresh_server_subscriptions(_server(session), force=True))
        assert _routes(home) == {}

    def test_a_failing_refresh_backs_off(self, home):
        _subscribe(FakeSession(_grant()))
        failures = {}
        delays = []
        for _ in range(3):
            session = FakeSession(MCPError(-32027, "CallbackEndpointError", {"reason": "timeout"}))
            next_due = asyncio.run(events.refresh_server_subscriptions(_server(session), force=True, failures=failures))
            delays.append(next_due - time.time())
        assert [round(delay / events._REFRESH_MIN_SLEEP_S) for delay in delays] == [1, 2, 4]
        state = _state(home)  # the grant has run out, so only the back-off holds the next attempt
        next(iter(state.values())).update(granted_at=time.time() - 7200, refresh_before=_grant(-3600)["refreshBefore"])
        (home / "mcp_event_subscriptions.json").write_text(json.dumps(state))
        session = FakeSession()
        next_due = asyncio.run(events.refresh_server_subscriptions(_server(session), force=False, failures=failures))
        assert session.sent == [] and 100 < next_due - time.time() <= 4 * events._REFRESH_MIN_SLEEP_S

    def test_a_disabled_route_is_left_to_lapse(self, home):
        _subscribe(FakeSession(_grant()))
        data = _routes(home)
        next(iter(data.values()))["enabled"] = False
        (home / "webhook_subscriptions.json").write_text(json.dumps(data))
        session = FakeSession()
        asyncio.run(events.refresh_server_subscriptions(_server(session), force=True))
        assert session.sent == []

    def test_a_route_removed_during_the_refresh_is_released(self, home):
        _subscribe(FakeSession(_grant()))

        class RacingSession(FakeSession):
            async def send_request(self, request, result_type):
                reply = await super().send_request(request, result_type)
                if request.method == "events/subscribe":  # another process unsubscribed meanwhile
                    (home / "webhook_subscriptions.json").write_text("{}")
                return reply

        session = RacingSession(_grant(), {})
        asyncio.run(events.refresh_server_subscriptions(_server(session), force=True))
        assert [sent["method"] for sent in session.sent] == ["events/subscribe", "events/unsubscribe"]
        assert session.sent[1]["params"]["delivery"] == {"url": session.sent[0]["params"]["delivery"]["url"]}
        assert _routes(home) == {}

    def test_a_new_subscription_wakes_the_refresh_loop(self, home, monkeypatch):
        passes = []

        async def recording_pass(server, *, force, failures=None):
            passes.append(force)
            return float("inf")  # nothing due: without a wake-up the loop sleeps for an hour
        monkeypatch.setattr(events, "refresh_server_subscriptions", recording_pass)

        async def scenario():
            loop = asyncio.create_task(events.run_refresh_loop(_server(FakeSession())))
            await async_wait_until(lambda: passes == [True], what="the first pass")
            await events._subscribe_event(FakeSession(_grant(seconds=600)), {
                "name": "feed.emission.created", "arguments": ARGS}, "ambush")
            await async_wait_until(lambda: len(passes) == 2, what="a pass for the new subscription")
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop
        asyncio.run(scenario())
        assert passes == [True, False]

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
