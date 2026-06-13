"""Regression tests for Home Assistant response-service support (#27256).

`todo.get_items`, `calendar.get_events`, `weather.get_forecasts` and
`conversation.process` are registered in Home Assistant with
``SupportsResponse.ONLY``: a REST call without ``?return_response=true``
gets rejected with HTTP 400 ("Service ... requires response data").

These tests cover:

* The new ``return_response`` parameter on ``_async_call_service`` and the
  ``ha_call_service`` schema.
* Auto-opt-in for the well-known response-only services so the model
  doesn't have to set the flag for the most common cases.
* Defensive auto-retry-once when HA's 400 explicitly signals the issue
  (covers user HA versions / custom integrations that aren't in the
  static list).
* Backward-compat for ordinary service calls (no flag, no query string,
  flat response shape).
* Parsing of the new response shape that includes ``service_response``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, patch

import pytest

from tools.homeassistant_tool import (
    _RESPONSE_ONLY_SERVICES,
    _async_call_service,
    _build_service_payload,
    _handle_call_service,
    _is_response_required_hint,
    _parse_service_response,
    _service_url,
    HA_CALL_SERVICE_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers — a tiny aiohttp session double good enough for our call paths.
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int, body: str, content_type: str = "application/json"):
        self.status = status
        self._body = body
        self.content_type = content_type

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._body

    async def json(self):
        return json.loads(self._body)

    def raise_for_status(self):
        if self.status >= 400:
            import aiohttp

            raise aiohttp.ClientResponseError(
                request_info=None,  # type: ignore[arg-type]
                history=(),
                status=self.status,
                message=self._body,
            )


class _FakeSession:
    """Replays scripted (status, body, content_type) tuples in order."""

    def __init__(self, scripted: List[Tuple[int, str, str]]):
        self._scripted = list(scripted)
        self.posts: List[Dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        if not self._scripted:
            raise AssertionError(f"Unexpected POST {url}; no more scripted responses")
        status, body, ct = self._scripted.pop(0)
        return _FakeResp(status, body, ct)


def _patch_aiohttp(scripted: List[Tuple[int, str, str]]):
    """Install a FakeSession factory that captures POSTs to HA."""
    session = _FakeSession(scripted)

    class _Factory:
        def __call__(self):  # ClientSession()
            return session

    import aiohttp

    return patch.object(aiohttp, "ClientSession", _Factory()), session


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _ha_env(monkeypatch):
    """All tests in this module need a HASS token + a deterministic URL."""
    monkeypatch.setenv("HASS_TOKEN", "test-token")
    monkeypatch.setenv("HASS_URL", "http://ha.local:8123")
    monkeypatch.setattr("tools.homeassistant_tool._HASS_TOKEN", "")
    monkeypatch.setattr("tools.homeassistant_tool._HASS_URL", "")


# ---------------------------------------------------------------------------
# Service URL construction
# ---------------------------------------------------------------------------


class TestServiceURL:
    def test_no_return_response_omits_query(self):
        url = _service_url("http://ha.local:8123", "light", "turn_on", False)
        assert url == "http://ha.local:8123/api/services/light/turn_on"

    def test_return_response_appends_query(self):
        url = _service_url("http://ha.local:8123", "todo", "get_items", True)
        assert url == "http://ha.local:8123/api/services/todo/get_items?return_response=true"


# ---------------------------------------------------------------------------
# 400-hint detector
# ---------------------------------------------------------------------------


class TestIsResponseRequiredHint:
    @pytest.mark.parametrize(
        "body",
        [
            '{"message": "Service todo.get_items requires response data"}',
            '{"message": "Set return_response=true to receive responses"}',
            "Service requires response data",
            "Please call with return_response=true",
        ],
    )
    def test_recognised_hints(self, body):
        assert _is_response_required_hint(400, body) is True

    @pytest.mark.parametrize(
        "status,body",
        [
            (400, '{"message": "Unknown service"}'),
            (400, '{"message": "Invalid entity_id"}'),
            (401, "Service requires response data"),  # auth error, don't retry
            (500, "return_response would have helped"),  # not a 400
            (400, ""),
            (200, "ok"),
        ],
    )
    def test_negatives(self, status, body):
        assert _is_response_required_hint(status, body) is False


# ---------------------------------------------------------------------------
# Static response-only allowlist
# ---------------------------------------------------------------------------


class TestResponseOnlyAllowlist:
    @pytest.mark.parametrize(
        "domain,service",
        [
            ("todo", "get_items"),
            ("calendar", "get_events"),
            ("weather", "get_forecasts"),
            ("conversation", "process"),
        ],
    )
    def test_known_services_are_listed(self, domain, service):
        assert (domain, service) in _RESPONSE_ONLY_SERVICES

    def test_regular_services_not_listed(self):
        assert ("light", "turn_on") not in _RESPONSE_ONLY_SERVICES
        assert ("todo", "add_item") not in _RESPONSE_ONLY_SERVICES
        assert ("todo", "remove_item") not in _RESPONSE_ONLY_SERVICES


# ---------------------------------------------------------------------------
# Response parser handles both old and new shapes
# ---------------------------------------------------------------------------


class TestParseServiceResponseNewShape:
    def test_dict_with_service_response_extracts_payload(self):
        ha_response = {
            "changed_states": [],
            "service_response": {
                "todo.mylist": {
                    "items": [
                        {"summary": "buy milk", "status": "needs_action"},
                        {"summary": "vacuum", "status": "completed"},
                    ]
                }
            },
        }
        parsed = _parse_service_response("todo", "get_items", ha_response)
        assert parsed["success"] is True
        assert parsed["service"] == "todo.get_items"
        assert parsed["affected_entities"] == []
        assert parsed["service_response"] == ha_response["service_response"]

    def test_dict_with_changed_states_and_response(self):
        ha_response = {
            "changed_states": [
                {"entity_id": "todo.mylist", "state": "5"},
            ],
            "service_response": {"todo.mylist": {"items": []}},
        }
        parsed = _parse_service_response("todo", "get_items", ha_response)
        assert parsed["affected_entities"] == [
            {"entity_id": "todo.mylist", "state": "5"}
        ]
        assert parsed["service_response"] == ha_response["service_response"]

    def test_legacy_list_response_still_works(self):
        ha_response = [
            {"entity_id": "light.kitchen", "state": "on"},
            {"entity_id": "light.bedroom", "state": "on"},
        ]
        parsed = _parse_service_response("light", "turn_on", ha_response)
        assert "service_response" not in parsed
        assert len(parsed["affected_entities"]) == 2

    def test_empty_dict_response(self):
        parsed = _parse_service_response("script", "run", {})
        assert parsed["affected_entities"] == []
        assert "service_response" not in parsed

    def test_none_response_no_crash(self):
        parsed = _parse_service_response("automation", "trigger", None)
        assert parsed["affected_entities"] == []

    def test_malformed_changed_states_items_skipped(self):
        ha_response = {
            "changed_states": [None, "not-a-dict", {"entity_id": "x", "state": "y"}],
            "service_response": {"x": 1},
        }
        parsed = _parse_service_response("foo", "bar", ha_response)
        assert parsed["affected_entities"] == [{"entity_id": "x", "state": "y"}]
        assert parsed["service_response"] == {"x": 1}


# ---------------------------------------------------------------------------
# _async_call_service network behaviour
# ---------------------------------------------------------------------------


class TestAsyncCallServiceQueryString:
    def test_regular_service_no_query(self):
        body = json.dumps([{"entity_id": "light.x", "state": "on"}])
        patcher, session = _patch_aiohttp([(200, body, "application/json")])
        with patcher:
            result = _run(
                _async_call_service("light", "turn_on", entity_id="light.x")
            )
        assert "?return_response" not in session.posts[0]["url"]
        assert result["success"] is True

    def test_known_response_only_service_auto_appends_query(self):
        body = json.dumps({
            "changed_states": [],
            "service_response": {"todo.mylist": {"items": []}},
        })
        patcher, session = _patch_aiohttp([(200, body, "application/json")])
        with patcher:
            result = _run(
                _async_call_service(
                    "todo", "get_items", entity_id="todo.mylist"
                )
            )
        # No explicit flag — auto-opt-in via the allowlist.
        assert session.posts[0]["url"].endswith("?return_response=true")
        assert result["service_response"] == {"todo.mylist": {"items": []}}

    def test_explicit_return_response_flag(self):
        body = json.dumps({
            "changed_states": [],
            "service_response": {"calendar.foo": {"events": []}},
        })
        patcher, session = _patch_aiohttp([(200, body, "application/json")])
        with patcher:
            _run(
                _async_call_service(
                    "calendar",
                    "get_events",
                    entity_id="calendar.foo",
                    return_response=True,
                )
            )
        assert session.posts[0]["url"].endswith("?return_response=true")

    def test_auto_retry_on_400_hint(self):
        """HA tells us this service needs return_response → we retry once."""
        first = (
            400,
            '{"message": "Service custom.fetch requires response data"}',
            "application/json",
        )
        second = (
            200,
            json.dumps({
                "changed_states": [],
                "service_response": {"value": 42},
            }),
            "application/json",
        )
        patcher, session = _patch_aiohttp([first, second])
        with patcher:
            result = _run(_async_call_service("custom", "fetch"))

        assert len(session.posts) == 2, "expected one retry, not more"
        assert "?return_response" not in session.posts[0]["url"]
        assert session.posts[1]["url"].endswith("?return_response=true")
        assert result["service_response"] == {"value": 42}

    def test_400_without_hint_does_not_retry(self):
        patcher, session = _patch_aiohttp([
            (400, '{"message": "Entity not found"}', "application/json"),
        ])
        with patcher:
            with pytest.raises(Exception):
                _run(_async_call_service("light", "turn_on", entity_id="light.x"))
        assert len(session.posts) == 1, "must not retry on unrelated 400"

    def test_known_response_service_does_not_retry_on_400(self):
        """If we already had return_response=true and HA still 400s, surface it."""
        patcher, session = _patch_aiohttp([
            (
                400,
                '{"message": "Service todo.get_items requires response data"}',
                "application/json",
            ),
        ])
        with patcher:
            with pytest.raises(Exception):
                _run(
                    _async_call_service(
                        "todo", "get_items", entity_id="todo.mylist"
                    )
                )
        # No retry — first request already used return_response=true.
        assert len(session.posts) == 1
        assert session.posts[0]["url"].endswith("?return_response=true")

    def test_non_json_body_returned_as_text(self):
        patcher, session = _patch_aiohttp([(200, "OK", "text/plain")])
        with patcher:
            result = _run(_async_call_service("script", "run"))
        # Plain-text bodies still parse cleanly into a success envelope.
        assert result["success"] is True
        assert result["affected_entities"] == []

    def test_empty_body_returned_as_none(self):
        patcher, session = _patch_aiohttp([(200, "", "application/json")])
        with patcher:
            result = _run(_async_call_service("script", "run"))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# End-to-end: _handle_call_service routes return_response correctly
# ---------------------------------------------------------------------------


class TestHandlerReturnResponseWiring:
    """Confirm the handler forwards return_response (incl. string coercion)."""

    def _patch_call(self):
        """Patch _async_call_service with an AsyncMock + bypass _run_async."""
        async_mock = AsyncMock(return_value={"success": True, "service": "x.y"})
        return patch(
            "tools.homeassistant_tool._async_call_service", async_mock
        ), async_mock

    @staticmethod
    def _passthrough_run_async():
        """Patch _run_async to actually drive the coroutine via asyncio.run."""
        return patch(
            "tools.homeassistant_tool._run_async",
            side_effect=lambda coro: asyncio.run(coro),
        )

    def _last_call_kwargs(self, async_mock):
        call = async_mock.await_args
        # positional: (domain, service, entity_id, data, return_response)
        return call.args, call.kwargs

    def test_bool_true_flows_through(self):
        patcher, async_mock = self._patch_call()
        with patcher, self._passthrough_run_async():
            _handle_call_service({
                "domain": "todo",
                "service": "get_items",
                "entity_id": "todo.mylist",
                "return_response": True,
            })
        args, _ = self._last_call_kwargs(async_mock)
        assert args[-1] is True

    def test_bool_false_flows_through(self):
        patcher, async_mock = self._patch_call()
        with patcher, self._passthrough_run_async():
            _handle_call_service({
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.x",
                "return_response": False,
            })
        args, _ = self._last_call_kwargs(async_mock)
        assert args[-1] is False

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
    ])
    def test_string_values_coerced(self, raw, expected):
        patcher, async_mock = self._patch_call()
        with patcher, self._passthrough_run_async():
            _handle_call_service({
                "domain": "todo",
                "service": "get_items",
                "entity_id": "todo.mylist",
                "return_response": raw,
            })
        args, _ = self._last_call_kwargs(async_mock)
        assert args[-1] is expected

    def test_default_is_false(self):
        patcher, async_mock = self._patch_call()
        with patcher, self._passthrough_run_async():
            _handle_call_service({
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.x",
            })
        args, _ = self._last_call_kwargs(async_mock)
        assert args[-1] is False


# ---------------------------------------------------------------------------
# Unrelated 400s surface as a clean tool error (not an AttributeError crash)
# ---------------------------------------------------------------------------


class TestHandlerUnrelated400:
    """An unrelated 400 must come back as a tool error, not crash the handler.

    Regression for the review on #27270: the error used to be raised as
    ``aiohttp.ClientResponseError(request_info=None, ...)``. Formatting that
    exception (``f"{e}"`` in the handler's ``except`` block) dereferences
    ``request_info.real_url`` and raises ``AttributeError``, so the handler's
    error path crashed instead of returning a JSON tool error.
    """

    @staticmethod
    def _passthrough_run_async():
        return patch(
            "tools.homeassistant_tool._run_async",
            side_effect=lambda coro: _run(coro),
        )

    def test_unrelated_400_returns_tool_error(self):
        patcher, session = _patch_aiohttp([
            (400, '{"message": "Entity not found"}', "application/json"),
        ])
        with patcher, self._passthrough_run_async():
            raw = _handle_call_service({
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.x",
            })
        # Must be a clean JSON tool error, not a raised exception.
        payload = json.loads(raw)
        assert "error" in payload
        assert "light.turn_on" in payload["error"]
        # Unrelated 400 must not trigger a retry.
        assert len(session.posts) == 1

    def test_api_error_stringifies_safely(self):
        """The synthesized HTTP error must format without raising."""
        from tools.homeassistant_tool import HomeAssistantAPIError

        err = HomeAssistantAPIError(400, '{"message": "Entity not found"}')
        # Any of these used to raise AttributeError under the old aiohttp path.
        assert str(err)
        assert f"{err}"
        assert err.status == 400
        assert "Entity not found" in err.message


# ---------------------------------------------------------------------------
# Schema advertises the new field
# ---------------------------------------------------------------------------


class TestSchemaSurface:
    def test_return_response_field_in_schema(self):
        props = HA_CALL_SERVICE_SCHEMA["parameters"]["properties"]
        assert "return_response" in props
        assert props["return_response"]["type"] == "boolean"

    def test_return_response_not_required(self):
        required = HA_CALL_SERVICE_SCHEMA["parameters"]["required"]
        assert "return_response" not in required

    def test_description_mentions_response_services(self):
        desc = HA_CALL_SERVICE_SCHEMA["description"].lower()
        assert "todo.get_items".lower() in desc
        assert "return_response" in desc


# ---------------------------------------------------------------------------
# Issue #27256 end-to-end reproduction
# ---------------------------------------------------------------------------


class TestIssue27256Repro:
    """The exact failing scenario described in the bug report."""

    def test_get_items_returns_items(self):
        body = json.dumps({
            "changed_states": [],
            "service_response": {
                "todo.mylist": {
                    "items": [
                        {"summary": "buy milk", "status": "needs_action"},
                        {"summary": "vacuum", "status": "completed"},
                    ]
                }
            },
        })
        patcher, session = _patch_aiohttp([(200, body, "application/json")])
        with patcher:
            with patch(
                "tools.homeassistant_tool._run_async",
                side_effect=lambda coro: _run(coro),
            ):
                raw = _handle_call_service({
                    "domain": "todo",
                    "service": "get_items",
                    "entity_id": "todo.mylist",
                    "data": '{"status": "needs_action"}',
                })

        result = json.loads(raw)["result"]
        assert result["success"] is True
        items = result["service_response"]["todo.mylist"]["items"]
        assert {i["summary"] for i in items} == {"buy milk", "vacuum"}
        # Confirm we used the query string the model never had to know about.
        assert session.posts[0]["url"].endswith("?return_response=true")
        # And the data dict round-tripped via the flat payload, like
        # add_item/remove_item already did successfully.
        assert session.posts[0]["json"] == {
            "entity_id": "todo.mylist",
            "status": "needs_action",
        }

    def test_add_item_unchanged(self):
        """`add_item` already worked — must not regress."""
        body = json.dumps([])
        patcher, session = _patch_aiohttp([(200, body, "application/json")])
        with patcher:
            with patch(
                "tools.homeassistant_tool._run_async",
                side_effect=lambda coro: _run(coro),
            ):
                raw = _handle_call_service({
                    "domain": "todo",
                    "service": "add_item",
                    "entity_id": "todo.mylist",
                    "data": '{"item": "buy milk"}',
                })

        result = json.loads(raw)["result"]
        assert result["success"] is True
        # No return_response opt-in for add_item — keeps wire compat.
        assert "?return_response" not in session.posts[0]["url"]
