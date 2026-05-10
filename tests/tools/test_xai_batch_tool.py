"""Unit tests for tools.xai_batch_tool."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

from tools import xai_batch_tool
from tools.xai_batch_tool import (
    XaiBatchError,
    XAI_BATCH_SCHEMA,
    check_xai_batch_requirements,
    xai_batch_chat,
)


# ---------------------------------------------------------------------------
# Fake httpx
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _FakeRouter:
    """Round-robin queue of (method, url-suffix-match, response) — order matters."""

    def __init__(self, scripted: List[Dict[str, Any]]):
        # Each entry: {"match": (method, suffix), "resp": _FakeResponse}
        self.scripted = list(scripted)
        self.calls: List[Dict[str, Any]] = []

    def request(self, method: str, url: str, *, headers: Dict[str, str], timeout: int, **kw: Any):
        self.calls.append({"method": method, "url": url, "headers": headers, "kw": kw})
        if not self.scripted:
            raise AssertionError(f"unexpected request: {method} {url}")
        spec = self.scripted.pop(0)
        m, suffix = spec["match"]
        if method != m or not url.endswith(suffix):
            raise AssertionError(
                f"expected {m} ...{suffix}, got {method} {url}"
            )
        return spec["resp"]


@pytest.fixture
def fake_http(monkeypatch):
    """Install a scripted httpx.request stub. Returns the install function."""
    holder: Dict[str, _FakeRouter] = {}

    def install(scripted: List[Dict[str, Any]]) -> _FakeRouter:
        router = _FakeRouter(scripted)
        holder["router"] = router
        monkeypatch.setattr(xai_batch_tool.httpx, "request", router.request)
        monkeypatch.setattr(xai_batch_tool.time, "sleep", lambda _s: None)
        return router

    return install


@pytest.fixture(autouse=True)
def _no_disk_config(monkeypatch):
    monkeypatch.setattr(xai_batch_tool, "_config_section", lambda: {})


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-test-batch")


def _resp_create(batch_id: str = "batch-abc"):
    return {"match": ("POST", "/batches"), "resp": _FakeResponse(200, {"batch_id": batch_id, "name": "x", "state": {"num_requests": 0}})}


def _resp_add():
    return {"match": ("POST", "/requests"), "resp": _FakeResponse(200, {})}


def _resp_state(num_pending: int):
    return {"match": ("GET", lambda u: True), "resp": None}  # placeholder; helper below


def _state_resp(num_pending: int, num_success: int = 0, num_error: int = 0):
    return _FakeResponse(200, {
        "batch_id": "batch-abc",
        "state": {
            "num_requests": num_pending + num_success + num_error,
            "num_pending": num_pending,
            "num_success": num_success,
            "num_error": num_error,
        },
    })


def _results_resp(items: List[Dict[str, Any]], page_token: Optional[str] = None):
    payload: Dict[str, Any] = {"results": items}
    if page_token:
        payload["pagination_token"] = page_token
    return _FakeResponse(200, payload)


# ---------------------------------------------------------------------------
# Requirements / schema
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        ok, why = check_xai_batch_requirements()
        assert ok is False
        assert "XAI_API_KEY" in why

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk")
        ok, _ = check_xai_batch_requirements()
        assert ok is True


class TestSchema:
    def test_schema_requires_requests(self):
        assert XAI_BATCH_SCHEMA["parameters"]["required"] == ["requests"]

    def test_schema_advertises_optional_params(self):
        props = XAI_BATCH_SCHEMA["parameters"]["properties"]
        for key in ("requests", "name", "model", "wait", "max_wait_seconds", "poll_interval_seconds"):
            assert key in props


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestArgValidation:
    def test_empty_requests_raises(self, api_key):
        with pytest.raises(ValueError):
            xai_batch_chat([])

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with pytest.raises(XaiBatchError):
            xai_batch_chat([{"prompt": "hi"}])

    def test_request_without_prompt_or_messages_raises(self, api_key, fake_http):
        # The tool should fail before any HTTP call. Install router with no scripted calls.
        fake_http([])
        with pytest.raises(ValueError):
            xai_batch_chat([{"system": "be brief"}])

    def test_negative_max_wait_raises(self, api_key):
        with pytest.raises(ValueError):
            xai_batch_chat([{"prompt": "hi"}], max_wait_seconds=-1)


# ---------------------------------------------------------------------------
# Submit + add semantics
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_submit_creates_batch_and_adds_requests(self, api_key, fake_http):
        router = fake_http([
            _resp_create("batch-1"),
            _resp_add(),
            {"match": ("GET", "/batches/batch-1"), "resp": _state_resp(num_pending=0, num_success=1)},
            # wait=False: no poll, no results — but with wait=False default we skip
            # Actually wait defaults True, so we go through poll + results
        ])
        # Add poll + results scripted entries:
        router.scripted.append({"match": ("GET", "/batches/batch-1/results"), "resp": _results_resp([
            {"custom_id": "req-00000-aaa", "response": {"choices": [{"message": {"content": "ok"}}]}}
        ])})

        out = xai_batch_chat([{"prompt": "hello", "request_id": "req-00000-aaa"}])
        assert out["batch_id"] == "batch-1"
        # Two HTTP calls before poll: create + add
        create_call = router.calls[0]
        add_call = router.calls[1]
        assert create_call["method"] == "POST" and create_call["url"].endswith("/batches")
        assert add_call["url"].endswith("/batches/batch-1/requests")
        # Inline payload shape
        sent = add_call["kw"]["json"]
        assert "batch_requests" in sent
        assert sent["batch_requests"][0]["batch_request_id"] == "req-00000-aaa"
        assert sent["batch_requests"][0]["batch_request"]["url"] == "/v1/chat/completions"

    def test_submit_uses_default_model(self, api_key, fake_http):
        router = fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=1)},
            {"match": ("GET", "/results"), "resp": _results_resp([])},
        ])
        xai_batch_chat([{"prompt": "hi", "request_id": "r1"}])
        sent = router.calls[1]["kw"]["json"]
        assert sent["batch_requests"][0]["batch_request"]["body"]["model"] == "grok-4.3"

    def test_submit_per_request_model_override(self, api_key, fake_http):
        router = fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=1)},
            {"match": ("GET", "/results"), "resp": _results_resp([])},
        ])
        xai_batch_chat(
            [{"prompt": "hi", "model": "grok-4.20-multi-agent-0309", "request_id": "r1"}],
            model="grok-4.3",
        )
        body = router.calls[1]["kw"]["json"]["batch_requests"][0]["batch_request"]["body"]
        assert body["model"] == "grok-4.20-multi-agent-0309"

    def test_submit_messages_overrides_prompt(self, api_key, fake_http):
        router = fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=1)},
            {"match": ("GET", "/results"), "resp": _results_resp([])},
        ])
        xai_batch_chat([{
            "messages": [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
            "request_id": "r1",
        }])
        msgs = router.calls[1]["kw"]["json"]["batch_requests"][0]["batch_request"]["body"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["content"] == "X"

    def test_submit_4xx_raises(self, api_key, fake_http):
        fake_http([
            {"match": ("POST", "/batches"), "resp": _FakeResponse(400, text="bad")},
        ])
        with pytest.raises(XaiBatchError) as exc:
            xai_batch_chat([{"prompt": "hi"}])
        assert "400" in str(exc.value)


# ---------------------------------------------------------------------------
# wait=False — return immediately
# ---------------------------------------------------------------------------

class TestNoWait:
    def test_returns_after_submit_and_state(self, api_key, fake_http):
        router = fake_http([
            _resp_create("batch-9"),
            _resp_add(),
            {"match": ("GET", "/batches/batch-9"), "resp": _state_resp(num_pending=2)},
        ])
        out = xai_batch_chat(
            [{"prompt": "a"}, {"prompt": "b"}],
            wait=False,
        )
        assert out["batch_id"] == "batch-9"
        assert "results" not in out
        assert out["state"]["num_pending"] == 2
        assert len(router.calls) == 3  # create + add + state


# ---------------------------------------------------------------------------
# Poll + results
# ---------------------------------------------------------------------------

class TestPoll:
    def test_polls_until_pending_zero(self, api_key, fake_http):
        router = fake_http([
            _resp_create(),
            _resp_add(),
            # pre-poll state (from line: state = _get_state(...))
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=2)},
            # poll iterations
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=1)},
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=2)},
            # results
            {"match": ("GET", "/results"), "resp": _results_resp([
                {"custom_id": "r0", "response": {"choices": [{"message": {"content": "first"}}]}},
                {"custom_id": "r1", "response": {"choices": [{"message": {"content": "second"}}]}},
            ])},
        ])
        out = xai_batch_chat([
            {"prompt": "a", "request_id": "r0"},
            {"prompt": "b", "request_id": "r1"},
        ], poll_interval_seconds=0.01)
        assert len(out["results"]) == 2
        assert out["results"][0]["request_id"] == "r0"
        assert out["results"][1]["request_id"] == "r1"

    def test_poll_timeout_raises(self, api_key, monkeypatch, fake_http):
        # Make monotonic jump past the budget after the first poll.
        ticks = iter([0.0, 0.0, 9999.0])
        monkeypatch.setattr(xai_batch_tool.time, "monotonic", lambda: next(ticks))
        fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=5)},
        ])
        with pytest.raises(XaiBatchError) as exc:
            xai_batch_chat([{"prompt": "a"}], max_wait_seconds=10, poll_interval_seconds=0.01)
        assert "not done after" in str(exc.value)


class TestResults:
    def test_pagination_walks_all_pages(self, api_key, fake_http):
        router = fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=3)},
            # Page 1 with token
            {"match": ("GET", "/results"), "resp": _results_resp(
                [{"custom_id": "r0", "response": {"x": 1}}, {"custom_id": "r1", "response": {"x": 2}}],
                page_token="next-page",
            )},
            # Page 2 final
            {"match": ("GET", "/results"), "resp": _results_resp(
                [{"custom_id": "r2", "response": {"x": 3}}],
            )},
        ])
        out = xai_batch_chat([
            {"prompt": "a", "request_id": "r0"},
            {"prompt": "b", "request_id": "r1"},
            {"prompt": "c", "request_id": "r2"},
        ], poll_interval_seconds=0.01)
        assert [r["request_id"] for r in out["results"]] == ["r0", "r1", "r2"]
        assert out["results"][2]["response"]["x"] == 3

    def test_missing_results_yield_none_response(self, api_key, fake_http):
        fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=1)},
            {"match": ("GET", "/results"), "resp": _results_resp([
                {"custom_id": "r0", "response": {"x": 1}},
                # r1 missing from results entirely
            ])},
        ])
        out = xai_batch_chat([
            {"prompt": "a", "request_id": "r0"},
            {"prompt": "b", "request_id": "r1"},
        ], poll_interval_seconds=0.01)
        assert out["results"][0]["response"] == {"x": 1}
        assert out["results"][1]["response"] is None


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

class TestHeaders:
    def test_authorization_and_user_agent(self, api_key, fake_http):
        router = fake_http([
            _resp_create(),
            _resp_add(),
            {"match": ("GET", "/batches/batch-abc"), "resp": _state_resp(num_pending=0, num_success=1)},
            {"match": ("GET", "/results"), "resp": _results_resp([])},
        ])
        xai_batch_chat([{"prompt": "hi", "request_id": "r0"}], poll_interval_seconds=0.01)
        for call in router.calls:
            assert call["headers"]["Authorization"] == "Bearer sk-test-batch"
            assert call["headers"]["User-Agent"].startswith("Hermes-Agent/")
