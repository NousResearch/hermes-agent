"""Unit tests for the TTM ingress skill (PR-F-H2).

All HTTP traffic is mocked through a stub ``client_factory`` so no network
calls are required. The tests assert the wire contract: headers, body
shape (matching ``ControlPlaneEventAppendRequest`` etc.), retry/backoff
on 5xx, immediate raise on 401, and token redaction in logs.
"""

import logging

import httpx
import pytest

from tools.ttm_ingress import (
    CANONICAL_EVENT_TYPES,
    ENV_INGRESS_BASE_URL,
    ENV_PRINCIPAL_TOKEN,
    ENV_RUNTIME_ID,
    ENV_RUN_ID,
    ENV_SCOPE_EPOCH,
    EVENT_RUN_DISPATCHED,
    EVENT_TASK_UPDATED,
    IngressAuthError,
    IngressClientError,
    IngressNotBoundError,
    IngressServerError,
    TtmIngress,
    _token_present,
)

PRINCIPAL_TOKEN = "tok_abcdef0123456789xyz"
RUN_ID = "run-pr-f-h2-test"
BASE_URL = "http://127.0.0.1:8000"
RUNTIME_ID = "hermes"


# ---------------------------------------------------------------------------
# Stub HTTP client — captures every request and returns canned responses
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        # Mirror httpx.Response.text fallback when no JSON.
        self.text = text or (str(self._json) if json_body else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _StubClient:
    """Drop-in replacement for ``httpx.Client`` used in tests."""

    def __init__(self, plan):
        # ``plan`` is a callable that returns the next _StubResponse, OR
        # a list popped left-to-right.
        self._plan = plan
        self.calls: list[tuple[str, dict, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json or {}, headers or {}))
        if callable(self._plan):
            return self._plan(url, json, headers)
        if not self._plan:
            raise AssertionError("StubClient ran out of canned responses")
        nxt = self._plan.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def get(self, url, headers=None):
        self.calls.append((url, {}, headers or {}))
        if callable(self._plan):
            return self._plan(url, None, headers)
        if not self._plan:
            raise AssertionError("StubClient ran out of canned responses")
        nxt = self._plan.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _client_factory(plan):
    """Return a factory that yields a fresh _StubClient per call.

    The TtmIngress code opens a new client for each POST (via
    context manager), so we share a *single* StubClient across factory
    invocations to keep call history together.
    """
    stub = _StubClient(plan)

    def _factory():
        return stub

    return _factory, stub


def _bind(ttm: TtmIngress) -> None:
    ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, BASE_URL, runtime_id=RUNTIME_ID)


# ---------------------------------------------------------------------------
# bind_run / unbind_run
# ---------------------------------------------------------------------------


class TestBinding:
    def test_bind_run_requires_run_id(self):
        ttm = TtmIngress()
        with pytest.raises(ValueError, match="run_id"):
            ttm.bind_run("", PRINCIPAL_TOKEN, BASE_URL)

    def test_bind_run_requires_token(self):
        ttm = TtmIngress()
        with pytest.raises(ValueError, match="principal_token"):
            ttm.bind_run(RUN_ID, "", BASE_URL)

    def test_bind_run_requires_base_url(self):
        ttm = TtmIngress()
        with pytest.raises(ValueError, match="ingress_base_url"):
            ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, "")

    def test_bind_run_strips_trailing_slash(self):
        factory, stub = _client_factory([_StubResponse(201, {"event_id": "ev-1"})])
        ttm = TtmIngress(client_factory=factory)
        ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, "http://127.0.0.1:8000/")
        ttm.post_event(RUN_ID, EVENT_RUN_DISPATCHED, {})
        url, _, _ = stub.calls[0]
        assert url == "http://127.0.0.1:8000/api/ingress/runtime/hermes/events"

    def test_unbind_drops_binding(self):
        ttm = TtmIngress()
        _bind(ttm)
        assert ttm.is_bound(RUN_ID)
        ttm.unbind_run(RUN_ID)
        assert not ttm.is_bound(RUN_ID)

    def test_call_without_binding_raises(self):
        ttm = TtmIngress()
        with pytest.raises(IngressNotBoundError):
            ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {})

    def test_rebind_replaces_token(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        ttm.bind_run(RUN_ID, "old-token-aaaa", BASE_URL)
        ttm.bind_run(RUN_ID, "new-token-bbbb", BASE_URL)
        ttm.post_event(RUN_ID, EVENT_RUN_DISPATCHED, {})
        _, _, headers = stub.calls[0]
        assert headers["Authorization"] == "Bearer new-token-bbbb"


# ---------------------------------------------------------------------------
# Token logging — only a presence marker, never the value or a prefix
# ---------------------------------------------------------------------------


class TestTokenPresenceLogging:
    def test_present_when_set(self):
        assert _token_present("tok_abcdef0123456789") == "set"

    def test_present_when_short(self):
        assert _token_present("tiny") == "set"

    def test_unset_when_empty(self):
        assert _token_present("") == "unset"

    def test_token_value_and_prefix_never_logged(self, caplog):
        # Even a partial prefix would be unsafe in long-running production
        # logs (post-mortem leakage). The presence marker is the only
        # token-derived signal we emit.
        factory, _ = _client_factory([_StubResponse(201, {"event_id": "ev-1"})])
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        with caplog.at_level(logging.DEBUG, logger="tools.ttm_ingress"):
            ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {"x": 1})
        for record in caplog.records:
            msg = record.getMessage()
            assert PRINCIPAL_TOKEN not in msg
            assert PRINCIPAL_TOKEN[:8] not in msg
            assert "tok_abcd" not in msg


# ---------------------------------------------------------------------------
# post_event — wire contract
# ---------------------------------------------------------------------------


class TestPostEvent:
    def test_success_returns_event_id(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-deadbeef"})]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        event_id = ttm.post_event(
            RUN_ID,
            EVENT_TASK_UPDATED,
            {"task": "writing tests"},
            summary="bumped task state",
        )
        assert event_id == "ev-deadbeef"
        assert len(stub.calls) == 1

    def test_request_url_and_headers(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        ttm.post_event(RUN_ID, EVENT_RUN_DISPATCHED, {"k": "v"})
        url, body, headers = stub.calls[0]
        assert url == f"{BASE_URL}/api/ingress/runtime/hermes/events"
        assert headers["Authorization"] == f"Bearer {PRINCIPAL_TOKEN}"
        assert headers["X-Runtime-Id"] == "hermes"
        assert headers["X-Run-Id"] == RUN_ID
        assert headers["Content-Type"] == "application/json"

    def test_body_matches_control_plane_event_append_request(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, BASE_URL, initial_scope_epoch=4)
        ttm.post_event(
            RUN_ID,
            EVENT_TASK_UPDATED,
            {"task_id": "T-7", "status": "in_progress"},
            summary="started T-7",
            actor_id="hermes",
        )
        _, body, _ = stub.calls[0]
        assert body == {
            "event_type": EVENT_TASK_UPDATED,
            "actor_type": "runtime",
            "actor_id": "hermes",
            "expected_scope_epoch": 4,
            "summary": "started T-7",
            "payload": {"task_id": "T-7", "status": "in_progress"},
        }

    def test_human_actor_omits_actor_id(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        ttm.post_event(
            RUN_ID,
            EVENT_APPROVAL_REQUESTED := "approval.requested",
            {},
            summary="op approval",
            actor_type="human",
            # actor_id passed but ignored for human
            actor_id="should-be-dropped",
        )
        _, body, _ = stub.calls[0]
        assert body["actor_type"] == "human"
        assert body["actor_id"] is None

    def test_default_summary_is_humanized_event_type(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        ttm.post_event(RUN_ID, "phase.entered", {})
        _, body, _ = stub.calls[0]
        assert body["summary"] == "phase entered"

    def test_explicit_scope_epoch_overrides_binding(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, BASE_URL, initial_scope_epoch=1)
        ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {}, scope_epoch=9)
        _, body, _ = stub.calls[0]
        assert body["expected_scope_epoch"] == 9

    def test_non_canonical_event_type_warns(self, caplog):
        factory, _ = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        with caplog.at_level(logging.WARNING, logger="tools.ttm_ingress"):
            ttm.post_event(RUN_ID, "made.up", {})
        assert any("non_canonical" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 401 — IngressAuthError, no retry
# ---------------------------------------------------------------------------


class TestAuthFailure:
    def test_401_raises_immediately(self, caplog):
        factory, stub = _client_factory(
            [
                _StubResponse(
                    401, text='{"detail":{"reason":"principal_token_invalid"}}'
                ),
                # Sentinel to fail the test if a retry happens.
                _StubResponse(201, {"event_id": "should-not-be-reached"}),
            ]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        with caplog.at_level(logging.WARNING, logger="tools.ttm_ingress"):
            with pytest.raises(IngressAuthError) as excinfo:
                ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {})
        assert excinfo.value.run_id == RUN_ID
        assert "principal_token_invalid" in excinfo.value.body
        assert len(stub.calls) == 1, "401 must not retry"
        assert any(
            "principal_token_rejected" in r.getMessage() for r in caplog.records
        )


# ---------------------------------------------------------------------------
# 4xx (non-401) — IngressClientError, no retry
# ---------------------------------------------------------------------------


class TestClientError:
    def test_409_raises_without_retry(self):
        factory, stub = _client_factory(
            [
                _StubResponse(409, text="scope_epoch changed"),
                _StubResponse(201, {"event_id": "should-not-be-reached"}),
            ]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        with pytest.raises(IngressClientError) as excinfo:
            ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {})
        assert excinfo.value.status_code == 409
        assert len(stub.calls) == 1


# ---------------------------------------------------------------------------
# 5xx — exponential backoff retry, then raise
# ---------------------------------------------------------------------------


class TestServerErrorRetry:
    def test_5xx_retries_then_succeeds(self):
        factory, stub = _client_factory(
            [
                _StubResponse(503, text="overloaded"),
                _StubResponse(502, text="bad gateway"),
                _StubResponse(201, {"event_id": "ev-late"}),
            ]
        )
        slept: list[float] = []
        ttm = TtmIngress(
            client_factory=factory, max_retries=3, retry_base_delay=0.1, sleep=slept.append
        )
        _bind(ttm)
        event_id = ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {})
        assert event_id == "ev-late"
        assert len(stub.calls) == 3
        assert slept == [0.1, 0.2]

    def test_5xx_exhausts_then_raises(self):
        factory, stub = _client_factory(
            [
                _StubResponse(500, text="boom-1"),
                _StubResponse(500, text="boom-2"),
                _StubResponse(500, text="boom-3"),
            ]
        )
        slept: list[float] = []
        ttm = TtmIngress(
            client_factory=factory, max_retries=3, retry_base_delay=0.1, sleep=slept.append
        )
        _bind(ttm)
        with pytest.raises(IngressServerError) as excinfo:
            ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {})
        assert excinfo.value.attempts == 3
        assert excinfo.value.last_status == 500
        assert "boom-3" in excinfo.value.last_error
        assert len(stub.calls) == 3
        # Slept twice (between attempt 1→2 and 2→3); no sleep after final.
        assert slept == [0.1, 0.2]

    def test_transport_error_retries_then_raises(self):
        factory, stub = _client_factory(
            [
                httpx.ConnectError("connection refused"),
                httpx.ConnectError("connection refused"),
                httpx.ConnectError("connection refused"),
            ]
        )
        slept: list[float] = []
        ttm = TtmIngress(
            client_factory=factory, max_retries=3, retry_base_delay=0.1, sleep=slept.append
        )
        _bind(ttm)
        with pytest.raises(IngressServerError) as excinfo:
            ttm.post_event(RUN_ID, EVENT_TASK_UPDATED, {})
        assert excinfo.value.attempts == 3
        assert "ConnectError" in excinfo.value.last_error
        assert len(stub.calls) == 3


# ---------------------------------------------------------------------------
# post_evidence — wire contract
# ---------------------------------------------------------------------------


class TestPostEvidence:
    def test_body_matches_control_plane_evidence_append_request(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"evidence_id": "evid-7"})]
        )
        ttm = TtmIngress(client_factory=factory)
        ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, BASE_URL, initial_scope_epoch=2)
        sha = "a" * 64
        evidence_id = ttm.post_evidence(
            RUN_ID,
            kind="test_results",
            subject="pytest passed",
            content_hash=sha.upper(),  # exercise lowercase normalization
            storage_ref="s3://artifacts/run-x/pytest.json",
            source_event_id="00000000-0000-0000-0000-000000000001",
            verdict="pass",
            verification_status="passed",
            evidence_id="evid-7",
        )
        assert evidence_id == "evid-7"
        url, body, _ = stub.calls[0]
        assert url == f"{BASE_URL}/api/ingress/runtime/hermes/evidence"
        assert body == {
            "evidence_id": "evid-7",
            "kind": "test_results",
            "subject": "pytest passed",
            "content_hash": sha,  # lowercased
            "storage_ref": "s3://artifacts/run-x/pytest.json",
            "expected_scope_epoch": 2,
            "source_event_id": "00000000-0000-0000-0000-000000000001",
            "verdict": "pass",
            "verification_status": "passed",
        }

    def test_evidence_id_auto_generated_when_omitted(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"evidence_id": "ignored"})]
        )
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        ttm.post_evidence(
            RUN_ID,
            kind="log",
            subject="line",
            content_hash="b" * 64,
            storage_ref="file:///tmp/x",
            source_event_id="00000000-0000-0000-0000-000000000002",
            verdict="pass",
        )
        _, body, _ = stub.calls[0]
        assert body["evidence_id"]  # non-empty UUID-ish string


# ---------------------------------------------------------------------------
# request_approval — wire contract
# ---------------------------------------------------------------------------


class TestRequestApproval:
    def test_body_matches_runtime_approval_request(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"approval_id": "apr-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        ttm.bind_run(RUN_ID, PRINCIPAL_TOKEN, BASE_URL, initial_scope_epoch=3)
        approval_id = ttm.request_approval(
            RUN_ID,
            approval_type="contract_lock",
            summary="Lock contract before write phase",
            notes_ref="docs/runs/x/contract.md",
            payload={"phase": "write"},
        )
        assert approval_id == "apr-1"
        url, body, _ = stub.calls[0]
        assert url == f"{BASE_URL}/api/ingress/runtime/hermes/approvals"
        assert body == {
            "approval_type": "contract_lock",
            "expected_scope_epoch": 3,
            "summary": "Lock contract before write phase",
            "notes_ref": "docs/runs/x/contract.md",
            "payload": {"phase": "write"},
        }


# ---------------------------------------------------------------------------
# Canonical event type registry
# ---------------------------------------------------------------------------


class TestCanonicalEvents:
    def test_canonical_set_is_complete(self):
        # Per RUNTIME-ADAPTER-CONTRACT.md §Events. Anything not in this
        # set will trigger a non_canonical WARNING but still POST.
        assert CANONICAL_EVENT_TYPES == frozenset(
            {
                "run.dispatched",
                "phase.entered",
                "phase.completed",
                "task.updated",
                "evidence.added",
                "approval.requested",
                "approval.granted",
                "approval.rejected",
                "runtime.error",
                "run.closed",
            }
        )


# ---------------------------------------------------------------------------
# bind_run_from_env — spawn-shim contract
# ---------------------------------------------------------------------------


class TestBindFromEnv:
    def test_returns_run_id_and_binds(self):
        ttm = TtmIngress()
        env = {
            ENV_RUN_ID: "run-from-env",
            ENV_PRINCIPAL_TOKEN: "tok_envenv_envenv_envenv_envenv",
            ENV_INGRESS_BASE_URL: "http://127.0.0.1:8000",
        }
        run_id = ttm.bind_run_from_env(env)
        assert run_id == "run-from-env"
        assert ttm.is_bound("run-from-env")

    def test_returns_none_when_required_var_missing(self):
        ttm = TtmIngress()
        # Missing TTM_INGRESS_BASE_URL — must return None, not raise.
        env = {
            ENV_RUN_ID: "run-x",
            ENV_PRINCIPAL_TOKEN: "tok_x",
        }
        assert ttm.bind_run_from_env(env) is None
        assert not ttm.is_bound("run-x")

    def test_empty_env_returns_none(self):
        ttm = TtmIngress()
        assert ttm.bind_run_from_env({}) is None

    def test_picks_up_optional_runtime_id_and_scope_epoch(self):
        factory, stub = _client_factory(
            [_StubResponse(201, {"event_id": "ev-1"})]
        )
        ttm = TtmIngress(client_factory=factory)
        env = {
            ENV_RUN_ID: "run-y",
            ENV_PRINCIPAL_TOKEN: "tok_yyyy_yyyy_yyyy_yyyy",
            ENV_INGRESS_BASE_URL: "http://127.0.0.1:8000",
            ENV_RUNTIME_ID: "hermes-shadow",
            ENV_SCOPE_EPOCH: "7",
        }
        ttm.bind_run_from_env(env)
        ttm.post_event("run-y", EVENT_TASK_UPDATED, {})
        url, body, headers = stub.calls[0]
        assert "/runtime/hermes-shadow/" in url
        assert headers["X-Runtime-Id"] == "hermes-shadow"
        assert body["expected_scope_epoch"] == 7

    def test_invalid_scope_epoch_raises(self):
        ttm = TtmIngress()
        env = {
            ENV_RUN_ID: "run-z",
            ENV_PRINCIPAL_TOKEN: "tok_z",
            ENV_INGRESS_BASE_URL: "http://127.0.0.1:8000",
            ENV_SCOPE_EPOCH: "not-an-int",
        }
        with pytest.raises(ValueError, match="TTM_SCOPE_EPOCH"):
            ttm.bind_run_from_env(env)

    def test_defaults_to_os_environ(self, monkeypatch):
        ttm = TtmIngress()
        monkeypatch.setenv(ENV_RUN_ID, "run-os")
        monkeypatch.setenv(ENV_PRINCIPAL_TOKEN, "tok_from_os_environ_xxxxx")
        monkeypatch.setenv(ENV_INGRESS_BASE_URL, "http://127.0.0.1:8000")
        run_id = ttm.bind_run_from_env()
        assert run_id == "run-os"


# ---------------------------------------------------------------------------
# get_run_state
# ---------------------------------------------------------------------------


class TestGetRunState:
    _state_body = {
        "run_id": RUN_ID,
        "stream_id": "galactus",
        "stream_version": "v1",
        "scope_epoch": 3,
        "status": "active",
        "execution_contract": {
            "required_tests": ["test_a"],
            "approval_policy": ["scope"],
        },
        "approvals": [
            {"approval_id": "ap-1", "approval_type": "scope", "status": "requested", "scope_epoch": 3}
        ],
    }

    def test_returns_state_dict_and_updates_scope_epoch(self):
        factory, stub = _client_factory([_StubResponse(200, self._state_body)])
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        result = ttm.get_run_state(RUN_ID)
        assert result["scope_epoch"] == 3
        assert result["status"] == "active"
        # scope_epoch auto-applied to binding
        assert ttm._binding(RUN_ID).scope_epoch == 3
        # Wire check: GET to the correct path with auth headers
        url, _, headers = stub.calls[0]
        assert f"/runs/{RUN_ID}/state" in url
        assert headers["Authorization"] == f"Bearer {PRINCIPAL_TOKEN}"
        assert headers["X-Run-Id"] == RUN_ID

    def test_401_raises_ingress_auth_error(self):
        factory, _ = _client_factory([_StubResponse(401, text="token_revoked")])
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        with pytest.raises(IngressAuthError):
            ttm.get_run_state(RUN_ID)

    def test_5xx_retries_then_raises_server_error(self):
        slept = []
        factory, stub = _client_factory(
            [_StubResponse(503)] * 3
        )
        ttm = TtmIngress(client_factory=factory, max_retries=3, sleep=slept.append)
        _bind(ttm)
        with pytest.raises(IngressServerError):
            ttm.get_run_state(RUN_ID)
        assert len(stub.calls) == 3
        assert len(slept) == 2

    def test_scope_epoch_update_skipped_if_field_missing(self):
        body_no_epoch = {k: v for k, v in self._state_body.items() if k != "scope_epoch"}
        factory, _ = _client_factory([_StubResponse(200, body_no_epoch)])
        ttm = TtmIngress(client_factory=factory)
        _bind(ttm)
        ttm.get_run_state(RUN_ID)
        # epoch stays at initial value (1) — no crash
        assert ttm._binding(RUN_ID).scope_epoch == 1

    def test_not_bound_raises(self):
        ttm = TtmIngress()
        with pytest.raises(IngressNotBoundError):
            ttm.get_run_state("unbound-run")
