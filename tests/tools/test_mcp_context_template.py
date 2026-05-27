"""Tests for ``${context:NAME}`` per-request interpolation in MCP-headers config.

Exercises both the resolution helpers (``_has_context_template``,
``_resolve_context_templates``, ``_split_static_and_templated_headers``)
and the per-request injection behavior via a real ``httpx.AsyncClient``
event hook — the latter is the contract that matters for downstream
consumers (Mercator's delegated-principal use case).
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import httpx
import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    register_session_context_var,
)
from tools.mcp_tool import (
    _has_context_template,
    _resolve_context_templates,
    _split_static_and_templated_headers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def custom_var():
    """Register a fresh ContextVar each test under a stable name."""
    var: ContextVar = ContextVar("TEST_PRINCIPAL", default=_UNSET)
    register_session_context_var("TEST_PRINCIPAL", var)
    try:
        yield var
    finally:
        var.set(_UNSET)
        _VAR_MAP.pop("TEST_PRINCIPAL", None)


# ---------------------------------------------------------------------------
# Unit tests: detection + resolution helpers
# ---------------------------------------------------------------------------

class TestHasContextTemplate:
    def test_detects_simple_template(self):
        assert _has_context_template("${context:FOO}")

    def test_detects_template_with_surrounding_text(self):
        assert _has_context_template("Bearer ${context:JWT}")

    def test_rejects_env_var_style(self):
        # Env vars are config-load-time, not per-request.
        assert not _has_context_template("${PLAIN_ENV}")

    def test_rejects_literal_string(self):
        assert not _has_context_template("plain-static-value")

    def test_rejects_non_string(self):
        assert not _has_context_template(None)
        assert not _has_context_template(42)
        assert not _has_context_template({"x": "${context:Y}"})


class TestResolveContextTemplates:
    def test_resolves_set_value(self, custom_var):
        custom_var.set("alice@example.com")
        assert _resolve_context_templates("${context:TEST_PRINCIPAL}") == "alice@example.com"

    def test_unset_resolves_to_empty(self, custom_var):
        # ContextVar default is _UNSET — resolver falls back to os.environ
        # (also unset for TEST_PRINCIPAL) and finally returns "".
        assert _resolve_context_templates("${context:TEST_PRINCIPAL}") == ""

    def test_mixed_literal_and_template(self, custom_var):
        custom_var.set("opaque-token-xyz")
        assert (
            _resolve_context_templates("Bearer ${context:TEST_PRINCIPAL}")
            == "Bearer opaque-token-xyz"
        )

    def test_multiple_templates_in_one_string(self, custom_var):
        # Register a second var inline for this test.
        other: ContextVar = ContextVar("TEST_OTHER", default=_UNSET)
        register_session_context_var("TEST_OTHER", other)
        try:
            custom_var.set("a")
            other.set("b")
            result = _resolve_context_templates(
                "${context:TEST_PRINCIPAL}|${context:TEST_OTHER}"
            )
            assert result == "a|b"
        finally:
            other.set(_UNSET)
            _VAR_MAP.pop("TEST_OTHER", None)

    def test_non_string_passes_through(self):
        assert _resolve_context_templates(42) == 42  # type: ignore[arg-type]

    def test_unknown_name_resolves_to_empty(self):
        # No var registered for THIS_NAME — falls back to os.environ,
        # which is unset in tests, so empty string.
        assert _resolve_context_templates("${context:NEVER_REGISTERED}") == ""


class TestSplitStaticAndTemplatedHeaders:
    def test_partitions_correctly(self):
        headers = {
            "Authorization": "Bearer static-token",
            "X-Delegated-Principal": "${context:PRINCIPAL_EMAIL}",
            "X-Other": "static-other",
            "X-Mixed": "prefix-${context:NAME}-suffix",
        }
        static, templated = _split_static_and_templated_headers(headers)
        assert static == {
            "Authorization": "Bearer static-token",
            "X-Other": "static-other",
        }
        assert templated == {
            "X-Delegated-Principal": "${context:PRINCIPAL_EMAIL}",
            "X-Mixed": "prefix-${context:NAME}-suffix",
        }

    def test_empty_input(self):
        assert _split_static_and_templated_headers({}) == ({}, {})

    def test_non_string_values_treated_as_static(self):
        # Edge case: a header value that's not a string (unusual but
        # possible from raw YAML) is treated as static — we don't try
        # to substitute on non-strings.
        headers = {"X-Number": 42, "X-Bool": True}
        static, templated = _split_static_and_templated_headers(headers)
        assert static == {"X-Number": 42, "X-Bool": True}
        assert templated == {}


# ---------------------------------------------------------------------------
# Integration test: the actual event-hook injection pattern using httpx
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_request_header_injection_via_event_hook(custom_var):
    """The pattern used inside ``_run_http`` — install a request event
    hook on httpx.AsyncClient that resolves ``${context:NAME}`` templates
    against the *calling task's* session context.  Verifies the per-request
    contract end-to-end (no MCP-stack mocking required)."""
    captured_headers: dict[str, str] = {}

    async def _capture_handler(request: httpx.Request) -> httpx.Response:
        captured_headers.clear()
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"ok": True})

    templated = {"X-Delegated-Principal": "${context:TEST_PRINCIPAL}"}

    async def _inject(request: httpx.Request) -> None:
        for name, template in templated.items():
            resolved = _resolve_context_templates(template)
            if resolved:
                request.headers[name] = resolved
            elif name in request.headers:
                del request.headers[name]

    transport = httpx.MockTransport(_capture_handler)
    async with httpx.AsyncClient(
        transport=transport,
        event_hooks={"request": [_inject]},
        headers={"Authorization": "Bearer static-token"},
    ) as client:
        # Request 1: principal "alice" — both headers present.
        custom_var.set("alice@example.com")
        await client.get("http://example.invalid/mcp")
        assert captured_headers["x-delegated-principal"] == "alice@example.com"
        assert captured_headers["authorization"] == "Bearer static-token"

        # Request 2: principal "bob" — verifies per-request resolution
        # (the same client emits a different header value on the next call).
        custom_var.set("bob@example.com")
        await client.get("http://example.invalid/mcp")
        assert captured_headers["x-delegated-principal"] == "bob@example.com"

        # Request 3: principal cleared — header is dropped (NOT sent empty).
        custom_var.set("")
        await client.get("http://example.invalid/mcp")
        assert "x-delegated-principal" not in captured_headers


@pytest.mark.asyncio
async def test_asyncio_task_isolation_for_per_request_headers(custom_var):
    """Two concurrent asyncio tasks setting different principals must each
    see their own value on outbound requests — no cross-task leak."""
    seen_per_task: dict[str, str] = {}

    async def _record_handler(request: httpx.Request) -> httpx.Response:
        # Pull the calling task's principal out of the request header.
        seen_per_task[request.url.path.lstrip("/")] = request.headers.get(
            "x-delegated-principal", "<missing>"
        )
        return httpx.Response(200)

    async def _inject(request: httpx.Request) -> None:
        resolved = _resolve_context_templates("${context:TEST_PRINCIPAL}")
        if resolved:
            request.headers["X-Delegated-Principal"] = resolved

    transport = httpx.MockTransport(_record_handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"request": [_inject]},
    ) as client:
        async def handler(path: str, principal: str, delay: float):
            custom_var.set(principal)
            await asyncio.sleep(delay)
            await client.get(f"http://example.invalid/{path}")

        await asyncio.gather(
            handler("alice-path", "alice@example.com", 0.02),
            handler("bob-path",   "bob@example.com",   0.01),
        )

    assert seen_per_task == {
        "alice-path": "alice@example.com",
        "bob-path":   "bob@example.com",
    }
