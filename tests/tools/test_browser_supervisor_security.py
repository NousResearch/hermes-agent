"""Security-focused unit tests for the browser CDP supervisor bridge."""

from __future__ import annotations

import pytest

from tools.browser_supervisor import (
    CDPSupervisor,
    DIALOG_BRIDGE_HOST,
    DIALOG_TEXT_MAX_CHARS,
    PENDING_DIALOGS_MAX,
)


def _bridge_params(url: str, request_id: str = "req-1") -> dict:
    return {"requestId": request_id, "request": {"url": url}}


@pytest.mark.asyncio
async def test_bridge_rejects_forged_magic_host_without_token():
    """A page request to the public bridge host is ignored without the token."""
    supervisor = CDPSupervisor("security-test", "ws://unused")
    cdp_calls: list[tuple[str, dict, str | None]] = []

    async def fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        cdp_calls.append((method, params or {}, session_id))
        return {}

    supervisor._cdp = fake_cdp  # type: ignore[method-assign]

    await supervisor._on_fetch_paused(
        _bridge_params(
            f"http://{DIALOG_BRIDGE_HOST}/?kind=alert&message=forged",
            request_id="forged-1",
        ),
        session_id="sid-1",
    )

    assert supervisor.snapshot().pending_dialogs == ()
    assert cdp_calls
    assert cdp_calls[0][0] == "Fetch.fulfillRequest"
    assert cdp_calls[0][1]["requestId"] == "forged-1"
    assert cdp_calls[0][1]["responseCode"] == 403


@pytest.mark.asyncio
async def test_bridge_accepts_tokenized_request_and_truncates_fields():
    """Only tokenized bridge URLs become dialogs, with bounded text fields."""
    supervisor = CDPSupervisor("security-test", "ws://unused")
    cdp_calls: list[tuple[str, dict, str | None]] = []

    async def fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        cdp_calls.append((method, params or {}, session_id))
        return {}

    supervisor._cdp = fake_cdp  # type: ignore[method-assign]
    long_message = "x" * (DIALOG_TEXT_MAX_CHARS + 100)
    url = (
        f"http://{DIALOG_BRIDGE_HOST}/{supervisor._bridge_token}/"
        f"?kind=prompt&message={long_message}&default_prompt={long_message}"
    )

    await supervisor._on_fetch_paused(_bridge_params(url), session_id="sid-1")

    dialogs = supervisor.snapshot().pending_dialogs
    assert len(dialogs) == 1
    assert dialogs[0].type == "prompt"
    assert len(dialogs[0].message) == DIALOG_TEXT_MAX_CHARS + 1
    assert dialogs[0].message.endswith("…")
    assert len(dialogs[0].default_prompt) == DIALOG_TEXT_MAX_CHARS + 1
    assert cdp_calls == []


@pytest.mark.asyncio
async def test_bridge_overflow_is_rejected_without_growing_pending_dialogs():
    """The supervisor rejects bridge requests beyond the pending-dialog cap."""
    supervisor = CDPSupervisor("security-test", "ws://unused")
    cdp_calls: list[tuple[str, dict, str | None]] = []

    async def fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
        cdp_calls.append((method, params or {}, session_id))
        return {}

    supervisor._cdp = fake_cdp  # type: ignore[method-assign]

    for i in range(PENDING_DIALOGS_MAX + 1):
        await supervisor._on_fetch_paused(
            _bridge_params(
                f"http://{DIALOG_BRIDGE_HOST}/{supervisor._bridge_token}/"
                f"?kind=alert&message={i}",
                request_id=f"req-{i}",
            ),
            session_id="sid-1",
        )

    assert len(supervisor.snapshot().pending_dialogs) == PENDING_DIALOGS_MAX
    assert cdp_calls
    assert cdp_calls[-1][0] == "Fetch.fulfillRequest"
    assert cdp_calls[-1][1]["requestId"] == f"req-{PENDING_DIALOGS_MAX}"
    assert cdp_calls[-1][1]["responseCode"] == 403
