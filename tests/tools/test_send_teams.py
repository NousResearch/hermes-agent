"""Tests for ``_send_teams`` — outbound media path through the running
Teams adapter.

Companion to ``tests/tools/test_running_adapters.py``. The registry is
the plumbing; ``_send_teams`` is the actual outbound call site that
``send_message_tool.py`` dispatches to when a caller sends to Teams
with ``MEDIA:/path`` payloads.

Why these tests use a mock adapter rather than a real ``TeamsAdapter``:
the real adapter requires Microsoft Bot Framework auth, an Azure
service URL, and an established ``ConversationReference`` — none of
which fits a unit test. The tests verify ``_send_teams`` correctly
*dispatches* to the registered adapter's ``send_*`` methods by file
extension, which is the contract the helper owes its caller.
"""

import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _run(coro):
    return asyncio.run(coro)


def _make_send_result(success=True, message_id="msg-1", error=None):
    """SendResult-shaped object — duck-typed so we don't import the
    base class (which would drag the full plugin-registry loader)."""
    return SimpleNamespace(
        success=success,
        message_id=message_id,
        error=error,
        retryable=False,
    )


def _register_mock_adapter():
    """Register a fresh mock adapter for ``teams`` and return it."""
    from tools._running_adapters import (
        clear_running_adapters,
        set_running_adapter,
    )

    clear_running_adapters()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=_make_send_result(message_id="text-1"))
    adapter.send_image_file = AsyncMock(return_value=_make_send_result(message_id="img-1"))
    adapter.send_video = AsyncMock(return_value=_make_send_result(message_id="vid-1"))
    adapter.send_voice = AsyncMock(return_value=_make_send_result(message_id="voice-1"))
    adapter.send_document = AsyncMock(return_value=_make_send_result(message_id="doc-1"))
    set_running_adapter("teams", adapter)
    return adapter


@pytest.fixture
def tmp_pdf(tmp_path):
    """A throwaway file with a non-image extension — exercises the
    document branch."""
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    return str(path)


@pytest.fixture
def tmp_png(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n")
    return str(path)


@pytest.fixture
def tmp_mp4(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftyp")
    return str(path)


@pytest.fixture
def tmp_ogg(tmp_path):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS")
    return str(path)


def test_send_teams_routes_pdf_to_send_document(tmp_pdf):
    """A non-image, non-video, non-voice path lands on send_document."""
    from tools.send_message_tool import _send_teams

    adapter = _register_mock_adapter()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="here is the report",
            media_files=[(tmp_pdf, False)],
        )
    )

    assert result.get("success") is True
    assert result.get("platform") == "teams"
    adapter.send.assert_awaited_once()
    adapter.send_document.assert_awaited_once()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


def test_send_teams_routes_png_to_send_image_file(tmp_png):
    from tools.send_message_tool import _send_teams

    adapter = _register_mock_adapter()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="",
            media_files=[(tmp_png, False)],
        )
    )

    assert result.get("success") is True
    adapter.send_image_file.assert_awaited_once()
    adapter.send_document.assert_not_awaited()


def test_send_teams_routes_mp4_to_send_video(tmp_mp4):
    from tools.send_message_tool import _send_teams

    adapter = _register_mock_adapter()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="",
            media_files=[(tmp_mp4, False)],
        )
    )

    assert result.get("success") is True
    adapter.send_video.assert_awaited_once()


def test_send_teams_routes_ogg_voice_to_send_voice(tmp_ogg):
    """``is_voice=True`` for an ogg/opus file picks the voice path."""
    from tools.send_message_tool import _send_teams

    adapter = _register_mock_adapter()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="",
            media_files=[(tmp_ogg, True)],
        )
    )

    assert result.get("success") is True
    adapter.send_voice.assert_awaited_once()


def test_send_teams_returns_error_when_no_running_adapter(tmp_pdf):
    """If the gateway hasn't connected the Teams adapter, the helper
    returns a clear error rather than crashing or silently dropping."""
    from tools._running_adapters import clear_running_adapters
    from tools.send_message_tool import _send_teams

    clear_running_adapters()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="hi",
            media_files=[(tmp_pdf, False)],
        )
    )

    assert "error" in result
    assert "teams" in result["error"].lower()
    assert "not connected" in result["error"].lower() or "running adapter" in result["error"].lower()


def test_send_teams_returns_error_when_media_file_missing():
    """Bad media path is reported, not silently swallowed."""
    from tools.send_message_tool import _send_teams

    _register_mock_adapter()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="",
            media_files=[("/nonexistent/path.pdf", False)],
        )
    )

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_send_teams_propagates_adapter_failure(tmp_pdf):
    """If the live adapter returns ``success=False``, surface the error."""
    from tools._running_adapters import (
        clear_running_adapters,
        set_running_adapter,
    )
    from tools.send_message_tool import _send_teams

    clear_running_adapters()
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=_make_send_result(success=True))
    adapter.send_document = AsyncMock(
        return_value=_make_send_result(success=False, error="upload denied")
    )
    set_running_adapter("teams", adapter)

    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="",
            media_files=[(tmp_pdf, False)],
        )
    )

    assert "error" in result
    assert "upload denied" in result["error"]


def test_send_teams_text_only_does_not_touch_media_methods():
    """A text-only call (no MEDIA tag) just hits ``send`` once."""
    from tools.send_message_tool import _send_teams

    adapter = _register_mock_adapter()
    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="just text",
            media_files=[],
        )
    )

    assert result.get("success") is True
    adapter.send.assert_awaited_once()
    adapter.send_document.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


# ── Cross-event-loop bridge ─────────────────────────────────────────────────
#
# In production the gateway runs the TeamsAdapter on its main event loop
# (call it loop A). The agent's tool dispatch (``model_tools._run_async``)
# detects an already-running outer loop and spins up a *separate worker
# loop* in a sidecar thread (call it loop B) so ``asyncio.run_until_complete``
# can drive the coroutine without nesting loops.
#
# The Microsoft Teams SDK's ``App`` object — built on loop A at gateway
# startup — internally caches ``asyncio.Event``/``Lock`` primitives bound
# to loop A. ``_send_teams`` invoked from loop B awaiting those primitives
# raises ``RuntimeError: ... is bound to a different event loop``.
#
# The fix: ``TeamsAdapter.connect()`` records ``self._loop`` and
# ``_send_teams`` bridges to that loop via
# ``asyncio.run_coroutine_threadsafe`` when the call site loop differs.
#
# The two tests below pin both branches of that bridge:
#   - same-loop: no bridging needed, the coroutine awaits inline
#   - cross-loop: we must hop to the adapter's captured loop
def test_send_teams_bridges_to_adapter_loop_when_called_from_different_loop(tmp_pdf):
    """If the running adapter has captured its own event loop and the
    caller is on a different loop, ``_send_teams`` must dispatch the
    underlying ``adapter.send*`` coroutines onto the adapter's loop —
    otherwise SDK-internal ``asyncio.Event`` instances bound to the
    adapter's loop blow up.
    """
    from tools._running_adapters import (
        clear_running_adapters,
        set_running_adapter,
    )
    from tools.send_message_tool import _send_teams

    # 1. Spin up a dedicated "gateway loop" in a sidecar thread.
    import threading

    adapter_loop = asyncio.new_event_loop()
    loop_ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(adapter_loop)
        loop_ready.set()
        adapter_loop.run_forever()

    loop_thread = threading.Thread(target=_run_loop, daemon=True)
    loop_thread.start()
    loop_ready.wait(timeout=2.0)

    try:
        # 2. Build an adapter with send_* coroutines that *assert* they
        #    are running on adapter_loop. If _send_teams forgets to bridge,
        #    these assertions fire and the test fails.
        clear_running_adapters()

        async def _send_on_adapter_loop(*args, **kwargs):
            assert asyncio.get_running_loop() is adapter_loop, (
                "send must run on the adapter's loop, not the caller's"
            )
            return _make_send_result(message_id="text-x")

        async def _send_doc_on_adapter_loop(*args, **kwargs):
            assert asyncio.get_running_loop() is adapter_loop, (
                "send_document must run on the adapter's loop"
            )
            return _make_send_result(message_id="doc-x")

        adapter = MagicMock()
        adapter._loop = adapter_loop  # the contract _send_teams must honor
        adapter.send = AsyncMock(side_effect=_send_on_adapter_loop)
        adapter.send_document = AsyncMock(side_effect=_send_doc_on_adapter_loop)
        adapter.send_image_file = AsyncMock(return_value=_make_send_result())
        adapter.send_video = AsyncMock(return_value=_make_send_result())
        adapter.send_voice = AsyncMock(return_value=_make_send_result())
        set_running_adapter("teams", adapter)

        # 3. Drive _send_teams from a *different* loop (the test's
        #    asyncio.run() loop) — mirrors how model_tools._run_async
        #    dispatches tool calls from the gateway's worker loop.
        result = _run(
            _send_teams(
                chat_id="a:abc",
                message="here it is",
                media_files=[(tmp_pdf, False)],
            )
        )

        assert result.get("success") is True, f"got {result!r}"
        assert result.get("platform") == "teams"
        adapter.send.assert_awaited_once()
        adapter.send_document.assert_awaited_once()
    finally:
        adapter_loop.call_soon_threadsafe(adapter_loop.stop)
        loop_thread.join(timeout=2.0)
        adapter_loop.close()


def test_send_teams_uses_inline_await_when_adapter_loop_matches(tmp_pdf):
    """If ``adapter._loop`` is the *same* loop the caller is already on,
    bridging is unnecessary — _send_teams should await directly. This
    pins the same-loop fast path so we don't accidentally introduce
    a thread-hop on every call (which would be an order-of-magnitude
    latency regression).
    """
    from tools._running_adapters import (
        clear_running_adapters,
        set_running_adapter,
    )
    from tools.send_message_tool import _send_teams

    async def _drive():
        clear_running_adapters()
        adapter = MagicMock()
        adapter._loop = asyncio.get_running_loop()
        adapter.send = AsyncMock(return_value=_make_send_result(message_id="t"))
        adapter.send_document = AsyncMock(return_value=_make_send_result(message_id="d"))
        adapter.send_image_file = AsyncMock(return_value=_make_send_result())
        adapter.send_video = AsyncMock(return_value=_make_send_result())
        adapter.send_voice = AsyncMock(return_value=_make_send_result())
        set_running_adapter("teams", adapter)

        return await _send_teams(
            chat_id="a:abc",
            message="hi",
            media_files=[(tmp_pdf, False)],
        )

    result = asyncio.run(_drive())
    assert result.get("success") is True


def test_send_teams_works_when_adapter_has_no_loop_attribute(tmp_pdf):
    """Backward compat: an adapter that hasn't (or can't) capture
    ``_loop`` must still work — _send_teams should fall back to inline
    await rather than crash with AttributeError. This protects the
    Yuanbao/Mattermost-style adapters that don't yet implement the
    loop-capture protocol."""
    from tools.send_message_tool import _send_teams

    adapter = _register_mock_adapter()
    # _register_mock_adapter() builds a bare MagicMock — accessing _loop on
    # it returns a MagicMock, not a real loop. Force the attribute away so
    # the helper sees the "no loop captured" shape.
    if hasattr(adapter, "_loop"):
        del adapter._loop

    result = _run(
        _send_teams(
            chat_id="a:abc",
            message="",
            media_files=[(tmp_pdf, False)],
        )
    )

    assert result.get("success") is True
    adapter.send_document.assert_awaited_once()
