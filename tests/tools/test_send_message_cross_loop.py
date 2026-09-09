"""Regression tests for the cross-event-loop deadlock fix in send_message.

When the agent's tool worker thread calls _send_via_adapter() while the
adapter's queues live on the gateway's main event loop, the send must be
dispatched via run_coroutine_threadsafe to the gateway loop — NOT awaited
directly on the worker loop (which would deadlock due to the selector never
being woken by cross-thread future.set_result).
"""

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from gateway.config import Platform


class TestSendViaAdapterCrossLoopDispatch:

    @pytest.mark.asyncio
    async def test_cross_loop_dispatches_to_gateway_loop(self, monkeypatch):
        """adapter.send() runs on gateway loop, not the caller's loop."""
        from tools.send_message_tool import _send_via_adapter

        send_loop_id = {}
        platform = Platform("wecom")

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                send_loop_id["loop"] = id(asyncio.get_running_loop())
                return SimpleNamespace(success=True, message_id="cross-ok")

        gateway_loop = asyncio.new_event_loop()
        started = threading.Event()

        def run_gateway():
            asyncio.set_event_loop(gateway_loop)
            started.set()
            gateway_loop.run_forever()

        t = threading.Thread(target=run_gateway, daemon=True)
        t.start()
        started.wait(timeout=2)

        try:
            runner = SimpleNamespace(
                adapters={platform: FakeAdapter()},
                _gateway_loop=gateway_loop,
            )
            fake_gateway_run = ModuleType("gateway.run")
            fake_gateway_run._gateway_runner_ref = lambda: runner
            monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

            result = await _send_via_adapter(
                platform,
                SimpleNamespace(extra={}),
                "wr_group_123",
                "hello from worker",
            )

            assert result == {"success": True, "message_id": "cross-ok"}
            # Verify send() ran on the gateway loop, not our current loop
            assert send_loop_id["loop"] == id(gateway_loop)
        finally:
            gateway_loop.call_soon_threadsafe(gateway_loop.stop)
            t.join(timeout=2)
            gateway_loop.close()

    @pytest.mark.asyncio
    async def test_same_loop_uses_direct_await(self, monkeypatch):
        """When current loop IS the gateway loop, adapter.send() is awaited
        directly — no run_coroutine_threadsafe (which would self-lock)."""
        from tools.send_message_tool import _send_via_adapter

        current_loop = asyncio.get_running_loop()
        platform = Platform("wecom")
        called_directly = {}

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                called_directly["loop"] = id(asyncio.get_running_loop())
                return SimpleNamespace(success=True, message_id="direct-ok")

        runner = SimpleNamespace(
            adapters={platform: FakeAdapter()},
            _gateway_loop=current_loop,
        )
        fake_gateway_run = ModuleType("gateway.run")
        fake_gateway_run._gateway_runner_ref = lambda: runner
        monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

        result = await _send_via_adapter(
            platform,
            SimpleNamespace(extra={}),
            "wr_group_456",
            "direct send",
        )

        assert result == {"success": True, "message_id": "direct-ok"}
        assert called_directly["loop"] == id(current_loop)

    @pytest.mark.asyncio
    async def test_gateway_loop_not_running_returns_error(self, monkeypatch):
        """When gateway loop exists but is stopped, return an error rather
        than attempting direct await on a loop-bound adapter."""
        from tools.send_message_tool import _send_via_adapter

        stopped_loop = asyncio.new_event_loop()
        stopped_loop.close()
        platform = Platform("wecom")

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                raise AssertionError("should not be called")

        runner = SimpleNamespace(
            adapters={platform: FakeAdapter()},
            _gateway_loop=stopped_loop,
        )
        fake_gateway_run = ModuleType("gateway.run")
        fake_gateway_run._gateway_runner_ref = lambda: runner
        monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

        result = await _send_via_adapter(
            platform,
            SimpleNamespace(extra={}),
            "wr_group_789",
            "should fail",
        )

        assert "error" in result
        assert "not running" in result["error"]

    @pytest.mark.asyncio
    async def test_shield_prevents_cancel_of_enqueued_send(self, monkeypatch):
        """asyncio.shield ensures that cancelling the caller does NOT cancel
        the already-dispatched send on the gateway loop."""
        from tools.send_message_tool import _send_via_adapter

        send_completed = asyncio.Event()
        send_result_holder = {}
        platform = Platform("wecom")

        class FakeAdapter:
            async def send(self, *, chat_id, content, metadata=None):
                # Simulate a slow send (token bucket wait)
                await asyncio.sleep(0.3)
                send_result_holder["sent"] = True
                send_completed.set()
                return SimpleNamespace(success=True, message_id="shielded")

        gateway_loop = asyncio.new_event_loop()
        started = threading.Event()

        def run_gateway():
            asyncio.set_event_loop(gateway_loop)
            started.set()
            gateway_loop.run_forever()

        t = threading.Thread(target=run_gateway, daemon=True)
        t.start()
        started.wait(timeout=2)

        try:
            runner = SimpleNamespace(
                adapters={platform: FakeAdapter()},
                _gateway_loop=gateway_loop,
            )
            fake_gateway_run = ModuleType("gateway.run")
            fake_gateway_run._gateway_runner_ref = lambda: runner
            monkeypatch.setitem(sys.modules, "gateway.run", fake_gateway_run)

            # Start the send, then cancel the caller task after a short delay
            async def do_send():
                return await _send_via_adapter(
                    platform,
                    SimpleNamespace(extra={}),
                    "wr_group_shield",
                    "shielded msg",
                )

            task = asyncio.create_task(do_send())
            await asyncio.sleep(0.1)  # let it dispatch to gateway loop
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            # The send on the gateway loop should still complete despite cancel
            fut = asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(send_completed.wait(), timeout=1.0),
                gateway_loop,
            )
            fut.result(timeout=2)
            assert send_result_holder.get("sent") is True
        finally:
            gateway_loop.call_soon_threadsafe(gateway_loop.stop)
            t.join(timeout=2)
            gateway_loop.close()
