"""Matrix sends run on their gateway owner's event loop."""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch
from gateway.config import Platform
from tools.send_message_senders import _send_matrix_via_adapter


def test_live_adapter_send_runs_on_gateway_owner_loop():
    """Control-socket sends must not use the request thread's event loop."""
    import threading

    owner_loop = asyncio.new_event_loop()
    owner_ready = threading.Event()

    def _run_owner_loop():
        asyncio.set_event_loop(owner_loop)
        owner_ready.set()
        owner_loop.run_forever()

    owner_thread = threading.Thread(target=_run_owner_loop, daemon=True)
    owner_thread.start()
    assert owner_ready.wait(timeout=2.0)

    observed_loops = []

    class LiveAdapter:
        async def send(self, chat_id, message, metadata=None):
            observed_loops.append(asyncio.get_running_loop())
            return SimpleNamespace(success=True, message_id="$encrypted")

    fake_runner = SimpleNamespace(
        adapters={Platform.MATRIX: LiveAdapter()},
        _gateway_loop=owner_loop,
    )

    try:
        with patch(
            "gateway.run._gateway_runner_ref",
            return_value=fake_runner,
        ), patch.dict(
            sys.modules,
            {"plugins.platforms.matrix.adapter": SimpleNamespace()},
        ):
            result = asyncio.run(
                _send_matrix_via_adapter(
                    SimpleNamespace(enabled=True, token="tok", extra={}),
                    "!room:example.com",
                    "hello",
                )
            )
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=2.0)
        owner_loop.close()

    assert result["success"] is True
    assert result["message_id"] == "$encrypted"
    assert observed_loops == [owner_loop]
