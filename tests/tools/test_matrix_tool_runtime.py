"""Owner-loop loss bounds Matrix requests without hiding dispatched writes."""

import asyncio
import json
import threading

import pytest

from agent.memory_provider import spawn_context_thread
from gateway.config import PlatformConfig
from gateway.session_context import clear_session_vars, set_session_vars
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools import matrix_tool_runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["admission", "preflight", "dispatch", "confirmed", "recovery"]
)
async def test_owner_loop_loss_preserves_outcomes_and_prevents_late_writes(
    monkeypatch, stage
):
    owner_loop = asyncio.new_event_loop()
    ready = threading.Event()
    stopped = threading.Event()
    restart = threading.Event()
    finished = threading.Event()
    release = asyncio.Event()
    clock = [0.0]
    events = []
    monkeypatch.setattr(matrix_tool_runtime, "_monotonic", lambda: clock[0])

    def owner():
        asyncio.set_event_loop(owner_loop)
        if stage != "admission":
            owner_loop.call_soon(ready.set)
            owner_loop.run_forever()
        stopped.set()
        restart.wait(15)

        async def cleanup():
            release.set()
            await asyncio.sleep(0)
            tasks = asyncio.all_tasks(owner_loop) - {asyncio.current_task()}
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        owner_loop.run_until_complete(cleanup())
        owner_loop.close()
        finished.set()

    adapter = MatrixAdapter(
        PlatformConfig(enabled=True, extra={"user_id": "@bot:server"})
    )
    tokens = set_session_vars(
        platform="matrix",
        chat_id="!room:server",
        user_id="@alice:server",
        transport_adapter=adapter,
        transport_loop=owner_loop,
    )
    thread = spawn_context_thread(owner, name="matrix-owner-lifecycle", daemon=True)
    thread.start()
    if stage != "admission":
        assert await asyncio.to_thread(ready.wait, 3)

    def request(interrupted, before_write):
        async def operation():
            events.append("entered")
            try:
                if stage in {"dispatch", "confirmed"}:
                    before_write()
                    events.append("write")
                clock[0] = 31.0
                owner_loop.call_soon(owner_loop.stop)
                if stage == "confirmed":
                    return {"success": True, "event_id": "$confirmed"}
                await release.wait()
                if not interrupted():
                    before_write()
                    events.append("write")
                return {"success": True, "event_id": "$late"}
            finally:
                events.append("drained")

        return operation()

    pending = asyncio.create_task(
        matrix_tool_runtime.run_matrix_mutation(
            owner_loop,
            request,
            operation_label="Matrix lifecycle request",
            next_step="Check the event before retrying",
        )
    )
    forced_restart = False
    try:
        assert await asyncio.to_thread(stopped.wait, 3)
        if stage == "admission":
            clock[0] = 31.0
        try:
            raw = await asyncio.wait_for(asyncio.shield(pending), timeout=3)
        except asyncio.TimeoutError:
            forced_restart = True
            restart.set()
            raw = await pending
        result = json.loads(raw)
        expected = {"error": "Matrix gateway loop is unavailable"}
        if stage == "dispatch":
            expected = {
                "error": "Matrix gateway loop is unavailable after the change was sent to the homeserver",
                "outcome": "unknown",
                "next_step": "Check the event before retrying",
            }
        if stage == "confirmed":
            expected = {"success": True, "event_id": "$confirmed"}
        assert (result, forced_restart, events.count("write"), "drained" in events) == (
            expected,
            False,
            int(stage in {"dispatch", "confirmed"}),
            stage == "confirmed",
        )
    finally:
        restart.set()
        await pending
        assert await asyncio.to_thread(finished.wait, 3)
        thread.join()
        clear_session_vars(tokens)
    assert events.count("write") == int(stage in {"dispatch", "confirmed"})
