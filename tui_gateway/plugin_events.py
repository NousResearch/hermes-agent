"""Serve-owned typed plugin events for one exact, connected native session."""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

_current_plugin_turn = ContextVar("desktop_plugin_turn", default=None)


@contextmanager
def desktop_plugin_turn(row, internal=False):
    lease = {"row": row, "active": True, "internal": internal}
    token = _current_plugin_turn.set(lease)
    try:
        yield
    finally:
        lease["active"] = False
        _current_plugin_turn.reset(token)

from gateway.internal_events import (
    GatewaySystemEvent, gateway_system_event_is_eligible,
    new_gateway_event_receipt, resolve_gateway_event_receipt,
)


class DesktopPluginConsumer:
    def __init__(self, manager, server):
        self.manager = manager
        self.server = server
        self.generation = uuid.uuid4().hex
        self.active = True

    def _resolve(self, destination):
        if not self.active or getattr(self.manager, "_desktop_consumer", None) is not self:
            return None
        if not isinstance(destination, dict) or set(destination) != {"profile_name", "session_id"}:
            return None
        if destination["profile_name"] != "default":
            return None
        with self.server._sessions_lock:
            matches = [(sid, row) for sid, row in self.server._sessions.items()
                       if getattr(row.get("agent"), "session_id", None) == destination["session_id"]
                       and Path(row.get("profile_home") or self.manager.home_path).resolve()
                       == Path(self.manager.home_path).resolve()
                       and getattr(row.get("agent"), "api_mode", None) == "codex_responses"
                       and getattr(row.get("agent"), "provider", None) == "openai-codex"
                       and not row.get("_closing")
                       and self.server._session_has_live_transport(row)]
            return matches[0] if len(matches) == 1 else None

    def readiness(self, destination):
        found = self._resolve(destination)
        if found is None:
            return {"ready": False, "status": "session_mismatch"}
        sid, row = found
        if row.get("_turn_cancel_requested"):
            return {"ready": False, "status": "cancelled"}
        return {"ready": True, "status": "ready", "generation": self.generation,
                "native_session_id": sid, "destination": dict(destination), "busy": bool(row.get("running"))}

    def current_destination(self):
        from agent.delegation_context import is_delegated_child_context
        row = self.server._current_runtime_session_record.get()
        lease = _current_plugin_turn.get()
        if (row is None or not lease or lease["row"] is not row or not lease["active"]
                or lease["internal"] or is_delegated_child_context()):
            return None
        target = {"profile_name": "default", "session_id": getattr(row.get("agent"), "session_id", None)}
        found = self._resolve(target)
        if found is None or found[1] is not row:
            return None
        return target

    def inject(self, content, event):
        receipt = new_gateway_event_receipt()
        def finish(status):
            resolve_gateway_event_receipt(receipt, status, event=event)
        if not isinstance(event, GatewaySystemEvent) or event.desktop_destination is None:
            finish("unauthorized")
            return receipt
        target = dataclasses.asdict(event.desktop_destination)
        if not self.active or getattr(self.manager, "_desktop_consumer", None) is not self:
            finish("cancelled")
            return receipt
        found = self._resolve(target)
        if found is None:
            # Unadmitted queued work may wait for an ordinary same-physical resume.
            # Old owners above are terminal; a live owner with no connected record
            # reports a bounded retry instead of discarding the durable event.
            finish("stopping")
            return receipt
        sid, row = found
        live_peers = getattr(self.server, "_session_live_transports", lambda record: [record.get("transport")])
        original_peers = tuple(live_peers(row))
        cancel_generation = int(row.get("_queued_prompt_generation", 0))
        def same_transport():
            return any(peer is original for peer in live_peers(row) for original in original_peers)
        def same_generation():
            return not row.get("_turn_cancel_requested") and int(row.get("_queued_prompt_generation", 0)) == cancel_generation
        def eligible():
            current = self._resolve(target)
            return current is not None and current[0] == sid and current[1] is row and same_transport() and same_generation() and gateway_system_event_is_eligible(event)
        with self.server._sessions_lock, row["history_lock"]:
            if not eligible():
                finish("cancelled")
                return receipt
            if row.get("running"):
                finish("busy")
                return receipt
            # An earlier process may have persisted the developer input before losing its
            # receipt. Never run the same event again just because its answer is uncertain.
            if any(isinstance(m, dict) and m.get("role") == "developer"
                   and (m.get("display_metadata") or {}).get("event_id") == event.event_id
                   and (m.get("display_metadata") or {}).get("plugin_id") == event.plugin_id
                   for m in row.get("history", [])):
                finish("cancelled")
                return receipt
            row["running"] = True
        guarded = dataclasses.replace(event, eligibility_check=eligible)
        def terminal_authorized():
            current = self._resolve(target)
            try:
                authorized = event.completion_check() is True if callable(event.completion_check) else gateway_system_event_is_eligible(event)
            except Exception:
                authorized = False
            return bool(current is not None and current[0] == sid and current[1] is row
                        and same_transport() and same_generation() and authorized)
        def write_terminal(payload):
            if not terminal_authorized():
                return False
            from tui_gateway.event_replay import _stamp_event
            from tui_gateway.transport import FanoutTransport
            frame = self.server._event_frame("message.complete", sid, payload)
            _stamp_event(frame)
            transport = row.get("transport")
            if isinstance(transport, FanoutTransport):
                return transport.write_confirmed(frame, peers=original_peers)
            writer = getattr(transport, "write_confirmed", None)
            return callable(writer) and writer(frame) is True
        def terminal(result):
            if not terminal_authorized():
                finish("cancelled")
            else:
                finish({"settled": "completed", "cancelled": "cancelled"}.get(result.get("status"), "agent_error"))
        try:
            started = self.server._run_prompt_submit(
                f"__plugin__{event.receipt_id}", sid, row, content, image_paths=[],
                gateway_system_event=guarded, terminal_callback=terminal,
                queued_prompt_generation=cancel_generation, terminal_frame_writer=write_terminal)
            if not started:
                finish("session_mismatch")
        except Exception:
            with row["history_lock"]:
                row["running"] = False
            finish("agent_error")
        return receipt


async def start_desktop_plugins():
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from tui_gateway import server
    discover_plugins()
    manager = get_plugin_manager()
    consumer = DesktopPluginConsumer(manager, server)
    manager._desktop_consumer = consumer
    tasks = {name: asyncio.create_task(factory(), name=name)
             for name, factory in getattr(manager, "_desktop_task_factories", {}).items()}
    manager._desktop_tasks = tasks
    return consumer, list(tasks.values())


async def stop_desktop_plugins(runtime):
    consumer, tasks = runtime
    consumer.active = False
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if getattr(consumer.manager, "_desktop_consumer", None) is consumer:
        consumer.manager._desktop_consumer = None
