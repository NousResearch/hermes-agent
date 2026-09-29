"""Config invalidation through real WS parsing/dispatch with in-memory peers."""

import asyncio
import json
import os
from pathlib import Path

from tui_gateway import server
from tui_gateway import ws as ws_mod


class ConfigPeer:
    """Only the ASGI socket boundary is fake; frames cross as serialized JSON."""

    def __init__(self):
        self.inbound = asyncio.Queue()
        self.outbound = asyncio.Queue()

    async def accept(self):
        pass

    async def close(self):
        pass

    async def send_text(self, text):
        self.outbound.put_nowait(json.loads(text))

    async def receive_text(self):
        text = await self.inbound.get()
        if text is None:
            raise ws_mod._WebSocketDisconnect()
        return text

    async def receive(self):
        return await asyncio.wait_for(self.outbound.get(), 5)

    async def config_get(self, key, request_id):
        self.inbound.put_nowait(json.dumps({
            "jsonrpc": "2.0", "id": request_id,
            "method": "config.get", "params": {"key": key},
        }))
        frame = await self.receive()
        assert frame.get("id") == request_id, frame
        assert "error" not in frame, frame
        return frame["result"]


def test_config_change_rehydrates_two_clients_through_ws_parser(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(server, "_live_transports", set())
    monkeypatch.setattr(server, "_sessions", {})
    for name in ("_change_sigs", "_change_checked_at", "_change_broadcast_at",
                 "_sessions_db_sig_cache"):
        monkeypatch.setattr(server, name, {})
    monkeypatch.setattr(server, "_pairing_roots_cache", None)
    monkeypatch.setattr(server, "_bot_relay_outbox_seen", 0)
    # Tick the production watcher explicitly; do not leave process-wide watchers
    # or unrelated maintenance running after this connection-lifecycle test.
    for name in ("_ensure_skin_watcher", "_ensure_lease_watcher",
                 "_start_backend_heartbeat_refresher", "_schedule_startup_orphan_sweep"):
        monkeypatch.setattr(server, name, lambda: None)
    config = home / "config.yaml"
    config.write_text("display: {tui_theme: dark, tui_compact: false}\n", encoding="utf-8")

    async def scenario():
        peers = [ConfigPeer(), ConfigPeer()]
        tasks = [asyncio.create_task(ws_mod.handle_ws(peer)) for peer in peers]
        try:
            for index, peer in enumerate(peers):
                assert (await peer.receive())["params"]["type"] == "gateway.ready"
                # A handled request is a registration barrier, unlike ready alone.
                assert (await peer.config_get("mtime", f"capability-{index}"))["change_events"] is True
                assert (await peer.config_get("theme", f"initial-{index}"))["value"] == "dark"
            assert len(server._live_transports) == len(peers)
            await asyncio.to_thread(server._broadcast_watched_changes, now=0)
            for tick, theme, compact, density in ((2, "light", "true", "on"), (4, "dark", "false", "off")):
                config.write_text(
                    f"display: {{tui_theme: {theme}, tui_compact: {compact}}}\n"
                    "api_key: never-broadcast-this\n", encoding="utf-8")
                os.utime(config, (2_000_000_000 + tick, 2_000_000_000 + tick))
                await asyncio.to_thread(server._broadcast_watched_changes, now=tick)
                for index, peer in enumerate(peers):
                    assert await peer.receive() == {
                        "jsonrpc": "2.0", "method": "event",
                        "params": {"type": "config.changed", "session_id": "", "payload": {}},
                    }
                    assert (await peer.config_get("theme", f"theme-{tick}-{index}"))["value"] == theme
                    assert (await peer.config_get("density", f"density-{tick}-{index}"))["value"] == density
                await asyncio.to_thread(server._broadcast_watched_changes, now=tick + 1)
                for index, peer in enumerate(peers):
                    # Ordered response barrier catches duplicate events and replies
                    # accidentally broadcast from the other client's requests.
                    assert (await peer.config_get("theme", f"barrier-{tick}-{index}"))["value"] == theme
        finally:
            for peer in peers:
                peer.inbound.put_nowait(None)
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert not server._live_transports

    asyncio.run(scenario())
