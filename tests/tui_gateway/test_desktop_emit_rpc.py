import json

from tools import desktop_ui


def test_desktop_emit_is_registered_and_emits_preview_open():
    seen = []
    desktop_ui.set_emitter(lambda sid, event, payload: seen.append((sid, event, payload.get("url"))))
    try:
        from tui_gateway import server

        table = getattr(server, "_methods", None) or {}
        handler = table.get("desktop.emit")
        assert handler is not None, sorted(table)[:5]

        reply = handler("1", {
            "event": "preview.open",
            "payload": {"url": "https://admin.mistral.ai/organization/usage", "label": "Mistral console"},
            "session_id": "win-1",
        })
        assert isinstance(reply, dict), reply
        assert seen and seen[-1][0] == "win-1" and seen[-1][1] == "preview.open", (seen, reply)
        assert seen[-1][2].endswith("/organization/usage"), seen

        refused = handler("2", {"event": "session.delete", "payload": {}, "session_id": "win-1"})
        assert "error" in refused, refused
    finally:
        desktop_ui.set_emitter(None)
