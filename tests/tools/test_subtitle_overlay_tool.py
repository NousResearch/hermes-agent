"""Tests for the GUI-surface ``subtitle_overlay`` tool."""

import json

from tools import subtitle_overlay_tool as so
from tools.registry import registry


def _run(**kwargs):
    kwargs.setdefault("callback", lambda _payload: json.dumps({"success": True}))
    return json.loads(so.subtitle_overlay_tool(**kwargs))


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Scoped by toolset, not by the backend's env — see AGENTS.md."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("subtitle_overlay")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    assert "desktop" in json.loads(so.subtitle_overlay_tool(action="start", callback=None))["error"]


def test_rejects_unknown_action():
    assert "action must be one of" in _run(action="pause")["error"]


def test_start_requires_a_language():
    assert "language" in _run(action="start")["error"]
    assert "language" in _run(action="start", language="   ")["error"]


def test_band_fraction_bounds():
    assert "band_fraction" in _run(action="start", language="pt", band_fraction=0)["error"]
    assert "band_fraction" in _run(action="start", language="pt", band_fraction=0.9)["error"]
    # Bool is an int in Python and must not pass as a fraction.
    assert "band_fraction" in _run(action="start", language="pt", band_fraction=True)["error"]
    assert "error" not in _run(action="start", language="pt", band_fraction=0.35)


def test_start_payload_carries_language_target_and_band():
    seen = {}

    def cb(payload):
        seen.update(payload)
        return json.dumps({"success": True})

    so.subtitle_overlay_tool(
        action="start", language="pt", target="Chrome", band_fraction=0.3, callback=cb
    )
    assert seen == {
        "action": "start",
        "language": "pt",
        "target": "Chrome",
        "band_fraction": 0.3,
    }


def test_stop_and_status_payloads_stay_minimal():
    seen = []

    def cb(payload):
        seen.append(payload)
        return json.dumps({"success": True})

    so.subtitle_overlay_tool(action="stop", callback=cb)
    so.subtitle_overlay_tool(action="status", callback=cb)
    assert seen == [{"action": "stop"}, {"action": "status"}]


def test_unanswered_bridge_is_reported_rather_than_faked_as_success():
    assert "error" in _run(action="status", callback=lambda _p: "")


def test_passes_renderer_json_through():
    payload = {"success": True, "running": True, "lines_translated": 12}
    assert _run(action="status", callback=lambda _p: json.dumps(payload)) == payload


def test_wraps_non_json_text():
    assert _run(action="stop", callback=lambda _p: "stopped") == {"text": "stopped"}


def test_callback_failure_is_reported():
    def _boom(_payload):
        raise RuntimeError("renderer went away")

    assert "renderer went away" in _run(action="stop", callback=_boom)["error"]
