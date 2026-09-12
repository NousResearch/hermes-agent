"""Tests for the GUI-surface ``read_window_below`` tool."""

import json

from tools import read_window_tool as rw
from tools.registry import registry


def test_lives_in_the_gui_surface_toolset(monkeypatch):
    """Mirrors read_terminal: scoped by toolset, not by the backend's env."""
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    entry = registry.get_entry("read_window_below")

    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert entry.check_fn is None


def test_requires_callback():
    """Outside the desktop GUI there is no bridge — a clear error, no crash."""
    result = json.loads(rw.read_window_below_tool(callback=None))
    assert "desktop" in result["error"]


def test_empty_answer_means_unavailable():
    result = json.loads(rw.read_window_below_tool(callback=lambda: ""))
    assert "error" in result


def test_passes_json_through():
    payload = {
        "window": {"app": "Figma", "title": "", "bounds": {"x": 0, "y": 38, "width": 1470, "height": 870}, "id": 13937},
        "frontmost": {"app": "Figma", "title": ""},
        "platform": "darwin",
    }
    result = json.loads(rw.read_window_below_tool(callback=lambda: json.dumps(payload)))
    assert result == payload


def test_wraps_non_json_text():
    result = json.loads(rw.read_window_below_tool(callback=lambda: "plain words"))
    assert result == {"text": "plain words"}


def test_callback_failure_is_reported():
    def _boom():
        raise RuntimeError("renderer went away")

    result = json.loads(rw.read_window_below_tool(callback=_boom))
    assert "renderer went away" in result["error"]


class TestAgentHost:
    """The window is on the user's screen; computer_use drives the gateway's
    host. On a remote backend those are different machines, and the answer is
    where the agent finds out — the HUD note can only tell it to look."""

    WINDOW = {"window": {"app": "Figma", "title": ""}, "platform": "darwin"}

    def _read(self, **extra):
        payload = {**self.WINDOW, **extra}

        return json.loads(rw.read_window_below_tool(callback=lambda: json.dumps(payload)))

    def test_local_session_says_nothing(self):
        """The desktop app omits the flag when the agent is on this machine, so
        the common case pays no tokens for a fact that is already true."""
        assert self._read() == self.WINDOW

    def test_remote_session_is_told_it_cannot_click_the_window(self):
        note = self._read(agent_on_this_machine=False)["agent_host"]["note"]

        assert "computer_use" in note
        assert "cannot click" in note

    def test_remote_session_names_the_machine_it_is_actually_on(self, monkeypatch):
        monkeypatch.setattr(rw.socket, "gethostname", lambda: "remote-box")
        agent_host = self._read(agent_on_this_machine=False)["agent_host"]

        assert agent_host["same_machine"] is False
        assert agent_host["name"] == "remote-box"
        assert "on remote-box" in agent_host["note"]

    def test_an_unresolvable_hostname_still_states_the_gap(self, monkeypatch):
        def _boom():
            raise OSError("no hostname")

        monkeypatch.setattr(rw.socket, "gethostname", _boom)
        agent_host = self._read(agent_on_this_machine=False)["agent_host"]

        assert agent_host["name"] is None
        assert "on another machine" in agent_host["note"]

    def test_the_wire_flag_is_not_passed_through_to_the_model(self):
        """Two ways to say the same thing invites the model to trust the terser
        one and skip the note that tells it what to do instead."""
        assert "agent_on_this_machine" not in self._read(agent_on_this_machine=False)

    def test_an_older_desktop_that_never_sends_the_flag_is_unchanged(self):
        assert self._read(agent_on_this_machine=True) == self.WINDOW
