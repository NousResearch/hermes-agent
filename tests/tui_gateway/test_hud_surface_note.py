"""HUD mode reaches the model as a per-turn note, not as a platform hint.

The floating HUD sits over whatever the user is really working in, so a request
typed there is usually about that other app — and usually wants to be carried
out IN that app. Without the note the model answers from its own browser and
panes, which is the wrong half of the screen.

The same desktop session can be driven from the app window on one turn and the
HUD on the next, so this cannot live in the system prompt — that has to stay
byte-stable for the life of a conversation. It rides the model-bound message
instead, beside the reaction and speech-interrupted notes.
"""

import threading
import types

import pytest

from agent.prompt_builder import hud_surface_note
from tui_gateway import server

FULL_KIT = {"read_window_below", "computer_use", "browser_navigate"}


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _session(*, tools=FULL_KIT, **extra):
    return {
        "agent": types.SimpleNamespace(valid_tool_names=set(tools)),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


class TestNoteContents:
    """Every tool the note names has to be one this agent actually has."""

    def test_points_at_the_window_below_and_at_working_in_it(self):
        note = hud_surface_note(FULL_KIT)

        assert "read_window_below" in note
        assert "computer_use" in note
        assert "browser_navigate" in note

    def test_no_note_at_all_without_the_tool_it_rests_on(self):
        assert hud_surface_note({"computer_use", "browser_navigate"}) == ""

    def test_identifies_the_app_even_when_it_cannot_drive_it(self):
        note = hud_surface_note({"read_window_below"})

        assert "read_window_below" in note
        assert "computer_use" not in note

    def test_browser_preference_needs_a_browser_to_prefer_over(self):
        note = hud_surface_note({"read_window_below", "computer_use"})

        assert "computer_use" in note
        assert "browser_navigate" not in note

    def test_no_tools_at_all(self):
        assert hud_surface_note(None) == ""


class TestTurnRouting:
    def test_hud_turn_gets_the_note(self):
        assert server._hud_surface_note(_session(client_surface="hud")) == hud_surface_note(FULL_KIT)

    def test_app_window_turn_gets_nothing(self):
        assert server._hud_surface_note(_session(client_surface="")) == ""

    def test_session_that_never_reported_a_surface_gets_nothing(self):
        """Every other client (TUI, dashboard, messaging) omits the field."""
        assert server._hud_surface_note(_session()) == ""

    def test_survives_an_agent_with_no_toolset_at_all(self):
        session = _session(client_surface="hud")
        session["agent"] = types.SimpleNamespace()

        assert server._hud_surface_note(session) == ""

    def test_note_survives_tool_search_when_mcp_tools_are_present(self):
        """valid_tool_names is the post-assembly list. MCP tools used to hide
        read_window_below there, so a HUD turn got no note at all."""
        from tools.registry import discover_builtin_tools, registry
        from tools.tool_search import ToolSearchConfig, assemble_tool_defs

        discover_builtin_tools()
        mcp_name = "mcp_hud_surface_probe"
        registry.register(
            name=mcp_name,
            handler=lambda args, **kw: "{}",
            schema=_tool_def(mcp_name)["function"],
            toolset="mcp-hud-surface-probe",
        )

        assembled = assemble_tool_defs(
            [_tool_def(name) for name in (*FULL_KIT, mcp_name)],
            context_length=200_000,
            config=ToolSearchConfig.from_raw({"enabled": "on"}),
        )
        names = {td["function"]["name"] for td in assembled.tool_defs}

        assert assembled.activated
        assert mcp_name not in names
        # Production computes the note from agent.valid_tool_names — the
        # GRANTED set — not from the visible post-assembly schemas. Under
        # #97979 the HUD kit (read_window_below, computer_use) is deferred
        # behind the bridge yet still granted/callable, so the note must
        # survive assembly unchanged.
        assert server._hud_surface_note(_session(tools=FULL_KIT, client_surface="hud")) == (
            hud_surface_note(FULL_KIT)
        )


class TestPrepending:
    """Notes ride the model input; the persisted prompt stays what was typed."""

    def test_plain_text_turn(self):
        assert server._prepend_note("go to x", "[Note: hi]") == "[Note: hi]\n\ngo to x"

    def test_multimodal_turn_keeps_its_parts(self):
        parts = [{"type": "text", "text": "go to x"}, {"type": "image_url", "image_url": {}}]

        assert server._prepend_note(parts, "[Note: hi]") == [
            {"type": "text", "text": "[Note: hi]"},
            *parts,
        ]

    def test_nothing_to_say_leaves_the_message_untouched(self):
        parts = [{"type": "text", "text": "go to x"}]

        assert server._prepend_note("go to x", "") == "go to x"
        assert server._prepend_note(parts, "") is parts


class TestSurfaceRecording:
    """``prompt.submit`` stamps the window each message was typed into."""

    @pytest.fixture
    def busy_session(self):
        # A running session takes the busy path, which returns before any of
        # the agent/DB machinery — enough to observe what submit recorded.
        session = _session(running=True)
        server._sessions["sid"] = session
        yield session
        server._sessions.pop("sid", None)

    def _submit(self, **params):
        return server._methods["prompt.submit"](
            "r1", {"session_id": "sid", "text": "what is this?", "queued": True, **params}
        )

    def test_hud_submit_is_recorded(self, busy_session):
        self._submit(surface="hud")

        assert busy_session["client_surface"] == "hud"

    def test_the_next_app_window_submit_clears_it(self, busy_session):
        """A stale 'hud' would tell the model the user is still floating."""
        self._submit(surface="hud")
        self._submit()

        assert busy_session["client_surface"] == ""

    def test_an_unknown_surface_is_not_hud(self, busy_session):
        self._submit(surface="pet-overlay")

        assert busy_session["client_surface"] == ""
