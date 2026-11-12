"""focus_app / capture for apps on another macOS Space (every fullscreen app has its own Space). #142478"""

from unittest.mock import MagicMock

import pytest

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend_capture import _select_capture_target


def _row(app, pid, wid, *, title="", on_screen=True, z=0, spaces=(1,)):
    return {"app_name": app, "pid": pid, "window_id": wid, "title": title, "is_on_screen": on_screen,
            "z_index": z, "space_ids": list(spaces)}


HERMES = _row("Hermes", 10, 1, title="Hermes", z=5)
# Claude fullscreen on its own Space: the untitled title strip is stacked above the titled content window.
CLAUDE_STRIP = _row("Claude", 20, 38792, on_screen=False, z=9, spaces=(435,))
CLAUDE = _row("Claude", 20, 38782, title="Claude", on_screen=False, z=8, spaces=(435,))
HELPER = _row("Claude", 20, 99, on_screen=False, z=7, spaces=())   # no Space id: hidden helper surface


def _backend(on_screen, every_space, bring=None):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._session = MagicMock()
    calls = []

    def call_tool(name, args, **_):
        calls.append((name, dict(args)))
        if name == "list_windows":
            rows = on_screen if args.get("on_screen_only") else every_space
            return {"data": "", "images": [], "structuredContent": {"windows": rows}, "isError": False}
        if name == "get_window_state":
            return {"data": "", "images": [], "structuredContent": None, "isError": False}
        raise AssertionError(f"unexpected tool call: {name}")

    backend._session.call_tool.side_effect = call_tool
    backend._session._has_tool.return_value = True
    raised = []

    def bring_to_front(*, pid, window_id=None):
        raised.append((pid, window_id))
        return bring or ActionResult(ok=True, action="bring_to_front", meta={"status": "activated"})

    backend.bring_to_front = bring_to_front
    return backend, calls, raised


def test_background_focus_names_the_other_space_and_targets_nothing():
    backend, _, raised = _backend([HERMES], [HERMES, CLAUDE_STRIP, CLAUDE])

    res = backend.focus_app("Claude")

    assert res.ok is False
    assert res.code == "window_on_other_space"
    assert "raise_window=true" in res.message
    assert backend._active_pid is None
    assert raised == []


def test_raise_window_switches_to_the_app_on_another_space():
    backend, _, raised = _backend([HERMES], [HERMES, CLAUDE_STRIP, CLAUDE, HELPER])

    res = backend.focus_app("Claude", raise_window=True)

    assert res.ok is True
    assert res.action == "focus_app"
    assert raised and raised[0][0] == 20
    assert backend._active_pid == 20


def test_app_on_the_current_space_never_queries_other_spaces():
    backend, calls, _ = _backend([HERMES, _row("Notes", 30, 3, title="Notes", z=4)], [])

    assert backend.focus_app("Notes").ok is True
    assert [a.get("on_screen_only") for n, a in calls if n == "list_windows"] == [True]


def test_rows_without_a_space_are_not_targets():
    backend, _, raised = _backend([HERMES], [HERMES, HELPER])

    res = backend.focus_app("Claude", raise_window=True)

    assert res.ok is False
    assert res.code != "window_on_other_space"
    assert raised == []


def _partial(pid, focused_window, frontmost_pid, matches=True):
    return ActionResult(ok=False, action="bring_to_front", effect="unverifiable",
                        escalation={"recommended": "foreground"}, meta={
                            "code": "bring_to_front_exact_window_unverified", "status": "partial",
                            "activated": False, "exact_window_effect": {"focused": True},
                            "observed": {"front_process_matches_target": matches,
                                         "focused_window_id": focused_window, "frontmost_pid": frontmost_pid}})


def test_fullscreen_raise_with_own_title_strip_above_counts_as_done():
    backend, _, _ = _backend([HERMES], [HERMES, CLAUDE], bring=_partial(20, 38782, 20))

    res = backend.focus_app("Claude", raise_window=True)

    assert res.ok is True
    assert res.code == "bring_to_front_focused_own_app_above"
    assert (res.verified, res.effect, res.escalation) == (True, "confirmed", None)
    assert res.meta["status"] == "activated"
    assert res.meta["driver_verdict"]["status"] == "partial"


@pytest.mark.parametrize("focused_window, frontmost_pid, matches", [
    (12345, 20, True),      # a different window has focus
    (38782, 99, True),      # another process is frontmost
    (38782, 20, False),     # driver says the front process is not the target
])
def test_partial_raise_stays_partial_unless_the_target_really_has_focus(focused_window, frontmost_pid, matches):
    backend, _, _ = _backend([HERMES], [HERMES, CLAUDE],
                             bring=_partial(20, focused_window, frontmost_pid, matches))

    res = backend.focus_app("Claude", raise_window=True)

    assert res.ok is False
    assert res.code != "bring_to_front_focused_own_app_above"


def _w(app, pid, wid, title, z):
    return {"app_name": app, "pid": pid, "window_id": wid, "title": title, "off_screen": False, "z_index": z}


@pytest.mark.platforms("macos")
def test_macos_capture_prefers_the_titled_window_behind_the_title_strip():
    strip, content = _w("Claude", 20, 38792, "", 9), _w("Claude", 20, 38782, "Claude", 8)

    assert _select_capture_target([strip, content], app_requested=True) == content
    # An exact target is never second-guessed, and another process's titled window is not a sibling.
    assert _select_capture_target([strip, content], app_requested=True, exact_target=True) == strip
    other = _w("Notes", 30, 3, "Notes", 8)
    assert _select_capture_target([strip, other], app_requested=False) == strip
