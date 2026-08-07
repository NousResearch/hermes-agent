"""Regression for the wlroots/Hyprland "no window ever matched" capture bug.

NousResearch/hermes-agent#74969.

``zwlr_foreign_toplevel_manager_v1`` — the protocol every wlroots compositor
(Hyprland, Sway, river) uses to enumerate windows — carries an app-id and a
title and *nothing else*. There is no PID in the protocol, so cua-driver
reports ``pid: null`` for **every** window on such a session::

    {"app_name": "google-chrome", "pid": null, "window_id": 4278190081,
     "title": "Pull requests - Google Chrome [google-chrome]", "z_index": null}

``_ingest_windows`` dropped every pid-less row (correct on X11, where only
panels and popups omit ``_NET_WM_PID``), so the normalized window list came
back empty and capture() surfaced either a silent ``0x0`` result or
``<no on-screen window matched app='Google Chrome'>``. The driver itself was
fine the whole time: ``get_window_state`` returns a full PNG plus AT-SPI tree
the moment it is handed a real pid, and it hard-rejects the call without one
(``Missing required integer field: pid``).

The fix recovers the pid out-of-band from the compositor (``hyprctl clients
-j``) before dropping the window, and refuses to guess when the mapping is
ambiguous.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

# 8x8 transparent PNG — decodes cleanly so capture() can size it.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NkGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)

MODULE = "tools.computer_use.cua_backend"

# Shape cua-driver actually emits on Hyprland (captured from a live session).
_WLROOTS_WINDOWS = [
    {
        "app_name": "google-chrome",
        "bounds": {"height": 0, "width": 0, "x": 0, "y": 0},
        "height": 0,
        "is_on_screen": True,
        "pid": None,
        "title": "Pull requests - Google Chrome [google-chrome]",
        "width": 0,
        "window_id": 4278190081,
        "x": 0,
        "y": 0,
        "z_index": None,
    },
    {
        "app_name": "kitty",
        "bounds": {"height": 0, "width": 0, "x": 0, "y": 0},
        "height": 0,
        "is_on_screen": True,
        "pid": None,
        "title": "hermes - kitty [kitty]",
        "width": 0,
        "window_id": 4278190080,
        "x": 0,
        "y": 0,
        "z_index": None,
    },
]

# Shape `hyprctl clients -j` actually emits, trimmed to the read fields.
_HYPR_CLIENTS = [
    {"pid": 57569, "class": "google-chrome", "mapped": True,
     "title": "Pull requests - Google Chrome"},
    {"pid": 54645, "class": "kitty", "mapped": True, "title": "hermes - kitty"},
]


# ---------------------------------------------------------------------------
# _strip_wayland_title_suffix
# ---------------------------------------------------------------------------

class TestStripTitleSuffix:
    @pytest.mark.parametrize("raw,expected", [
        ("Pull requests - Google Chrome [google-chrome]", "Pull requests - Google Chrome"),
        ("hermes - kitty [kitty]", "hermes - kitty"),
        ("No suffix here", "No suffix here"),
        # Only the trailing label is a suffix; brackets mid-title stay put.
        ("[Draft] PR title [firefox]", "[Draft] PR title"),
        ("", ""),
    ])
    def test_strips_only_the_trailing_app_label(self, raw, expected):
        from tools.computer_use.cua_backend import _strip_wayland_title_suffix

        assert _strip_wayland_title_suffix(raw) == expected

    def test_non_string_is_empty(self):
        from tools.computer_use.cua_backend import _strip_wayland_title_suffix

        assert _strip_wayland_title_suffix(None) == ""


# ---------------------------------------------------------------------------
# _resolve_wayland_pid — the matching policy, including its refusals
# ---------------------------------------------------------------------------

class TestResolveWaylandPid:
    def _toplevels(self):
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.shutil.which", return_value="/usr/bin/hyprctl"), \
             patch(f"{MODULE}.subprocess.run") as run:
            run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(_HYPR_CLIENTS),
            )
            return _hyprland_toplevels()

    def test_matches_on_exact_title(self):
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        pid = _resolve_wayland_pid(
            "google-chrome",
            "Pull requests - Google Chrome [google-chrome]",
            self._toplevels(),
        )
        assert pid == 57569

    def test_falls_back_to_app_id_when_title_drifted(self):
        """Titles change as the user switches tabs between the two probes."""
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        pid = _resolve_wayland_pid(
            "kitty", "some other title [kitty]", self._toplevels(),
        )
        assert pid == 54645

    def test_app_id_match_allows_multiple_windows_of_one_process(self):
        """A browser with several windows is one PID — still unambiguous."""
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        toplevels = [
            {"pid": 900, "app": "firefox", "title": "window one"},
            {"pid": 900, "app": "firefox", "title": "window two"},
        ]
        assert _resolve_wayland_pid("firefox", "drifted", toplevels) == 900

    def test_refuses_to_guess_between_distinct_processes(self):
        """Two PIDs behind one app-id must NOT resolve — capturing/clicking
        the wrong process is worse than reporting no match."""
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        toplevels = [
            {"pid": 900, "app": "firefox", "title": "window one"},
            {"pid": 901, "app": "firefox", "title": "window two"},
        ]
        assert _resolve_wayland_pid("firefox", "drifted", toplevels) is None

    def test_ambiguous_title_falls_through_to_app_id(self):
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        toplevels = [
            {"pid": 900, "app": "firefox", "title": "same"},
            {"pid": 901, "app": "chrome", "title": "same"},
        ]
        # Title is ambiguous (two PIDs), but the app-id is not.
        assert _resolve_wayland_pid("chrome", "same", toplevels) == 901

    def test_no_toplevels_resolves_to_none(self):
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        assert _resolve_wayland_pid("firefox", "title", []) is None

    def test_unknown_app_resolves_to_none(self):
        from tools.computer_use.cua_backend import _resolve_wayland_pid

        assert _resolve_wayland_pid(
            "inkscape", "Untitled [inkscape]", self._toplevels(),
        ) is None


# ---------------------------------------------------------------------------
# _hyprland_toplevels — the probe must be inert off Hyprland and never raise
# ---------------------------------------------------------------------------

class TestHyprlandToplevels:
    def test_no_probe_without_hyprland_signature(self):
        """X11/GNOME sessions must not pay for (or be changed by) the probe."""
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {}, clear=True), \
             patch(f"{MODULE}.sys.platform", "linux"), \
             patch(f"{MODULE}.subprocess.run") as run:
            assert _hyprland_toplevels() == []
            run.assert_not_called()

    def test_no_probe_off_linux(self):
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.sys.platform", "darwin"), \
             patch(f"{MODULE}.subprocess.run") as run:
            assert _hyprland_toplevels() == []
            run.assert_not_called()

    def test_missing_binary_is_not_fatal(self):
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.sys.platform", "linux"), \
             patch(f"{MODULE}.shutil.which", return_value=None):
            assert _hyprland_toplevels() == []

    @pytest.mark.parametrize("stdout", ["", "not json", "{}", "null", "[1, 2]"])
    def test_malformed_payload_is_not_fatal(self, stdout):
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.sys.platform", "linux"), \
             patch(f"{MODULE}.shutil.which", return_value="/usr/bin/hyprctl"), \
             patch(f"{MODULE}.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout=stdout)
            assert _hyprland_toplevels() == []

    def test_subprocess_failure_is_not_fatal(self):
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.sys.platform", "linux"), \
             patch(f"{MODULE}.shutil.which", return_value="/usr/bin/hyprctl"), \
             patch(f"{MODULE}.subprocess.run", side_effect=OSError("boom")):
            assert _hyprland_toplevels() == []

    def test_skips_unmapped_and_pidless_clients(self):
        from tools.computer_use.cua_backend import _hyprland_toplevels

        clients = [
            {"pid": 1, "class": "a", "mapped": False, "title": "hidden"},
            {"pid": None, "class": "b", "mapped": True, "title": "no pid"},
            "not a dict",
            {"pid": 42, "class": "Kitty", "mapped": True, "title": "Real"},
        ]
        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.sys.platform", "linux"), \
             patch(f"{MODULE}.shutil.which", return_value="/usr/bin/hyprctl"), \
             patch(f"{MODULE}.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout=json.dumps(clients))
            out = _hyprland_toplevels()

        # Case-folded for comparison against the driver's app_name/title.
        assert out == [{"pid": 42, "app": "kitty", "title": "real"}]

    def test_invocation_is_argv_not_shell(self):
        """No shell: compositor-controlled strings never reach a shell."""
        from tools.computer_use.cua_backend import _hyprland_toplevels

        with patch.dict("os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "sig"}), \
             patch(f"{MODULE}.sys.platform", "linux"), \
             patch(f"{MODULE}.shutil.which", return_value="/usr/bin/hyprctl"), \
             patch(f"{MODULE}.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="[]")
            _hyprland_toplevels()

        args, kwargs = run.call_args
        assert args[0] == ["/usr/bin/hyprctl", "clients", "-j"]
        assert kwargs.get("shell") is None or kwargs.get("shell") is False
        assert kwargs.get("timeout") == 2
        assert kwargs.get("check") is False


# ---------------------------------------------------------------------------
# _ingest_windows — the bug locus
# ---------------------------------------------------------------------------

class TestIngestWindowsOnWlroots:
    def test_wlroots_windows_survive_ingestion(self):
        """The regression: every window was dropped, so capture saw nothing."""
        from tools.computer_use.cua_backend import _ingest_windows

        with patch(f"{MODULE}._hyprland_toplevels") as probe:
            probe.return_value = [
                {"pid": 57569, "app": "google-chrome",
                 "title": "pull requests - google chrome"},
                {"pid": 54645, "app": "kitty", "title": "hermes - kitty"},
            ]
            out = _ingest_windows(_WLROOTS_WINDOWS)

        assert [w["app_name"] for w in out] == ["google-chrome", "kitty"]
        assert [w["pid"] for w in out] == [57569, 54645]
        assert [w["window_id"] for w in out] == [4278190081, 4278190080]
        # z_index: null must normalise to 0, not crash the later sort.
        assert all(w["z_index"] == 0 for w in out)
        # The compositor is probed once for the whole batch, not per window.
        assert probe.call_count == 1

    def test_probe_is_skipped_when_every_pid_is_present(self):
        from tools.computer_use.cua_backend import _ingest_windows

        with patch(f"{MODULE}._hyprland_toplevels") as probe:
            out = _ingest_windows([
                {"app_name": "Firefox", "pid": 4321, "window_id": 77, "z_index": 1},
            ])
            probe.assert_not_called()
        assert out[0]["pid"] == 4321

    def test_unresolvable_window_is_still_dropped(self):
        """X11 behaviour is unchanged: no recovery available => drop it."""
        from tools.computer_use.cua_backend import _ingest_windows

        with patch(f"{MODULE}._hyprland_toplevels", return_value=[]):
            out = _ingest_windows([
                {"app_name": "Desktop", "pid": None, "window_id": 1, "z_index": 0},
                {"app_name": "Firefox", "pid": 4321, "window_id": 77, "z_index": 1},
            ])

        assert [w["app_name"] for w in out] == ["Firefox"]

    def test_partially_resolvable_batch_keeps_only_identified_windows(self):
        from tools.computer_use.cua_backend import _ingest_windows

        with patch(f"{MODULE}._hyprland_toplevels") as probe:
            probe.return_value = [{"pid": 55, "app": "kitty", "title": "t"}]
            out = _ingest_windows([
                {"app_name": "unknown-app", "pid": None, "window_id": 1},
                {"app_name": "kitty", "pid": None, "window_id": 2},
            ])

        assert [(w["app_name"], w["pid"]) for w in out] == [("kitty", 55)]

    def test_window_without_window_id_is_always_dropped(self):
        """No window_id means no screenshot/click target, pid or not."""
        from tools.computer_use.cua_backend import _ingest_windows

        with patch(f"{MODULE}._hyprland_toplevels") as probe:
            out = _ingest_windows([
                {"app_name": "kitty", "pid": 55, "window_id": None},
            ])
            probe.assert_not_called()
        assert out == []

    def test_original_order_is_preserved(self):
        """Callers sort by z_index with a stable sort; when the compositor
        reports no stacking order, list order is the only signal left."""
        from tools.computer_use.cua_backend import _ingest_windows

        with patch(f"{MODULE}._hyprland_toplevels") as probe:
            probe.return_value = [{"pid": 55, "app": "kitty", "title": "t"}]
            out = _ingest_windows([
                {"app_name": "kitty", "pid": None, "window_id": 1},
                {"app_name": "Firefox", "pid": 4321, "window_id": 2},
            ])

        assert [w["app_name"] for w in out] == ["kitty", "Firefox"]


# ---------------------------------------------------------------------------
# capture() end-to-end on a simulated Hyprland session
# ---------------------------------------------------------------------------

def _backend_with_windows(raw_windows):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.capabilities_discovered = True
    session._has_tool.return_value = True

    def _call_tool(name, args, *a, **k):
        if name == "list_windows":
            return {"structuredContent": {"windows": raw_windows}}
        if name == "screenshot":
            return {
                "structuredContent": {
                    "screenshot_png_b64": _PNG_B64,
                    "screenshot_mime_type": "image/png",
                }
            }
        return {}

    session.call_tool.side_effect = _call_tool
    backend._session = session
    return backend


def test_capture_by_app_finds_the_wlroots_window():
    """Before the fix this returned `<no on-screen window matched ...>`."""
    backend = _backend_with_windows(_WLROOTS_WINDOWS)

    with patch(f"{MODULE}._hyprland_toplevels") as probe:
        probe.return_value = [
            {"pid": 57569, "app": "google-chrome",
             "title": "pull requests - google chrome"},
            {"pid": 54645, "app": "kitty", "title": "hermes - kitty"},
        ]
        cap = backend.capture(mode="vision", app="google-chrome")

    assert cap.app == "google-chrome"
    assert cap.png_b64 == _PNG_B64
    assert base64.b64decode(cap.png_b64)
    # The pid recovered from the compositor is what actions will be sent to.
    assert backend._active_pid == 57569
    assert backend._active_window_id == 4278190081


def test_capture_reports_no_match_when_pid_is_unrecoverable():
    """Fail closed and say so, rather than capturing an unrelated window."""
    backend = _backend_with_windows(_WLROOTS_WINDOWS)

    with patch(f"{MODULE}._hyprland_toplevels", return_value=[]):
        cap = backend.capture(mode="vision", app="google-chrome")

    assert cap.width == 0 and cap.height == 0
    assert cap.png_b64 is None
    assert backend._active_pid is None


# ---------------------------------------------------------------------------
# app= matching against a Wayland app id
# ---------------------------------------------------------------------------

class TestAppIdMatching:
    """Wayland reports the app *id* (`google-chrome`), never the display name.

    `capture(app="Google Chrome")` therefore matched nothing: not exactly
    (`google chrome` != `google-chrome`) and not as a substring either, since
    the separators differ. That produced the reported
    `<no on-screen window matched app='Google Chrome'>` even once the window
    list itself was populated.
    """

    def _match(self, windows, query):
        from tools.computer_use.cua_backend import CuaDriverBackend

        backend = CuaDriverBackend()
        # list_apps enumerates every process on Linux; keep it out of the way.
        with patch.object(CuaDriverBackend, "list_apps", return_value=[]):
            return backend._match_windows_for_app(windows, query)

    def _window(self, app_name):
        return {
            "app_name": app_name, "pid": 1, "window_id": 1,
            "off_screen": False, "title": f"Some page [{app_name}]",
            "z_index": 0,
        }

    @pytest.mark.parametrize("query", [
        "Google Chrome", "google chrome", "google-chrome", "GOOGLE CHROME",
    ])
    def test_display_name_matches_hyphenated_app_id(self, query):
        assert len(self._match([self._window("google-chrome")], query)) == 1

    def test_reverse_dns_app_id_matches_its_tail(self):
        assert len(self._match([self._window("org.mozilla.firefox")], "Firefox")) == 1

    def test_underscored_app_id_matches_display_name(self):
        assert len(self._match([self._window("libre_office_writer")],
                               "LibreOffice Writer")) == 0
        assert len(self._match([self._window("libre_office_writer")],
                               "Libre Office Writer")) == 1

    def test_folding_does_not_outrank_the_exact_tier(self):
        """The pre-existing guarantee: `Code` must not select `Visual Studio
        Code` merely because it is frontmost. Folding is an *exact* tier, so
        it must not start matching across distinct names.

        (A lone `Visual Studio Code` still matches `Code` through the
        substring tier further down — that is long-standing behaviour and is
        deliberately left alone.)
        """
        windows = [
            self._window("Visual Studio Code"),
            self._window("Code"),
        ]
        out = self._match(windows, "Code")
        assert [w["app_name"] for w in out] == ["Code"]

    def test_folding_alone_does_not_bridge_distinct_names(self):
        from tools.computer_use.cua_backend import _app_name_aliases

        assert not (_app_name_aliases("Code") & _app_name_aliases("Visual Studio Code"))

    def test_unrelated_app_still_does_not_match(self):
        assert self._match([self._window("google-chrome")], "Firefox") == []

    def test_exact_app_name_still_wins_over_folding(self):
        windows = [
            self._window("google-chrome"),
            self._window("Google Chrome"),
        ]
        out = self._match(windows, "Google Chrome")
        assert [w["app_name"] for w in out] == ["Google Chrome"]


# ---------------------------------------------------------------------------
# Diagnostics: an empty window list and a bare timeout must both be legible
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_raw_window_summary_counts_without_leaking_titles(self):
        from tools.computer_use.cua_backend import _describe_raw_windows

        summary = _describe_raw_windows(_WLROOTS_WINDOWS)

        assert "2 raw entries" in summary
        assert "2 missing pid" in summary
        assert "google-chrome" in summary
        # Window titles carry private content (chats, documents, URLs) and
        # must never reach the log.
        assert "Pull requests" not in summary

    def test_empty_raw_window_summary(self):
        from tools.computer_use.cua_backend import _describe_raw_windows

        assert _describe_raw_windows([]) == "0 raw entries"

    def test_bridge_timeout_becomes_an_actionable_message(self):
        """`fut.result(timeout=...)` raises TimeoutError with an EMPTY str(),
        which reached the model as a bare "capture failed:" (#74969)."""
        import concurrent.futures

        from tools.computer_use.cua_backend import _CuaDriverSession

        session = _CuaDriverSession.__new__(_CuaDriverSession)
        session._started = True
        session._require_started = lambda: None
        session._call_tool_async = lambda name, args: None
        session._bridge = MagicMock()
        session._bridge.run.side_effect = concurrent.futures.TimeoutError()

        with pytest.raises(RuntimeError) as excinfo:
            session.call_tool("get_window_state", {}, timeout=30.0)

        message = str(excinfo.value)
        assert message.strip(), "the whole point is that it is not empty"
        assert "get_window_state" in message
        assert "30" in message

    def test_tool_layer_never_returns_an_empty_error_string(self):
        from tools.computer_use import tool as cu_tool

        class _Silent(Exception):
            pass

        with patch.object(cu_tool, "_get_backend", return_value=MagicMock()), \
             patch.object(cu_tool, "_dispatch", side_effect=_Silent()):
            out = json.loads(cu_tool.handle_computer_use({"action": "capture"}))

        assert out["error"] == "capture failed: _Silent"
