"""Tests for browser tab management (#71375).

A click can open a second tab — an OAuth consent screen, a ``target="_blank"``
link, ``window.open()``. Nothing let the model see that a second tab existed,
return to the tab it came from, or close one it was done with.

The agent-browser payloads asserted here were captured from a real
``agent-browser@0.26.0`` run (the version pinned in package.json), not invented:

    $ agent-browser --json tab list
    {"success":true,"data":{"tabs":[{"active":true,"label":null,"tabId":"t1",
      "title":"example.com","type":"page","url":"https://example.com/"}]},"error":null}
    $ agent-browser --json tab t1
    {"success":true,"data":{"label":null,"tabId":"t1","title":"Example Domain",
      "url":"https://example.com/"},"error":null}
    $ agent-browser --json tab close
    {"success":true,"data":{"closed":true,"label":null,"tabId":"t1"},"error":null}
"""
import json

import pytest

from tools import browser_tool


@pytest.fixture()
def local_backend(monkeypatch):
    """Force the agent-browser path and capture the CLI calls it makes."""
    calls = []
    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda t: t)
    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda _t: False)
    return calls


def _cli(calls, response):
    def _run(task_id, command, args=None, **_kw):
        calls.append((command, list(args or [])))
        return response
    return _run


class TestList:
    def test_lists_open_tabs(self, local_backend, monkeypatch):
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": True,
            "data": {"tabs": [
                {"active": False, "tabId": "t1", "title": "origin", "url": "https://a.test/"},
                {"active": True, "tabId": "t2", "title": "consent", "url": "https://b.test/"},
            ]},
        }))

        out = json.loads(browser_tool.browser_tabs(task_id="x"))

        assert local_backend == [("tab", ["list"])]
        assert out["success"] and out["count"] == 2
        assert out["tabs"][1] == {
            "tab_id": "t2", "url": "https://b.test/", "title": "consent", "active": True,
        }

    def test_cli_failure_is_reported(self, local_backend, monkeypatch):
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": False, "error": "no browser session",
        }))
        out = json.loads(browser_tool.browser_tabs(task_id="x"))
        assert out["success"] is False and "no browser session" in out["error"]


class TestSwitch:
    def test_switches_to_the_named_tab(self, local_backend, monkeypatch):
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": True,
            "data": {"tabId": "t1", "title": "Example Domain", "url": "https://a.test/"},
        }))

        out = json.loads(browser_tool.browser_tabs(action="switch", tab_id="t1", task_id="x"))

        assert local_backend == [("tab", ["t1"])], (
            "switch must pass the bare tab id as the CLI's `tab <n>` form"
        )
        assert out["success"] and out["switched_to"] == "t1"
        assert out["url"] == "https://a.test/"

    def test_switch_without_tab_id_is_rejected_before_touching_the_cli(self, local_backend, monkeypatch):
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {"success": True}))
        out = json.loads(browser_tool.browser_tabs(action="switch", task_id="x"))
        assert out["success"] is False and "requires tab_id" in out["error"]
        assert local_backend == []

    def test_a_blocked_switch_returns_to_the_previous_tab(self, local_backend, monkeypatch):
        """The CLI can only read a tab's URL by making it current.

        So the switch happens, then the guard fires. Leaving the session parked
        on the blocked page while reporting a refusal would make the message
        false and hand that page to the next snapshot.
        """
        def _run(task_id, command, args=None, **_kw):
            local_backend.append((command, list(args or [])))
            a = list(args or [])
            if a == ["list"]:
                return {"success": True, "data": {"tabs": [
                    {"tabId": "t1", "active": True, "url": "https://safe.test/"},
                    {"tabId": "t2", "active": False, "url": "http://169.254.169.254/"},
                ]}}
            return {"success": True, "data": {"tabId": a[0] if a else ""}}

        monkeypatch.setattr(browser_tool, "_run_browser_command", _run)
        monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda _t: True)
        monkeypatch.setattr(
            browser_tool, "_current_page_private_url", lambda _t: "http://169.254.169.254/"
        )

        out = json.loads(browser_tool.browser_tabs(action="switch", tab_id="t2", task_id="x"))

        assert out["success"] is False
        assert local_backend[-1] == ("tab", ["t1"]), (
            "the session was left parked on the blocked tab — the refusal "
            f"message is false. calls were: {local_backend}"
        )
        assert "Returned to tab t1" in out["error"]

    def test_no_tab_listing_when_the_guard_is_off(self, local_backend, monkeypatch):
        """With nothing to undo, a switch must not pay for an extra round-trip."""
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": True, "data": {"tabId": "t1", "url": "https://a.test/"},
        }))
        json.loads(browser_tool.browser_tabs(action="switch", tab_id="t1", task_id="x"))
        assert local_backend == [("tab", ["t1"])]

    def test_switching_onto_a_private_address_is_blocked(self, local_backend, monkeypatch):
        """A script-opened tab was never seen by the navigate preflight.

        Same reason browser_back re-checks after history navigation: switching
        makes another page current, and that page may point at an internal or
        cloud-metadata address.
        """
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": True, "data": {"tabId": "t2", "url": "http://169.254.169.254/"},
        }))
        monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", lambda _t: True)
        monkeypatch.setattr(
            browser_tool, "_current_page_private_url", lambda _t: "http://169.254.169.254/"
        )

        out = json.loads(browser_tool.browser_tabs(action="switch", tab_id="t2", task_id="x"))

        assert out["success"] is False, (
            "switching onto a private/internal address was allowed — the SSRF "
            "floor must fire here as it does for browser_back"
        )
        assert "169.254.169.254" in out["error"]


class TestClose:
    def test_closes_the_active_tab_by_default(self, local_backend, monkeypatch):
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": True, "data": {"closed": True, "tabId": "t1"},
        }))
        out = json.loads(browser_tool.browser_tabs(action="close", task_id="x"))
        assert local_backend == [("tab", ["close"])]
        assert out["success"] and out["closed"] == "t1"

    def test_closes_a_named_tab(self, local_backend, monkeypatch):
        monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {
            "success": True, "data": {"closed": True, "tabId": "t3"},
        }))
        json.loads(browser_tool.browser_tabs(action="close", tab_id="t3", task_id="x"))
        assert local_backend == [("tab", ["close", "t3"])]


def test_unknown_action_is_rejected(local_backend, monkeypatch):
    monkeypatch.setattr(browser_tool, "_run_browser_command", _cli(local_backend, {"success": True}))
    out = json.loads(browser_tool.browser_tabs(action="teleport", task_id="x"))
    assert out["success"] is False and "Unknown action" in out["error"]
    assert local_backend == []


class TestRegisteredForTheModel:
    def test_schema_is_declared(self):
        from tools.browser_tool import _BROWSER_SCHEMA_MAP

        schema = _BROWSER_SCHEMA_MAP.get("browser_tabs")
        assert schema is not None, "browser_tabs has no schema, so the model never sees it"
        assert set(schema["parameters"]["properties"]) == {"action", "tab_id"}
        assert schema["parameters"]["properties"]["action"]["enum"] == ["list", "switch", "close"]

    def test_listed_in_the_browser_toolset(self):
        """Declared but absent from the toolset means it is never offered."""
        import toolsets

        src_lists = [v for v in vars(toolsets).values() if isinstance(v, (list, tuple, set, frozenset))]
        assert any("browser_tabs" in v for v in src_lists if "browser_back" in v), (
            "browser_tabs is missing from the browser toolset that ships browser_back"
        )


class TestCamofoxFollowsANewTab:
    """The bug the issue reports, on the backend where it actually happens.

    Camofox pins ``session["tab_id"]`` at navigate time (browser_camofox.py) and
    every later call posts to ``/tabs/<that id>/...``. A click that opens a
    consent screen therefore leaves the session addressing the page the model
    already left, and every following snapshot/click/type hits the wrong tab.

    (The default agent-browser backend does not have this bug — its CLI already
    activates the new tab, verified against agent-browser@0.26.0 — so the fix is
    scoped to Camofox.)
    """

    @pytest.fixture()
    def camo(self, monkeypatch):
        from tools import browser_camofox as cf

        session = {"user_id": "u1", "tab_id": "old", "session_key": "k"}
        monkeypatch.setattr(cf, "_get_session", lambda _t: session)
        monkeypatch.setattr(cf, "get_camofox_url", lambda: "http://camofox.test")
        monkeypatch.setattr(cf, "_camofox_private_page_block", lambda *_a, **_k: None)
        monkeypatch.setattr(cf, "_post", lambda *_a, **_k: {"url": "https://consent.test/"})
        return cf, session

    def _tabs(self, cf, monkeypatch, sequence):
        """Serve successive GET /tabs responses (before-click, after-click)."""
        calls = iter(sequence)
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": next(calls)})

    def test_click_that_opens_a_tab_switches_the_session_to_it(self, camo, monkeypatch):
        cf, session = camo
        self._tabs(cf, monkeypatch, [
            [{"tabId": "old"}],                      # before the click
            [{"tabId": "old"}, {"tabId": "new"}],    # the click opened one
        ])

        out = json.loads(cf.camofox_click("@e1", task_id="x"))

        assert session["tab_id"] == "new", (
            "the session stayed pinned to the tab the model already left — "
            "every following snapshot/click would hit the wrong page (#71375)"
        )
        assert out["followed_new_tab"] == "new"

    def test_ordinary_click_leaves_the_tab_alone(self, camo, monkeypatch):
        cf, session = camo
        self._tabs(cf, monkeypatch, [[{"tabId": "old"}], [{"tabId": "old"}]])

        out = json.loads(cf.camofox_click("@e1", task_id="x"))

        assert session["tab_id"] == "old"
        assert "followed_new_tab" not in out

    def test_a_pre_existing_second_tab_is_not_hijacked(self, camo, monkeypatch):
        """Only a tab that appeared *because of* this click may be adopted."""
        cf, session = camo
        self._tabs(cf, monkeypatch, [
            [{"tabId": "old"}, {"tabId": "other"}],
            [{"tabId": "old"}, {"tabId": "other"}],
        ])

        json.loads(cf.camofox_click("@e1", task_id="x"))

        assert session["tab_id"] == "old"

    def test_tab_bookkeeping_never_fails_the_click(self, camo, monkeypatch):
        """A broken /tabs endpoint must not turn a successful click into an error."""
        cf, session = camo

        def _boom(*_a, **_k):
            raise RuntimeError("camofox /tabs is down")

        monkeypatch.setattr(cf, "_get", _boom)

        out = json.loads(cf.camofox_click("@e1", task_id="x"))

        assert out["success"] is True
        assert session["tab_id"] == "old"

    def test_camofox_switch_rejects_an_unknown_tab(self, camo, monkeypatch):
        cf, session = camo
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": [{"tabId": "old"}]})

        out = json.loads(cf.camofox_tabs("switch", "nope", task_id="x"))

        assert out["success"] is False
        assert session["tab_id"] == "old"

    def test_camofox_switch_accepts_a_known_tab(self, camo, monkeypatch):
        cf, session = camo
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": [{"tabId": "old"}, {"tabId": "t2"}]})

        out = json.loads(cf.camofox_tabs("switch", "t2", task_id="x"))

        assert out["success"] and session["tab_id"] == "t2"

class TestCamofoxSafety:
    """Two gaps the local path already guarded against.

    The SSRF recheck was written for the local switch path in this same change
    and then not applied to the Camofox path — same bug class, second backend.
    """

    @pytest.fixture()
    def camo(self, monkeypatch):
        from tools import browser_camofox as cf

        session = {"user_id": "u1", "tab_id": "old", "session_key": "k"}
        monkeypatch.setattr(cf, "_get_session", lambda _t: session)
        monkeypatch.setattr(cf, "get_camofox_url", lambda: "http://camofox.test")
        monkeypatch.setattr(cf, "_camofox_private_page_block", lambda *_a, **_k: None)
        monkeypatch.setattr(cf, "_post", lambda *_a, **_k: {"url": "https://ok.test/"})
        return cf, session

    def _guard(self, monkeypatch, blocked_for=()):
        """Turn the SSRF guard on and block the listed tab ids."""
        import tools.browser_tool as bt

        monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda _t: True)
        monkeypatch.setattr(
            bt, "_camofox_current_page_private_url",
            lambda tab_id, _uid: "http://169.254.169.254/" if tab_id in blocked_for else None,
        )

    # ── enumeration failure must not authorise a switch ──
    def test_switch_refuses_when_the_tab_list_cannot_be_read(self, camo, monkeypatch):
        cf, session = camo

        def _boom(*_a, **_k):
            raise RuntimeError("camofox /tabs is down")

        monkeypatch.setattr(cf, "_get", _boom)

        out = json.loads(cf.camofox_tabs("switch", "anything", task_id="x"))

        assert out["success"] is False, (
            "an unreadable tab list was treated as permission to switch to an "
            "unverified id — every later call would then address that tab"
        )
        assert session["tab_id"] == "old"

    def test_list_reports_a_read_failure_instead_of_an_empty_list(self, camo, monkeypatch):
        cf, _session = camo

        def _boom(*_a, **_k):
            raise RuntimeError("down")

        monkeypatch.setattr(cf, "_get", _boom)
        out = json.loads(cf.camofox_tabs("list", task_id="x"))
        assert out["success"] is False

    def test_genuinely_empty_list_is_still_a_success(self, camo, monkeypatch):
        cf, _session = camo
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": []})
        out = json.loads(cf.camofox_tabs("list", task_id="x"))
        assert out["success"] is True and out["count"] == 0

    # ── private-address guard on switch ──
    def test_switch_to_a_private_tab_is_refused_and_keeps_the_current_tab(self, camo, monkeypatch):
        cf, session = camo
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": [{"tabId": "old"}, {"tabId": "t2"}]})
        self._guard(monkeypatch, blocked_for={"t2"})

        out = json.loads(cf.camofox_tabs("switch", "t2", task_id="x"))

        assert out["success"] is False and "169.254.169.254" in out["error"]
        assert session["tab_id"] == "old", (
            "the session moved to the blocked tab anyway — the probe must run "
            "before the assignment, not after"
        )

    def test_switch_to_a_public_tab_still_works_under_the_guard(self, camo, monkeypatch):
        cf, session = camo
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": [{"tabId": "old"}, {"tabId": "t2"}]})
        self._guard(monkeypatch, blocked_for=set())

        out = json.loads(cf.camofox_tabs("switch", "t2", task_id="x"))

        assert out["success"] and session["tab_id"] == "t2"

    # ── private-address guard on auto-follow ──
    def test_click_does_not_follow_a_tab_opened_at_a_private_address(self, camo, monkeypatch):
        """A page can open a tab pointing at an intranet host.

        Adopting it would make that page the session's current tab and hand its
        content to the next snapshot.
        """
        cf, session = camo
        seq = iter([[{"tabId": "old"}], [{"tabId": "old"}, {"tabId": "evil"}]])
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": next(seq)})
        self._guard(monkeypatch, blocked_for={"evil"})

        out = json.loads(cf.camofox_click("@e1", task_id="x"))

        assert session["tab_id"] == "old", (
            "the click auto-follow adopted a tab at a private address"
        )
        assert "followed_new_tab" not in out
        assert out["success"] is True  # the click itself still succeeded

    def test_click_still_follows_a_public_new_tab_under_the_guard(self, camo, monkeypatch):
        cf, session = camo
        seq = iter([[{"tabId": "old"}], [{"tabId": "old"}, {"tabId": "new"}]])
        monkeypatch.setattr(cf, "_get", lambda *_a, **_k: {"tabs": next(seq)})
        self._guard(monkeypatch, blocked_for=set())

        json.loads(cf.camofox_click("@e1", task_id="x"))

        assert session["tab_id"] == "new"


class TestExposedOnTheCodexCallback:
    def test_browser_tabs_is_in_the_app_server_allowlist(self):
        """codex_app_server dispatches browser tools through this explicit list."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

        assert "browser_tabs" in EXPOSED_TOOLS, (
            "browser_tabs is missing from the codex_app_server callback allowlist, "
            "so that runtime silently loses the tool"
        )
        assert "browser_back" in EXPOSED_TOOLS  # sibling still there
