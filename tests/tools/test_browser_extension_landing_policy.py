"""Where an extension page landed.

``_extension_url_guard`` judges the URL that was asked for, before anything is sent. A redirect, a clicked link
or a submitted form can land somewhere else, so the URL the controller reports back goes through the same
policy the built-in lane applies after a redirect (the cloud-metadata floor and the operator's website policy,
not the private-address rule). A page that lands on a blocked address is withheld whole: the model gets an
error naming the host, and nothing the page showed (snapshot, title, text, URL path or query). The tab stays on
that page (the extension cannot blank it), so Hermes remembers it and refuses later calls on it.
"""

import json

import pytest

from tools import browser_tool
from tools.browser_extension_router import route_browser_tool
from tools.browser_tool_extension_landing import reset_withheld
from tools.registry import registry

SECRET_PAGE = {
    "url": "https://blocked.test/account/private?session=abc123",
    "title": "Private account page",
    "snapshot": {"text": "heading 'Private account page' button 'Delete account'", "refs": [{"ref": "@e1"}],
                 "truncated": False, "element_count": 2},
}
PAGE_LEAKS = ("account/private", "session=abc123", "Private account page", "Delete account")
LEAKS = PAGE_LEAKS + ("snapshot", "@e1")


class FakeBroker:
    """A bound controller that answers every command with ``result`` (a dict, as the real broker returns).
    ``replies`` answers named actions instead: a result, or a list of results used one after the other (the
    last one repeats); an exception is raised."""

    def __init__(self, result, replies=None):
        self.result = result
        self.replies = dict(replies or {})
        self.dispatched = []

    def scope_for_session(self, **identity):
        return object()

    def select(self, scope, action):
        return object()

    def dispatch(self, scope, *, action, arguments, tool_call_id=""):
        self.dispatched.append((action, arguments))
        reply = self.replies.get(action, self.result)
        if isinstance(reply, list):
            reply = reply.pop(0) if len(reply) > 1 else reply[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def forget_withheld_tabs(monkeypatch):
    """No withheld tab carries over between tests, and a new tab's settle polling does not sleep."""
    import tools.browser_tool_extension_landing as landing

    reset_withheld()
    monkeypatch.setattr(landing.time, "sleep", lambda seconds: None)
    yield
    reset_withheld()


@pytest.fixture(autouse=True)
def operator_policy(monkeypatch):
    """blocked.test is on the operator's website blocklist."""
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: (
        {"message": "Blocked by website policy", "host": "blocked.test", "rule": "blocked.test", "source": "config"}
        if "blocked.test" in url else None))


@pytest.fixture
def lane(monkeypatch):
    """Builds a bound extension controller that replies with the given result."""
    import gateway.browser_control_broker as gw
    import tools.browser_extension_router as router

    holder = {}

    def make(result, replies=None):
        holder["broker"] = FakeBroker(result, replies)
        return holder["broker"]

    monkeypatch.setattr(gw, "browser_control_enabled", lambda: True)
    monkeypatch.setattr(gw, "get_browser_control_broker", lambda: holder["broker"])
    monkeypatch.setattr(router, "_bound_identity", lambda: ("session", "principal", "transport"))
    return make


def _raw(name, args):
    return registry.get_entry(name).handler(args, task_id="task", session_id="session")


def _call(name, args):
    return json.loads(_raw(name, args))


def _assert_withheld(out, host):
    assert out["success"] is False and out["code"] == "redirect_blocked"
    assert out["blocked_host"] == host and f"({host})" in out["error"]
    assert "landed on a blocked address" in out["error"] and "Do not continue on this page" in out["error"]
    text = json.dumps(out)
    assert not [leak for leak in LEAKS if leak in text], text
    assert not {"snapshot", "title", "url", "text"} & set(out)


@pytest.mark.parametrize("name, args, action", [
    ("browser_navigate", {"url": "https://example.test/start"}, "browser_navigate"),
    ("browser_back", {}, "browser_back"),
])
def test_a_navigation_that_redirects_to_a_blocked_host_withholds_the_page(lane, name, args, action):
    lane({"status": "navigated", "navigated": True, **SECRET_PAGE})

    out = _call(name, args)

    _assert_withheld(out, "blocked.test")
    assert out["blocked_by_policy"] == {"host": "blocked.test", "rule": "blocked.test", "source": "config"}


@pytest.mark.parametrize("landed", [
    "http://169.254.169.254/latest/meta-data/iam?token=abc", "http://metadata.google.internal/computeMetadata/v1/",
])
def test_a_redirect_to_a_cloud_metadata_address_is_withheld_and_names_only_the_host(lane, landed):
    lane({"status": "navigated", "url": landed, "title": "Private account page",
          "snapshot": SECRET_PAGE["snapshot"]})

    out = _call("browser_navigate", {"url": "https://example.test/start"})

    host = landed.split("/")[2]
    _assert_withheld(out, host)
    assert "latest/meta-data" not in json.dumps(out) and "computeMetadata" not in json.dumps(out)
    assert "blocked_by_policy" not in out


def test_a_tab_that_opens_on_a_blocked_host_is_reported_without_its_url(lane):
    lane({"tab": {"id": 31, "windowId": 2, "active": False, "url": SECRET_PAGE["url"]}})

    out = _call("browser_tab_create", {"url": "https://example.test/start"})

    _assert_withheld(out, "blocked.test")
    assert out["tab_id"] == 31 and "browser_tab_close" in out["error"]


REDIRECTING_TAB = {"tab": {"id": 31, "windowId": 2, "active": False, "url": ""}}


def _tab_list(*urls):
    return [{"tabs": [{"id": 31, "active": False, "title": "t", "url": url}]} for url in urls]


def test_a_new_tab_that_redirects_to_a_blocked_host_is_judged_once_it_has_landed(lane):
    """The controller reports a created tab before its navigation has committed, with no address yet."""
    broker = lane(REDIRECTING_TAB, {"browser_tabs": _tab_list("", "", SECRET_PAGE["url"])})

    out = _call("browser_tab_create", {"url": "https://example.test/redirect"})

    _assert_withheld(out, "blocked.test")
    assert out["tab_id"] == 31 and "browser_tab_close" in out["error"]
    assert [action for action, _ in broker.dispatched] == ["browser_tab_create"] + ["browser_tabs"] * 3


def test_a_new_tab_that_landed_on_an_allowed_page_reports_its_address(lane):
    lane(REDIRECTING_TAB, {"browser_tabs": _tab_list("about:blank", "https://example.test/landed")})

    out = _call("browser_tab_create", {"url": "https://example.test/start"})

    assert out == {"tab": {"id": 31, "windowId": 2, "active": False, "url": "https://example.test/landed"}}


def test_a_new_tab_whose_address_never_arrives_is_reported_as_the_controller_did(lane):
    broker = lane(REDIRECTING_TAB, {"browser_tabs": _tab_list("")})

    assert _call("browser_tab_create", {"url": "https://example.test/start"}) == REDIRECTING_TAB
    assert [action for action, _ in broker.dispatched].count("browser_tabs") == 6


def test_a_controller_that_cannot_list_tabs_leaves_a_new_tab_as_reported(lane):
    lane(REDIRECTING_TAB, {"browser_tabs": RuntimeError("not offered")})

    assert _call("browser_tab_create", {"url": "https://example.test/start"}) == REDIRECTING_TAB


def test_a_blocked_new_tab_is_refused_for_reads_only_on_that_tab_never_the_current_one(lane):
    lane(REDIRECTING_TAB, {"browser_tabs": _tab_list(SECRET_PAGE["url"]),
                           "browser_snapshot": {"url": "https://example.test/ok", "text": "fine", "refs": []}})
    _assert_withheld(_call("browser_tab_create", {"url": "https://example.test/redirect"}), "blocked.test")

    assert _call("browser_snapshot", {})["text"] == "fine"


def test_a_batch_that_ends_on_a_blocked_host_keeps_its_progress_and_drops_the_page(lane):
    lane({
        "status": "completed", "steps": [
            {"index": 0, "action": "click", "status": "clicked"},
            {"index": 1, "action": "navigate", "status": "navigated", "message": "Private account page"},
        ],
        "completed": 2, "remaining": 0, "navigated": True, **SECRET_PAGE,
    })

    out = _call("browser_batch", {"actions": [{"action": "click", "ref": "@e1"}, {"action": "navigate", "url": "https://example.test/x"}]})

    _assert_withheld(out, "blocked.test")
    assert out["status"] == "completed" and out["completed"] == 2 and out["remaining"] == 0
    assert out["steps"] == [
        {"index": 0, "action": "click", "status": "clicked"}, {"index": 1, "action": "navigate", "status": "navigated"}]


@pytest.mark.parametrize("name, args, result", [
    ("browser_navigate", {"url": "https://example.test/start"},
     {"status": "navigated", "url": "https://example.test/landed?q=1", "title": "Landed", "snapshot": {"text": "hello"}}),
    ("browser_back", {}, {"status": "navigated-back", "url": "https://example.test/previous", "title": "Prev"}),
    ("browser_tab_create", {"url": "https://example.test/start"},
     {"tab": {"id": 4, "windowId": 1, "active": False, "url": "https://example.test/start"}}),
    ("browser_batch", {"actions": [{"action": "back"}]},
     {"status": "completed", "steps": [{"index": 0, "action": "back", "status": "navigated-back"}], "completed": 1,
      "remaining": 0, "url": "https://example.test/previous", "title": "Prev", "snapshot": {"text": "hello"}}),
])
def test_an_allowed_landing_is_untouched(lane, name, args, result):
    lane(result)

    assert _call(name, args) == result


def test_a_landing_on_a_private_address_is_not_judged_by_the_private_address_rule(lane):
    page = {"status": "navigated", "url": "http://192.168.1.10/admin", "title": "Router", "snapshot": {"text": "hi"}}
    lane(page)

    assert _call("browser_navigate", {"url": "http://192.168.1.10/admin"}) == page


@pytest.mark.parametrize("name, args, result", [
    ("browser_navigate", {"url": "https://example.test/start"}, {"status": "already-open"}),
    ("browser_batch", {"actions": [{"action": "back"}]}, {"status": "stopped", "steps": [], "completed": 0, "remaining": 1,
                                                          "snapshot_error": "page_unavailable"}),
])
def test_a_result_that_reports_no_landed_url_is_untouched(lane, name, args, result):
    lane(result)

    assert _call(name, args) == result


def test_a_string_result_from_the_controller_is_checked_too():
    broker = FakeBroker(json.dumps({"status": "navigated", **SECRET_PAGE}))
    out = json.loads(route_browser_tool(
        "browser_navigate", {"url": "https://example.test"}, fallback=lambda: pytest.fail("extension lane"),
        broker=broker, enabled=True, principal_id="principal", transport_family="transport",
        result_guard=browser_tool._extension_landing_guard))

    _assert_withheld(out, "blocked.test")


BLOCKED_PAGE = {"url": SECRET_PAGE["url"], "title": SECRET_PAGE["title"], "snapshot": SECRET_PAGE["snapshot"]}
PREVIOUS_PAGE = {"status": "navigated-back", "url": "https://example.test/previous", "title": "Prev"}


@pytest.mark.parametrize("name, args, status", [
    ("browser_click", {"ref": "@e1"}, "clicked"),
    ("browser_press", {"key": "Enter"}, "pressed"),
    ("browser_type", {"ref": "@e1", "text": "hi"}, "typed"),
    ("browser_fill", {"ref": "@e1", "value": "hi"}, "filled"),
    ("browser_select", {"ref": "@e1", "value": "a"}, "selected"),
    ("browser_hover", {"ref": "@e1"}, "hovered"),
    ("browser_scroll", {"direction": "down"}, "scrolled"),
    ("browser_scroll_to", {"ref": "@e1"}, "scrolled-to-target"),
])
def test_an_action_that_lands_on_a_blocked_page_is_withheld(lane, name, args, status):
    """A click on a link, a submitted form or a key press can end on a blocked page, and the extension returns
    that page's address, title and snapshot with the result."""
    lane({"status": status, "navigated": True, **BLOCKED_PAGE})

    out = _call(name, args)

    _assert_withheld(out, "blocked.test")


def test_a_click_that_lands_on_a_cloud_metadata_page_is_withheld(lane):
    lane({"status": "clicked", "navigated": True, "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
          "title": "iam", "snapshot": SECRET_PAGE["snapshot"]})

    out = _call("browser_click", {"ref": "@e1"})

    _assert_withheld(out, "169.254.169.254")
    assert "security-credentials" not in json.dumps(out)


def test_a_snapshot_of_a_blocked_page_is_withheld_and_the_tab_is_not_moved(lane):
    """A read reports the page the tab is already on; Hermes never navigates the person's tab for it."""
    broker = lane({"url": SECRET_PAGE["url"], "title": "Private account page", "text": "Delete account", "refs": []})

    out = _call("browser_snapshot", {})

    _assert_withheld(out, "blocked.test")
    assert [action for action, _ in broker.dispatched] == ["browser_snapshot"]


def test_an_allowed_action_result_is_untouched(lane):
    page = {"status": "clicked", "navigated": True, "url": "https://example.test/next", "title": "Next",
            "snapshot": {"text": "hello"}}
    lane(page)

    assert _call("browser_click", {"ref": "@e1"}) == page


def test_a_navigation_that_lands_on_a_blocked_page_is_taken_back_when_it_can_be(lane):
    broker = lane({"status": "navigated", "navigated": True, **BLOCKED_PAGE}, {"browser_back": PREVIOUS_PAGE})

    out = _call("browser_navigate", {"url": "https://example.test/start"})

    _assert_withheld(out, "blocked.test")
    assert "taken back to the previous page" in out["error"] and "still on that page" not in out["error"]
    assert [action for action, _ in broker.dispatched] == ["browser_navigate", "browser_back"]
    # The tab is off the page, so it works again.
    broker.replies["browser_snapshot"] = {"url": PREVIOUS_PAGE["url"], "text": "fine", "refs": []}
    assert _call("browser_snapshot", {})["text"] == "fine"
    assert [action for action, _ in broker.dispatched][-1] == "browser_snapshot"


def test_a_click_that_did_not_navigate_is_not_taken_back(lane):
    """Only a call that moved the page can undo the move; a click that stayed on the page just reports it."""
    broker = lane({"status": "clicked", "navigated": False, **BLOCKED_PAGE})

    _assert_withheld(_call("browser_click", {"ref": "@e1"}), "blocked.test")

    assert [action for action, _ in broker.dispatched] == ["browser_click"]


@pytest.mark.parametrize("back", [
    {"status": "navigated-back", "url": "https://blocked.test/earlier", "title": "x"},  # still blocked
    {"status": "navigated-back"},  # no page to go back to, no address
    RuntimeError("no history"),
])
def test_a_tab_that_cannot_be_taken_back_is_refused_until_it_leaves_the_page(lane, back):
    broker = lane({"status": "navigated", "navigated": True, **BLOCKED_PAGE}, {"browser_back": back})
    out = _call("browser_navigate", {"url": "https://example.test/start"})
    assert "still on that page" in out["error"]
    sent = len(broker.dispatched)

    for name, args in [("browser_snapshot", {}), ("browser_screenshot", {}), ("browser_click", {"ref": "@e1"}),
                       ("browser_type", {"ref": "@e1", "text": "x"}), ("browser_press", {"key": "Enter"}),
                       ("browser_batch", {"actions": [{"action": "click", "ref": "@e1"}]})]:
        refused = _call(name, args)
        assert refused["success"] is False and refused["code"] == "tab_withheld", name
        assert refused["blocked_host"] == "blocked.test" and name in refused["error"]
        assert not [leak for leak in PAGE_LEAKS if leak in json.dumps(refused)]
    assert len(broker.dispatched) == sent  # nothing reached the browser


def test_a_withheld_tab_can_still_leave_the_page_or_be_closed(lane):
    broker = lane({"status": "navigated", "navigated": True, **BLOCKED_PAGE}, {"browser_back": RuntimeError("no history")})
    _call("browser_navigate", {"url": "https://example.test/start"})
    assert _call("browser_snapshot", {})["code"] == "tab_withheld"

    # Navigating elsewhere reaches the browser, and a clean landing lifts the block.
    broker.replies["browser_navigate"] = {"status": "navigated", "url": "https://example.test/ok", "title": "Ok"}
    assert _call("browser_navigate", {"url": "https://example.test/ok"})["status"] == "navigated"
    broker.replies["browser_snapshot"] = {"url": "https://example.test/ok", "text": "fine", "refs": []}
    assert _call("browser_snapshot", {})["text"] == "fine"


def test_a_batch_that_starts_by_leaving_the_page_is_not_refused(lane):
    broker = lane({"status": "navigated", "navigated": True, **BLOCKED_PAGE}, {"browser_back": RuntimeError("no history")})
    _call("browser_navigate", {"url": "https://example.test/start"})
    broker.replies["browser_batch"] = {"status": "completed", "steps": [], "completed": 1, "remaining": 0,
                                       "url": "https://example.test/ok", "snapshot": {"text": "fine"}}

    out = _call("browser_batch", {"actions": [{"action": "back"}, {"action": "click", "ref": "@e1"}]})

    assert out["status"] == "completed"
    assert _call("browser_snapshot", {})["code"] != "tab_withheld"


def test_a_tab_list_names_a_blocked_tab_by_its_host_only(lane):
    listing = {"tabs": [
        {"id": 1, "active": True, "title": "Fine", "url": "https://example.test/a"},
        {"id": 2, "active": False, "title": "Private account page", "url": SECRET_PAGE["url"]},
    ]}
    lane(listing)

    out = _call("browser_tabs", {})

    assert out["tabs"][0] == {"id": 1, "active": True, "title": "Fine", "url": "https://example.test/a"}
    assert out["tabs"][1] == {"id": 2, "active": False, "withheld": True, "blocked_host": "blocked.test"}
    assert not [leak for leak in LEAKS if leak in json.dumps(out)]


def test_a_tab_list_without_blocked_tabs_is_untouched(lane):
    listing = {"tabs": [{"id": 1, "active": True, "title": "Fine", "url": "https://example.test/a"}]}
    lane(listing)

    assert _call("browser_tabs", {}) == listing


def test_the_built_in_lane_is_unchanged(monkeypatch):
    """Extension control off: the legacy handler's own result goes back as it is, whatever its URL says."""
    import gateway.browser_control_broker as gw

    monkeypatch.setattr(gw, "browser_control_enabled", lambda: False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: {
        "message": "Blocked by website policy", "host": "blocked.test", "rule": "blocked.test", "source": "config"})
    legacy = json.dumps({"success": True, **SECRET_PAGE})
    monkeypatch.setattr(browser_tool, "browser_navigate", lambda url, task_id=None: legacy)

    assert _raw("browser_navigate", {"url": "https://example.test/start"}) == legacy


def test_the_result_guard_runs_only_on_the_extension_lane_and_after_dispatch():
    seen = []

    def guard(action, args, result, route):
        seen.append((action, args, result))
        assert route.key == (None, "principal", "transport") and callable(route.send)
        return "guarded"

    broker = FakeBroker({"status": "navigated"})
    out = route_browser_tool(
        "browser_navigate", {"url": "https://example.test"}, fallback=lambda: pytest.fail("extension lane"),
        broker=broker, enabled=True, principal_id="principal", transport_family="transport", result_guard=guard)
    assert out == "guarded"
    assert seen == [("browser_navigate", {"url": "https://example.test"}, {"status": "navigated"})]

    seen.clear()
    assert route_browser_tool(
        "browser_navigate", {"url": "https://example.test"}, fallback=lambda: "legacy", broker=broker, enabled=False,
        result_guard=guard) == "legacy"
    assert seen == []


@pytest.fixture
def stuck_tab(lane):
    """Tab 7 navigated to a blocked page and could not be taken back."""
    broker = lane({"status": "navigated", "navigated": True, **BLOCKED_PAGE}, {"browser_back": RuntimeError("no history")})
    out = _call("browser_navigate", {"url": "https://example.test/start", "tab_id": 7})
    assert "still on that page" in out["error"]
    broker.dispatched.clear()
    return broker


def test_a_navigation_in_a_chosen_tab_is_taken_back_in_that_tab(lane):
    broker = lane({"status": "navigated", "navigated": True, **BLOCKED_PAGE}, {"browser_back": PREVIOUS_PAGE})

    out = _call("browser_navigate", {"url": "https://example.test/start", "tab_id": 7})

    assert "taken back to the previous page" in out["error"]
    assert broker.dispatched[-1] == ("browser_back", {"tab_id": 7})


def test_only_the_tab_that_landed_on_a_blocked_page_is_refused(stuck_tab):
    for name, args in [("browser_snapshot", {"tab_id": 7}), ("browser_screenshot", {"tab_id": 7}),
                       ("browser_click", {"ref": "@e1", "tab_id": 7}),
                       ("browser_batch", {"actions": [{"action": "click", "ref": "@e1"}], "tab_id": 7})]:
        refused = _call(name, args)
        assert refused["code"] == "tab_withheld" and refused["blocked_host"] == "blocked.test", name
    assert stuck_tab.dispatched == []

    stuck_tab.replies["browser_snapshot"] = {"url": "https://example.test/ok", "text": "fine", "refs": []}
    assert _call("browser_snapshot", {})["text"] == "fine"  # the current tab
    assert _call("browser_snapshot", {"tab_id": 8})["text"] == "fine"  # another tab


def test_a_chosen_tab_leaves_a_blocked_page_by_navigating_or_by_closing(stuck_tab):
    stuck_tab.replies["browser_navigate"] = {"status": "navigated", "url": "https://example.test/ok", "title": "Ok"}
    stuck_tab.replies["browser_snapshot"] = {"url": "https://example.test/ok", "text": "fine", "refs": []}
    assert _call("browser_navigate", {"url": "https://example.test/ok", "tab_id": 7})["status"] == "navigated"
    assert _call("browser_snapshot", {"tab_id": 7})["text"] == "fine"

    stuck_tab.replies["browser_navigate"] = {"status": "navigated", "navigated": True, **BLOCKED_PAGE}
    _call("browser_navigate", {"url": "https://example.test/start", "tab_id": 7})
    assert _call("browser_snapshot", {"tab_id": 7})["code"] == "tab_withheld"
    stuck_tab.replies["browser_tab_close"] = {"status": "tab-closed"}
    assert _call("browser_tab_close", {"tab_id": 7}) == {"status": "tab-closed"}
    assert _call("browser_snapshot", {"tab_id": 7})["text"] == "fine"


def test_a_new_tab_on_a_blocked_page_is_refused_by_its_id_until_it_is_closed(lane):
    lane(REDIRECTING_TAB, {"browser_tabs": _tab_list(SECRET_PAGE["url"]),
                           "browser_tab_close": {"status": "tab-closed"}})
    _assert_withheld(_call("browser_tab_create", {"url": "https://example.test/redirect"}), "blocked.test")

    refused = _call("browser_screenshot", {"tab_id": 31})
    assert refused["code"] == "tab_withheld" and refused["blocked_host"] == "blocked.test"
    assert _call("browser_tab_close", {"tab_id": 31}) == {"status": "tab-closed"}
    assert _call("browser_screenshot", {"tab_id": 31}).get("code") != "tab_withheld"


def test_a_tab_list_that_shows_a_withheld_tab_on_an_allowed_page_lifts_the_block(stuck_tab):
    stuck_tab.replies["browser_tabs"] = {"tabs": [{"id": 7, "active": False, "title": "t", "url": "https://example.test/ok"}]}
    stuck_tab.replies["browser_snapshot"] = {"url": "https://example.test/ok", "text": "fine", "refs": []}

    _call("browser_tabs", {})

    assert _call("browser_snapshot", {"tab_id": 7})["text"] == "fine"


def test_a_tab_list_that_shows_a_tab_on_a_blocked_page_refuses_it(lane):
    broker = lane({"tabs": [{"id": 9, "active": False, "title": "Private account page", "url": SECRET_PAGE["url"]}]})
    _call("browser_tabs", {})

    assert _call("browser_snapshot", {"tab_id": 9})["code"] == "tab_withheld"
    assert [action for action, _ in broker.dispatched] == ["browser_tabs"]
