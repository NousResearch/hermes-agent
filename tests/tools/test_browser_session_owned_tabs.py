"""Tabs browser_exec opens on a shared browser close when its session is cleaned up, and no others do.

Regression for #21532: on a ``/browser connect`` / ``browser.cdp_url`` browser every ``new_tab()`` stayed
open forever, because cleanup only ever released browsers Hermes launched. The recorder runs inside the
real browser-harness CLI (only its daemon transport is faked); the close runs against an in-process CDP
WebSocket server, so the real ``Target.getTargets`` / ``Target.closeTarget`` traffic is what is asserted.
"""
import asyncio
import json
import os
import socket
import sys
import textwrap
import threading
import time
import types

import pytest

import tools.browser_use_cli as bu_cli
from tests.tools.test_browser_cdp_tool import _CDPServer
from tools import browser_session as bs
from tools import browser_session_cloud as bs_cloud
from tools import browser_session_lifecycle as lifecycle
from tools import browser_session_owned_tabs as owned_tabs
from tools import browser_use_tab_recorder as recorder

USER_TAB = "USER-TAB"

# sitecustomize on the harness's PYTHONPATH: the real browser_harness CLI runs (helpers, new_tab, close_tab,
# switch_tab), only its daemon transport is faked. Created targets are numbered NEW-1, NEW-2, ...
_FAKE_TRANSPORT = textwrap.dedent("""\
    import itertools, json, os
    import browser_harness.helpers as h

    LOG = os.environ["FAKE_CDP_LOG"]
    ids = itertools.count(int(os.environ.get("FAKE_FIRST_ID", "1")))

    def cdp(method, session_id=None, _response_timeout=5.0, **params):
        with open(LOG, "a") as f:
            f.write(json.dumps({"method": method, **params}) + "\\n")
        if method == "Target.createTarget":
            return {"targetId": "NEW-%d" % next(ids)}
        if method == "Target.attachToTarget":
            return {"sessionId": "S-" + params["targetId"]}
        if method == "Target.closeTarget":
            return {"success": True}
        if method == "Page.captureScreenshot":
            return {"data": "aGVybWVz"}
        return {}

    h.cdp = cdp
    h._send = lambda req, **kw: {}
    h.current_tab = lambda: {"targetId": "USER-TAB", "target_id": "USER-TAB",
                             "url": "https://user.example/", "title": "user"}
    import browser_harness.admin as admin, browser_harness.telemetry as telemetry
    admin.ensure_daemon = admin.print_update_banner = lambda *a, **k: None
    admin.daemon_alive = lambda *a, **k: True
    telemetry.is_enabled = lambda: False
    telemetry.capture_cli_event = lambda *a, **k: None
""")


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch, tmp_path):
    for name in ("_active_sessions", "_session_last_activity", "_session_owner_homes", "_cleanup_failures",
                 "_owned_routes", "_calls_in_flight"):
        monkeypatch.setattr(bs, name, {})
    monkeypatch.setattr(bs, "_teardowns", set())
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: None)  # the janitor is driven by hand
    monkeypatch.delenv("BU_NAME", raising=False)
    monkeypatch.delenv("BU_AUTOSPAWN", raising=False)
    bs_cloud.reset_caches()
    yield
    bs_cloud.reset_caches()


class _Browser:
    """An in-process CDP endpoint whose open page targets are ``self.open``."""

    def __init__(self, server: _CDPServer, url: str, *targets: str):
        self.server, self.url, self.open = server, url, set(targets)
        self.refuse: set[str] = set()
        server.on("Target.getTargets", lambda p, s: {"targetInfos": [
            {"targetId": t, "type": "page", "url": "about:blank"} for t in sorted(self.open)]})
        server.on("Target.closeTarget", self._close)

    def _close(self, params, _session):
        target = params["targetId"]
        if target in self.refuse:
            return RuntimeError(f"cannot close {target}")
        self.open.discard(target)
        return {"success": True}

    def closed(self) -> list[str]:
        return [m["params"]["targetId"] for m in self.server.received() if m["method"] == "Target.closeTarget"]


@pytest.fixture
def browser():
    servers = []

    def start(*targets: str) -> _Browser:
        server = _CDPServer()
        url = server.start()
        servers.append(server)
        return _Browser(server, url, *targets)

    yield start
    for server in servers:
        server.stop()


@pytest.fixture
def run_exec(monkeypatch, tmp_path):
    """browser_exec on the CDP-override route, through the real harness CLI with the fake transport."""
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu_cli, "_exec_route_is_sidecar", {})
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: None)
    fake_dir = tmp_path / "fake_transport"
    fake_dir.mkdir()
    (fake_dir / "sitecustomize.py").write_text(_FAKE_TRANSPORT, encoding="utf-8")
    site_dir = bu_cli._harness_site_dir()
    assert site_dir, "browser-harness is a core dependency of the test environment"
    monkeypatch.setattr(bu_cli, "_harness_site_dir", lambda: os.pathsep.join([str(fake_dir), site_dir]))
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [sys.executable, "-m", "browser_harness.run"])
    log = tmp_path / "cdp.jsonl"
    monkeypatch.setenv("FAKE_CDP_LOG", str(log))

    def run(endpoint: str, code: str, *, task_id: str = "turn-1", session: str = "", first_id: int = 1):
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: endpoint)
        monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: endpoint)
        monkeypatch.setenv("FAKE_FIRST_ID", str(first_id))
        return json.loads(bu_cli.browser_exec(code, session=session, task_id=task_id, timeout_s=60))

    return run


def _owned(route_key: str) -> set[str]:
    route = bs._owned_routes.get(route_key)
    return set().union(*route.targets.values()) if route else set()


# ── 1. Recorder: exact created ids, from inside the real harness ────────────────────────────────────

def test_new_tab_records_exactly_the_created_tab_and_close_tab_removes_it(run_exec):
    endpoint = "ws://127.0.0.1:1/devtools/browser/x"
    result = run_exec(endpoint, "a = new_tab(); b = new_tab(); close_tab(a); print('TABS', a, b)")

    assert result["success"] is True, result
    assert "TABS NEW-1 NEW-2" in result["output"]
    assert bs._owned_routes["turn-1"].targets == {endpoint: {"NEW-2"}}


def test_switching_to_the_users_tab_records_nothing(run_exec):
    result = run_exec("ws://127.0.0.1:1/devtools/browser/x", f"switch_tab({USER_TAB!r}); print('ok')")

    assert result["success"] is True, result
    assert _owned("turn-1") == set()


def test_a_call_that_raises_or_dies_still_reports_its_tabs(run_exec):
    raised = run_exec("ws://127.0.0.1:1/devtools/browser/x", "new_tab()\nraise RuntimeError('boom')")
    died = run_exec("ws://127.0.0.1:1/devtools/browser/x", "new_tab()\nimport os; os._exit(3)", first_id=7)

    assert raised["success"] is False and died["exit_code"] == 3
    assert _owned("turn-1") == {"NEW-1", "NEW-7"}


def test_a_later_call_closing_an_earlier_calls_tab_releases_it(run_exec):
    run_exec("ws://127.0.0.1:1/devtools/browser/x", "new_tab()", session="r7k2")
    run_exec("ws://127.0.0.1:1/devtools/browser/x", "close_tab('NEW-1')", session="r7k2")

    assert _owned("bu-named-r7k2") == set()


def test_an_unwritable_report_never_fails_the_models_code(run_exec, monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(owned_tabs, "tab_report_path", lambda endpoint, private: str(blocker / "tabs.json"))
    shot = tmp_path / "shot.png"

    result = run_exec("ws://127.0.0.1:1/devtools/browser/x",
                      f"new_tab(); new_tab(); print('SHOT', capture_screenshot({str(shot)!r}))")

    assert result["success"] is True, result
    assert f"SHOT {shot}" in result["output"] and shot.read_bytes() == b"hermes"
    assert result["stderr"].count("cannot record the tabs") == 1


def test_the_report_variable_reaches_neither_the_models_code_nor_its_subprocesses(run_exec):
    code = ("import os, subprocess, sys\n"
            "print('SELF', os.environ.get('_HERMES_BU_TAB_REPORT'))\n"
            "print(subprocess.run([sys.executable, '-c', \"import os; print('CHILD', os.environ.get('_HERMES_BU_TAB_REPORT'))\"],"
            " capture_output=True, text=True).stdout.strip())\n"
            "new_tab()")
    result = run_exec("ws://127.0.0.1:1/devtools/browser/x", code)

    assert result["success"] is True, result
    assert "SELF None" in result["output"] and "CHILD None" in result["output"]
    assert _owned("turn-1") == {"NEW-1"}  # the recorder still holds its path


def test_a_report_variable_in_hermess_own_environment_is_never_passed_on(run_exec, monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text("browser:\n  cdp_close_agent_tabs: false\n", encoding="utf-8")
    stale = tmp_path / "stale.json"
    monkeypatch.setenv(recorder.REPORT_ENV, str(stale))

    result = run_exec("ws://127.0.0.1:1/devtools/browser/x",
                      "import os; new_tab(); print('SEEN', os.environ.get('_HERMES_BU_TAB_REPORT'))")

    assert result["success"] is True, result
    assert "SEEN None" in result["output"]
    assert not stale.exists()


def test_concurrent_cdp_calls_record_every_tab(tmp_path, monkeypatch):
    """Two threads in one call: each event's rewrite is serialised, so no write loses another's tab."""
    report = tmp_path / "tabs.json"
    monkeypatch.setenv(recorder.REPORT_ENV, str(report))
    ids = iter(["T-1", "T-2"])
    helpers = types.SimpleNamespace(cdp=lambda method, **params: {"targetId": next(ids)})
    recorder.install(helpers)
    persist, writers, overlapped = recorder._persist, threading.Semaphore(1), threading.Event()

    def slow_persist(path, state):
        if not writers.acquire(blocking=False):  # another writer is in here too
            overlapped.set()
            return persist(path, state)
        try:
            overlapped.wait(0.5)  # hold on, giving the other writer every chance to enter
            persist(path, state)
        finally:
            writers.release()
    monkeypatch.setattr(recorder, "_persist", slow_persist)
    results, errors = [], []

    def create():
        try:
            results.append(helpers.cdp("Target.createTarget", url="about:blank"))
        except Exception as exc:  # health: allow BLE001 -- surfaced by the assertion below
            errors.append(exc)
    threads = [threading.Thread(target=create) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == [] and sorted(r["targetId"] for r in results) == ["T-1", "T-2"]
    assert not overlapped.is_set()
    assert sorted(json.loads(report.read_text(encoding="utf-8"))["created"]) == ["T-1", "T-2"]


def test_a_private_provider_browser_records_nothing(run_exec, monkeypatch):
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: object())
    monkeypatch.setattr(bs, "get_session_info", lambda key: {"cdp_url": "ws://127.0.0.1:1/devtools/browser/p"})
    result = run_exec("", "new_tab()")

    assert result["success"] is True, result
    assert bs._owned_routes == {}


# ── 2. Close selection ──────────────────────────────────────────────────────────────────────────────

def _own(route_key: str, endpoint: str, *targets: str, home_key: str = "home") -> None:
    bs._owned_routes[route_key] = owned_tabs.OwnedRoute(home_key=home_key, targets={endpoint: set(targets)})


def test_close_sends_closetarget_only_for_owned_tabs_still_open(browser):
    b = browser(USER_TAB, "OWNED-A")
    _own("turn-1", b.url, "OWNED-A", "OWNED-B-ALREADY-GONE")

    lifecycle.cleanup_browser("turn-1")

    assert b.closed() == ["OWNED-A"]
    assert b.open == {USER_TAB}
    assert "turn-1" not in bs._owned_routes


def _age(route_key: str) -> None:
    bs._session_last_activity[route_key] = time.time() - bs.BROWSER_SESSION_INACTIVITY_TIMEOUT - 5


def test_a_tab_the_browser_refuses_to_close_is_retried_by_the_janitor(browser):
    b = browser(USER_TAB, "OWNED-A", "OWNED-C")
    b.refuse.add("OWNED-C")
    _own("bu-named-r7k2", b.url, "OWNED-A", "OWNED-C")
    _age("bu-named-r7k2")

    lifecycle.cleanup_inactive_sessions()
    assert b.open == {USER_TAB, "OWNED-C"}
    assert _owned("bu-named-r7k2") == {"OWNED-C"}
    assert "bu-named-r7k2" in bs._session_last_activity  # still scheduled

    b.refuse.clear()
    _age("bu-named-r7k2")
    lifecycle.cleanup_inactive_sessions()

    assert sorted(b.closed()) == ["OWNED-A", "OWNED-C", "OWNED-C"]
    assert b.open == {USER_TAB}
    assert bs._owned_routes == {} and bs._session_last_activity == {}


def _dead_endpoint() -> str:
    with socket.socket() as probe:  # a port nothing listens on any more
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"ws://127.0.0.1:{port}/devtools/browser/gone"


def test_the_janitor_drops_an_unreachable_browsers_ids_after_three_passes():
    _own("bu-named-r7k2", _dead_endpoint(), "OWNED-A")
    lifecycle.update_session_activity("bu-named-r7k2")

    for _ in range(owned_tabs._MAX_UNREACHABLE - 1):
        _age("bu-named-r7k2")
        lifecycle.cleanup_inactive_sessions()
        assert _owned("bu-named-r7k2") == {"OWNED-A"}
        assert "bu-named-r7k2" in bs._session_last_activity
    _age("bu-named-r7k2")
    lifecycle.cleanup_inactive_sessions()

    assert bs._owned_routes == {}
    assert bs._session_last_activity == {} and bs._session_owner_homes == {}


def test_a_stalled_close_is_retried_without_counting_against_the_browser_or_starving_its_other_tabs(
        browser, monkeypatch):
    """A close that stalls is not an unreachable browser: its id is kept however often it stalls. The pass
    goes on to the next tab, so the healthy one closes whichever tab a batch starts at (other routes' batches
    advance the rotation in between); another browser in the batch is not held up."""
    import itertools
    from tools import browser_cdp_tool
    stalls, fine = browser(USER_TAB, "A-STALLS", "B-HEALTHY"), browser(USER_TAB, "C-CLOSES")
    real_call = browser_cdp_tool._cdp_call

    async def cdp_call(ws_url, method, params, target_id, timeout):
        if method == "Target.closeTarget" and params["targetId"] == "A-STALLS":
            raise TimeoutError("Timed out waiting for Target.closeTarget")  # what _cdp_call raises on a stall
        return await real_call(ws_url, method, params, target_id, timeout)
    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", cdp_call)
    _own("turn-1", stalls.url, "A-STALLS", "B-HEALTHY")
    _own("turn-1@home-a", fine.url, "C-CLOSES")
    monkeypatch.setattr(owned_tabs, "_rotation", itertools.count())

    for _ in range(owned_tabs._MAX_UNREACHABLE):
        lifecycle.cleanup_browser("turn-1")
        next(owned_tabs._rotation)  # another route's batch in between: the healthy tab must not need a parity

    assert fine.open == {USER_TAB} and stalls.open == {USER_TAB, "A-STALLS"}
    assert _owned("turn-1") == {"A-STALLS"} and bs._owned_routes["turn-1"].misses == {}


def test_a_browser_that_stalls_on_every_close_does_not_take_the_batch_from_another_browser(browser, monkeypatch):
    """A browser that lists its targets but hangs on every close ends its pass after two stalls in a row (a real
    stall costs up to 5 s, so trying every tab could spend the whole batch budget), and a healthy browser in the
    same batch still closes its tabs; the stalled ids stay recorded for the next cleanup."""
    import itertools
    from tools import browser_cdp_tool
    wedged, fine = browser(USER_TAB, "W-1", "W-2", "W-3", "W-4"), browser(USER_TAB, "H-1", "H-2")
    real_call = browser_cdp_tool._cdp_call
    stalled: list[str] = []

    async def cdp_call(ws_url, method, params, target_id, timeout):
        if method == "Target.closeTarget" and params["targetId"].startswith("W-"):
            stalled.append(params["targetId"])  # each real stall costs up to _CLOSE_TIMEOUT_S of the batch
            raise TimeoutError("Timed out waiting for Target.closeTarget")
        return await real_call(ws_url, method, params, target_id, timeout)
    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", cdp_call)
    _own("turn-1", wedged.url, "W-1", "W-2", "W-3", "W-4")
    _own("turn-1@home-a", fine.url, "H-1", "H-2")
    first = sorted([wedged.url, fine.url]).index(wedged.url)
    monkeypatch.setattr(owned_tabs, "_rotation", itertools.count(first))  # the wedged browser goes first

    lifecycle.cleanup_browser("turn-1")

    assert len(stalled) == owned_tabs._MAX_STALLS_PER_PASS
    assert fine.open == {USER_TAB}
    assert _owned("turn-1") == {"W-1", "W-2", "W-3", "W-4"} and bs._owned_routes["turn-1"].misses == {}


class _Blackhole:
    """A port that accepts connections and never answers (a hung or firewalled CDP endpoint)."""

    def __init__(self, scheme: str = "ws", path: str = "/devtools/browser/hung"):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.accepted: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()
        self.url = f"{scheme}://127.0.0.1:{self.sock.getsockname()[1]}{path}"

    def _accept(self):
        while True:
            try:
                self.accepted.append(self.sock.accept()[0])
            except OSError:
                return

    def close(self):
        self.sock.close()
        for conn in self.accepted:
            conn.close()


@pytest.mark.parametrize("batch", ["turn", "janitor"])
def test_a_batch_dials_a_hung_browser_once_for_all_its_routes(monkeypatch, batch):
    monkeypatch.setattr(owned_tabs, "_CLOSE_TIMEOUT_S", 2.0)
    hung = _Blackhole()
    keys = ("turn-1", "turn-1@home-a", "turn-1@home-b")
    try:
        for n, key in enumerate(keys):
            _own(key, hung.url, f"TAB-{n}")
            _age(key)

        {"turn": lambda: lifecycle.cleanup_browser("turn-1"), "janitor": lifecycle.cleanup_inactive_sessions}[batch]()

        assert len(hung.accepted) == 1
        assert {key: route.misses for key, route in bs._owned_routes.items()} == {key: {hung.url: 1} for key in keys}
    finally:
        hung.close()


def test_exit_dials_a_hung_browser_once_however_many_exit_passes_run(monkeypatch):
    """The TUI and gateway run the exit cleanup two or three times: only the first dials."""
    monkeypatch.setattr(owned_tabs, "_CLOSE_TIMEOUT_S", 2.0)
    monkeypatch.setattr("tools.browser_launcher.stop_all_browsers", lambda: None)
    hung = _Blackhole()
    try:
        _own("turn-1", hung.url, "TAB-1")
        _own("bu-named-r7k2", hung.url, "TAB-2")
        lifecycle.cleanup_all_browsers()
        lifecycle.cleanup_all_browsers()

        assert len(hung.accepted) == 1 and bs._owned_routes == {}
    finally:
        hung.close()


def test_no_request_starts_after_the_batch_budget_and_the_rest_is_left_for_later(monkeypatch):
    """Three hung discovery endpoints and a 2.5 s budget: the batch returns at the budget (no request
    starts after it; each has its own timeout), and the endpoints it ran out of time for keep their ids
    without a miss."""
    monkeypatch.setattr(owned_tabs, "_BATCH_BUDGET_S", 2.5)
    hung = [_Blackhole("http", "") for _ in range(3)]
    try:
        for n, endpoint in enumerate(hung):
            _own(f"turn-1@home-{n}", endpoint.url, f"TAB-{n}")
        started = time.monotonic()
        lifecycle.cleanup_browser("turn-1")
        elapsed = time.monotonic() - started

        assert elapsed < owned_tabs._BATCH_BUDGET_S + 2.0
        assert all(route.misses == {} and route.targets for route in bs._owned_routes.values())
    finally:
        for endpoint in hung:
            endpoint.close()


def test_an_acknowledged_close_whose_re_check_times_out_keeps_the_tab(browser, monkeypatch):
    """closeTarget answers success=true but the tab is still open and the re-check times out: the id
    was never seen gone, so it stays for the next cleanup."""
    from tools import browser_cdp_tool
    b = browser(USER_TAB, "A")
    real_call, asked = browser_cdp_tool._cdp_call, []

    async def cdp_call(ws_url, method, params, target_id, timeout):
        asked.append(method)
        if method == "Target.closeTarget":
            return {"success": True}  # acknowledged, not closed
        if asked.count("Target.getTargets") > 1:
            raise TimeoutError("Timed out waiting for Target.getTargets")
        return await real_call(ws_url, method, params, target_id, timeout)
    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", cdp_call)
    _own("turn-1", b.url, "A")

    lifecycle.cleanup_browser("turn-1")

    assert "Target.closeTarget" in asked and _owned("turn-1") == {"A"}


def test_a_close_answering_success_false_at_the_deadline_keeps_the_tab(browser, monkeypatch):
    from tools import browser_cdp_tool
    b = browser(USER_TAB, "A")
    real_call, clock = browser_cdp_tool._cdp_call, {"skew": 0.0}
    monkeypatch.setattr(owned_tabs, "time", types.SimpleNamespace(monotonic=lambda: time.monotonic() + clock["skew"]))

    async def cdp_call(ws_url, method, params, target_id, timeout):
        if method == "Target.closeTarget":
            clock["skew"] = owned_tabs._BATCH_BUDGET_S  # the batch budget runs out while the browser answers
            return {"success": False}
        return await real_call(ws_url, method, params, target_id, timeout)
    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", cdp_call)
    _own("turn-1", b.url, "A")

    lifecycle.cleanup_browser("turn-1")

    assert b.open == {USER_TAB, "A"} and _owned("turn-1") == {"A"}


def test_a_browser_the_budget_never_reached_is_reached_first_by_the_next_batch(browser, monkeypatch):
    import itertools
    monkeypatch.setattr(owned_tabs, "_BATCH_BUDGET_S", 2.0)
    hung, fine = _Blackhole(), browser(USER_TAB, "LATE-TAB")
    # the first batch starts at the hung browser, the next one at the other
    monkeypatch.setattr(owned_tabs, "_rotation", itertools.count(sorted([hung.url, fine.url]).index(hung.url)))
    try:
        _own("turn-1", hung.url, "EARLY-TAB")
        _own("turn-1@home-a", fine.url, "LATE-TAB")
        lifecycle.cleanup_browser("turn-1")
        assert fine.server.received() == [] and _owned("turn-1@home-a") == {"LATE-TAB"}
        assert bs._owned_routes["turn-1@home-a"].misses == {} and bs._owned_routes["turn-1"].misses == {}
        lifecycle.cleanup_browser("turn-1")

        assert fine.open == {USER_TAB} and "turn-1@home-a" not in bs._owned_routes
    finally:
        hung.close()


# ── 3. Lifecycle wiring ─────────────────────────────────────────────────────────────────────────────

def test_per_turn_cleanup_closes_the_turns_tabs_with_no_session_record(run_exec, browser, reloads):
    b = browser(USER_TAB)
    for n in range(1, 6):  # five calls, each opening a tab, as models do
        result = run_exec(b.url, "new_tab()", first_id=n)
        assert result["success"] is True, result
        b.open.add(f"NEW-{n}")  # the fake transport's tab, now open in the browser
    assert bs._active_sessions == {}

    lifecycle.cleanup_browser("turn-1")

    assert b.open == {USER_TAB}
    assert sorted(b.closed()) == [f"NEW-{n}" for n in range(1, 6)]
    assert _owned("turn-1") == set()


def test_a_named_sessions_tabs_survive_the_turn_and_its_calls_alone_make_the_janitor_close_them(run_exec, browser):
    """No session record and no hand-seeded activity: browser_exec's own report is what the janitor sees."""
    b = browser(USER_TAB)
    result = run_exec(b.url, "new_tab()", session="r7k2")
    b.open.add("NEW-1")
    assert result["success"] is True, result

    lifecycle.cleanup_browser("turn-1")
    lifecycle.cleanup_inactive_sessions()  # still fresh: nothing closes
    assert b.open == {USER_TAB, "NEW-1"}
    _age("bu-named-r7k2")
    lifecycle.cleanup_inactive_sessions()

    assert b.open == {USER_TAB} and bs._owned_routes == {}


def test_a_profile_switch_closes_that_profiles_tabs_on_the_old_endpoint_once(browser, monkeypatch):
    """``/browser connect`` reaps around publishing the new endpoint: the first reap closes the tabs on the
    endpoint they were opened on and drops their records, so the second dials nothing; the other
    profile's tabs stay."""
    from hermes_constants import hermes_home_key
    old, new = browser(USER_TAB, "OLD-TAB", "STUCK", "B-TAB"), browser("NEW-USER-TAB")
    old.refuse.add("STUCK")
    _own("bu-named-r7k2", old.url, "OLD-TAB", "STUCK", home_key=hermes_home_key())
    _own("turn-1@home-b", old.url, "B-TAB", home_key="home-b")
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: new.url)

    lifecycle.cleanup_profile_browsers(hermes_home_key())
    dialled = len(old.server.received())
    lifecycle.cleanup_profile_browsers(hermes_home_key())

    assert old.open == {USER_TAB, "STUCK", "B-TAB"} and "bu-named-r7k2" not in bs._owned_routes
    assert len(old.server.received()) == dialled and new.server.received() == []
    assert _owned("turn-1@home-b") == {"B-TAB"}


def test_the_janitor_never_closes_a_route_while_a_call_drives_it(run_exec, browser, monkeypatch):
    """A browser_exec call may run past the inactivity timeout: its route is in flight, not idle. No
    teardown but exit's touches it (tabs, supervisor, backend); once the call is done the janitor does."""
    stopped, released = [], []
    monkeypatch.setattr("tools.browser_session_cdp.stop_cdp_supervisor", stopped.append)
    monkeypatch.setattr(lifecycle, "release_session_backend", lambda key, info: released.append(key))
    b = browser(USER_TAB)
    assert run_exec(b.url, "new_tab()", session="r7k2")["success"] is True
    b.open.add("NEW-1")
    bs._active_sessions["bu-named-r7k2"] = {"launcher_key": "persistent", "features": {}}
    real_run, seen = bu_cli._run_cli_killing_process_group, []

    def long_call(cmd, code, env, timeout):
        assert time.time() - bs._session_last_activity["bu-named-r7k2"] < 60  # touched as the call starts
        _age("bu-named-r7k2")  # the call has outlasted the inactivity timeout
        lifecycle.cleanup_inactive_sessions()
        lifecycle.cleanup_single_session("bu-named-r7k2")
        seen.append(set(b.open))
        return real_run(cmd, code, env, timeout)
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", long_call)
    assert run_exec(b.url, "print('still working')", session="r7k2")["success"] is True

    assert seen == [{USER_TAB, "NEW-1"}] and b.closed() == [] and stopped == released == []
    assert bs._calls_in_flight == {}
    _age("bu-named-r7k2")
    lifecycle.cleanup_inactive_sessions()
    assert b.open == {USER_TAB} and released == ["bu-named-r7k2"]


def test_the_janitor_skips_a_route_a_call_used_after_its_idle_snapshot(browser, monkeypatch):
    b = browser(USER_TAB, "A")
    _own("bu-named-r7k2", b.url, "A")
    _age("bu-named-r7k2")
    real_check = lifecycle._human_holds_shared_browser

    def call_runs_after_the_snapshot(key):
        with owned_tabs.route_call(key):
            pass
        return real_check(key)
    monkeypatch.setattr(lifecycle, "_human_holds_shared_browser", call_runs_after_the_snapshot)

    lifecycle.cleanup_inactive_sessions()

    assert b.closed() == [] and _owned("bu-named-r7k2") == {"A"}


@pytest.mark.parametrize("later_call_elsewhere", [False, True])
def test_the_janitor_leaves_a_real_profile_route_alone_while_a_human_holds_the_bot_desktop(
        run_exec, browser, reloads, monkeypatch, later_call_elsewhere):
    """The real-profile browser runs on the Bot Desktop with no session record: the lease that keeps the
    janitor off a browser a human took over (#110064) must reach its route too, also once a later call of the
    route went to another browser while the tab stayed recorded. After handback it closes."""
    from tools.bot_desktop import lease
    b, other = browser(USER_TAB), browser("OTHER-USER-TAB")
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: True)
    monkeypatch.setattr("tools.browser_session_real_profile.real_profile_cdp", lambda: (b.url, None))
    assert run_exec("", "new_tab()")["success"] is True
    b.open.add("NEW-1")
    if later_call_elsewhere:  # an operator CDP override owns the route's next call (_resolve_real_profile_cdp)
        assert run_exec(other.url, "print(1)")["success"] is True
    assert bs._active_sessions == {} and _owned("turn-1") == {"NEW-1"}

    monkeypatch.setattr(lease, "human_holds", lambda: True)
    stops = len(reloads.log)
    _age("turn-1")
    lifecycle.cleanup_inactive_sessions()
    assert b.open == {USER_TAB, "NEW-1"} and len(reloads.log) == stops

    monkeypatch.setattr(lease, "human_holds", lambda: False)
    _age("turn-1")
    lifecycle.cleanup_inactive_sessions()
    assert b.open == {USER_TAB} and bs._owned_routes == {}


def test_a_call_that_starts_during_a_teardown_waits_for_it(browser, monkeypatch):
    """Teardown first, then a call on the same route: the call must not drive tabs being closed."""
    b = browser(USER_TAB, "A")
    _own("turn-1", b.url, "A")
    real_close, closing, resume, events = owned_tabs._close_on, threading.Event(), threading.Event(), []

    def paused_close(*args):
        closing.set()  # the teardown has claimed the route and is about to close its tabs
        resume.wait(10)
        events.append("closing")
        return real_close(*args)
    monkeypatch.setattr(owned_tabs, "_close_on", paused_close)
    cleanup = threading.Thread(target=lifecycle.cleanup_browser, args=("turn-1",))
    cleanup.start()
    assert closing.wait(10)

    def call():
        with owned_tabs.route_call("turn-1"):
            events.append("call")
    caller = threading.Thread(target=call)
    caller.start()
    caller.join(0.5)
    assert caller.is_alive() and events == []  # blocked behind the teardown
    resume.set()
    cleanup.join(10)
    caller.join(10)

    assert events == ["closing", "call"]
    assert b.open == {USER_TAB} and bs._teardowns == set() and bs._calls_in_flight == {}


def test_a_call_still_blocked_by_a_teardown_fails_retryably_instead_of_running_under_it(run_exec, monkeypatch):
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(owned_tabs, "_TEARDOWN_WAIT_S", 2.0)
    ran = []
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", lambda *a: ran.append(a))
    bs._teardowns.add("turn-1")  # a cleanup of this route that does not finish

    result = run_exec("ws://127.0.0.1:1/devtools/browser/x", "new_tab()")

    assert "retry" in result["error"] and ran == []
    assert list((get_hermes_home() / "cache" / "browser-use" / "tabs").glob("*")) == []


def test_each_key_of_a_batch_is_free_again_as_soon_as_its_own_cleanup_is_done(monkeypatch):
    monkeypatch.setattr(owned_tabs, "_TEARDOWN_WAIT_S", 10.0)
    for key in ("turn-1", "turn-1@home-a"):
        bs._active_sessions[key] = {"features": {}}
    entered = threading.Event()

    def stop_supervisor(key):
        if key == "turn-1@home-a":  # the batch's second key: the first is done
            def call():
                with owned_tabs.route_call("turn-1"):
                    entered.set()
            caller = threading.Thread(target=call)
            caller.start()
            caller.join(5)
            assert entered.is_set()
    monkeypatch.setattr("tools.browser_session_cdp.stop_cdp_supervisor", stop_supervisor)

    lifecycle.cleanup_browser("turn-1")

    assert entered.is_set() and bs._active_sessions == {}


def test_a_call_that_finishes_leaves_an_overlapping_calls_hold_on_the_route(browser):
    """Two calls on one route overlap: the first to finish must not drop the other's hold, or the turn end
    closes the tabs the second call is still driving."""
    b = browser(USER_TAB, "A")
    _own("turn-1", b.url, "A")
    first, second = owned_tabs.route_call("turn-1"), owned_tabs.route_call("turn-1")
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)

    lifecycle.cleanup_browser("turn-1")
    assert b.closed() == [] and _owned("turn-1") == {"A"}  # the second call still holds it

    second.__exit__(None, None, None)
    lifecycle.cleanup_browser("turn-1")
    assert b.open == {USER_TAB} and bs._calls_in_flight == {}


def test_a_release_that_raises_frees_every_key_of_the_batch(monkeypatch):
    """The first key's release raises: the batch's other key is not left claimed, so a later call on it
    runs instead of waiting out the teardown and failing with TeardownBusy."""
    monkeypatch.setattr(owned_tabs, "_TEARDOWN_WAIT_S", 2.0)

    def release(key):
        if key == "turn-1":
            raise RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        owned_tabs.teardown(["turn-1", "turn-2"], release, stop_daemons=False)

    assert bs._teardowns == set()
    with owned_tabs.route_call("turn-2"):  # would raise TeardownBusy with turn-2 still claimed
        pass


def test_a_cleanup_while_the_backend_resolves_leaves_the_calls_session_alone(run_exec, browser, monkeypatch):
    """The call holds its route before its backend resolves: a cleanup in between releases nothing the
    call is about to drive."""
    released = []
    monkeypatch.setattr(lifecycle, "release_session_backend", lambda key, info: released.append(key))
    bs._active_sessions["turn-1"] = {"launcher_key": "persistent", "features": {}}
    real_report_path = owned_tabs.tab_report_path

    def cleanup_then_report(endpoint, private):
        lifecycle.cleanup_browser("turn-1")
        return real_report_path(endpoint, private)
    monkeypatch.setattr(owned_tabs, "tab_report_path", cleanup_then_report)
    b = browser(USER_TAB)

    assert run_exec(b.url, "print(1)")["success"] is True
    assert released == [] and "turn-1" in bs._active_sessions


def test_a_turn_end_during_a_hybrid_sidecar_call_leaves_the_sidecar_browser_alone(monkeypatch):
    import subprocess
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_session_cdp.get_cdp_override_raw", lambda: "")
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: object())
    monkeypatch.setattr(bs_cloud, "auto_local_for_private_urls", lambda: True)
    monkeypatch.setattr(bu_cli, "_exec_route_is_sidecar", {})
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/bin/true"])
    sidecar = {"cdp_url": "ws://127.0.0.1:47000/devtools/browser/s", "features": {"local": True}}
    monkeypatch.setattr(bs, "get_session_info", lambda key: bs._active_sessions.setdefault(key, sidecar))
    released = []
    monkeypatch.setattr(lifecycle, "release_session_backend", lambda key, info: released.append(key))

    def call_during_turn_end(cmd, code, env, timeout):
        lifecycle.cleanup_browser("t")  # another thread's turn end, as far as the claims go
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", call_during_turn_end)

    result = json.loads(bu_cli.browser_exec('new_tab("http://192.168.0.1:9120/login")', task_id="t"))

    assert result["success"] is True, result
    assert released == [] and "t::local" in bs._active_sessions


# ── 3b. An unnamed route's own harness daemon ───────────────────────────────────────────────────────

@pytest.fixture
def reloads(monkeypatch):
    stopped: list = []
    failing: set = set()

    def reload(daemons, timeout=None):
        stopped.extend(daemons)
        return {d for d in daemons if d[-1] not in failing}
    monkeypatch.setattr(bu_cli, "_reload_daemons", reload)
    stopped_names = types.SimpleNamespace(log=stopped, failing=failing)
    return stopped_names


def test_an_unnamed_routes_own_daemon_lives_across_turns_and_stops_when_the_session_ends(run_exec, browser, reloads):
    b = browser(USER_TAB)
    name = bu_cli.route_daemon_name("turn-1")
    for turn in (1, 2):
        result = run_exec(b.url, "import os; new_tab(); print('NAME', os.environ['BU_NAME'])", first_id=turn)
        b.open.add(f"NEW-{turn}")
        assert result["success"] is True, result
        assert f"NAME {name}" in result["output"] and bu_cli._SESSION_RE.match(name) is None  # never a user name
        lifecycle.cleanup_browser("turn-1")
        assert b.open == {USER_TAB} and reloads.log == []  # only the call's tabs: the daemon stays
        assert bs._owned_routes["turn-1"].daemon[-1] == name

    reloads.failing.add(name)  # a stop that fails keeps the daemon tracked and the route scheduled
    _age("turn-1")
    lifecycle.cleanup_inactive_sessions()
    assert bs._owned_routes["turn-1"].daemon[-1] == name and "turn-1" in bs._session_last_activity
    reloads.failing.clear()
    _age("turn-1")
    lifecycle.cleanup_inactive_sessions()

    assert [d[-1] for d in reloads.log] == [name, name]
    assert bs._owned_routes == {} and bu_cli._driven_daemons == {}


def test_each_routes_daemon_stays_one_over_many_turns_and_exit_stops_them(run_exec, browser, reloads, monkeypatch):
    monkeypatch.setattr("tools.browser_launcher.stop_all_browsers", lambda: None)
    b = browser(USER_TAB)
    for _turn in range(5):
        for task in ("task-a", "task-b"):
            assert run_exec(b.url, "print(1)", task_id=task)["success"] is True
            lifecycle.cleanup_browser(task)
        assert len({route.daemon for route in bs._owned_routes.values()}) == 2 and reloads.log == []

    lifecycle.cleanup_all_browsers()

    assert sorted(d[-1] for d in reloads.log) == sorted(bu_cli.route_daemon_name(t) for t in ("task-a", "task-b"))
    assert bs._owned_routes == {} and bu_cli._driven_daemons == {}


def test_named_and_private_routes_keep_their_daemon_names(run_exec, monkeypatch, browser, reloads):
    b = browser(USER_TAB)
    named = run_exec(b.url, "import os; print('NAME', os.environ['BU_NAME'])", session="r7k2")
    monkeypatch.setattr(bs_cloud, "get_cloud_provider", lambda: object())
    monkeypatch.setattr(bs, "get_session_info", lambda key: {"cdp_url": "ws://127.0.0.1:1/devtools/browser/p"})
    private = run_exec("", "import os; print('NAME', os.environ.get('BU_NAME'))", task_id="turn-2")

    assert "NAME r7k2" in named["output"] and "NAME None" in private["output"]
    assert [(k, r.daemon[-1]) for k, r in bs._owned_routes.items()] == [("bu-named-r7k2", "r7k2")]  # stopped at session end


@pytest.mark.parametrize("route", ["turn-1", "bu-named-r7k2"])
def test_a_daemons_own_tab_closes_with_its_session_even_when_its_stop_is_cut_short(
        browser, reloads, monkeypatch, tmp_path, route):
    """``--reload`` SIGTERMs a daemon slow to exit before it closes its dedicated tab: the batch closes it."""
    b = browser(USER_TAB, "DAEMON-TAB")
    monkeypatch.setattr(bu_cli, "_base_subprocess_env", lambda: {"BH_HOME": str(tmp_path)})
    name = bu_cli.route_daemon_name(route) if route == "turn-1" else "r7k2"
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / f"bu-{name}.log").write_text(f"named daemon {name}: created dedicated tab (DAEMON-TAB)\n",
                                                     encoding="utf-8")
    bs._owned_routes[route] = owned_tabs.OwnedRoute(home_key="home", daemon=bu_cli.daemon_identity({"BU_NAME": name}),
                                                    daemon_endpoint=b.url)
    lifecycle.cleanup_browser("turn-1")
    assert b.open == {USER_TAB, "DAEMON-TAB"}  # turn end: the daemon and its tab stay
    _age(route)

    lifecycle.cleanup_inactive_sessions()

    assert b.open == {USER_TAB} and bs._owned_routes == {} and [d[-1] for d in reloads.log] == [name]


def test_generated_route_daemons_get_their_own_socket_under_an_operator_runtime_dir(tmp_path, monkeypatch):
    """With ``BH_RUNTIME_DIR`` set and no ``BH_RUNTIME_DIR_SHARED``, the harness names every daemon's
    socket ``bu.sock``: two routes' daemons would be one. Asked of the real harness."""
    import subprocess

    from tools import browser_harness_dirs
    short = []
    monkeypatch.setattr(browser_harness_dirs, "_claim_short_dir", lambda key, budget: short.append(budget) or None)
    sockets = []
    for task in ("task-a", "task-b"):
        env = {"BH_RUNTIME_DIR": str(tmp_path / "rt"), "BH_TMP_DIR": str(tmp_path / "tmp")}
        assert bu_cli._name_daemon(env, "", task, False, "ws://127.0.0.1:1/devtools/browser/x", False)
        probe = "import os; from browser_harness import _ipc; print(_ipc._sock_path(os.environ['BU_NAME']))"
        out = subprocess.run([sys.executable, "-c", probe], env={**os.environ, **env, "PYTHONPATH": bu_cli._harness_site_dir() or ""},
                             capture_output=True, text=True, timeout=60, check=True).stdout.strip()
        sockets.append(out)

    assert len(set(sockets)) == 2 and all(bu_cli.route_daemon_name(t) in s for t, s in zip(("task-a", "task-b"), sockets))
    assert len(short) == 2  # the deep pytest dir overflows sun_path with the longer name: a short dir is sought


def test_a_stopped_route_daemons_log_is_deleted_and_no_other(tmp_path, monkeypatch):
    import subprocess
    logs = tmp_path / "tmp"
    logs.mkdir()
    route, user = bu_cli.route_daemon_name("cron-1"), "work"
    for name in (route, user, bu_cli.route_daemon_name("cron-2")):
        (logs / f"bu-{name}.log").write_text("log", encoding="utf-8")
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["browser-harness"])
    monkeypatch.setattr(bu_cli, "_base_subprocess_env", lambda: {"BH_HOME": str(tmp_path)})
    exits = {route: 0, user: 0, bu_cli.route_daemon_name("cron-2"): 1}
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: subprocess.CompletedProcess(cmd, exits[env["BU_NAME"]]))

    stopped = bu_cli._reload_daemons([bu_cli.daemon_identity({"BU_NAME": n}) for n in exits])

    assert {d[-1] for d in stopped} == {route, user}
    assert sorted(p.name for p in logs.iterdir()) == sorted(f"bu-{n}.log" for n in (user, bu_cli.route_daemon_name("cron-2")))


def test_only_a_route_daemon_stop_waits_30_s(monkeypatch):
    import subprocess
    timeouts: list = []
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["browser-harness"])
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: timeouts.append(timeout) or subprocess.CompletedProcess(cmd, 0))
    daemon = bu_cli.daemon_identity({"BU_NAME": bu_cli.route_daemon_name("t")})
    bu_cli._driven_daemons[daemon] = "home"
    bu_cli.stop_harness_daemons()  # the exit sweep and the republish stop keep 15 s
    bs._owned_routes["t"] = owned_tabs.OwnedRoute(home_key="home", daemon=daemon, daemon_endpoint="http://x")
    owned_tabs.stop_route_daemon("t")
    assert timeouts == [15, 30] and "t" not in bs._owned_routes


def test_a_daemon_stop_that_finishes_after_a_newer_daemon_was_recorded_keeps_the_newer_one(monkeypatch):
    """An exit or swap pass does not wait for calls: a call can record a new daemon for the route while the
    old one is being stopped. The stop's success must clear only the daemon it stopped."""
    old = bu_cli.daemon_identity({"BU_NAME": bu_cli.route_daemon_name("t")})
    new = bu_cli.daemon_identity({"BU_NAME": bu_cli.route_daemon_name("t@home-b")})
    bs._owned_routes["t"] = owned_tabs.OwnedRoute(home_key="home", daemon=old, daemon_endpoint="http://old")

    def reload(daemons, timeout=None):
        route = bs._owned_routes["t"]
        route.daemon, route.daemon_endpoint = new, "http://new"  # a call recorded its daemon meanwhile
        return set(daemons)
    monkeypatch.setattr(bu_cli, "_reload_daemons", reload)

    owned_tabs.stop_route_daemon("t")

    assert bs._owned_routes["t"].daemon == new and bs._owned_routes["t"].daemon_endpoint == "http://new"


def test_route_daemons_are_scheduled_for_the_janitor_even_without_a_report_file(run_exec, browser, reloads, monkeypatch):
    """No report file can be made: each first-use call's own daemon is still scheduled (activity, owner
    home, janitor started) and the janitor stops every one of them."""
    started = []
    monkeypatch.setattr(lifecycle, "start_cleanup_thread", lambda: started.append(1))
    monkeypatch.setattr(owned_tabs, "tab_report_path", lambda endpoint, private: None)
    b = browser(USER_TAB)
    tasks = [f"task-{n}" for n in range(4)]
    for task in tasks:
        assert run_exec(b.url, "print(1)", task_id=task)["success"] is True

    assert set(bs._session_last_activity) == set(bs._session_owner_homes) == set(tasks) == set(bs._owned_routes)
    assert started
    for task in tasks:
        _age(task)
    lifecycle.cleanup_inactive_sessions()

    assert sorted(d[-1] for d in reloads.log) == sorted(bu_cli.route_daemon_name(t) for t in tasks)
    assert bs._owned_routes == {}


def test_a_daemon_moving_to_a_new_browser_keeps_the_old_endpoint_until_its_stop_succeeds(run_exec, browser, reloads):
    old, new = browser(USER_TAB), browser(USER_TAB)
    assert run_exec(old.url, "print(1)", session="r7k2")["success"] is True

    reloads.failing.add("r7k2")  # the daemon latched the old browser and will not stop yet
    failed = run_exec(new.url, "print('ran')", session="r7k2")
    assert "retry" in failed["error"] and bs._owned_routes["bu-named-r7k2"].daemon_endpoint == old.url
    assert "bu-named-r7k2" in bs._session_last_activity  # the janitor retries the stop

    reloads.failing.clear()
    moved = run_exec(new.url, "print('ran')", session="r7k2")

    assert moved["success"] is True and "ran" in moved["output"]
    assert bs._owned_routes["bu-named-r7k2"].daemon_endpoint == new.url
    assert [d[-1] for d in reloads.log] == ["r7k2", "r7k2"]


def test_one_tasks_turn_end_leaves_another_tasks_running_call_alone(run_exec, browser, reloads):
    """Two unnamed tasks on one shared browser: each drives its own daemon, so ending task A's turn
    neither closes task B's tab nor touches the daemon B's call is running on."""
    b = browser(USER_TAB)
    assert run_exec(b.url, "new_tab()", task_id="task-a")["success"] is True
    assert run_exec(b.url, "new_tab()", task_id="task-b", first_id=2)["success"] is True
    b.open.update({"NEW-1", "NEW-2"})
    assert bs._owned_routes["task-a"].daemon != bs._owned_routes["task-b"].daemon

    with owned_tabs.route_call("task-b"):
        lifecycle.cleanup_browser("task-a")
        lifecycle.cleanup_browser("task-b")  # B's own turn cannot end its running call either
        assert b.open == {USER_TAB, "NEW-2"}
    lifecycle.cleanup_browser("task-b")

    assert b.open == {USER_TAB} and reloads.log == []


def test_a_backend_swap_closes_the_recorded_tabs_and_stops_the_route_daemons(run_exec, browser, reloads, monkeypatch):
    """CLI ``/browser connect|disconnect`` with no call running: every route's tabs close on the old
    browser and its own daemon stops; nothing is left to dial afterwards."""
    monkeypatch.setattr("tools.browser_launcher.stop_all_browsers", lambda: None)
    b = browser(USER_TAB)
    assert run_exec(b.url, "new_tab()")["success"] is True
    b.open.add("NEW-1")

    lifecycle.cleanup_all_browsers()

    assert b.open == {USER_TAB} and [d[-1] for d in reloads.log] == [bu_cli.route_daemon_name("turn-1")]
    assert bs._owned_routes == {}


@pytest.mark.parametrize("swap", ["all", "profile"])
def test_a_backend_swap_keeps_a_route_daemon_whose_stop_failed_for_a_retry(run_exec, browser, reloads, monkeypatch, swap):
    """A swap drops the tab records, but a daemon whose stop failed stays recorded with the endpoint it
    latched: the next call gets a retryable error instead of the old browser, and the janitor retries."""
    monkeypatch.setattr("tools.browser_launcher.stop_all_browsers", lambda: None)
    b = browser(USER_TAB)
    name = bu_cli.route_daemon_name("turn-1")
    assert run_exec(b.url, "new_tab()")["success"] is True
    b.open.add("NEW-1")
    endpoint = bs._owned_routes["turn-1"].daemon_endpoint
    reloads.failing.add(name)

    if swap == "all":
        lifecycle.cleanup_all_browsers()
    else:
        lifecycle.cleanup_profile_browsers(bs._owned_routes["turn-1"].home_key)

    assert b.open == {USER_TAB}  # the recorded tabs still close
    route = bs._owned_routes["turn-1"]
    assert route.daemon[-1] == name and route.daemon_endpoint == endpoint and not route.targets
    assert "turn-1" in bs._session_last_activity  # scheduled for the janitor
    reloads.failing.clear()
    _age("turn-1")
    lifecycle.cleanup_inactive_sessions()
    assert [d[-1] for d in reloads.log] == [name, name] and bs._owned_routes == {}


@pytest.mark.parametrize("serve_ends", ["stdin-eof", "broken-pipe"])
def test_the_stdio_entry_exit_closes_the_agents_tabs_on_a_hostname_endpoint(browser, monkeypatch, serve_ends):
    """At ``atexit`` asyncio cannot resolve ``localhost`` any more: the stdio entry's exit closes the tabs
    before it gets there, also when a failed stdout write ends it with ``sys.exit(0)`` (``_write_or_exit``),
    and before it gives up the profile's ownership."""
    from gateway.runtime_ownership import process_ownership
    from tui_gateway import entry
    b = browser(USER_TAB, "A")
    released = []

    def serve():
        if serve_ends == "broken-pipe":
            raise SystemExit(0)
    monkeypatch.setattr("tools.browser_launcher.stop_all_browsers", lambda: None)
    monkeypatch.setattr(entry, "_claim_launch_home", lambda: "home")
    monkeypatch.setattr(entry, "_serve", serve)
    monkeypatch.setattr(process_ownership, "release", lambda home: released.append(set(b.open)))
    _own("bu-named-r7k2", b.url.replace("127.0.0.1", "localhost"), "A")

    if serve_ends == "broken-pipe":
        with pytest.raises(SystemExit):
            entry.main()
    else:
        entry.main()

    assert b.open == {USER_TAB} and bs._owned_routes == {}
    assert released == [{USER_TAB}]  # the tabs were closed before ownership went


def test_exit_closes_every_route_even_with_a_call_in_flight(browser, monkeypatch):
    monkeypatch.setattr("tools.browser_launcher.stop_all_browsers", lambda: None)
    monkeypatch.setattr(lifecycle, "reap_orphaned_browser_sessions", lambda: 0)
    monkeypatch.setattr(bs, "_cleanup_done", False)
    b = browser(USER_TAB, "A", "B")
    _own("turn-1", b.url, "A")
    _own("bu-named-r7k2", b.url, "B")
    with owned_tabs.route_call("bu-named-r7k2"):
        lifecycle.emergency_cleanup_all_sessions()  # exit

    assert b.open == {USER_TAB}
    assert bs._owned_routes == {}


# ── 4. Loop safety ──────────────────────────────────────────────────────────────────────────────────

def test_close_runs_under_a_running_event_loop_and_never_through_a_cdp_supervisor(browser, monkeypatch):
    from tools import browser_supervisor

    class _Registry:
        def stop(self, key):
            stopped.append(key)

        def __getattr__(self, name):
            raise AssertionError(f"supervisor registry used for the close: {name}")

    stopped = []
    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
    b = browser(USER_TAB, "A")
    _own("turn-1", b.url, "A")

    async def gateway_turn_end():
        lifecycle.cleanup_browser("turn-1")  # the gateway calls cleanup with its loop running

    asyncio.run(gateway_turn_end())

    assert b.open == {USER_TAB} and stopped == []  # a route with no session record keeps its supervisor


# ── 5. Session names ────────────────────────────────────────────────────────────────────────────────

def test_session_named_default_is_rejected(monkeypatch):
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: ["/bin/true"])
    result = json.loads(bu_cli.browser_exec("print(1)", session="default"))

    assert "reserved" in result["error"]


# ── 6. Opt-out ──────────────────────────────────────────────────────────────────────────────────────

def test_cdp_close_agent_tabs_false_records_and_closes_nothing(run_exec, browser):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text("browser:\n  cdp_close_agent_tabs: false\n", encoding="utf-8")
    b = browser(USER_TAB)

    result = run_exec(b.url, "new_tab()")
    b.open.add("NEW-1")
    lifecycle.cleanup_browser("turn-1")

    assert result["success"] is True, result
    assert b.open == {USER_TAB, "NEW-1"} and b.server.received() == []
