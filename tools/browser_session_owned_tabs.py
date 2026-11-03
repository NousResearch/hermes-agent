"""Tabs ``browser_exec`` opened on a shared browser, and closing exactly those when its session ends.

A shared browser (``/browser connect``, ``BROWSER_CDP_URL``, ``browser.cdp_url``, an operator ``BU_CDP_*``
endpoint, the real-profile or a persistent-profile Chromium) outlives every Hermes session, so a tab the
agent leaves open stays open until the user closes it (#21532). Each such call runs with the tab recorder
(``browser_use_tab_recorder``), which reports the exact ``targetId`` of every tab the call created; the
ids are merged here per route key (``browser_use_cli._backend_cache_key``) and closed by the existing
lifecycle: per-turn ``cleanup_browser`` for the task's unnamed routes, the inactivity janitor for named
sessions, ``cleanup_profile_browsers`` before a ``/browser connect`` switch, and ``cleanup_all_browsers``
(exit, CLI ``/browser connect|disconnect``).

An unnamed route on a shared browser drives its own harness daemon (``browser_use_cli.route_daemon_name``),
so it works in the daemon's own tab instead of the browser's first page (the user's). That daemon lives
across turns (turn end closes only the tabs the calls opened) and is stopped when the route's session
ends: the janitor, a profile switch or backend swap, exit.

A call claims its route, and the route's hybrid ``::local`` sidecar key, before its backend resolves and
until it returns (``route_call``); a teardown claims each key it cleans up (``teardown``). A call waits for
a running teardown of its keys (a retryable error after ``_TEARDOWN_WAIT_S``); the turn-end and janitor
teardowns skip a key a call holds. A backend swap, profile switch, dead-backend replacement or exit tears
everything down, running calls included. One batch asks each endpoint for its targets once for all of its
routes, with one connection per CDP request; no request starts after the batch's time budget runs out
(each request has its own timeout). Only recorded ids are ever closed, so a tab the user opened, or one
the agent merely switched to, stays. Tabs that would not close stay registered and the janitor retries
them; ids on an endpoint whose target list could not be read (connection refused, a hung endpoint, a CDP
error on ``Target.getTargets``) for three cleanups are dropped. The close runs on its own one-shot
browser-level connections (``browser_cdp_tool._cdp_call`` under ``_run_async``), never on a CDP
supervisor's loop. ``browser.cdp_close_agent_tabs: false`` turns it off.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional

from agent.deadline import poll_until
from agent.redact import redact_cdp_url
from hermes_constants import get_hermes_home, hermes_home_key
from tools.browser_use_tab_recorder import REPORT_ENV, TMP_SUFFIX
from utils import is_truthy_value, read_json_or_empty

logger = logging.getLogger(__name__)

_CLOSE_TIMEOUT_S = 5.0
_DISCOVERY_TIMEOUT_S = 10.0
# Chrome answers Target.closeTarget before the target is gone: the re-check polls this long for it to go.
_GONE_WITHIN_S = 3.0
_GONE_POLL_S = 0.25
# Cleanups that could not read a browser's target list before its ids are dropped: a browser that stays unreachable
# was closed or restarted (its tabs went with it), and a dead endpoint must not cost every cleanup a dial.
_MAX_UNREACHABLE = 3
# One batch's discovery, connects, closes and re-checks over every endpoint together: no request starts
# after it runs out (a request already running is bounded by its own timeout), and work it leaves undone
# is kept for the next cleanup.
_BATCH_BUDGET_S = 30.0
# Consecutive stalled closes that end an endpoint's pass: a browser that lists its targets but hangs on every
# close (a hung renderer) would otherwise spend the whole batch budget and starve the other endpoints.
_MAX_STALLS_PER_PASS = 2
# How long a browser_exec call waits for a teardown of its route before it fails with a retryable error.
_TEARDOWN_WAIT_S = 120.0
_rotation = itertools.count()  # each batch starts at a different endpoint and tab: one stall starves none


@dataclass
class OwnedRoute:
    """One route's resources on a shared browser (``browser_session._owned_routes``)."""

    home_key: str
    targets: dict[str, set[str]] = field(default_factory=dict)  # endpoint -> ids of tabs the route opened
    misses: dict[str, int] = field(default_factory=dict)  # endpoint -> cleanups that could not read its targets
    daemon: Optional[tuple] = None  # the route's own harness daemon (daemon_identity), until it stops
    daemon_endpoint: str = ""  # the endpoint it latched at spawn
    # set for every route whose browser runs on the Bot Desktop display (local or lightpanda engine, e.g. the real
    # profile): the janitor honours a human's lease on that display for it
    bot_desktop: bool = False


@dataclass(eq=False)
class Call:
    """One browser_exec call, holding its route keys from before its backend resolves until it returns."""

    keys: tuple[str, ...]
    endpoint: str = ""
    report: Optional[str] = None
    bot_desktop: bool = False

    def track(self, env: dict, endpoint: str, private_browser: bool, own_daemon: bool, named: bool = False,
              bot_desktop: bool = False) -> None:
        """Record what the call drives (BU_NAME is final) and point the recorder at a fresh report. On a
        shared browser the route's daemon (its own, or a named session's) is recorded with the route and
        the route scheduled for the janitor, so the session's end stops it and closes its tab, whether or
        not a report file could be made. A daemon that latched another endpoint at spawn (a relaunched
        browser) is stopped first; until that stop succeeds the old endpoint stays recorded and the call
        fails with a retryable error (``TeardownBusy``) instead of running on the old browser."""
        from tools.browser_harness_dirs import daemon_identity
        self.endpoint, self.bot_desktop, daemon = endpoint, bot_desktop, daemon_identity(env)
        env.pop(REPORT_ENV, None)  # only this call's own report, never one inherited from Hermes's environment
        if own_daemon or (named and owns_tabs_on(endpoint, private_browser)):
            _track_daemon(self.keys[0], daemon, endpoint)
        self.report = tab_report_path(endpoint, private_browser)
        if self.report:
            env[REPORT_ENV] = self.report


def _track_daemon(route_key: str, daemon: tuple, endpoint: str) -> None:
    bs = _bs()
    with bs._cleanup_lock:
        route = _route(route_key)
        moved = route.daemon == daemon and route.daemon_endpoint not in ("", endpoint)
        if not moved:
            route.daemon, route.daemon_endpoint = daemon, endpoint
    if moved:
        from tools.browser_use_cli import ROUTE_DAEMON_STOP_TIMEOUT_S, stop_driven_daemons
        if daemon not in stop_driven_daemons([daemon], ROUTE_DAEMON_STOP_TIMEOUT_S):
            _touch_if_tracked(route_key)  # still recorded on the old endpoint: the janitor retries the stop
            raise TeardownBusy(f"The browser session {route_key!r} could not be moved to the new browser yet; "
                               "retry the call.")
        with bs._cleanup_lock:
            route = _route(route_key)
            route.daemon, route.daemon_endpoint = daemon, endpoint
    _touch_if_tracked(route_key)


class TeardownBusy(RuntimeError):
    """A cleanup of the call's route did not finish within ``_TEARDOWN_WAIT_S``, or its daemon could not
    be moved to a new endpoint: the call can be retried."""


def _bs():
    from tools import browser_session
    return browser_session


def _close_agent_tabs_enabled() -> bool:
    """``browser.cdp_close_agent_tabs`` (default on)."""
    from tools.browser_session_cloud import browser_cfg
    return browser_cfg("cdp_close_agent_tabs", True, lambda v: is_truthy_value(v, default=True))


def owns_tabs_on(endpoint: str, private_browser: bool) -> bool:
    """Whether a call on ``endpoint`` records and closes its tabs: a shared browser, opt-out not set."""
    return bool(endpoint) and not private_browser and _close_agent_tabs_enabled()


def tab_report_path(endpoint: str, private_browser: bool) -> Optional[str]:
    """A fresh 0600 report file for one call on a shared CDP browser, or None (``owns_tabs_on`` is false,
    or the file cannot be made): the recorder is only installed when this returns a path."""
    if not owns_tabs_on(endpoint, private_browser):
        return None
    try:
        directory = Path(get_hermes_home()) / "cache" / "browser-use" / "tabs"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="tabs-", suffix=".json", dir=directory)
    except OSError as exc:
        logger.warning("browser_exec: cannot record the tabs this call opens: %s", exc)
        return None
    os.close(fd)
    return path


def _read_report(path: str) -> tuple[set[str], set[str]]:
    """``(created, closed)`` ids from a report; the file (and a half-written temp) is removed."""
    try:
        state = read_json_or_empty(path)
    finally:
        for leftover in (path, path + TMP_SUFFIX):
            try:
                os.unlink(leftover)
            except OSError:
                pass

    def ids(name: str) -> set[str]:
        value = state.get(name)
        return {t for t in value if isinstance(t, str) and t} if isinstance(value, list) else set()
    return ids("created"), ids("closed")


def _route(route_key: str) -> OwnedRoute:
    """``route_key``'s record, created on first use (caller holds the lock)."""
    routes = _bs()._owned_routes
    if route_key not in routes:
        routes[route_key] = OwnedRoute(home_key=hermes_home_key())
    return routes[route_key]


def record_tab_report(route_key: str, endpoint: str, path: Optional[str]) -> None:
    """Merge one call's report into ``route_key``'s owned tabs and touch the route's activity, so the
    inactivity janitor closes them once the route goes idle."""
    if not path:
        return
    created, closed = _read_report(path)
    bs = _bs()
    with bs._cleanup_lock:
        if created or route_key in bs._owned_routes:
            owned = _route(route_key).targets.setdefault(endpoint, set())
            owned.update(created)
            owned.difference_update(closed)
            _prune(route_key)
    _touch_if_tracked(route_key)


def _touch_if_tracked(route_key: str) -> None:
    bs = _bs()
    with bs._cleanup_lock:
        tracked = route_key in bs._owned_routes
    if tracked:
        from tools.browser_session_lifecycle import start_cleanup_thread, update_session_activity
        start_cleanup_thread()
        update_session_activity(route_key)


@contextlib.contextmanager
def route_call(route_key: str) -> Generator[Call]:
    """Hold ``route_key`` and its hybrid sidecar key for one ``browser_exec`` call, from before its backend
    resolves until it returns: turn-end and janitor cleanups skip what it drives; a backend swap (CLI/TUI
    ``/browser connect|disconnect``), profile switch, dead-backend replacement and exit still release it.
    A teardown of either that is already running finishes first; after ``_TEARDOWN_WAIT_S`` the call fails
    (``TeardownBusy``) instead of running under it. Activity is touched at entry (a call may outlast the
    inactivity timeout) and at the end, when the call's report is merged."""
    bs = _bs()
    call = Call(keys=(route_key, route_key + bs.LOCAL_SUFFIX))
    with bs._teardown_released:
        if not bs._teardown_released.wait_for(lambda: bs._teardowns.isdisjoint(call.keys), _TEARDOWN_WAIT_S):
            raise TeardownBusy(f"A cleanup of browser session {route_key!r} is still running; retry the call.")
        for key in call.keys:
            bs._calls_in_flight.setdefault(key, []).append(call)
    try:
        _touch_if_tracked(route_key)
        yield call
    finally:
        try:
            record_tab_report(route_key, call.endpoint, call.report)  # a timed-out or killed call reports too
        finally:
            with bs._cleanup_lock:
                if route := bs._owned_routes.get(route_key):
                    route.bot_desktop |= call.bot_desktop  # sticky while it owns anything: the janitor's lease check
                for key in call.keys:
                    calls = bs._calls_in_flight.pop(key, [])
                    if left := [c for c in calls if c is not call]:
                        bs._calls_in_flight[key] = left


def teardown(keys: Iterable[str], release: Callable[[str], object] = lambda _key: None, *,
             include_in_flight: bool = False, idle_before: Optional[float] = None,
             stop_daemons: bool = True) -> list[str]:
    """Clean up ``keys`` as one batch and return the keys it claimed. A key another teardown holds is
    skipped; so is one a call holds (it stays scheduled), unless ``include_in_flight`` (exit, a backend
    swap or profile switch, or a dead backend being replaced: the tab records go after this pass, whatever
    closed; a daemon whose stop failed stays recorded for a retry). ``idle_before`` (janitor): a key active
    since that time is skipped. The claimed keys' tabs close first (one discovery per endpoint for the
    batch, one connection per CDP request); then, per key, its own daemon stops
    (``stop_daemons``) and ``release(key)`` runs, and that key's claim goes."""
    bs = _bs()
    claimed, busy = [], []
    with bs._cleanup_lock:
        for key in dict.fromkeys(keys):
            if key in bs._teardowns:
                continue
            if idle_before is not None and bs._session_last_activity.get(key, 0.0) > idle_before:
                continue  # a call ran after the janitor's snapshot
            if key in bs._calls_in_flight and not include_in_flight:
                busy.append(key)
                continue
            bs._teardowns.add(key)
            claimed.append(key)
    from tools.browser_session_lifecycle import _best_effort, update_session_activity
    for key in busy:
        update_session_activity(key)  # retried by the janitor once the call is done
    pending = list(claimed)
    try:
        if claimed and stop_daemons:
            _best_effort("finding the routes' daemon tabs", lambda: _own_daemon_tabs(claimed))
        if claimed:
            _best_effort("closing the tabs browser_exec opened", lambda: _close_batch(claimed, drop_records=include_in_flight))
        while pending:
            key = pending[0]
            try:
                if stop_daemons:
                    _best_effort("stopping the route's harness daemon", lambda: stop_route_daemon(key))
                release(key)
            finally:
                _unclaim([pending.pop(0)])  # each key's claim goes as soon as its own cleanup is done
    finally:
        _unclaim(pending)
    return claimed


def _own_daemon_tabs(keys: list[str]) -> None:
    """Add each stopping route daemon's own tab to the route's tabs, so this batch closes it: the daemon
    would close it itself on a clean stop, but a slow stop is cut short (``route_daemon_tab_ids``)."""
    bs = _bs()
    with bs._cleanup_lock:
        daemons = {key: (route.daemon, route.daemon_endpoint) for key in keys
                   if (route := bs._owned_routes.get(key)) and route.daemon and route.daemon_endpoint}
    if not daemons:
        return
    from tools.browser_use_cli import route_daemon_tab_ids
    found = {key: route_daemon_tab_ids(daemon) for key, (daemon, _endpoint) in daemons.items()}
    with bs._cleanup_lock:
        for key, ids in found.items():
            if ids and (route := bs._owned_routes.get(key)):
                route.targets.setdefault(daemons[key][1], set()).update(ids)


def _unclaim(keys: list[str]) -> None:
    if not keys:
        return
    bs = _bs()
    with bs._teardown_released:
        bs._teardowns.difference_update(keys)
        bs._teardown_released.notify_all()


def stop_route_daemon(route_key: str) -> None:
    """Stop ``route_key``'s own harness daemon. It stays recorded, with the endpoint it latched, until a
    stop succeeds, on every path (a backend swap too): the next cleanup retries it and a call meanwhile
    gets a retryable error instead of the old browser. At process death the owner-death supervisor reaps a
    daemon whose stop failed. Callers hold the key's claim."""
    bs = _bs()
    with bs._cleanup_lock:
        route = bs._owned_routes.get(route_key)
        daemon = route.daemon if route else None
    if daemon is None:
        return
    from tools.browser_use_cli import ROUTE_DAEMON_STOP_TIMEOUT_S, stop_driven_daemons
    stopped = stop_driven_daemons([daemon], ROUTE_DAEMON_STOP_TIMEOUT_S)
    with bs._cleanup_lock:
        route = bs._owned_routes.get(route_key)
        if route is not None and route.daemon == daemon and daemon in stopped:
            route.daemon, route.daemon_endpoint = None, ""
            _prune(route_key)


def _prune(route_key: str) -> None:
    """Drop empty endpoint sets, and the route once it owns nothing (caller holds the lock)."""
    routes = _bs()._owned_routes
    route = routes.get(route_key)
    if route is None:
        return
    for endpoint in [ep for ep, ids in route.targets.items() if not ids]:
        del route.targets[endpoint]
        route.misses.pop(endpoint, None)
    if not route.targets and route.daemon is None:
        del routes[route_key]


class _OutOfTime(Exception):
    """The batch budget ran out before the next request could start."""


def _rotated(items: list, turn: int) -> list:
    return items[turn % len(items):] + items[:turn % len(items)] if items else items


def _close_on(endpoint: str, owned: set[str], deadline: float, turn: int) -> Optional[tuple[bool, set[str]]]:
    """Close ``owned`` tabs still open on ``endpoint`` -> ``(connected, ids seen absent)``, or None when the
    budget ran out before the endpoint was asked. An id not seen absent afterwards (a close that stalls, is
    refused or answers success=false, or a re-check that times out) is kept."""
    from tools.browser_cdp_tool import _WS_AVAILABLE, WebSocketException, _cdp_call, _run_async
    from tools.browser_session_cdp import resolve_cdp_override
    if not _WS_AVAILABLE:
        return False, set()

    def budget(cap: float) -> float:
        left = min(cap, deadline - time.monotonic())
        if left <= 0:
            raise _OutOfTime
        return left

    def cdp(ws: str, method: str, params: dict) -> dict:
        return _run_async(_cdp_call(ws, method, params, None, budget(_CLOSE_TIMEOUT_S)))

    def open_ids() -> set[str]:
        return {str(t.get("targetId")) for t in cdp(ws_url, "Target.getTargets", {}).get("targetInfos", [])
                if isinstance(t, dict)}

    failures = (OSError, RuntimeError, TimeoutError, ValueError, WebSocketException)
    try:
        ws_url = resolve_cdp_override(endpoint, timeout=budget(_DISCOVERY_TIMEOUT_S))
        still = owned & open_ids()
    except _OutOfTime:
        return None
    except failures as exc:
        if time.monotonic() >= deadline:
            return None  # cut short by the batch budget, not a browser that cannot be reached
        logger.warning("Could not reach %s to close %d browser tab(s) Hermes opened: %s",
                       redact_cdp_url(endpoint), len(owned), redact_cdp_url(str(exc)))
        return False, set()
    try:
        stalls = 0
        for target in _rotated(sorted(still), turn):
            try:
                cdp(ws_url, "Target.closeTarget", {"targetId": target})
                stalls = 0
            except (RuntimeError, ValueError) as exc:  # refused: the re-check below keeps the id
                logger.debug("Target.closeTarget %s refused: %s", target, exc)
            except failures as exc:  # stalled: the id is kept too, but a browser that stalls on every close
                logger.debug("Target.closeTarget %s stalled: %s", target, exc)  # does not get the whole budget
                stalls += 1
                if stalls >= _MAX_STALLS_PER_PASS:
                    break
        still = _still_open(owned, open_ids, deadline)
    except (_OutOfTime, *failures) as exc:  # out of budget, or a stalled re-check: the rest waits for the next cleanup
        logger.warning("Closing the browser tabs Hermes opened on %s stalled: %s; retrying at the next cleanup",
                       redact_cdp_url(endpoint), redact_cdp_url(str(exc)) or "out of time")
        # only ids seen absent count as gone: an acknowledged close (even success=true) is not a closed tab
        return True, owned - still
    if still:
        logger.warning("%d browser tab(s) Hermes opened on %s are still open; retrying at the next cleanup",
                       len(still), redact_cdp_url(endpoint))
    return True, owned - still


def _still_open(owned: set[str], open_ids: Callable[[], set[str]], deadline: float) -> set[str]:
    wait = max(0.0, min(_GONE_WITHIN_S, deadline - time.monotonic()))
    if poll_until(lambda: not (owned & open_ids()), wait, _GONE_POLL_S):
        return set()
    return owned & open_ids()


def _close_batch(keys: list[str], drop_records: bool = False) -> None:
    bs = _bs()
    with bs._cleanup_lock:
        snapshot = {key: {ep: set(ids) for ep, ids in route.targets.items()}
                    for key in keys if (route := bs._owned_routes.get(key))}
    by_endpoint: dict[str, set[str]] = {}
    for targets in snapshot.values():
        for endpoint, ids in targets.items():
            by_endpoint.setdefault(endpoint, set()).update(ids)
    deadline, turn = time.monotonic() + _BATCH_BUDGET_S, next(_rotation)
    outcomes: dict[str, Optional[tuple[bool, set[str]]]] = {}
    for endpoint in _rotated(sorted(by_endpoint), turn):
        if time.monotonic() >= deadline:
            break  # not asked this time: its ids wait for the next cleanup, no miss counted
        outcomes[endpoint] = _close_on(endpoint, by_endpoint[endpoint], deadline, turn)
    with bs._cleanup_lock:
        for key, targets in snapshot.items():
            route = bs._owned_routes.get(key)
            if route is not None:
                for endpoint, owned in targets.items():
                    _settle(route, endpoint, owned, outcomes.get(endpoint), drop_records)
                _prune(key)


def _settle(route: OwnedRoute, endpoint: str, owned: set[str], outcome: Optional[tuple[bool, set[str]]],
            drop_records: bool) -> None:
    """Apply one endpoint's outcome to a route (caller holds the lock)."""
    if drop_records:  # swap, profile switch, dead-backend replacement, exit: a later pass must not dial again
        route.targets.pop(endpoint, None)
        return
    if outcome is None:
        return
    connected, gone = outcome
    if not connected:
        route.misses[endpoint] = route.misses.get(endpoint, 0) + 1
        if route.misses[endpoint] < _MAX_UNREACHABLE:
            return
        gone = owned  # given up on: see _MAX_UNREACHABLE
    route.misses.pop(endpoint, None)
    route.targets.get(endpoint, set()).difference_update(gone)
