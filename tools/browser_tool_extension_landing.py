"""Where an extension page landed: the check after the fact for the browser extension lane.

``_extension_url_guard`` (in ``tools.browser_tool``) judges the URL that was asked for, before anything is
sent. A redirect, a clicked link or a submitted form can end somewhere else, so the URL the controller reports
back is judged too, by the same policy, for every result that carries a page address (navigate, back, click,
press, type, fill, select, hover, scroll, snapshot, the page a batch ended on and each tab in a tab list).
A page that fails is withheld whole, and the tab that is showing it is remembered: the extension cannot blank
a tab the way the built-in browser does, so every later call on that tab is refused until it moves to an
allowed page or is closed.

A tab is remembered by its id, however a call named it. The controller's results do not say which tab a call
ran in (only ``tab_id`` in the command does), so for a call that named none Hermes reads the tab list and finds
the tab that is showing the withheld page; when that is not one tab, the record stays "the current tab,
unknown" and every call that could be on it is refused: one that names no tab, and one that names any tab.
Asking about the current tab is the same: when the connection controls one tab that is it, and when it controls
several Hermes cannot tell which a call without ``tab_id`` would use, so it refuses while any tab is withheld.

Facade-owned helpers are read through ``_bt`` (``tools.browser_tool``, resolved per call) — no import cycle.
"""

import json
import logging
import threading
import time
import urllib.parse
from collections import OrderedDict
from typing import Any, Optional

from tools import browser_tool_args as _args
from tools.browser_extension_router import RouteContext
from tools.browser_tool_origin import origin_module as _origin

logger = logging.getLogger(__name__)

# What a withheld batch result keeps: how far the batch got, never anything read from the page.
_BATCH_PROGRESS_KEYS = ("status", "stopped_reason", "completed", "remaining")
_STEP_KEYS = ("index", "action", "status", "error")
# Calls that can run on a withheld tab: they leave the page (or the tab) rather than use it. Everything else
# (snapshot, screenshot, click, type, scroll, ...) is refused there.
_LEAVING_ACTIONS = frozenset({
    "browser_navigate", "browser_back", "browser_tab_create", "browser_tabs", "browser_tab_close",
    "browser_tab_activate", "browser_tab_group", "browser_tab_ungroup",
})
_LEAVING_STEPS = frozenset({"navigate", "back"})
# A new tab's address is only known once its navigation commits; ask the browser for it a few times.
_SETTLE_POLLS = 6
_SETTLE_INTERVAL_S = 0.5
_UNSETTLED_URLS = frozenset({"", "about:blank"})
# Bounds on what is remembered: lanes, and withheld tabs per lane.
_MAX_LANES = 256
_MAX_TABS = 64
# The extension lists at most this many tabs, so a shorter list is the whole set and a longer one may not be.
_LISTING_CAP = 100
# The cloud-metadata floor resolves a hostname, and an address is checked on every result and for every listed
# tab, so each host's answer is kept this long rather than asking the gateway's resolver every time.
_FLOOR_TTL_S = 60.0
_MAX_FLOOR_HOSTS = 512


class _Withheld:
    """The tabs, per lane, that last showed a page the policy withheld: ``{lane key: {tab id or None: host}}``.
    ``None`` is the current tab when Hermes could not tell which tab id that is; a tab Hermes could identify is
    recorded under its id, whether the call named it or not."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lanes: OrderedDict[tuple, OrderedDict[Optional[int], str]] = OrderedDict()

    def add(self, key: tuple, tab: Optional[int], host: str) -> None:
        with self._lock:
            tabs = self._lanes.setdefault(key, OrderedDict())
            tabs.pop(tab, None)
            tabs[tab] = host
            while len(tabs) > _MAX_TABS:
                tabs.popitem(last=False)
            self._lanes.move_to_end(key)
            while len(self._lanes) > _MAX_LANES:
                self._lanes.popitem(last=False)

    def discard(self, key: tuple, tab: Optional[int]) -> None:
        with self._lock:
            tabs = self._lanes.get(key)
            if tabs is not None:
                tabs.pop(tab, None)
                if not tabs:
                    del self._lanes[key]

    def host(self, key: tuple, tab: Optional[int]) -> Optional[str]:
        """The host a call that names ``tab`` (``None``: names none) would be refused for, from what is
        recorded for the tab itself or for an unidentified current tab. A call that names a tab cannot be told
        apart from that unidentified tab, so it is refused too; whether a call that names none is on a
        recorded tab by id is for the caller to resolve."""
        with self._lock:
            tabs = self._lanes.get(key) or {}
            return tabs.get(tab) or tabs.get(None)

    def tabs(self, key: tuple) -> dict[int, str]:
        """The tabs recorded by id, ``{tab id: host}``."""
        with self._lock:
            return {tab: host for tab, host in (self._lanes.get(key) or {}).items() if tab is not None}

    def any(self, key: tuple) -> bool:
        with self._lock:
            return bool(self._lanes.get(key))

    def clear(self) -> None:
        with self._lock:
            self._lanes.clear()


_withheld = _Withheld()


class _FloorAnswers:
    """Recent answers of the cloud-metadata floor, per host: ``{host: (expires at, blocked)}``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._answers: OrderedDict[str, tuple[float, bool]] = OrderedDict()

    def blocked(self, url: str) -> bool:
        try:
            host = urllib.parse.urlsplit(url).hostname or ""
        except ValueError:
            host = ""
        if not host:
            return _origin()._is_always_blocked_url(url)
        now = time.monotonic()
        with self._lock:
            answer = self._answers.get(host)
            if answer is not None and answer[0] > now:
                return answer[1]
        blocked = bool(_origin()._is_always_blocked_url(url))
        with self._lock:
            self._answers[host] = (now + _FLOOR_TTL_S, blocked)
            self._answers.move_to_end(host)
            while len(self._answers) > _MAX_FLOOR_HOSTS:
                self._answers.popitem(last=False)
        return blocked

    def clear(self) -> None:
        with self._lock:
            self._answers.clear()


_floor = _FloorAnswers()


def reset_withheld() -> None:
    """Forget every withheld tab and every cached floor answer (tests; a new process starts empty)."""
    _withheld.clear()
    _floor.clear()


def _tab_arg(args: dict[str, Any]) -> Optional[int]:
    """The tab a call runs in (or, for the tab tools, is about): its ``tab_id`` when it names one, else
    ``None``, the current tab."""
    tab = args.get("tab_id")
    return tab if isinstance(tab, int) and not isinstance(tab, bool) and tab > 0 else None


def _decode(result: Any) -> Optional[dict[str, Any]]:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except ValueError:
            return None
    return payload if isinstance(payload, dict) else None


def _landed_url(action: str, payload: dict[str, Any]) -> str:
    """The URL an extension result says the call landed on, or ``""`` when it reports none."""
    if action == "browser_tab_create":
        payload = payload.get("tab") if isinstance(payload.get("tab"), dict) else {}
    url = payload.get("url")
    return url if isinstance(url, str) else ""


def _policy_error(url: str) -> Optional[dict]:
    """The block the landed-URL policy puts on ``url``: the cloud-metadata floor and the operator's website
    policy, but not the private-address rule, which the extension lane skips on purpose. The floor's answer for
    a host is reused for a minute (``_FloorAnswers``); the URL that was asked for is checked uncached before
    anything is sent."""
    return _origin()._url_policy_error(url, auto_local=True, always_blocked=_floor.blocked) if url else None


def _blocked_host(landed: str, policy_error: dict) -> str:
    """The host to name: the one the website policy matched, else the landed URL's. A hostname and nothing
    else, so a page cannot put prose in front of the model through its address."""
    try:
        host = (policy_error.get("blocked_by_policy") or {}).get("host") or urllib.parse.urlsplit(landed).hostname or ""
    except ValueError:  # a URL too malformed to name a host
        return ""
    return "".join(ch for ch in str(host) if ch.isascii() and (ch.isalnum() or ch in ".:-_"))[:253]


_STAYS_PUT = (" The tab is still on that page and Hermes will not read or act on it: go to another page "
              "(browser_navigate or browser_back) or close it (browser_tab_close).")


def redirect_blocked_result(action: str, payload: dict[str, Any], landed: str, policy_error: dict,
                            *, moved_back: bool = False) -> dict[str, Any]:
    """What the model gets in place of a page that landed on a blocked address. It names the host and nothing
    else the page or its URL said: no path, query, title, snapshot or text."""
    host = _blocked_host(landed, policy_error)
    named = f" ({host})" if host else ""
    result: dict[str, Any] = {
        "success": False,
        "code": "redirect_blocked",
        "error": f"Blocked: the page landed on a blocked address{named}. Its content was withheld, and the action "
                 "itself has already run. Do not continue on this page: do not act on it or read it.",
        "blocked_host": host,
    }
    if moved_back:
        result["error"] += " The tab was taken back to the previous page."
    elif action == "browser_tab_create":
        result["error"] += " The tab stays open; browser_tab_close closes it. Hermes will not read or act on it."
    else:
        result["error"] += _STAYS_PUT
    matched = policy_error.get("blocked_by_policy")
    if isinstance(matched, dict):
        result["blocked_by_policy"] = {key: matched.get(key) for key in ("host", "rule", "source")}
    if action == "browser_tab_create":
        tab_id = (payload.get("tab") or {}).get("id")
        if isinstance(tab_id, int) and not isinstance(tab_id, bool):
            result["tab_id"] = tab_id
    elif action == "browser_batch":
        result.update({key: payload[key] for key in _BATCH_PROGRESS_KEYS if key in payload})
        if isinstance(payload.get("steps"), list):
            result["steps"] = [{key: step[key] for key in _STEP_KEYS if key in step}
                               for step in payload["steps"] if isinstance(step, dict)]
    return result


def _int_id(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _tab_listing_url(listing: Any, tab_id: int) -> str:
    payload = _decode(listing) or {}
    for tab in payload.get("tabs") if isinstance(payload.get("tabs"), list) else []:
        if isinstance(tab, dict) and tab.get("id") == tab_id:
            return tab["url"] if isinstance(tab.get("url"), str) else ""
    return ""


def _settle_new_tab(route: RouteContext, tab_id: int) -> str:
    """The address a new tab committed to. The controller reports a created tab the moment it exists, before
    any redirect has run, so the tab is read back until it has an address (``""`` when it never does or the
    controller cannot list tabs)."""
    for attempt in range(_SETTLE_POLLS):
        if attempt:
            time.sleep(_SETTLE_INTERVAL_S)
        try:
            url = _tab_listing_url(route.send("browser_tabs", {}), tab_id)
        except Exception:
            logger.debug("could not read back new tab %s", tab_id, exc_info=True)
            return ""
        if url not in _UNSETTLED_URLS:
            return url
    return ""


def _move_back(route: RouteContext, tab: Optional[int]) -> bool:
    """Take a tab off a withheld page with one browser_back, the nearest thing to the built-in lane's
    about:blank (the extension only navigates to http(s) pages). True when it landed on an allowed page."""
    try:
        outcome = _decode(route.send("browser_back", {} if tab is None else {"tab_id": tab})) or {}
    except Exception:
        logger.debug("could not move a withheld tab off its page", exc_info=True)
        return False
    landed = _landed_url("browser_back", outcome)
    return bool(landed) and _policy_error(landed) is None


def _moves_the_page(action: str, args: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Whether this call may itself have taken the tab to the page it reports (so moving back undoes it). A
    read, or a tab that was already there, is not moved: it is only remembered."""
    if action in ("browser_navigate", "browser_back"):
        return True
    if action == "browser_batch":
        steps = args.get("actions") if isinstance(args.get("actions"), list) else []
        return any(_args.step_action(step) in _LEAVING_STEPS for step in steps)
    return payload.get("navigated") is True


def _guard_tab_list(key: tuple, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The tab list with each tab on a blocked address replaced by its id and host, or ``None`` when no tab
    was. Also keeps the withheld tabs current: a tab seen on an allowed address is no longer withheld."""
    tabs = payload.get("tabs")
    if not isinstance(tabs, list):
        return None
    listed, changed = [], False
    for tab in tabs:
        url = tab.get("url") if isinstance(tab, dict) and isinstance(tab.get("url"), str) else ""
        tab_id = tab.get("id") if isinstance(tab, dict) and isinstance(tab.get("id"), int) \
            and not isinstance(tab.get("id"), bool) else None
        error = _policy_error(url)
        if error is None:
            if tab_id is not None and url:
                _withheld.discard(key, tab_id)
            listed.append(tab)
            continue
        host = _blocked_host(url, error)
        if tab_id is not None:
            _withheld.add(key, tab_id, host)
        listed.append({"id": tab.get("id"), "active": tab.get("active"), "withheld": True, "blocked_host": host})
        changed = True
    return {**payload, "tabs": listed} if changed else None


def _read_tabs(route: RouteContext) -> Optional[dict[str, Any]]:
    """The controller's tab list (``{"tabs": [...]}``), or ``None`` when it cannot be read."""
    try:
        payload = _decode(route.send("browser_tabs", {}))
    except Exception:
        logger.debug("could not read the tab list", exc_info=True)
        return None
    return payload if payload is not None and isinstance(payload.get("tabs"), list) else None


def _sync_record(key: tuple, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Bring the withheld record in line with a tab list, which says what each tab shows right now: a tab on a
    blocked address is recorded by id, one that moved on or was closed is lifted, and an unidentified current
    tab is lifted once no listed tab is on a blocked address. Returns the list with blocked tabs reduced to
    their id and host (``None`` when there are none), as :func:`_guard_tab_list` does."""
    redacted = _guard_tab_list(key, payload)
    tabs = payload["tabs"]
    if len(tabs) < _LISTING_CAP:
        listed = {_int_id(tab.get("id")) for tab in tabs if isinstance(tab, dict)}
        for gone in set(_withheld.tabs(key)) - listed:
            _withheld.discard(key, gone)
        if not set(_withheld.tabs(key)) & listed:
            _withheld.discard(key, None)
    return redacted


def _identify_withheld_tab(route: RouteContext, blocked_host: str) -> Optional[int]:
    """The id of the tab a call that named none just withheld. The controller's results do not carry it, so the
    tab list is read: the tab on a blocked address at ``blocked_host``, when exactly one is. Every tab on a
    blocked address is recorded by id while the list is at hand. ``None`` when the list cannot be read or does
    not point at one tab."""
    payload = _read_tabs(route)
    if payload is None:
        return None
    _sync_record(route.key, payload)
    candidates = []
    for tab in payload["tabs"]:
        url = tab.get("url") if isinstance(tab, dict) and isinstance(tab.get("url"), str) else ""
        error = _policy_error(url)
        if error is not None and _int_id(tab.get("id")) and _blocked_host(url, error) == blocked_host:
            candidates.append(tab["id"])
    return candidates[0] if len(candidates) == 1 else None


def extension_landing_guard(action: str, args: dict[str, Any], result: Any, route: Optional[RouteContext] = None) -> Any:
    """Router result guard for the extension lane. Any address the controller reports (the page a navigation,
    click, key press or read ended on, a new tab's, each tab in a tab list) goes through the policy the
    built-in lane applies after a redirect: the cloud-metadata floor and the operator's website policy, but
    not the private-address rule, which the extension lane skips on purpose. A page that fails it is withheld
    whole, so nothing it showed reaches the model, and its tab is remembered so later calls on it are refused
    (see :func:`extension_withheld_guard`). Anything that reports no address passes through untouched."""
    payload = _decode(result)
    if payload is None:
        return result
    key = route.key if route is not None else ()
    tab = _tab_arg(args)
    _bt = _origin()
    dump = (lambda value: _bt._dumps(value)) if isinstance(result, str) else (lambda value: value)

    if action == "browser_tab_close":
        _withheld.discard(key, _tab_arg(args))
        return result
    if action == "browser_tabs":
        listed = _sync_record(key, payload) if isinstance(payload.get("tabs"), list) else None
        return result if listed is None else dump(listed)

    landed = _landed_url(action, payload)
    if action == "browser_tab_create":
        new_tab = payload.get("tab") if isinstance(payload.get("tab"), dict) else {}
        tab = new_tab.get("id") if isinstance(new_tab.get("id"), int) and not isinstance(new_tab.get("id"), bool) else None
        if not landed and tab is not None and route is not None:
            landed = _settle_new_tab(route, tab)
    policy_error = _policy_error(landed)
    if policy_error is None:
        if landed:
            _withheld.discard(key, tab)
            if action == "browser_tab_create" and not _landed_url(action, payload):
                return dump({**payload, "tab": {**payload["tab"], "url": landed}})
        return result

    blocked_host = _blocked_host(landed, policy_error)
    if tab is None and route is not None and action != "browser_tab_create":
        tab = _identify_withheld_tab(route, blocked_host)  # stays None when it cannot be told
    _withheld.add(key, tab, blocked_host)
    moved_back = False
    if route is not None and action != "browser_tab_create" and _moves_the_page(action, args, payload):
        moved_back = _move_back(route, tab)
        if moved_back:
            _withheld.discard(key, tab)
    blocked = redirect_blocked_result(action, payload, landed, policy_error, moved_back=moved_back)
    logger.warning("%s landed on a blocked address (host %s); page withheld", action, blocked["blocked_host"])
    return dump(blocked)


def extension_withheld_guard(action: str, args: dict[str, Any], route: Optional[RouteContext] = None) -> Optional[str]:
    """Router guard for the extension lane: a refusal (JSON) for a call that would use a tab still showing a
    page the landed-URL policy withheld, else ``None``. Calls that leave the page or the tab (navigate, back,
    open, list, close, switch, group) go through, and so does a batch that starts by leaving the page, whose
    own result is judged again."""
    if route is None or not _withheld.any(route.key) or action in _LEAVING_ACTIONS:
        return None
    if action == "browser_batch":
        steps = args.get("actions") if isinstance(args.get("actions"), list) else []
        if steps and _args.step_action(steps[0]) in _LEAVING_STEPS:
            return None
    tab = _tab_arg(args)
    host = _withheld_host(route, tab)
    if host is None:
        return None
    _bt = _origin()
    named = f" ({host})" if host else ""
    pick = " Pass tab_id to act on a different tab." if tab is None else ""
    return _bt._dumps(_bt._err(
        f"Blocked: this tab is still on a page Hermes withheld{named}, so {action} was not sent. Go to another "
        f"page with browser_navigate or browser_back, or close the tab with browser_tab_close.{pick}",
        code="tab_withheld", blocked_host=host))


def _withheld_host(route: RouteContext, tab: Optional[int]) -> Optional[str]:
    """The host a call is refused for, or ``None`` when it may go ahead. A call that names a tab is refused
    when that tab is recorded, or when the current tab is recorded but unidentified. A call that names none
    acts on the current tab, whose id is not known here: with one tab under control that is it, and with
    several the call is refused while any of them is withheld, because it could be on that one."""
    host = _withheld.host(route.key, tab)
    if host is not None or tab is not None:
        return host
    payload = _read_tabs(route)
    if payload is not None:
        _sync_record(route.key, payload)
        if len(payload["tabs"]) == 1:
            return _withheld.tabs(route.key).get(_int_id(payload["tabs"][0].get("id")))
    return next(iter(_withheld.tabs(route.key).values()), None)
